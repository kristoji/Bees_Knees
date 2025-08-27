"""Compute clustering centroids directly from previously cached sequential LLM dataset(s).

Why:
  The original `train_clustering.py` expects a flat CSV. But the LLM data
  generation pipeline already persisted rich samples (pickled list of dicts)
  containing board & move embedding sequences. Recomputing those embeddings
  just to cluster is wasteful.

What this script does:
  1. Load one or more sequential cache pickle files (each is a list of samples).
  2. Extract embeddings for boards and/or moves (configurable) into flat numpy arrays.
  3. Optionally subsample (fraction or max count) to control memory/time.
  4. Run KMeans over a k-range (default 2..64) with k-means++ init & multiple n_init.
  5. Score each k via silhouette (and optionally inertia) selecting the best.
  6. Persist centroids + metadata to pickle + companion txt summary.
  7. Optionally augment the input caches with computed cluster assignments.

Outputs:
  For each requested embedding type (board, move) you get:
    clustering_models/<type>/cluster_centroids_kmeans_best.pkl
    clustering_models/<type>/cluster_centroids_kmeans_best.txt

  If --augment-caches is specified, also creates:
    <original_cache_name>_clustered.pkl (or custom suffix)

The pickle dict schema:
  {
    'centroids': np.ndarray [k, d],
    'k': int,
    'metrics': { 'silhouette_per_k': {k: val}, 'inertia_per_k': {...} },
    'scaler': {'mean': mean_vec, 'scale': scale_vec}  # if scaling applied
    'embedding_type': 'board' | 'move',
    'created_from_caches': [list of cache paths],
    'created_at': iso timestamp,
    'feature_dim': d,
  }

Usage examples:
  # Basic clustering without cache augmentation
  python -m src.compute_centroids_from_cache \
      --caches data/sequential_hive_llm_dataset/train_sequential_cache.pkl \
               data/sequential_hive_llm_dataset/validation_sequential_cache.pkl \
      --types board move \
      --k-min 8 --k-max 64 \
      --sample-fraction 0.5

  # Clustering with different sample fractions per type
  python -m src.compute_centroids_from_cache \
      --caches data/sequential_hive_llm_dataset/train_sequential_cache.pkl \
      --types board move \
      --board-sample-fraction 0.3 \
      --move-sample-fraction 0.7

  # Clustering with automatic cache augmentation
  python -m src.compute_centroids_from_cache \
      --caches data/sequential_hive_llm_dataset/train_sequential_cache.pkl \
      --types board move \
      --augment-caches \
      --augment-output-suffix _with_clusters \
      --similarity-metric cosine

  # Clustering with subset fitting for large datasets
  python -m src.compute_centroids_from_cache \
      --caches data/sequential_hive_llm_dataset/train_sequential_cache.pkl \
      --types board move \
      --fit-on-subset \
      --subset-fraction 0.1 \
      --subset-max 50000

Note: With --augment-caches, you no longer need to run the separate 
augment_sequential_cache_with_clusters.py script.

The --fit-on-subset option is useful for very large datasets where KMeans 
computation becomes prohibitive. It fits the clustering model on a subset 
of the data but still assigns cluster IDs to all data points. Silhouette 
scores are computed on the full dataset for proper evaluation.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def _load_cache(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _gather_embeddings(samples: List[dict], embedding_type: str) -> np.ndarray:
    """Flatten sequences into a single 2D numpy array [N, D].

    embedding_type:
      'board' -> uses sample['board_embeddings_sequence'] (shape [T, *, D]).
      'move'  -> uses sample['chosen_move_embeddings_sequence'].
    Accepts either numpy arrays or torch tensors; converts to numpy float32.
    """
    collected = []
    for s in samples:
        key = 'board_embeddings_sequence' if embedding_type == 'board' else 'chosen_move_embeddings_sequence'
        arr = s.get(key)
        if arr is None:
            continue
        # Convert torch -> numpy if needed
        try:
            import torch  # local import to avoid hard dependency if not installed at runtime
            if isinstance(arr, torch.Tensor):
                arr_np = arr.detach().cpu().numpy()
            else:
                arr_np = np.asarray(arr)
        except Exception:
            arr_np = np.asarray(arr)

        # Collapse possible singleton dimensions (e.g., [T,1,D])
        if arr_np.ndim == 3 and arr_np.shape[1] == 1:
            arr_np = arr_np[:, 0, :]
        elif arr_np.ndim != 2:
            # Skip unexpected shapes
            continue
        collected.append(arr_np)

    if not collected:
        raise ValueError(f"No embeddings collected for type '{embedding_type}'.")
    all_emb = np.concatenate(collected, axis=0)
    return all_emb.astype(np.float32)


def _subsample(X: np.ndarray, sample_fraction: float, sample_max: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    target = n
    if 0 < sample_fraction < 1.0:
        target = int(n * sample_fraction)
    if sample_max and sample_max > 0:
        target = min(target, sample_max)
    if target < n:
        idx = rng.choice(n, target, replace=False)
        return X[idx]
    return X


def _cluster(X: np.ndarray, k_min: int, k_max: int, n_init: int, max_iter: int, random_state: int, 
             fit_on_subset: bool = False, subset_fraction: float = 0.1, subset_max: int = 50000) -> Tuple[np.ndarray, Dict]:
    """Run KMeans for k in [k_min, k_max] and select best by silhouette.
    
    Args:
        X: Full dataset [N, D]
        fit_on_subset: If True, fit KMeans only on a subset but evaluate silhouette on full data
        subset_fraction: Fraction of data to use for fitting when fit_on_subset=True
        subset_max: Maximum number of samples for fitting when fit_on_subset=True
    """
    rng = np.random.default_rng(random_state)
    
    # Determine subset for fitting if requested
    if fit_on_subset:
        n_total = X.shape[0]
        n_subset = min(int(n_total * subset_fraction), subset_max)
        if n_subset < n_total:
            subset_idx = rng.choice(n_total, n_subset, replace=False)
            X_fit = X[subset_idx]
            print(f"  Fitting KMeans on subset: {n_subset}/{n_total} samples ({n_subset/n_total*100:.1f}%)")
        else:
            X_fit = X
            print(f"  Subset size >= total size, using all data for fitting")
    else:
        X_fit = X
    
    sil_scores = {}
    inertia_scores = {}
    models = {}

    for k in tqdm(range(k_min, k_max + 1), desc="Clustering k sweep"):
        # Fit on subset (or full data)
        km = KMeans(n_clusters=k, init='k-means++', n_init=n_init, max_iter=max_iter, random_state=random_state, verbose=0)
        km.fit(X_fit)
        
        # Always predict on full data for silhouette evaluation
        labels = km.predict(X)
        
        try:
            # Evaluate silhouette on full dataset
            sil = silhouette_score(X, labels)
        except Exception:
            sil = float('nan')
        sil_scores[k] = sil
        
        # Store inertia from the fitted model (on subset if applicable)
        inertia_scores[k] = km.inertia_
        models[k] = km

    # Choose best k (highest silhouette; fallback to lowest inertia if all nan)
    valid_sil = {k: v for k, v in sil_scores.items() if not np.isnan(v)}
    if valid_sil:
        best_k = max(valid_sil, key=valid_sil.get)
    else:
        best_k = min(inertia_scores, key=inertia_scores.get)

    best_model = models[best_k]
    meta = {
        'silhouette_per_k': sil_scores,
        'inertia_per_k': inertia_scores,
        'best_k': best_k,
        'best_silhouette': sil_scores.get(best_k),
        'best_inertia': inertia_scores.get(best_k),
        'fit_on_subset': fit_on_subset,
        'subset_info': {
            'subset_fraction': subset_fraction,
            'subset_max': subset_max,
            'actual_subset_size': X_fit.shape[0] if fit_on_subset else X.shape[0],
            'total_size': X.shape[0]
        } if fit_on_subset else None
    }
    return best_model.cluster_centers_.astype(np.float32), meta


def _save_outputs(centroids: np.ndarray, meta: Dict, embedding_type: str, caches: List[str], scaler: StandardScaler, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, 'cluster_centroids_kmeans_best.pkl')
    txt_path = os.path.join(out_dir, 'cluster_centroids_kmeans_best.txt')

    payload = {
        'centroids': centroids,
        'k': meta['best_k'],
        'metrics': meta,
        'scaler': {'mean': scaler.mean_.astype(np.float32), 'scale': scaler.scale_.astype(np.float32)},
        'embedding_type': embedding_type,
        'created_from_caches': caches,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'feature_dim': centroids.shape[1],
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(payload, f)

    with open(txt_path, 'w') as f:
        f.write(f"CLUSTERING SUMMARY ({embedding_type})\n")
        f.write('=' * 60 + '\n')
        f.write(f"Created at (UTC): {payload['created_at']}\n")
        f.write(f"Caches: {len(caches)} files\n")
        for c in caches:
            f.write(f"  - {c}\n")
        f.write(f"Feature dim: {payload['feature_dim']}\n")
        f.write(f"Best k: {payload['k']}\n")
        f.write(f"Best silhouette: {payload['metrics']['best_silhouette']}\n")
        f.write(f"Best inertia: {payload['metrics']['best_inertia']}\n")
        
        # Add subset fitting information
        if payload['metrics'].get('fit_on_subset'):
            subset_info = payload['metrics']['subset_info']
            f.write(f"\nSubset fitting: ENABLED\n")
            f.write(f"Subset fraction: {subset_info['subset_fraction']}\n")
            f.write(f"Subset max: {subset_info['subset_max']}\n")
            f.write(f"Actual subset size: {subset_info['actual_subset_size']}\n")
            f.write(f"Total data size: {subset_info['total_size']}\n")
            f.write(f"Subset percentage: {subset_info['actual_subset_size']/subset_info['total_size']*100:.1f}%\n")
        else:
            f.write(f"\nSubset fitting: DISABLED (fitted on all data)\n")
        
        f.write('\nSilhouette per k:\n')
        for k, v in sorted(payload['metrics']['silhouette_per_k'].items()):
            f.write(f"  k={k}: {v}\n")
        f.write('\nInertia per k:\n')
        for k, v in sorted(payload['metrics']['inertia_per_k'].items()):
            f.write(f"  k={k}: {v}\n")
        f.write('\nScaler mean (first 8): ' + ', '.join(f"{m:.4f}" for m in payload['scaler']['mean'][:8]) + '\n')
        f.write('Scaler scale (first 8): ' + ', '.join(f"{s:.4f}" for s in payload['scaler']['scale'][:8]) + '\n')

    print(f"Saved centroids: {pkl_path}")
    print(f"Saved summary:   {txt_path}")


def main():  # noqa: C901
    ap = argparse.ArgumentParser(description="Compute KMeans centroids from sequential cache(s) without regenerating embeddings")
    ap.add_argument('--caches', nargs='+', required=True, help='One or more sequential cache pickle paths')
    ap.add_argument('--types', nargs='+', default=['board'], choices=['board', 'move'], help='Embedding types to cluster')
    # Global fallback k range
    ap.add_argument('--k-min', type=int, default=11, help='Global minimum k (used if per-type not provided)')
    ap.add_argument('--k-max', type=int, default=16, help='Global maximum k (used if per-type not provided)')
    # Optional per-type overrides
    ap.add_argument('--board-k-min', type=int, default=10, help='Override minimum k for board embeddings')
    ap.add_argument('--board-k-max', type=int, default=12, help='Override maximum k for board embeddings')
    ap.add_argument('--move-k-min', type=int, default=10, help='Override minimum k for move embeddings')
    ap.add_argument('--move-k-max', type=int, default=14, help='Override maximum k for move embeddings')
    ap.add_argument('--n-init', type=int, default=8)
    ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--sample-fraction', type=float, default=1.0, help='Global fraction of embeddings to use (0-1] (used if per-type not provided)')
    ap.add_argument('--sample-max', type=int, default=0, help='Absolute cap on samples (0 = no cap)')
    # Optional per-type sample fraction overrides
    ap.add_argument('--board-sample-fraction', type=float, default=None, help='Override sample fraction for board embeddings')
    ap.add_argument('--move-sample-fraction', type=float, default=None, help='Override sample fraction for move embeddings')
    ap.add_argument('--random-seed', type=int, default=42)
    ap.add_argument('--output-root', type=str, default='models/clustering', help='Root folder for output subdirs')
    
    # New arguments for automatic cache augmentation
    ap.add_argument('--augment-caches', action='store_true', default=False, 
                    help='Automatically augment input caches with computed cluster assignments')
    ap.add_argument('--augment-output-suffix', type=str, default='_clustered', 
                    help='Suffix to add to cache filenames when saving augmented versions')
    ap.add_argument('--similarity-metric', type=str, default='cosine', choices=['cosine', 'euclidean', 'dot'],
                    help='Similarity metric for cluster assignment')
    
    # New arguments for subset clustering
    ap.add_argument('--fit-on-subset', action='store_true', default=False,
                    help='Fit KMeans only on a subset of data but assign clusters to all data')
    ap.add_argument('--subset-fraction', type=float, default=0.1,
                    help='Fraction of data to use for fitting KMeans when --fit-on-subset is enabled (default: 0.1)')
    ap.add_argument('--subset-max', type=int, default=50000,
                    help='Maximum number of samples for fitting when --fit-on-subset is enabled (default: 50000)')
    
    args = ap.parse_args()

    rng = np.random.default_rng(args.random_seed)
    all_samples = []
    print("Loading caches ...")
    for path in args.caches:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        samples = _load_cache(path)
        print(f"  {path}: {len(samples)} samples")
        all_samples.extend(samples)
    print(f"Total aggregated samples: {len(all_samples)}")

    # Store computed centroids for potential cache augmentation
    computed_centroids = {}

    for emb_type in args.types:
        # Determine local k range
        if emb_type == 'board':
            k_min = args.board_k_min if args.board_k_min is not None else args.k_min
            k_max = args.board_k_max if args.board_k_max is not None else args.k_max
            sample_fraction = args.board_sample_fraction if args.board_sample_fraction is not None else args.sample_fraction
        else:  # move
            k_min = args.move_k_min if args.move_k_min is not None else args.k_min
            k_max = args.move_k_max if args.move_k_max is not None else args.k_max
            sample_fraction = args.move_sample_fraction if args.move_sample_fraction is not None else args.sample_fraction
        if k_min > k_max:
            raise ValueError(f"Invalid k range for {emb_type}: k_min ({k_min}) > k_max ({k_max})")

        print(f"\n=== Processing embedding type: {emb_type} (k range {k_min}-{k_max}, sample fraction {sample_fraction}) ===")
        X = _gather_embeddings(all_samples, emb_type)
        print(f"Raw collected shape: {X.shape}")
        X = _subsample(X, sample_fraction, args.sample_max, rng)
        print(f"After subsample: {X.shape}")

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("Running KMeans sweep ...")
        start = time.time()
        centroids, meta = _cluster(X_scaled, k_min, k_max, args.n_init, args.max_iter, args.random_seed,
                                 fit_on_subset=args.fit_on_subset, 
                                 subset_fraction=args.subset_fraction,
                                 subset_max=args.subset_max)
        elapsed = time.time() - start
        print(f"Best k={meta['best_k']} (sil={meta['best_silhouette']}) in {elapsed:.1f}s")

        # Transform centroids back to original scale for downstream cosine use
        centroids_unscaled = centroids * scaler.scale_ + scaler.mean_
        out_dir = os.path.join(args.output_root, f"{emb_type}s")
        _save_outputs(centroids_unscaled, meta, emb_type, args.caches, scaler, out_dir)
        
        # Store for potential cache augmentation
        if args.augment_caches:
            computed_centroids[emb_type] = centroids_unscaled

    # Augment caches if requested
    if args.augment_caches and computed_centroids:
        print(f"\n=== Augmenting caches with computed cluster assignments ===")
        _augment_caches_with_clusters(
            cache_paths=args.caches,
            computed_centroids=computed_centroids,
            similarity_metric=args.similarity_metric,
            output_suffix=args.augment_output_suffix
        )

    print("\nAll requested clustering jobs completed.")


def _augment_caches_with_clusters(cache_paths: List[str], computed_centroids: Dict[str, np.ndarray], 
                                 similarity_metric: str, output_suffix: str):
    """Augment input caches with cluster assignments using computed centroids."""
    
    try:
        import torch
        _HAVE_TORCH = True
    except ImportError:
        _HAVE_TORCH = False
    
    def _to_numpy(x):
        if x is None:
            return None
        if _HAVE_TORCH and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _reshape_2d(arr: np.ndarray) -> Optional[np.ndarray]:
        if arr is None:
            return None
        if arr.ndim == 3 and arr.shape[1] == 1:  # [T,1,D]
            return arr[:, 0, :]
        if arr.ndim == 2:
            return arr
        return None  # unexpected shape

    def _assign(emb: np.ndarray, cents: np.ndarray, metric: str) -> np.ndarray:
        # emb: [N,D]; cents: [K,D]
        if metric == 'cosine':
            # normalize
            emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
            cent_n = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
            sims = emb_n @ cent_n.T  # [N,K]
            return sims.argmax(axis=1).astype(np.int32)
        elif metric == 'euclidean':
            # (a-b)^2 = a^2 + b^2 - 2ab
            a2 = (emb**2).sum(axis=1, keepdims=True)
            b2 = (cents**2).sum(axis=1)[None, :]
            d2 = a2 + b2 - 2 * (emb @ cents.T)
            return d2.argmin(axis=1).astype(np.int32)
        else:  # dot
            scores = emb @ cents.T
            return scores.argmax(axis=1).astype(np.int32)

    def _assign_nested(list_of_emb_arrays, cents: np.ndarray, metric: str) -> List[List[int]]:
        out: List[List[int]] = []
        for arr in list_of_emb_arrays:
            if arr is None or len(arr) == 0:
                out.append([])
                continue
            arr_np = _to_numpy(arr)
            if arr_np.ndim == 3 and arr_np.shape[1] == 1:
                arr_np = arr_np[:, 0, :]
            if arr_np.ndim != 2:
                out.append([])
                continue
            assignments = _assign(arr_np, cents, metric)
            out.append(assignments.tolist())
        return out
    
    for cache_path in cache_paths:
        print(f"Augmenting cache: {cache_path}")
        
        # Load original cache
        with open(cache_path, 'rb') as f:
            samples = pickle.load(f)

        # Remove any previous centroid mapping entries to avoid duplicates
        # (identified by sentinel key 'is_cluster_centroid_mapping')
        original_len = len(samples)
        samples = [s for s in samples if not (isinstance(s, dict) and s.get('is_cluster_centroid_mapping'))]
        removed = original_len - len(samples)
        if removed > 0:
            print(f"  Removed {removed} existing centroid mapping entr(y/ies) to prevent duplication")
        
        # Augment each sample with cluster assignments
        for sample in samples:
            # Board cluster assignments
            if 'board' in computed_centroids:
                board_cents = computed_centroids['board']
                
                # Board embeddings sequence
                board_emb = sample.get('board_embeddings_sequence')
                if board_emb is not None:
                    board_emb_np = _to_numpy(board_emb)
                    board_emb_2d = _reshape_2d(board_emb_np)
                    if board_emb_2d is not None:
                        board_clusters = _assign(board_emb_2d, board_cents, similarity_metric)
                        sample['board_cluster_ids_sequence'] = board_clusters
                
                # Next board embeddings sequence
                next_board_emb = sample.get('next_board_embeddings_sequence')
                if next_board_emb is not None:
                    next_board_emb_np = _to_numpy(next_board_emb)
                    next_board_emb_2d = _reshape_2d(next_board_emb_np)
                    if next_board_emb_2d is not None:
                        next_board_clusters = _assign(next_board_emb_2d, board_cents, similarity_metric)
                        sample['next_board_cluster_ids_sequence'] = next_board_clusters
            
            # Move cluster assignments  
            if 'move' in computed_centroids:
                move_cents = computed_centroids['move']
                
                # Chosen move embeddings sequence
                move_emb = sample.get('chosen_move_embeddings_sequence')
                if move_emb is not None:
                    move_emb_np = _to_numpy(move_emb)
                    move_emb_2d = _reshape_2d(move_emb_np)
                    if move_emb_2d is not None:
                        # Check dimension compatibility
                        if move_emb_2d.shape[1] == move_cents.shape[1]:
                            move_clusters = _assign(move_emb_2d, move_cents, similarity_metric)
                            sample['chosen_move_cluster_ids_sequence'] = move_clusters
                        else:
                            print(f"  Warning: Move embedding dimension ({move_emb_2d.shape[1]}) != centroid dimension ({move_cents.shape[1]})")
                            sample['chosen_move_cluster_ids_sequence'] = np.full(len(move_emb_2d), -1, dtype=np.int32)
                
                # Legal move embeddings sequence (if available)
                legal_move_emb_seq = sample.get('legal_move_embeddings_sequence')
                if legal_move_emb_seq is not None:
                    legal_clusters = _assign_nested(legal_move_emb_seq, move_cents, similarity_metric)
                    sample['legal_move_cluster_ids_sequence'] = legal_clusters
            
            # Add metadata
            sample['similarity_metric'] = similarity_metric
            sample['num_board_centroids'] = len(computed_centroids.get('board', [])) if 'board' in computed_centroids else 0
            sample['num_move_centroids'] = len(computed_centroids.get('move', [])) if 'move' in computed_centroids else 0
        
        # Save augmented cache
        base_path = os.path.splitext(cache_path)[0]
        ext = os.path.splitext(cache_path)[1]
        output_path = f"{base_path}{output_suffix}{ext}"
        
        with open(output_path, 'wb') as f:
            pickle.dump(samples, f)
        
        print(f"  Saved augmented cache: {output_path}")

        # Append (in a second step) a single centroid mapping entry so that
        # downstream data loaders can quickly access centroid embeddings
        # without reloading separate clustering artifacts. This mapping is
        # global for the dataset, so we store it once.
        mapping_entry = {
            'is_cluster_centroid_mapping': True,
            'similarity_metric': similarity_metric,
            'created_at': datetime.utcnow().isoformat() + 'Z',
        }
        if 'board' in computed_centroids:
            board_cents = computed_centroids['board']
            # Provide both array and explicit id->embedding dict (list of lists) for flexibility
            mapping_entry['board_centroids'] = board_cents.astype(np.float32)
            mapping_entry['board_cluster_id_to_embedding'] = {int(i): board_cents[i].astype(np.float32) for i in range(board_cents.shape[0])}
            mapping_entry['num_board_centroids'] = board_cents.shape[0]
            mapping_entry['board_embedding_dim'] = board_cents.shape[1]
        if 'move' in computed_centroids:
            move_cents = computed_centroids['move']
            mapping_entry['move_centroids'] = move_cents.astype(np.float32)
            mapping_entry['move_cluster_id_to_embedding'] = {int(i): move_cents[i].astype(np.float32) for i in range(move_cents.shape[0])}
            mapping_entry['num_move_centroids'] = move_cents.shape[0]
            mapping_entry['move_embedding_dim'] = move_cents.shape[1]

        # Save mapping entry as a separate pickle alongside augmented cache for easier direct loading
        mapping_pickle_path = f"{base_path}{output_suffix}_centroid_mapping.pkl"
        with open(mapping_pickle_path, 'wb') as f:
            pickle.dump(mapping_entry, f)
        print(f"  Saved centroid mapping entry: {mapping_pickle_path}")


if __name__ == '__main__':
    main()
