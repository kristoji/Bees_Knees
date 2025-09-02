#!/usr/bin/env python3
import argparse
import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class MoveAnalysis:
    """Container for move analysis results."""
    move_norm: float
    state_distance: float
    next_state_distance: float
    move_original: np.ndarray
    move_projected: np.ndarray
    current_board: np.ndarray
    next_board: np.ndarray
    cosine_sim_to_projection: float


class MoveAutoencoder(nn.Module):
    """Autoencoder for board state transitions with latent space similar to move embeddings."""
    
    def __init__(self, board_dim: int, latent_dim: int):
        super().__init__()
        input_dim = board_dim * 2  # Concatenated current and next board
        hidden_dim = latent_dim  # Use the embedding dimension as hidden dimension
        
        # Encoder (from concatenated boards to latent space)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder (from latent space back to concatenated boards)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, current_board: torch.Tensor, next_board: torch.Tensor):
        # Concatenate board states
        concatenated = torch.cat([current_board, next_board], dim=-1)
        
        # Encode to latent representation
        latent = self.encoder(concatenated)
        
        # Decode back to concatenated board states
        reconstructed = self.decoder(latent)
        
        # Split the reconstruction back into current and next board
        board_dim = current_board.shape[1]
        reconstructed_current = reconstructed[:, :board_dim]
        reconstructed_next = reconstructed[:, board_dim:]
        
        return latent, reconstructed_current, reconstructed_next
    
    def encode(self, current_board: torch.Tensor, next_board: torch.Tensor) -> torch.Tensor:
        """Encode board states to latent space only."""
        concatenated = torch.cat([current_board, next_board], dim=-1)
        return self.encoder(concatenated)


def load_and_extract_embeddings(cache_path: str, sample_fraction: float = 1.0) -> Tuple[np.ndarray, ...]:
    """Load cache and extract embedding sequences."""
    print(f"Loading cache from {cache_path}")
    
    with open(cache_path, 'rb') as f:
        samples = pickle.load(f)
    
    current_boards = []
    nextBoards = []
    moves = []
    
    for sample in samples:
        board_emb = sample.get('board_embeddings_sequence')
        next_board_emb = sample.get('next_board_embeddings_sequence')
        move_emb = sample.get('chosen_move_embeddings_sequence')
        
        if board_emb is None or next_board_emb is None or move_emb is None:
            continue
        
        # Convert to numpy
        if hasattr(board_emb, 'detach'):
            board_emb = board_emb.detach().cpu().numpy()
        if hasattr(next_board_emb, 'detach'):
            next_board_emb = next_board_emb.detach().cpu().numpy()
        if hasattr(move_emb, 'detach'):
            move_emb = move_emb.detach().cpu().numpy()
        
        board_emb = np.asarray(board_emb)
        next_board_emb = np.asarray(next_board_emb)
        move_emb = np.asarray(move_emb)
        
        # Handle 3D arrays
        if board_emb.ndim == 3 and board_emb.shape[1] == 1:
            board_emb = board_emb[:, 0, :]
        if next_board_emb.ndim == 3 and next_board_emb.shape[1] == 1:
            next_board_emb = next_board_emb[:, 0, :]
        if move_emb.ndim == 3 and move_emb.shape[1] == 1:
            move_emb = move_emb[:, 0, :]
        
        min_len = min(len(board_emb), len(next_board_emb), len(move_emb))
        if min_len > 0:
            current_boards.append(board_emb[:min_len])
            nextBoards.append(next_board_emb[:min_len])
            moves.append(move_emb[:min_len])
    
    current_boards = np.concatenate(current_boards, axis=0).astype(np.float32)
    next_boards = np.concatenate(nextBoards, axis=0).astype(np.float32)
    moves = np.concatenate(moves, axis=0).astype(np.float32)
    
    # Subsample if requested
    if sample_fraction < 1.0:
        n_samples = int(len(current_boards) * sample_fraction)
        idx = np.random.choice(len(current_boards), n_samples, replace=False)
        current_boards = current_boards[idx]
        next_boards = next_boards[idx]
        moves = moves[idx]
    
    print(f"Loaded {len(current_boards)} move transitions")
    return current_boards, next_boards, moves


def train_autoencoder_model(current_boards: np.ndarray, 
                           next_boards: np.ndarray, 
                           moves: np.ndarray,  # Original moves used for dimensionality
                           epochs: int = 50) -> Tuple[nn.Module, np.ndarray]:
    """Train autoencoder from board states with a latent space of move embedding dimension."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training autoencoder on {device}...")
    print(f"Using move embedding dimension {moves.shape[1]} for both latent space and hidden layers")
    
    # Convert to tensors
    current_t = torch.FloatTensor(current_boards).to(device)
    next_t = torch.FloatTensor(next_boards).to(device)
    
    # Create autoencoder model
    model = MoveAutoencoder(
        board_dim=current_boards.shape[1],
        latent_dim=moves.shape[1]  # Use move embedding dimension for both latent and hidden
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        latent, reconstructed_current, reconstructed_next = model(current_t, next_t)
        
        # Compute reconstruction loss
        loss_current = criterion(reconstructed_current, current_t)
        loss_next = criterion(reconstructed_next, next_t)
        loss = loss_current + loss_next
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f} " +
                  f"(Current: {loss_current.item():.6f}, Next: {loss_next.item():.6f})")
    
    # Get latent representations (projected moves)
    model.eval()
    with torch.no_grad():
        projected = model.encode(current_t, next_t).cpu().numpy()
    
    return model, projected


def analyze_norm_vs_state_distance(current_boards: np.ndarray,
                                  next_boards: np.ndarray,
                                  original_moves: np.ndarray,
                                  projected_moves: np.ndarray) -> Dict[str, Any]:
    """Analyze relationship between move norm and state space distance."""
    
    print("\nAnalyzing norm vs state distance relationships...")
    
    # Compute board state distances (Euclidean)
    state_distances = np.linalg.norm(next_boards - current_boards, axis=1)
    
    # Compute move norms
    original_norms = np.linalg.norm(original_moves, axis=1)
    projected_norms = np.linalg.norm(projected_moves, axis=1)
    
    # Compute correlations
    corr_original_pearson, p_orig_p = pearsonr(state_distances, original_norms)
    corr_original_spearman, p_orig_s = spearmanr(state_distances, original_norms)
    
    corr_projected_pearson, p_proj_p = pearsonr(state_distances, projected_norms)
    corr_projected_spearman, p_proj_s = spearmanr(state_distances, projected_norms)
    
    # Compute move differences (for understanding what the move represents)
    # This is the "ideal" move if it directly encoded state difference
    state_diff_moves = next_boards - current_boards
    state_diff_norms = np.linalg.norm(state_diff_moves, axis=1)
    
    results = {
        'state_distances': state_distances,
        'original_norms': original_norms,
        'projected_norms': projected_norms,
        'state_diff_norms': state_diff_norms,
        'correlations': {
            'original_pearson': corr_original_pearson,
            'original_spearman': corr_original_spearman,
            'projected_pearson': corr_projected_pearson,
            'projected_spearman': corr_projected_spearman,
            'original_p_value': p_orig_p,
            'projected_p_value': p_proj_p
        }
    }
    
    return results


def analyze_directional_consistency(original_moves: np.ndarray,
                                   projected_moves: np.ndarray,
                                   n_samples: int = 1000) -> Dict[str, Any]:
    """Analyze if similar moves point in similar directions in both spaces."""
    
    print("\nAnalyzing directional consistency...")
    
    # Sample for efficiency
    if len(original_moves) > n_samples:
        print(f"  Sampling {n_samples} moves for directional consistency analysis")
        idx = np.random.choice(len(original_moves), n_samples, replace=False)
        orig_sample = original_moves[idx]
        proj_sample = projected_moves[idx]
    else:
        orig_sample = original_moves
        proj_sample = projected_moves
    
    # Normalize to unit vectors (direction only)
    orig_normalized = orig_sample / (np.linalg.norm(orig_sample, axis=1, keepdims=True) + 1e-9)
    proj_normalized = proj_sample / (np.linalg.norm(proj_sample, axis=1, keepdims=True) + 1e-9)
    
    # Compute pairwise cosine similarities
    orig_similarities = cosine_similarity(orig_normalized)
    proj_similarities = cosine_similarity(proj_normalized)
    
    # Flatten and compute correlation
    upper_tri_idx = np.triu_indices_from(orig_similarities, k=1)
    orig_sim_flat = orig_similarities[upper_tri_idx]
    proj_sim_flat = proj_similarities[upper_tri_idx]
    
    direction_correlation, p_value = spearmanr(orig_sim_flat, proj_sim_flat)
    
    # Compute per-move directional alignment
    per_move_alignment = np.sum(orig_normalized * proj_normalized, axis=1)
    
    return {
        'direction_correlation': direction_correlation,
        'p_value': p_value,
        'per_move_alignment': per_move_alignment,
        'mean_alignment': np.mean(per_move_alignment),
        'std_alignment': np.std(per_move_alignment)
    }


def analyze_isotropy(embeddings: np.ndarray, name: str, max_samples: int = 10000) -> Dict[str, float]:
    """Analyze isotropy (uniformity) of embedding distribution."""
    
    print(f"\nAnalyzing isotropy for {name} embeddings ({len(embeddings)} vectors)...")
    
    # Sample to keep memory usage reasonable
    if len(embeddings) > max_samples:
        print(f"  Sampling {max_samples} embeddings for isotropy analysis to avoid memory issues")
        idx = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
    
    # Normalize embeddings
    normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    
    # Compute average cosine similarity in batches to save memory
    batch_size = 1000
    n_batches = (len(normalized) + batch_size - 1) // batch_size
    total_sim = 0
    count = 0
    
    print(f"  Computing cosine similarities in {n_batches} batches...")
    for i in range(n_batches):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, len(normalized))
        batch_i = normalized[start_i:end_i]
        
        # Compute similarities between this batch and all embeddings
        # This still creates a large matrix but much smaller than before
        batch_sims = cosine_similarity(batch_i, normalized)
        
        # Remove self-similarities
        for j in range(len(batch_i)):
            batch_sims[j, start_i + j] = 0
        
        # Accumulate absolute similarities
        total_sim += np.sum(np.abs(batch_sims))
        count += batch_sims.size - len(batch_i)  # Subtract diagonal elements
    
    avg_similarity = total_sim / count if count > 0 else 0
    
    # Compute eigenvalue spread (another isotropy measure)
    print("  Computing covariance matrix and eigenvalues...")
    cov = np.cov(embeddings.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Keep positive only
    
    # Ratio of largest to smallest eigenvalue (condition number)
    condition_number = eigenvalues[-1] / eigenvalues[0] if len(eigenvalues) > 0 else np.inf
    
    # Effective rank (participation ratio)
    eigenvalues_normalized = eigenvalues / np.sum(eigenvalues)
    effective_rank = np.exp(-np.sum(eigenvalues_normalized * np.log(eigenvalues_normalized + 1e-10)))
    
    print(f"  Isotropy analysis complete: avg_cosine_sim={avg_similarity:.4f}, effective_rank={effective_rank:.1f}")
    
    return {
        'avg_cosine_similarity': avg_similarity,
        'condition_number': condition_number,
        'effective_rank': effective_rank,
        'embedding_dim': embeddings.shape[1]
    }


def analyze_move_magnitude_distribution(norm_analysis: Dict) -> Dict[str, Any]:
    """Analyze the distribution of move magnitudes."""
    
    # Define magnitude categories based on percentiles
    original_norms = norm_analysis['original_norms']
    projected_norms = norm_analysis['projected_norms']
    state_distances = norm_analysis['state_distances']
    
    # Categorize moves by state distance
    distance_percentiles = np.percentile(state_distances, [25, 50, 75])
    
    short_moves_idx = state_distances <= distance_percentiles[0]
    medium_moves_idx = (state_distances > distance_percentiles[0]) & (state_distances <= distance_percentiles[2])
    long_moves_idx = state_distances > distance_percentiles[2]
    
    results = {
        'short_moves': {
            'original_norm_mean': np.mean(original_norms[short_moves_idx]),
            'projected_norm_mean': np.mean(projected_norms[short_moves_idx]),
            'count': np.sum(short_moves_idx)
        },
        'medium_moves': {
            'original_norm_mean': np.mean(original_norms[medium_moves_idx]),
            'projected_norm_mean': np.mean(projected_norms[medium_moves_idx]),
            'count': np.sum(medium_moves_idx)
        },
        'long_moves': {
            'original_norm_mean': np.mean(original_norms[long_moves_idx]),
            'projected_norm_mean': np.mean(projected_norms[long_moves_idx]),
            'count': np.sum(long_moves_idx)
        },
        'distance_thresholds': distance_percentiles
    }
    
    return results


def create_comprehensive_plots(current_boards: np.ndarray,
                              next_boards: np.ndarray,
                              original_moves: np.ndarray,
                              projected_moves: np.ndarray,
                              norm_analysis: Dict,
                              direction_analysis: Dict,
                              isotropy_orig: Dict,
                              isotropy_proj: Dict,
                              magnitude_dist: Dict,
                              output_dir: str):
    """Create comprehensive visualization of all analyses."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(24, 16))
    
    # 1. Norm vs State Distance Scatter
    ax1 = plt.subplot(3, 4, 1)
    scatter1 = ax1.scatter(norm_analysis['state_distances'], 
                          norm_analysis['original_norms'],
                          alpha=0.3, s=1, c='blue', label='Original')
    ax1.set_xlabel('Board State Distance')
    ax1.set_ylabel('Move Embedding Norm')
    ax1.set_title(f'Original Moves\nCorr: {norm_analysis["correlations"]["original_spearman"]:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(norm_analysis['state_distances'], norm_analysis['original_norms'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(norm_analysis['state_distances'].min(), 
                         norm_analysis['state_distances'].max(), 100)
    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2)
    
    ax2 = plt.subplot(3, 4, 2)
    scatter2 = ax2.scatter(norm_analysis['state_distances'], 
                          norm_analysis['projected_norms'],
                          alpha=0.3, s=1, c='red', label='Projected')
    ax2.set_xlabel('Board State Distance')
    ax2.set_ylabel('Move Embedding Norm')
    ax2.set_title(f'Projected Moves\nCorr: {norm_analysis["correlations"]["projected_spearman"]:.3f}')
    ax2.grid(True, alpha=0.3)
    
    z2 = np.polyfit(norm_analysis['state_distances'], norm_analysis['projected_norms'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(x_trend, p2(x_trend), "r--", alpha=0.5, linewidth=2)
    
    # 2. Direct Norm Comparison
    ax3 = plt.subplot(3, 4, 3)
    ax3.scatter(norm_analysis['original_norms'], norm_analysis['projected_norms'],
               alpha=0.3, s=1, c=norm_analysis['state_distances'], cmap='viridis')
    ax3.set_xlabel('Original Move Norm')
    ax3.set_ylabel('Projected Move Norm')
    ax3.set_title('Norm Comparison\n(colored by state distance)')
    ax3.plot([0, max(norm_analysis['original_norms'].max(), norm_analysis['projected_norms'].max())],
            [0, max(norm_analysis['original_norms'].max(), norm_analysis['projected_norms'].max())],
            'k--', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    plt.colorbar(ax3.collections[0], ax=ax3, label='State Distance')
    
    # 3. Magnitude Distribution by Move Type
    ax4 = plt.subplot(3, 4, 4)
    categories = ['Short', 'Medium', 'Long']
    x = np.arange(len(categories))
    width = 0.35
    
    original_means = [magnitude_dist['short_moves']['original_norm_mean'],
                     magnitude_dist['medium_moves']['original_norm_mean'],
                     magnitude_dist['long_moves']['original_norm_mean']]
    projected_means = [magnitude_dist['short_moves']['projected_norm_mean'],
                      magnitude_dist['medium_moves']['projected_norm_mean'],
                      magnitude_dist['long_moves']['projected_norm_mean']]
    
    ax4.bar(x - width/2, original_means, width, label='Original', color='blue', alpha=0.7)
    ax4.bar(x + width/2, projected_means, width, label='Projected', color='red', alpha=0.7)
    ax4.set_xlabel('Move Distance Category')
    ax4.set_ylabel('Mean Embedding Norm')
    ax4.set_title('Norm by Move Distance')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Directional Alignment Distribution
    ax5 = plt.subplot(3, 4, 5)
    ax5.hist(direction_analysis['per_move_alignment'], bins=50, alpha=0.7, 
            color='purple', edgecolor='black')
    ax5.axvline(direction_analysis['mean_alignment'], color='red', 
               linestyle='--', linewidth=2, label=f'Mean: {direction_analysis["mean_alignment"]:.3f}')
    ax5.set_xlabel('Cosine Similarity (Original vs Projected Direction)')
    ax5.set_ylabel('Count')
    ax5.set_title(f'Directional Alignment\nCorr: {direction_analysis["direction_correlation"]:.3f}')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Norm Distribution Comparison
    ax6 = plt.subplot(3, 4, 6)
    ax6.hist(norm_analysis['original_norms'], bins=50, alpha=0.5, 
            color='blue', label='Original', density=True)
    ax6.hist(norm_analysis['projected_norms'], bins=50, alpha=0.5, 
            color='red', label='Projected', density=True)
    ax6.set_xlabel('Embedding Norm')
    ax6.set_ylabel('Density')
    ax6.set_title('Norm Distribution Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 2D t-SNE visualization
    ax7 = plt.subplot(3, 4, 7)
    print("\nComputing t-SNE for original moves...")
    n_tsne = min(5000, len(original_moves))
    idx_tsne = np.random.choice(len(original_moves), n_tsne, replace=False)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    orig_tsne = tsne.fit_transform(original_moves[idx_tsne])
    
    scatter7 = ax7.scatter(orig_tsne[:, 0], orig_tsne[:, 1], 
                          c=norm_analysis['state_distances'][idx_tsne],
                          cmap='viridis', s=1, alpha=0.5)
    ax7.set_title('Original Moves t-SNE\n(colored by state distance)')
    plt.colorbar(scatter7, ax=ax7)
    
    ax8 = plt.subplot(3, 4, 8)
    print("Computing t-SNE for projected moves...")
    proj_tsne = tsne.fit_transform(projected_moves[idx_tsne])
    scatter8 = ax8.scatter(proj_tsne[:, 0], proj_tsne[:, 1], 
                          c=norm_analysis['state_distances'][idx_tsne],
                          cmap='viridis', s=1, alpha=0.5)
    ax8.set_title('Projected Moves t-SNE\n(colored by state distance)')
    plt.colorbar(scatter8, ax=ax8)
    
    # 9. Isotropy comparison
    ax9 = plt.subplot(3, 4, 9)
    metrics = ['Avg Cosine Sim', 'Condition Number\n(log scale)', 'Effective Rank\n(normalized)']
    original_vals = [isotropy_orig['avg_cosine_similarity'], 
                    np.log10(isotropy_orig['condition_number']),
                    isotropy_orig['effective_rank'] / isotropy_orig['embedding_dim']]
    projected_vals = [isotropy_proj['avg_cosine_similarity'],
                     np.log10(isotropy_proj['condition_number']),
                     isotropy_proj['effective_rank'] / isotropy_proj['embedding_dim']]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax9.bar(x - width/2, original_vals, width, label='Original', color='blue', alpha=0.7)
    ax9.bar(x + width/2, projected_vals, width, label='Projected', color='red', alpha=0.7)
    ax9.set_ylabel('Value')
    ax9.set_title('Isotropy Metrics')
    ax9.set_xticks(x)
    ax9.set_xticklabels(metrics, rotation=45, ha='right')
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    
    # 10. State Distance Distribution
    ax10 = plt.subplot(3, 4, 10)
    ax10.hist(norm_analysis['state_distances'], bins=50, alpha=0.7, 
             color='green', edgecolor='black')
    for i, thresh in enumerate(magnitude_dist['distance_thresholds']):
        ax10.axvline(thresh, color='red', linestyle='--', alpha=0.5,
                    label=f'{[25, 50, 75][i]}th percentile')
    ax10.set_xlabel('Board State Distance')
    ax10.set_ylabel('Count')
    ax10.set_title('State Distance Distribution')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. Norm Ratio Analysis
    ax11 = plt.subplot(3, 4, 11)
    norm_ratio = norm_analysis['projected_norms'] / (norm_analysis['original_norms'] + 1e-9)
    ax11.scatter(norm_analysis['state_distances'], norm_ratio, 
                alpha=0.3, s=1, c='orange')
    ax11.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax11.set_xlabel('Board State Distance')
    ax11.set_ylabel('Projected Norm / Original Norm')
    ax11.set_title('Norm Scaling Factor')
    ax11.grid(True, alpha=0.3)
    
    # 12. Summary Statistics Table
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('tight')
    ax12.axis('off')
    
    summary_data = [
        ['Metric', 'Original', 'Projected'],
        ['Mean Norm', f"{np.mean(norm_analysis['original_norms']):.3f}", 
         f"{np.mean(norm_analysis['projected_norms']):.3f}"],
        ['Std Norm', f"{np.std(norm_analysis['original_norms']):.3f}",
         f"{np.std(norm_analysis['projected_norms']):.3f}"],
        ['State Dist Corr', f"{norm_analysis['correlations']['original_spearman']:.3f}",
         f"{norm_analysis['correlations']['projected_spearman']:.3f}"],
        ['Isotropy (Avg Cos)', f"{isotropy_orig['avg_cosine_similarity']:.3f}",
         f"{isotropy_proj['avg_cosine_similarity']:.3f}"],
        ['Effective Rank', f"{isotropy_orig['effective_rank']:.1f}",
         f"{isotropy_proj['effective_rank']:.1f}"],
        ['Direction Alignment', '-', f"{direction_analysis['mean_alignment']:.3f}"]
    ]
    
    table = ax12.table(cellText=summary_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax12.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    plt.suptitle('Move Embedding Semantic Analysis: Original vs Projected', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'semantic_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved semantic analysis plot to {plot_path}")
    
    # Print detailed insights
    print_insights(norm_analysis, direction_analysis, isotropy_orig, isotropy_proj, magnitude_dist)


def print_insights(norm_analysis: Dict, direction_analysis: Dict, 
                   isotropy_orig: Dict, isotropy_proj: Dict, 
                   magnitude_dist: Dict):
    """Print detailed insights from the analysis."""
    
    print("\n" + "="*80)
    print("SEMANTIC ANALYSIS INSIGHTS")
    print("="*80)
    
    print("\n1. NORM-DISTANCE RELATIONSHIP:")
    print("-" * 40)
    orig_corr = norm_analysis['correlations']['original_spearman']
    proj_corr = norm_analysis['correlations']['projected_spearman']
    
    if abs(orig_corr) < 0.3:
        print(f"✓ Original moves show WEAK correlation ({orig_corr:.3f}) with state distance")
        print("  → Move embeddings encode more than just distance")
    elif abs(orig_corr) < 0.7:
        print(f"✓ Original moves show MODERATE correlation ({orig_corr:.3f}) with state distance")
        print("  → Move embeddings partially encode distance but have other factors")
    else:
        print(f"✓ Original moves show STRONG correlation ({orig_corr:.3f}) with state distance")
        print("  → Move embeddings strongly encode state distance")
    
    if abs(proj_corr) > abs(orig_corr):
        print(f"✓ Projected moves have STRONGER correlation ({proj_corr:.3f})")
        print("  → Board concatenation naturally captures distance information")
    else:
        print(f"✓ Projected moves have WEAKER correlation ({proj_corr:.3f})")
        print("  → Projection learns to suppress pure distance encoding")
    
    print("\n2. DIRECTIONAL CONSISTENCY:")
    print("-" * 40)
    align = direction_analysis['mean_alignment']
    if align > 0.8:
        print(f"✓ HIGH directional alignment ({align:.3f})")
        print("  → Both representations encode similar move directions")
    elif align > 0.5:
        print(f"✓ MODERATE directional alignment ({align:.3f})")
        print("  → Representations share some directional information")
    else:
        print(f"✓ LOW directional alignment ({align:.3f})")
        print("  → Representations encode different aspects of moves")
    
    print("\n3. ISOTROPY (REPRESENTATION UNIFORMITY):")
    print("-" * 40)
    if isotropy_orig['avg_cosine_similarity'] < isotropy_proj['avg_cosine_similarity']:
        print("✓ Original embeddings are MORE isotropic (uniformly distributed)")
        print("  → Original moves explore the space more uniformly")
    else:
        print("✓ Projected embeddings are MORE isotropic")
        print("  → Projection creates more uniform distribution")
    
    print(f"  Original effective rank: {isotropy_orig['effective_rank']:.1f}/{isotropy_orig['embedding_dim']}")
    print(f"  Projected effective rank: {isotropy_proj['effective_rank']:.1f}/{isotropy_proj['embedding_dim']}")
    
    print("\n4. MOVE MAGNITUDE PATTERNS:")
    print("-" * 40)
    for category in ['short_moves', 'medium_moves', 'long_moves']:
        orig_norm = magnitude_dist[category]['original_norm_mean']
        proj_norm = magnitude_dist[category]['projected_norm_mean']
        ratio = proj_norm / orig_norm if orig_norm > 0 else 0
        
        category_name = category.replace('_', ' ').title()
        print(f"{category_name}:")
        print(f"  Original norm: {orig_norm:.3f}")
        print(f"  Projected norm: {proj_norm:.3f}")
        print(f"  Ratio: {ratio:.3f}")
    
    print("\n5. KEY DIFFERENCES:")
    print("-" * 40)
    
    # Check if norms scale proportionally
    short_ratio = magnitude_dist['short_moves']['projected_norm_mean'] / magnitude_dist['short_moves']['original_norm_mean']
    long_ratio = magnitude_dist['long_moves']['projected_norm_mean'] / magnitude_dist['long_moves']['original_norm_mean']
    
    if abs(short_ratio - long_ratio) < 0.1:
        print("✓ Norms scale PROPORTIONALLY across move distances")
        print("  → Both representations handle distance similarly")
    else:
        print("✓ Norms scale DIFFERENTLY across move distances")
        print(f"  → Short moves scale by {short_ratio:.2f}x, long moves by {long_ratio:.2f}x")
        if short_ratio > long_ratio:
            print("  → Projection amplifies short moves more than long moves")
        else:
            print("  → Projection amplifies long moves more than short moves")
    
    print("\n" + "="*80)


def create_pca_clusters_3d(original_moves: np.ndarray, 
                          projected_moves: np.ndarray,
                          state_distances: np.ndarray,
                          output_dir: str,
                          n_clusters: int = 8,
                          max_samples: int = 10000):
    """Create 3D PCA visualization with clustered data points."""
    
    print("\nCreating 3D PCA visualization with clusters...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample data if needed to keep visualization manageable
    if len(original_moves) > max_samples:
        print(f"  Sampling {max_samples} points for PCA visualization")
        idx = np.random.choice(len(original_moves), max_samples, replace=False)
        original_moves_sample = original_moves[idx]
        projected_moves_sample = projected_moves[idx]
        state_distances_sample = state_distances[idx]
    else:
        original_moves_sample = original_moves
        projected_moves_sample = projected_moves
        state_distances_sample = state_distances
    
    # Standardize data for better PCA and clustering
    print("  Standardizing data...")
    scaler_orig = StandardScaler()
    original_scaled = scaler_orig.fit_transform(original_moves_sample)
    
    scaler_proj = StandardScaler()
    projected_scaled = scaler_proj.fit_transform(projected_moves_sample)
    
    # Perform PCA for dimensionality reduction
    print("  Computing PCA (3 components)...")
    pca_orig = PCA(n_components=3)
    pca_proj = PCA(n_components=3)
    
    original_pca = pca_orig.fit_transform(original_scaled)
    projected_pca = pca_proj.fit_transform(projected_scaled)
    
    # Perform K-means clustering
    print(f"  Clustering data into {n_clusters} clusters...")
    kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_proj = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    clusters_orig = kmeans_orig.fit_predict(original_scaled)
    clusters_proj = kmeans_proj.fit_predict(projected_scaled)
    
    # Create 3D visualizations
    print("  Creating 3D PCA plots...")
    
    # Original moves
    fig_orig = plt.figure(figsize=(12, 10))
    ax_orig = fig_orig.add_subplot(111, projection='3d')
    
    # Plot points colored by cluster
    scatter_orig = ax_orig.scatter(
        original_pca[:, 0], original_pca[:, 1], original_pca[:, 2],
        c=clusters_orig, cmap='viridis', s=5, alpha=0.7
    )
    
    # Plot cluster centers
    centers_orig_pca = pca_orig.transform(scaler_orig.transform(kmeans_orig.cluster_centers_))
    ax_orig.scatter(
        centers_orig_pca[:, 0], centers_orig_pca[:, 1], centers_orig_pca[:, 2],
        marker='x', s=100, c='red', label='Cluster Centers'
    )
    
    # Add labels and legend
    ax_orig.set_xlabel(f'PC1 ({pca_orig.explained_variance_ratio_[0]:.2%} variance)')
    ax_orig.set_ylabel(f'PC2 ({pca_orig.explained_variance_ratio_[1]:.2%} variance)')
    ax_orig.set_zlabel(f'PC3 ({pca_orig.explained_variance_ratio_[2]:.2%} variance)')
    ax_orig.set_title('Original Move Embeddings - PCA with K-means Clustering')
    
    # Add colorbar for clusters
    cbar_orig = plt.colorbar(scatter_orig, ax=ax_orig, pad=0.1)
    cbar_orig.set_label('Cluster')
    
    plt.tight_layout()
    orig_path = os.path.join(output_dir, 'original_moves_pca_3d_clusters.png')
    plt.savefig(orig_path, dpi=150, bbox_inches='tight')
    plt.show()  # Show the original moves plot
    plt.close()
    
    # Projected moves
    fig_proj = plt.figure(figsize=(12, 10))
    ax_proj = fig_proj.add_subplot(111, projection='3d')
    
    # Plot points colored by cluster
    scatter_proj = ax_proj.scatter(
        projected_pca[:, 0], projected_pca[:, 1], projected_pca[:, 2],
        c=clusters_proj, cmap='viridis', s=5, alpha=0.7
    )
    
    # Plot cluster centers
    centers_proj_pca = pca_proj.transform(scaler_proj.transform(kmeans_proj.cluster_centers_))
    ax_proj.scatter(
        centers_proj_pca[:, 0], centers_proj_pca[:, 1], centers_proj_pca[:, 2],
        marker='x', s=100, c='red', label='Cluster Centers'
    )
    
    # Add labels and legend
    ax_proj.set_xlabel(f'PC1 ({pca_proj.explained_variance_ratio_[0]:.2%} variance)')
    ax_proj.set_ylabel(f'PC2 ({pca_proj.explained_variance_ratio_[1]:.2%} variance)')
    ax_proj.set_zlabel(f'PC3 ({pca_proj.explained_variance_ratio_[2]:.2%} variance)')
    ax_proj.set_title('Projected Move Embeddings - PCA with K-means Clustering')
    
    # Add colorbar for clusters
    cbar_proj = plt.colorbar(scatter_proj, ax=ax_proj, pad=0.1)
    cbar_proj.set_label('Cluster')
    
    plt.tight_layout()
    proj_path = os.path.join(output_dir, 'projected_moves_pca_3d_clusters.png')
    plt.savefig(proj_path, dpi=150, bbox_inches='tight')
    plt.show()  # Show the projected moves plot
    plt.close()
    
    # State distance colored visualization (original embeddings)
    fig_dist = plt.figure(figsize=(12, 10))
    ax_dist = fig_dist.add_subplot(111, projection='3d')
    
    # Normalize state distances for better coloring
    norm_distances = (state_distances_sample - state_distances_sample.min()) / (state_distances_sample.max() - state_distances_sample.min())
    
    scatter_dist = ax_dist.scatter(
        original_pca[:, 0], original_pca[:, 1], original_pca[:, 2],
        c=norm_distances, cmap='plasma', s=5, alpha=0.7
    )
    
    ax_dist.set_xlabel(f'PC1 ({pca_orig.explained_variance_ratio_[0]:.2%} variance)')
    ax_dist.set_ylabel(f'PC2 ({pca_orig.explained_variance_ratio_[1]:.2%} variance)')
    ax_dist.set_zlabel(f'PC3 ({pca_orig.explained_variance_ratio_[2]:.2%} variance)')
    ax_dist.set_title('Original Move Embeddings - PCA Colored by State Distance')
    
    cbar_dist = plt.colorbar(scatter_dist, ax=ax_dist, pad=0.1)
    cbar_dist.set_label('State Distance (normalized)')
    
    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'original_moves_pca_3d_distances.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.show()  # Show the state distance plot
    plt.close()
    
    print(f"  3D PCA visualizations saved to {output_dir}")
    return {
        'original_variance_explained': pca_orig.explained_variance_ratio_,
        'projected_variance_explained': pca_proj.explained_variance_ratio_,
        'original_clusters': clusters_orig,
        'projected_clusters': clusters_proj
    }


def create_cluster_analysis_plots(original_moves: np.ndarray,
                                 projected_moves: np.ndarray,
                                 pca_results: Dict,
                                 output_dir: str,
                                 max_samples: int = 10000):
    """Create additional analysis plots for clustering and PCA results."""
    
    print("\nCreating cluster analysis and PCA variance plots...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample data consistently if needed to match the 3D visualization sampling
    if len(original_moves) > max_samples:
        print(f"  Sampling {max_samples} points for consistency with 3D visualization")
        idx = np.random.choice(len(original_moves), max_samples, replace=False)
        original_moves_sample = original_moves[idx]
        projected_moves_sample = projected_moves[idx]
        # Make sure clusters match the sampled data
        original_clusters_sample = pca_results['original_clusters']
        projected_clusters_sample = pca_results['projected_clusters']
    else:
        original_moves_sample = original_moves
        projected_moves_sample = projected_moves
        original_clusters_sample = pca_results['original_clusters']
        projected_clusters_sample = pca_results['projected_clusters']
    
    # 1. Cluster Size Distribution
    fig_sizes, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original clusters
    cluster_counts_orig = np.bincount(original_clusters_sample)
    cluster_ids_orig = np.arange(len(cluster_counts_orig))
    
    ax1.bar(cluster_ids_orig, cluster_counts_orig, color='blue', alpha=0.7)
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Original Move Embedding Cluster Sizes')
    ax1.set_xticks(cluster_ids_orig)
    for i, count in enumerate(cluster_counts_orig):
        ax1.text(i, count + max(cluster_counts_orig) * 0.02, str(count), 
                ha='center', va='bottom', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Projected clusters
    cluster_counts_proj = np.bincount(projected_clusters_sample)
    cluster_ids_proj = np.arange(len(cluster_counts_proj))
    
    ax2.bar(cluster_ids_proj, cluster_counts_proj, color='red', alpha=0.7)
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Projected Move Embedding Cluster Sizes')
    ax2.set_xticks(cluster_ids_proj)
    for i, count in enumerate(cluster_counts_proj):
        ax2.text(i, count + max(cluster_counts_proj) * 0.02, str(count), 
                ha='center', va='bottom', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_sizes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. PCA Explained Variance Analysis
    # Run PCA with more components to show variance distribution
    print("  Computing extended PCA for variance analysis...")
    n_components = min(50, min(original_moves_sample.shape[1], projected_moves_sample.shape[1]))
    
    scaler_orig = StandardScaler()
    original_scaled = scaler_orig.fit_transform(original_moves_sample)
    pca_orig_full = PCA(n_components=n_components)
    pca_orig_full.fit(original_scaled)
    
    scaler_proj = StandardScaler()
    projected_scaled = scaler_proj.fit_transform(projected_moves_sample)
    pca_proj_full = PCA(n_components=n_components)
    pca_proj_full.fit(projected_scaled)
    
    # Create explained variance plots
    fig_var, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original embeddings variance
    components = np.arange(1, len(pca_orig_full.explained_variance_ratio_) + 1)
    cumulative_var_orig = np.cumsum(pca_orig_full.explained_variance_ratio_)
    
    ax1.bar(components, pca_orig_full.explained_variance_ratio_, color='blue', alpha=0.7)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(components, cumulative_var_orig, 'r-', marker='o', markersize=3)
    
    # Mark 80%, 90%, 95% explained variance thresholds
    for threshold in [0.8, 0.9, 0.95]:
        components_needed = np.argmax(cumulative_var_orig >= threshold) + 1
        ax1_twin.axhline(threshold, color='black', linestyle='--', alpha=0.3)
        ax1_twin.text(components[-1] * 0.8, threshold, f'{threshold:.0%}', 
                    ha='center', va='bottom', fontsize=8)
        ax1.axvline(components_needed, color='green', linestyle='--', alpha=0.5)
        ax1.text(components_needed, max(pca_orig_full.explained_variance_ratio_) * 0.8, 
                f'{components_needed}', ha='center', va='center', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.2"))
    
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1_twin.set_ylabel('Cumulative Explained Variance')
    ax1.set_title('Original Move Embedding PCA Explained Variance')
    ax1.set_xticks(np.arange(1, n_components + 1, step=max(1, n_components // 10)))
    ax1.grid(alpha=0.3)
    
    # Projected embeddings variance
    components = np.arange(1, len(pca_proj_full.explained_variance_ratio_) + 1)
    cumulative_var_proj = np.cumsum(pca_proj_full.explained_variance_ratio_)
    
    ax2.bar(components, pca_proj_full.explained_variance_ratio_, color='red', alpha=0.7)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(components, cumulative_var_proj, 'b-', marker='o', markersize=3)
    
    # Mark 80%, 90%, 95% explained variance thresholds
    for threshold in [0.8, 0.9, 0.95]:
        components_needed = np.argmax(cumulative_var_proj >= threshold) + 1
        ax2_twin.axhline(threshold, color='black', linestyle='--', alpha=0.3)
        ax2_twin.text(components[-1] * 0.8, threshold, f'{threshold:.0%}', 
                    ha='center', va='bottom', fontsize=8)
        ax2.axvline(components_needed, color='green', linestyle='--', alpha=0.5)
        ax2.text(components_needed, max(pca_proj_full.explained_variance_ratio_) * 0.8, 
                f'{components_needed}', ha='center', va='center', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.2"))
    
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2_twin.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Projected Move Embedding PCA Explained Variance')
    ax2.set_xticks(np.arange(1, n_components + 1, step=max(1, n_components // 10)))
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'), dpi=150, bbox_inches='tight')
    plt.show()  # Show the variance plot
    plt.close()
    
    # 3. 3D Visualization of the first 3 principal components from the extended PCA
    # using the same colors as the clustering visualization
    
    # Original PCA with cluster coloring - use consistent sampling
    pca_orig_data = pca_orig_full.transform(original_scaled)[:, :3]
    
    fig_3d = plt.figure(figsize=(14, 12))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    scatter = ax_3d.scatter(
        pca_orig_data[:, 0], pca_orig_data[:, 1], pca_orig_data[:, 2],
        c=original_clusters_sample, cmap='tab10', s=5, alpha=0.7
    )
    
    # Add variance information to axis labels
    var_ratio = pca_orig_full.explained_variance_ratio_[:3]
    ax_3d.set_xlabel(f'PC1 ({var_ratio[0]:.2%} variance)')
    ax_3d.set_ylabel(f'PC2 ({var_ratio[1]:.2%} variance)')
    ax_3d.set_zlabel(f'PC3 ({var_ratio[2]:.2%} variance)')
    ax_3d.set_title('Original Move Embeddings - Extended PCA with Cluster Coloring')
    
    # Add a legend with cluster IDs
    legend1 = ax_3d.legend(*scatter.legend_elements(),
                         title="Clusters", loc="upper right")
    ax_3d.add_artist(legend1)
    
    # Add a text with the total explained variance
    total_var = sum(var_ratio)
    ax_3d.text2D(0.05, 0.95, f'Total explained variance: {total_var:.2%}',
                transform=ax_3d.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'extended_pca_3d_clusters.png'), dpi=150, bbox_inches='tight')
    plt.show()  # Show the 3D plot
    plt.close()
    
    print(f"  Additional analysis plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Self-supervised semantic analysis of move embeddings")
    parser.add_argument('--cache', type=str, required=True,
                       help='Path to sequential cache pickle file')
    parser.add_argument('--sample-fraction', type=float, default=0.2,
                       help='Fraction of data to use for analysis')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension for autoencoder (None for linear)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs for projection')
    parser.add_argument('--output-dir', type=str, default='output/semantic_analysis',
                       help='Output directory for plots')
    parser.add_argument('--random-seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Load data
    current_boards, next_boards, original_moves = load_and_extract_embeddings(
        args.cache, args.sample_fraction
    )
    
    # Train autoencoder instead of projection model
    model, projected_moves = train_autoencoder_model(
        current_boards, next_boards, original_moves,
        epochs=args.epochs
    )
    
    # Perform analyses (same as before, but now with autoencoder's latent representation)
    norm_analysis = analyze_norm_vs_state_distance(
        current_boards, next_boards, original_moves, projected_moves
    )
    
    direction_analysis = analyze_directional_consistency(
        original_moves, projected_moves, 
        n_samples=min(5000, len(original_moves))  # Limit sample size for memory efficiency
    )
    
    isotropy_orig = analyze_isotropy(original_moves, "Original")
    isotropy_proj = analyze_isotropy(projected_moves, "Projected")
    
    magnitude_dist = analyze_move_magnitude_distribution(norm_analysis)
    
    # Create 3D PCA visualization with clusters
    max_samples = 10000
    pca_results = create_pca_clusters_3d(
        original_moves, projected_moves, 
        norm_analysis['state_distances'],
        args.output_dir,
        n_clusters=8,
        max_samples=max_samples
    )
    
    # Create additional cluster and PCA analysis plots
    create_cluster_analysis_plots(
        original_moves, projected_moves,
        pca_results, args.output_dir,
        max_samples=max_samples
    )
    
    # Create comprehensive visualizations
    create_comprehensive_plots(
        current_boards, next_boards, original_moves, projected_moves,
        norm_analysis, direction_analysis, isotropy_orig, isotropy_proj,
        magnitude_dist, args.output_dir
    )
    
    print("\nSelf-supervised analysis complete!")


if __name__ == '__main__':
    main()