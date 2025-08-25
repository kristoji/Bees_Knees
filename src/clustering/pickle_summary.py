"""
Summary script to examine pickle files - combines basic structure analysis with cluster insights.

This script provides a comprehensive overview of both original and clustered cache files.
"""

import os
import pickle
import numpy as np
from typing import Any, Dict, List
from collections import Counter
import torch


def format_shape(obj):
    """Format object shape information."""
    if obj is None:
        return "None"
    
    if hasattr(obj, 'shape'):
        if isinstance(obj, torch.Tensor):
            return f"torch.Tensor{tuple(obj.shape)}"
        elif isinstance(obj, np.ndarray):
            return f"numpy.array{tuple(obj.shape)}"
    
    if isinstance(obj, list):
        if len(obj) == 0:
            return "list(empty)"
        else:
            return f"list({len(obj)})"
    
    if isinstance(obj, (int, float, str, bool)):
        return f"{type(obj).__name__}({obj})"
    
    return f"{type(obj).__name__}"


def analyze_pickle_file(file_path: str, file_label: str):
    """Analyze a single pickle file."""
    print(f"\n{'='*60}")
    print(f"📁 {file_label}: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Loaded successfully")
        print(f"📊 Number of game samples: {len(data)}")
        
        # Count total moves across all samples
        if len(data) > 0:
            total_moves = 0
            for sample in data:
                if 'processed_moves' in sample:
                    total_moves += sample['processed_moves']
                elif 'board_embeddings_sequence' in sample:
                    # Fallback: count sequence length
                    seq = sample['board_embeddings_sequence']
                    if hasattr(seq, '__len__'):
                        total_moves += len(seq)
            
            print(f"📊 Total moves/sequences: {total_moves}")
            if len(data) > 0:
                print(f"📊 Average moves per game: {total_moves / len(data):.1f}")
        
        if len(data) > 0:
            sample = data[0]
            print(f"🔑 Sample keys: {len(list(sample.keys()))}")
            
            # Basic info
            basic_info = {}
            for key in ['game_file', 'winner', 'total_game_moves', 'processed_moves']:
                if key in sample:
                    basic_info[key] = sample[key]
            
            print(f"📋 First sample info:")
            for key, value in basic_info.items():
                print(f"   {key}: {value}")
            
            # Embedding shapes
            embedding_keys = ['board_embeddings_sequence', 'chosen_move_embeddings_sequence', 'next_board_embeddings_sequence']
            print(f"🧠 Embedding shapes:")
            for key in embedding_keys:
                if key in sample:
                    shape = format_shape(sample[key])
                    print(f"   {key}: {shape}")
            
            # Clustering info (if present)
            cluster_keys = ['board_cluster_ids_sequence', 'chosen_move_cluster_ids_sequence', 'next_board_cluster_ids_sequence']
            cluster_found = [key for key in cluster_keys if key in sample]
            
            if cluster_found:
                print(f"🎯 Clustering data:")
                for key in cluster_found:
                    shape = format_shape(sample[key])
                    print(f"   {key}: {shape}")
                
                # Clustering metadata
                meta_keys = ['num_board_centroids', 'num_move_centroids', 'similarity_metric']
                for key in meta_keys:
                    if key in sample:
                        print(f"   {key}: {sample[key]}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def compare_files(original_data, clustered_data):
    """Compare original and clustered data."""
    print(f"\n{'='*60}")
    print(f"🔍 COMPARISON ANALYSIS")
    print(f"{'='*60}")
    
    if original_data is None or clustered_data is None:
        print("❌ Cannot compare - one or both files failed to load")
        return
    
    print(f"📊 Sample counts: Original={len(original_data)}, Clustered={len(clustered_data)}")
    
    if len(original_data) != len(clustered_data):
        print("⚠️  Different number of samples!")
        return
    
    if len(original_data) == 0:
        print("❌ No samples to compare")
        return
    
    # Compare first sample
    orig_sample = original_data[0]
    clust_sample = clustered_data[0]
    
    orig_keys = set(orig_sample.keys())
    clust_keys = set(clust_sample.keys())
    
    common_keys = orig_keys & clust_keys
    new_keys = clust_keys - orig_keys
    
    print(f"🔑 Keys: {len(orig_keys)} original, {len(clust_keys)} clustered")
    print(f"✅ Common keys: {len(common_keys)}")
    print(f"🆕 New keys in clustered: {len(new_keys)}")
    
    if new_keys:
        print(f"📝 New clustering fields:")
        for key in sorted(new_keys):
            shape = format_shape(clust_sample[key])
            print(f"   {key}: {shape}")


def cluster_summary(clustered_data):
    """Provide clustering summary statistics."""
    if not clustered_data or len(clustered_data) == 0:
        return
    
    print(f"\n{'='*60}")
    print(f"📈 CLUSTERING SUMMARY")
    print(f"{'='*60}")
    
    sample = clustered_data[0]
    
    # Clustering configuration
    print(f"⚙️  Configuration:")
    print(f"   Board centroids: {sample.get('num_board_centroids', 'N/A')}")
    print(f"   Move centroids: {sample.get('num_move_centroids', 'N/A')}")
    print(f"   Similarity metric: {sample.get('similarity_metric', 'N/A')}")
    
    # Analyze cluster usage across all samples
    cluster_fields = {
        'board_cluster_ids_sequence': 'Board clusters',
        'chosen_move_cluster_ids_sequence': 'Move clusters', 
        'next_board_cluster_ids_sequence': 'Next board clusters'
    }
    
    for field, label in cluster_fields.items():
        if field in sample:
            all_clusters = []
            unique_per_sample = []
            
            for s in clustered_data:
                if field in s:
                    clusters = s[field]
                    all_clusters.extend(clusters.tolist())
                    unique_per_sample.append(len(np.unique(clusters)))
            
            if all_clusters:
                all_clusters = np.array(all_clusters)
                unique_total = len(np.unique(all_clusters))
                most_common = Counter(all_clusters).most_common(3)
                
                print(f"\n📊 {label}:")
                print(f"   Total unique used: {unique_total}")
                print(f"   Range: {all_clusters.min()} to {all_clusters.max()}")
                print(f"   Avg unique per game: {np.mean(unique_per_sample):.1f}")
                print(f"   Most common: {[f'{c}({n}x)' for c, n in most_common]}")


def main():
    """Main analysis function."""
    print("🔬 PICKLE FILE ANALYSIS SUMMARY")
    print("=" * 70)
    
    # File paths
    base_dir = "data/LLM_dataset"
    original_file = os.path.join(base_dir, "train_sequential_cache.pkl")
    clustered_file = os.path.join(base_dir, "train_sequential_cache_clustered.pkl")
    
    print(f"📂 Base directory: {base_dir}")
    
    # Analyze both files
    original_data = analyze_pickle_file(original_file, "ORIGINAL FILE")
    clustered_data = analyze_pickle_file(clustered_file, "CLUSTERED FILE")
    
    # Compare files
    compare_files(original_data, clustered_data)
    
    # Clustering summary
    cluster_summary(clustered_data)
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*70}")
    
    # Quick recommendations
    print("\n💡 FINDINGS:")
    if original_data and clustered_data:
        if len(clustered_data) > 0:
            sample = clustered_data[0]
            
            print(f"• Successfully added cluster assignments to {len(clustered_data)} game sequences")
            print(f"• Each game has ~{sample.get('processed_moves', '?')} moves with cluster IDs")
            print(f"• Using {sample.get('num_board_centroids', '?')} board clusters and {sample.get('num_move_centroids', '?')} move clusters")
            print(f"• Cluster assignments use {sample.get('similarity_metric', '?')} similarity metric")
            print(f"• Ready for LLM training with discrete cluster tokens")
    
    print("\n🚀 NEXT STEPS:")
    print("• Use clustered file for training transformer models")
    print("• Cluster IDs can be used as discrete tokens in LLM input sequences")
    print("• Consider analyzing cluster transition patterns for move prediction")
    
    # Print final sample counts
    print(f"\n📊 FINAL COUNTS:")
    
    if original_data is not None:
        print(f"• Original dataset game samples: {len(original_data)}")
        # Count total moves in original
        total_orig_moves = 0
        for sample in original_data:
            if 'processed_moves' in sample:
                total_orig_moves += sample['processed_moves']
            elif 'board_embeddings_sequence' in sample:
                seq = sample['board_embeddings_sequence']
                if hasattr(seq, '__len__'):
                    total_orig_moves += len(seq)
        print(f"• Original total moves/sequences: {total_orig_moves}")
    
    if clustered_data is not None:
        print(f"• Clustered dataset game samples: {len(clustered_data)}")
        # Count total moves in clustered
        total_clust_moves = 0
        for sample in clustered_data:
            if 'processed_moves' in sample:
                total_clust_moves += sample['processed_moves']
            elif 'board_embeddings_sequence' in sample:
                seq = sample['board_embeddings_sequence']
                if hasattr(seq, '__len__'):
                    total_clust_moves += len(seq)
        print(f"• Clustered total moves/sequences: {total_clust_moves}")


if __name__ == "__main__":
    main()
