#!/usr/bin/env python3
"""Analyze overlap distributions for GeoTransformer and OverlapPredator datasets."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_pickle_file(filepath):
    """Load a pickle file and return its contents."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_overlaps_geotransformer(data):
    """Extract overlap values from GeoTransformer format (list of dicts)."""
    if isinstance(data, list):
        overlaps = [entry['overlap'] for entry in data if 'overlap' in entry]
        return np.array(overlaps)
    return None

def extract_overlaps_overlappredator(data):
    """Extract overlap values from OverlapPredator format (dict of lists/arrays)."""
    if isinstance(data, dict) and 'overlap' in data:
        overlap_data = data['overlap']
        if isinstance(overlap_data, np.ndarray):
            return overlap_data
        elif isinstance(overlap_data, list):
            return np.array(overlap_data)
    return None

def analyze_overlap_distribution(overlaps, title, ax):
    """Analyze and plot overlap distribution."""
    if overlaps is None or len(overlaps) == 0:
        print(f"No overlap data for {title}")
        return None

    # Calculate counts
    high_overlap = np.sum((overlaps > 0.3) & (overlaps <= 1.0))
    low_overlap = np.sum((overlaps > 0.1) & (overlaps <= 0.3))
    very_low_overlap = np.sum(overlaps <= 0.1)

    total = len(overlaps)

    # Print statistics
    print(f"\n{title}:")
    print(f"  Total instances: {total}")
    print(f"  Overlap > 0.3 and <= 1.0: {high_overlap} ({high_overlap/total*100:.1f}%)")
    print(f"  Overlap > 0.1 and <= 0.3: {low_overlap} ({low_overlap/total*100:.1f}%)")
    print(f"  Overlap <= 0.1: {very_low_overlap} ({very_low_overlap/total*100:.1f}%)")
    print(f"  Min overlap: {overlaps.min():.4f}")
    print(f"  Max overlap: {overlaps.max():.4f}")
    print(f"  Mean overlap: {overlaps.mean():.4f}")
    print(f"  Median overlap: {np.median(overlaps):.4f}")

    # Create histogram
    ax.hist(overlaps, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.1, color='red', linestyle='--', label='0.1 threshold')
    ax.axvline(x=0.3, color='green', linestyle='--', label='0.3 threshold')
    ax.set_xlabel('Overlap Value')
    ax.set_ylabel('Count')
    ax.set_title(f'{title}\n(n={total})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text box with statistics
    stats_text = f'> 0.3: {high_overlap} ({high_overlap/total*100:.1f}%)\n'
    stats_text += f'0.1-0.3: {low_overlap} ({low_overlap/total*100:.1f}%)\n'
    stats_text += f'≤ 0.1: {very_low_overlap} ({very_low_overlap/total*100:.1f}%)'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return {'high': high_overlap, 'low': low_overlap, 'very_low': very_low_overlap, 'total': total}

def main():
    """Main analysis function."""

    # Define file paths
    geo_files = {
        'Train': '/home/daniel/repos/pcr-repos/GeoTransformer/data/3DMatch/metadata/train.pkl',
        'Val': '/home/daniel/repos/pcr-repos/GeoTransformer/data/3DMatch/metadata/val.pkl',
        '3DMatch Test': '/home/daniel/repos/pcr-repos/GeoTransformer/data/3DMatch/metadata/3DMatch.pkl',
        '3DLoMatch Test': '/home/daniel/repos/pcr-repos/GeoTransformer/data/3DMatch/metadata/3DLoMatch.pkl',
    }

    overlap_files = {
        'Train': '/home/daniel/repos/pcr-repos/OverlapPredator/configs/indoor/train_info.pkl',
        'Val': '/home/daniel/repos/pcr-repos/OverlapPredator/configs/indoor/val_info.pkl',
        '3DMatch Test': '/home/daniel/repos/pcr-repos/OverlapPredator/configs/indoor/3DMatch.pkl',
        '3DLoMatch Test': '/home/daniel/repos/pcr-repos/OverlapPredator/configs/indoor/3DLoMatch.pkl',
    }

    # Create figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle('Overlap Distribution Comparison: GeoTransformer vs OverlapPredator', fontsize=16, fontweight='bold')

    print("=" * 80)
    print("GEOTRANSFORMER OVERLAP ANALYSIS")
    print("=" * 80)

    # Analyze GeoTransformer
    for idx, (name, filepath) in enumerate(geo_files.items()):
        data = load_pickle_file(filepath)
        if data is not None:
            overlaps = extract_overlaps_geotransformer(data)
            analyze_overlap_distribution(overlaps, f"GeoTransformer - {name}", axes[idx, 0])

    print("\n" + "=" * 80)
    print("OVERLAPPREDATOR OVERLAP ANALYSIS")
    print("=" * 80)

    # Analyze OverlapPredator
    for idx, (name, filepath) in enumerate(overlap_files.items()):
        data = load_pickle_file(filepath)
        if data is not None:
            overlaps = extract_overlaps_overlappredator(data)
            analyze_overlap_distribution(overlaps, f"OverlapPredator - {name}", axes[idx, 1])

    # Adjust layout and save
    plt.tight_layout()
    output_path = '/home/daniel/repos/iVISION-PCR/overlap_distributions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.show()

    # Create summary comparison table
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 80)

    print("\n" + "-" * 80)
    print(f"{'Dataset':<30} {'Total':<10} {'High (>0.3)':<20} {'Low (0.1-0.3)':<20} {'Very Low (≤0.1)':<20}")
    print("-" * 80)

    # Re-analyze for summary table
    for name in geo_files.keys():
        # GeoTransformer
        geo_data = load_pickle_file(geo_files[name])
        if geo_data:
            geo_overlaps = extract_overlaps_geotransformer(geo_data)
            if geo_overlaps is not None:
                high = np.sum((geo_overlaps > 0.3) & (geo_overlaps <= 1.0))
                low = np.sum((geo_overlaps > 0.1) & (geo_overlaps <= 0.3))
                very_low = np.sum(geo_overlaps <= 0.1)
                total = len(geo_overlaps)
                print(f"{'GeoTransformer ' + name:<30} {total:<10} {f'{high} ({high/total*100:.1f}%)':<20} {f'{low} ({low/total*100:.1f}%)':<20} {f'{very_low} ({very_low/total*100:.1f}%)':<20}")

        # OverlapPredator
        overlap_data = load_pickle_file(overlap_files[name])
        if overlap_data:
            overlap_overlaps = extract_overlaps_overlappredator(overlap_data)
            if overlap_overlaps is not None:
                high = np.sum((overlap_overlaps > 0.3) & (overlap_overlaps <= 1.0))
                low = np.sum((overlap_overlaps > 0.1) & (overlap_overlaps <= 0.3))
                very_low = np.sum(overlap_overlaps <= 0.1)
                total = len(overlap_overlaps)
                print(f"{'OverlapPredator ' + name:<30} {total:<10} {f'{high} ({high/total*100:.1f}%)':<20} {f'{low} ({low/total*100:.1f}%)':<20} {f'{very_low} ({very_low/total*100:.1f}%)':<20}")

    print("-" * 80)

if __name__ == "__main__":
    main()