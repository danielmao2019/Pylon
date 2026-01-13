#!/usr/bin/env python3
"""Script to analyze and compare metadata files from GeoTransformer and OverlapPredator."""

import pickle
import json
import os
from typing import Dict, Any, List
import numpy as np

def load_pickle_file(filepath: str) -> Any:
    """Load a pickle file and return its contents."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def analyze_structure(data: Any, name: str = "root", max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
    """Analyze the structure of data."""
    if current_depth >= max_depth:
        return {"type": str(type(data).__name__), "truncated": True}

    analysis = {"type": str(type(data).__name__)}

    if isinstance(data, dict):
        analysis["keys"] = list(data.keys())[:10]  # First 10 keys
        analysis["num_keys"] = len(data)
        if len(data) > 0:
            analysis["sample_values"] = {}
            for key in list(data.keys())[:3]:  # Analyze first 3 keys
                analysis["sample_values"][key] = analyze_structure(data[key], f"{name}.{key}", max_depth, current_depth + 1)

    elif isinstance(data, (list, tuple)):
        analysis["length"] = len(data)
        if len(data) > 0:
            analysis["first_element"] = analyze_structure(data[0], f"{name}[0]", max_depth, current_depth + 1)
            if len(data) > 1 and isinstance(data[0], dict):
                # Check if all elements have same structure
                first_keys = set(data[0].keys()) if isinstance(data[0], dict) else None
                if first_keys and all(isinstance(d, dict) and set(d.keys()) == first_keys for d in data[:min(10, len(data))]):
                    analysis["consistent_structure"] = True
                    analysis["common_keys"] = list(first_keys)

    elif isinstance(data, np.ndarray):
        analysis["shape"] = data.shape
        analysis["dtype"] = str(data.dtype)

    elif isinstance(data, (int, float, str, bool, type(None))):
        if current_depth < 2:  # Only show values at shallow depths
            analysis["value"] = data

    return analysis

def compare_files():
    """Compare metadata files from both repositories."""

    results = {}

    # GeoTransformer files
    geo_base = "/home/daniel/repos/pcr-repos/GeoTransformer/data/3DMatch/metadata/"
    geo_files = {
        "3DMatch.pkl": os.path.join(geo_base, "3DMatch.pkl"),
        "3DLoMatch.pkl": os.path.join(geo_base, "3DLoMatch.pkl"),
        "train.pkl": os.path.join(geo_base, "train.pkl"),
        "val.pkl": os.path.join(geo_base, "val.pkl"),
    }

    # OverlapPredator files
    overlap_base = "/home/daniel/repos/pcr-repos/OverlapPredator/configs/indoor/"
    overlap_files = {
        "3DMatch.pkl": os.path.join(overlap_base, "3DMatch.pkl"),
        "3DLoMatch.pkl": os.path.join(overlap_base, "3DLoMatch.pkl"),
        "train_info.pkl": os.path.join(overlap_base, "train_info.pkl"),
        "val_info.pkl": os.path.join(overlap_base, "val_info.pkl"),
    }

    print("=" * 80)
    print("ANALYZING GEOTRANSFORMER FILES")
    print("=" * 80)

    geo_data = {}
    for name, filepath in geo_files.items():
        print(f"\nAnalyzing {name}...")
        data = load_pickle_file(filepath)
        if data is not None:
            geo_data[name] = data
            analysis = analyze_structure(data, name)
            results[f"GeoTransformer_{name}"] = analysis
            print(json.dumps(analysis, indent=2))

    print("\n" + "=" * 80)
    print("ANALYZING OVERLAPPREDATOR FILES")
    print("=" * 80)

    overlap_data = {}
    for name, filepath in overlap_files.items():
        print(f"\nAnalyzing {name}...")
        data = load_pickle_file(filepath)
        if data is not None:
            overlap_data[name] = data
            analysis = analyze_structure(data, name)
            results[f"OverlapPredator_{name}"] = analysis
            print(json.dumps(analysis, indent=2))

    # Additional comparisons
    print("\n" + "=" * 80)
    print("DETAILED COMPARISONS")
    print("=" * 80)

    # Compare 3DMatch.pkl files
    if "3DMatch.pkl" in geo_data and "3DMatch.pkl" in overlap_data:
        print("\n### Comparing 3DMatch.pkl files:")
        geo_3dmatch = geo_data["3DMatch.pkl"]
        overlap_3dmatch = overlap_data["3DMatch.pkl"]

        if isinstance(geo_3dmatch, list) and isinstance(overlap_3dmatch, list):
            print(f"GeoTransformer: {len(geo_3dmatch)} entries")
            print(f"OverlapPredator: {len(overlap_3dmatch)} entries")

            if len(geo_3dmatch) > 0 and len(overlap_3dmatch) > 0:
                print("\nGeoTransformer first entry keys:", list(geo_3dmatch[0].keys()) if isinstance(geo_3dmatch[0], dict) else "Not a dict")
                print("OverlapPredator first entry keys:", list(overlap_3dmatch[0].keys()) if isinstance(overlap_3dmatch[0], dict) else "Not a dict")

    # Compare train files
    if "train.pkl" in geo_data and "train_info.pkl" in overlap_data:
        print("\n### Comparing training metadata files:")
        geo_train = geo_data["train.pkl"]
        overlap_train = overlap_data["train_info.pkl"]

        if isinstance(geo_train, list):
            print(f"GeoTransformer train.pkl: {len(geo_train)} entries")
            if len(geo_train) > 0 and isinstance(geo_train[0], dict):
                print(f"Sample keys: {list(geo_train[0].keys())}")

        if isinstance(overlap_train, list):
            print(f"OverlapPredator train_info.pkl: {len(overlap_train)} entries")
            if len(overlap_train) > 0 and isinstance(overlap_train[0], dict):
                print(f"Sample keys: {list(overlap_train[0].keys())}")

    return results

if __name__ == "__main__":
    compare_files()