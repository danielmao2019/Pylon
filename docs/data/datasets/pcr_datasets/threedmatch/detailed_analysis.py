#!/usr/bin/env python3
"""Detailed analysis of metadata files."""

import pickle
import os
import numpy as np

def load_pickle_file(filepath: str):
    """Load a pickle file and return its contents."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def print_detailed_info():
    """Print detailed information about the metadata files."""
    
    # Load GeoTransformer files
    print("=" * 80)
    print("GEOTRANSFORMER DETAILED ANALYSIS")
    print("=" * 80)
    
    # Load 3DMatch.pkl from GeoTransformer
    geo_3dmatch = load_pickle_file("/home/daniel/repos/pcr-repos/GeoTransformer/data/3DMatch/metadata/3DMatch.pkl")
    if geo_3dmatch and len(geo_3dmatch) > 0:
        print("\n### GeoTransformer 3DMatch.pkl structure:")
        print(f"Total entries: {len(geo_3dmatch)}")
        print(f"First entry type: {type(geo_3dmatch[0])}")
        if isinstance(geo_3dmatch[0], dict):
            print(f"Keys in first entry: {list(geo_3dmatch[0].keys())}")
            print("\nSample first entry:")
            entry = geo_3dmatch[0]
            for key, value in entry.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: numpy array with shape {value.shape}, dtype {value.dtype}")
                elif isinstance(value, str):
                    print(f"  {key}: '{value[:100]}...' (string)")
                else:
                    print(f"  {key}: {value} ({type(value).__name__})")
    
    # Load train.pkl from GeoTransformer
    geo_train = load_pickle_file("/home/daniel/repos/pcr-repos/GeoTransformer/data/3DMatch/metadata/train.pkl")
    if geo_train and len(geo_train) > 0:
        print("\n### GeoTransformer train.pkl structure:")
        print(f"Total entries: {len(geo_train)}")
        print(f"First entry type: {type(geo_train[0])}")
        if isinstance(geo_train[0], dict):
            print(f"Keys in first entry: {list(geo_train[0].keys())}")
    
    print("\n" + "=" * 80)
    print("OVERLAPPREDATOR DETAILED ANALYSIS")
    print("=" * 80)
    
    # Load 3DMatch.pkl from OverlapPredator
    overlap_3dmatch = load_pickle_file("/home/daniel/repos/pcr-repos/OverlapPredator/configs/indoor/3DMatch.pkl")
    if overlap_3dmatch:
        print("\n### OverlapPredator 3DMatch.pkl structure:")
        print(f"Type: {type(overlap_3dmatch)}")
        if isinstance(overlap_3dmatch, dict):
            print(f"Top-level keys: {list(overlap_3dmatch.keys())}")
            for key, value in overlap_3dmatch.items():
                if isinstance(value, list):
                    print(f"  {key}: list of {len(value)} items")
                    if len(value) > 0:
                        if isinstance(value[0], np.ndarray):
                            print(f"    First item: numpy array with shape {value[0].shape}, dtype {value[0].dtype}")
                        else:
                            print(f"    First item type: {type(value[0])}")
                            if isinstance(value[0], str):
                                print(f"    Sample: '{value[0][:100]}...'")
                elif isinstance(value, np.ndarray):
                    print(f"  {key}: numpy array with shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {type(value).__name__}")
    
    # Load train_info.pkl from OverlapPredator
    overlap_train = load_pickle_file("/home/daniel/repos/pcr-repos/OverlapPredator/configs/indoor/train_info.pkl")
    if overlap_train:
        print("\n### OverlapPredator train_info.pkl structure:")
        print(f"Type: {type(overlap_train)}")
        if isinstance(overlap_train, dict):
            print(f"Top-level keys: {list(overlap_train.keys())}")
            for key, value in overlap_train.items():
                if isinstance(value, list):
                    print(f"  {key}: list of {len(value)} items")
                    if len(value) > 0 and not isinstance(value[0], np.ndarray):
                        print(f"    First item: {value[0][:100] if isinstance(value[0], str) else value[0]}")
                elif isinstance(value, np.ndarray):
                    print(f"  {key}: numpy array with shape {value.shape}, dtype {value.dtype}")
    
    # Compare paths
    print("\n" + "=" * 80)
    print("PATH COMPARISON")
    print("=" * 80)
    
    if geo_3dmatch and overlap_3dmatch:
        # GeoTransformer paths
        if len(geo_3dmatch) > 0 and isinstance(geo_3dmatch[0], dict):
            geo_path0 = geo_3dmatch[0].get('pcd0', '')
            geo_path1 = geo_3dmatch[0].get('pcd1', '')
            print("\nGeoTransformer sample paths:")
            print(f"  pcd0: {geo_path0}")
            print(f"  pcd1: {geo_path1}")
        
        # OverlapPredator paths
        if isinstance(overlap_3dmatch, dict) and 'src' in overlap_3dmatch:
            src_list = overlap_3dmatch['src']
            tgt_list = overlap_3dmatch['tgt']
            if len(src_list) > 0:
                print("\nOverlapPredator sample paths:")
                print(f"  src: {src_list[0]}")
                print(f"  tgt: {tgt_list[0]}")

if __name__ == "__main__":
    print_detailed_info()