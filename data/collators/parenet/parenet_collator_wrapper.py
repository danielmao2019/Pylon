"""
PARENet Collator Wrapper for Pylon API Compatibility.

This module provides Pylon-compatible wrapper functions around the original PARENet collation logic.
The wrapper handles the integration between Pylon's collator-based preprocessing and PARENet's 
original trainer-based dynamic neighbor computation.
"""

from typing import List, Any, Dict
from data.collators.parenet.data import registration_collate_fn_stack_mode, precompute_neibors
from data.collators.base_collator import BaseCollator


def pylon_to_parenet_adapter(pylon_data_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Pylon dataset format to PARENet collation format.
    
    Pylon format: 
        inputs: {src_pc: {pos, features}, tgt_pc: {pos, features}}
        labels: {transform}
        
    PARENet format:
        ref_points, src_points, ref_feats, src_feats, transform
    """
    parenet_data_dicts = []
    
    for pylon_dict in pylon_data_dicts:
        # Extract data from Pylon nested structure
        src_pc = pylon_dict['inputs']['src_pc']
        tgt_pc = pylon_dict['inputs']['tgt_pc']
        
        # Convert to PARENet flat structure
        parenet_dict = {
            'ref_points': tgt_pc['pos'],  # Target becomes reference
            'src_points': src_pc['pos'],  # Source remains source  
            'ref_feats': tgt_pc['features'],
            'src_feats': src_pc['features'],
            'transform': pylon_dict['labels']['transform'],
        }
        
        parenet_data_dicts.append(parenet_dict)
    
    return parenet_data_dicts


def parenet_collate_fn(
    pylon_data_dicts: List[Dict[str, Any]], 
    num_stages: int, 
    voxel_size: float, 
    num_neighbors: List[int], 
    subsample_ratio: float, 
    precompute_data: bool = True
) -> Dict[str, Any]:
    """Pylon-compatible PARENet collate function.
    
    This wrapper adapts PARENet's original design to work with Pylon's collator-based preprocessing:
    - Original PARENet: precompute_subsample() in collator + precompute_neibors() in trainer
    - Pylon Integration: Both operations moved to collator for framework compatibility
    """
    # Convert Pylon format to PARENet format
    parenet_data_dicts = pylon_to_parenet_adapter(pylon_data_dicts)
    
    # Use original PARENet collation function (only does subsampling)
    collated_dict = registration_collate_fn_stack_mode(
        parenet_data_dicts, num_stages, voxel_size, num_neighbors, subsample_ratio, precompute_data
    )
    
    # ========================================================================
    # INTEGRATION ADAPTER: Move neighbor computation from trainer to collator
    # ========================================================================
    #
    # ORIGINAL PARENet DESIGN:
    #   1. Collator: precompute_subsample() - only grid subsampling
    #   2. Trainer: precompute_neibors() - dynamic neighbor computation 
    #   3. Model: expects both subsampling + neighbor data
    #
    # PYLON FRAMEWORK CONSTRAINT:
    #   - All preprocessing must happen in collator
    #   - Trainer has no access to model-specific parameters (num_stages, num_neighbors)
    #   - Model expects fully preprocessed data
    #
    # INTEGRATION SOLUTION:
    #   - Use original PARENet collation (subsampling only) 
    #   - Add original PARENet neighbor computation in wrapper
    #   - Preserves original source code unchanged
    #   - Bridges architectural difference between frameworks
    #
    # WITHOUT THIS: PARENet model would fail with KeyError: 'neighbors'
    # WITH THIS: PARENet model gets complete expected data structure
    # ========================================================================
    if precompute_data and 'points' in collated_dict and 'lengths' in collated_dict:
        # Call original PARENet neighbor computation function
        # This computes: neighbors (K-NN), subsampling (cross-scale down), upsampling (cross-scale up)
        neighbor_dict = precompute_neibors(
            collated_dict['points'],    # Multi-scale point hierarchies from subsampling
            collated_dict['lengths'],   # Batch lengths for each scale  
            num_stages,                 # Number of hierarchical scales (e.g., 4)
            num_neighbors               # K-NN neighbors per scale (e.g., [32, 32, 32, 32])
        )
        # Add neighbor data to collated dictionary
        # Result adds: 'neighbors', 'subsampling', 'upsampling' keys
        collated_dict.update(neighbor_dict)
    
    # Extract labels for criterion (keep transform in inputs for model)
    labels = {}
    if 'transform' in collated_dict:
        labels['transform'] = collated_dict['transform']  # Don't pop, keep in inputs too
    
    # Properly collate meta_info using BaseCollator's default collation logic
    # This preserves the 'idx' field that BaseDataset adds to meta_info
    meta_info = {}
    if pylon_data_dicts:
        meta_info_values = [dp['meta_info'] for dp in pylon_data_dicts]
        # Extract individual keys from meta_info and collate them properly
        if meta_info_values and isinstance(meta_info_values[0], dict):
            for key in meta_info_values[0].keys():
                values = [meta_info[key] for meta_info in meta_info_values]
                meta_info[key] = BaseCollator._default_collate(values, 'meta_info', key)
    
    # Wrap in Pylon structure
    return {
        'inputs': collated_dict,
        'labels': labels,
        'meta_info': meta_info,
    }
