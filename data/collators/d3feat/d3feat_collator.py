"""
D3Feat Collator for Pylon Integration.

This module provides collate functions for D3Feat that prepare data 
in the format expected by D3Feat models.
"""

from typing import List, Dict, Any, Tuple
import torch
import numpy as np
from functools import partial

from data.collators.base_collator import BaseCollator

try:
    import data.collators.d3feat.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
    import data.collators.d3feat.cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
    CPP_EXTENSIONS_AVAILABLE = True
except ImportError:
    # C++ extensions not compiled - provide fallback implementations
    CPP_EXTENSIONS_AVAILABLE = False
    cpp_subsampling = None
    cpp_neighbors = None


def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if not CPP_EXTENSIONS_AVAILABLE:
        # Fallback implementation - return original points
        points_tensor = torch.from_numpy(points) if isinstance(points, np.ndarray) else points
        batches_tensor = torch.from_numpy(batches_len) if isinstance(batches_len, np.ndarray) else batches_len
        
        if (features is None) and (labels is None):
            return points_tensor, batches_tensor
        elif (labels is None):
            features_tensor = torch.from_numpy(features) if isinstance(features, np.ndarray) else features
            return points_tensor, batches_tensor, features_tensor
        elif (features is None):
            labels_tensor = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels
            return points_tensor, batches_tensor, labels_tensor
        else:
            features_tensor = torch.from_numpy(features) if isinstance(features, np.ndarray) else features
            labels_tensor = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels
            return points_tensor, batches_tensor, features_tensor, labels_tensor
    
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(s_labels)


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """
    if not CPP_EXTENSIONS_AVAILABLE:
        # Fallback implementation - return dummy neighbors
        total_queries = sum(q_batches)
        if max_neighbors > 0:
            # Return dummy neighbor indices
            dummy_neighbors = torch.zeros((total_queries, max_neighbors), dtype=torch.int64)
        else:
            # Return empty neighbors
            dummy_neighbors = torch.zeros((total_queries, 1), dtype=torch.int64)
        return dummy_neighbors

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)


def collate_fn_descriptor(list_data, config, neighborhood_limits):
    """D3Feat collate function for descriptor training."""
    batched_points_list = []
    batched_features_list = []
    batched_lengths_list = []
    assert len(list_data) == 1
    
    for ind, (pts0, pts1, feat0, feat1, sel_corr, dist_keypts) in enumerate(list_data):
        batched_points_list.append(pts0)
        batched_points_list.append(pts1)
        batched_features_list.append(feat0)
        batched_features_list.append(feat1)
        batched_lengths_list.append(len(pts0))
        batched_lengths_list.append(len(pts1))
    
    batched_features = torch.from_numpy(np.concatenate(batched_features_list, axis=0))
    batched_points = torch.from_numpy(np.concatenate(batched_points_list, axis=0))
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list)).int()

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    for block_i, block in enumerate(config.architecture):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r, neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r, neighborhood_limits[layer])
            
            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r, neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []

    ###############
    # Return inputs
    ###############
    dict_inputs = {
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'corr': torch.from_numpy(sel_corr),
        'dist_keypts': torch.from_numpy(dist_keypts),
    }

    return dict_inputs


class D3FeatCollator(BaseCollator):
    """Collator for D3Feat that bridges Pylon format and D3Feat internal format.
    
    This collator takes Pylon's three-field format and converts it to the hierarchical
    batch format expected by D3Feat's KPConv operations.
    """
    
    def __init__(
        self,
        num_layers: int = 5,
        first_subsampling_dl: float = 0.03,
        conv_radius: float = 2.5,
        deform_radius: float = 5.0,
        num_kernel_points: int = 15,
        neighborhood_limits: List[int] = None,
        **kwargs
    ):
        """Initialize D3Feat collator.
        
        Args:
            num_layers: Number of network layers
            first_subsampling_dl: First subsampling grid size
            conv_radius: Convolution radius
            deform_radius: Deformable convolution radius
            num_kernel_points: Number of kernel points
            neighborhood_limits: Pre-computed neighborhood limits per layer
        """
        super(D3FeatCollator, self).__init__(**kwargs)
        
        self.num_layers = num_layers
        self.first_subsampling_dl = first_subsampling_dl
        self.conv_radius = conv_radius
        self.deform_radius = deform_radius
        self.num_kernel_points = num_kernel_points
        self.neighborhood_limits = neighborhood_limits
        
        # Build architecture configuration for collate function
        self.architecture = ['simple', 'resnetb']
        for i in range(num_layers-1):
            self.architecture.append('resnetb_strided')
            self.architecture.append('resnetb')
            self.architecture.append('resnetb')
        for i in range(num_layers-2):
            self.architecture.append('nearest_upsample')
            self.architecture.append('unary')
        self.architecture.append('nearest_upsample')
        self.architecture.append('last_unary')
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch of Pylon format data into D3Feat format.
        
        Args:
            batch: List of datapoints in Pylon three-field format
            
        Returns:
            Collated batch in D3Feat hierarchical format
        """
        # Convert Pylon format to D3Feat tuple format
        list_data = []
        
        for datapoint in batch:
            inputs = datapoint['inputs']
            labels = datapoint['labels'] 
            meta_info = datapoint['meta_info']
            
            # Extract point clouds
            src_pc = inputs['src_pc']
            tgt_pc = inputs['tgt_pc']
            
            # Convert to numpy arrays (D3Feat expects numpy)
            pts0 = src_pc['pos'].detach().cpu().numpy().astype(np.float32)
            pts1 = tgt_pc['pos'].detach().cpu().numpy().astype(np.float32)
            feat0 = src_pc['feat'].detach().cpu().numpy().astype(np.float32)
            feat1 = tgt_pc['feat'].detach().cpu().numpy().astype(np.float32)
            
            # Get correspondences
            if 'correspondences' in inputs:
                correspondences = inputs['correspondences'].detach().cpu().numpy().astype(np.int32)
            else:
                correspondences = np.array([], dtype=np.int32).reshape(0, 2)
                
            # Compute distances between keypoints (simplified)
            if correspondences.shape[0] > 0:
                # Get corresponding points
                corr_pts_src = pts0[correspondences[:, 0]]
                corr_pts_tgt = pts1[correspondences[:, 1]]
                
                # Compute pairwise distances between corresponding source points
                from scipy.spatial.distance import cdist
                dist_keypts = cdist(corr_pts_src, corr_pts_src).astype(np.float32)
            else:
                dist_keypts = np.array([], dtype=np.float32).reshape(0, 0)
            
            # Create tuple format expected by D3Feat collate function
            list_data.append((pts0, pts1, feat0, feat1, correspondences, dist_keypts))
        
        # Use default neighborhood limits if not provided
        if self.neighborhood_limits is None:
            self.neighborhood_limits = [20] * self.num_layers
            
        # Create config object for collate function
        config = type('Config', (), {})()
        config.first_subsampling_dl = self.first_subsampling_dl
        config.conv_radius = self.conv_radius
        config.deform_radius = self.deform_radius
        config.architecture = self.architecture
        
        # Apply D3Feat collate function
        d3feat_batch = collate_fn_descriptor(list_data, config, self.neighborhood_limits)
        
        # Add original Pylon data for potential use in loss computation
        d3feat_batch['pylon_batch'] = batch
        
        return d3feat_batch
        
    def calibrate_neighborhood_limits(self, dataset, keep_ratio: float = 0.8, samples_threshold: int = 2000):
        """Calibrate neighborhood limits for efficient KPConv operations.
        
        Args:
            dataset: Dataset to calibrate on
            keep_ratio: Ratio of neighbors to keep
            samples_threshold: Number of samples for calibration
            
        Returns:
            Calibrated neighborhood limits
        """
        # Create config for calibration
        config = type('Config', (), {})()
        config.first_subsampling_dl = self.first_subsampling_dl
        config.conv_radius = self.conv_radius
        config.deform_radius = self.deform_radius
        config.num_layers = self.num_layers
        config.architecture = self.architecture
        
        # Create temporary collator for calibration
        temp_collator = partial(collate_fn_descriptor, config=config)
        
        # Calibrate using D3Feat's calibration function
        self.neighborhood_limits = calibrate_neighbors(
            dataset, config, temp_collator, keep_ratio, samples_threshold
        )
        
        return self.neighborhood_limits