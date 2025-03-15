from typing import List, Dict, Any
import torch
from data.collators import BaseCollator
from utils.ops import transpose_buffer


class SiameseKPConvCollator(BaseCollator):
    """
    A collator for the SiameseKPConv model, handling point cloud pairs and their corresponding change maps.
    
    This collator takes individual point cloud pairs and batches them together, 
    ensuring that batch indices are correctly assigned to each point.
    """
    
    @staticmethod
    def _batch_point_cloud_data(point_clouds: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Helper function to batch point clouds.
        
        Args:
            point_clouds: List of point cloud dictionaries with 'pos' and 'feat' keys
            
        Returns:
            Dictionary with 'pos', 'feat', and 'batch' tensors
        """
        all_pos = []
        all_x = []
        all_batch = []
        
        for i, pc in enumerate(point_clouds):
            num_points = pc['pos'].shape[0]
            
            # Get positions and features
            all_pos.append(pc['pos'])
            all_x.append(pc['feat'])
            
            # Create batch index tensor
            batch = torch.full((num_points,), i, dtype=torch.long, device=pc['pos'].device)
            all_batch.append(batch)
        
        # Concatenate all tensors
        batched_pos = torch.cat(all_pos, dim=0)
        batched_x = torch.cat(all_x, dim=0)
        batched_batch = torch.cat(all_batch, dim=0)
        
        # Consistency check
        assert batched_pos.shape[0] == batched_x.shape[0] == batched_batch.shape[0], \
            f"Inconsistent tensor sizes: batched_pos={batched_pos.shape[0]}, batched_x={batched_x.shape[0]}, batched_batch={batched_batch.shape[0]}"
        
        return {
            'pos': batched_pos,
            'feat': batched_x,
            'batch': batched_batch
        }
    
    def __call__(self, datapoints: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """
        Process a batch of datapoints for the SiameseKPConv model.
        
        Args:
            datapoints: List of individual sample dictionaries, each containing:
                - inputs: Dict with 'pc_0', 'pc_1', etc.
                - labels: Dict with 'change_map'
                - meta_info: Dict with metadata
                
        Returns:
            Batched data with proper batch indices, all in tensor format
        """
        # Transpose the top level to get {'inputs': [...], 'labels': [...], 'meta_info': [...]}
        datapoints_dict = transpose_buffer(datapoints)
        assert set(datapoints_dict.keys()) == {'inputs', 'labels', 'meta_info'}, \
            f"Expected keys 'inputs', 'labels', 'meta_info', got {datapoints_dict.keys()}"
        
        # Process inputs - batch the point clouds
        inputs_dict = transpose_buffer(datapoints_dict["inputs"])
        assert set(inputs_dict.keys()) >= {'pc_0', 'pc_1'}, \
            f"Inputs must contain at least 'pc_0' and 'pc_1', got {inputs_dict.keys()}"
        
        batched_inputs = {
            'pc_0': self._batch_point_cloud_data(inputs_dict['pc_0']),
            'pc_1': self._batch_point_cloud_data(inputs_dict['pc_1']),
        }
        
        # Process labels - concatenate the change maps
        labels_dict = transpose_buffer(datapoints_dict["labels"])
        assert 'change_map' in labels_dict, \
            f"Labels must contain 'change_map', got {labels_dict.keys()}"
        
        batched_labels = {
            'change': torch.cat(labels_dict['change_map'], dim=0)
        }
        
        # Verify consistency between point clouds and change maps
        assert batched_inputs['pc_1']['pos'].shape[0] == batched_labels['change'].shape[0], \
            f"Batched pc_1 has {batched_inputs['pc_1']['pos'].shape[0]} points, but batched change map has {batched_labels['change'].shape[0]} points"
        
        # Process meta information
        meta_info_dict = transpose_buffer(datapoints_dict["meta_info"])
        batched_meta_info = {}
        for key, values in meta_info_dict.items():
            batched_meta_info[key] = self._default_collate(values=values)
        
        return {
            "inputs": batched_inputs,
            "labels": batched_labels,
            "meta_info": batched_meta_info
        }
