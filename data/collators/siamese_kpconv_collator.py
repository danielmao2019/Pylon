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
    def _batch_point_cloud_data(point_clouds: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Helper function to batch point clouds.
        
        Args:
            point_clouds: List of point cloud tensors
            
        Returns:
            Dictionary with 'pos', 'x', and 'batch' tensors
        """
        batch_size = len(point_clouds)
        all_pos = []
        all_x = []
        all_batch = []
        
        for i, pc in enumerate(point_clouds):
            num_points = pc.shape[0]
            
            # First 3 columns are positions, rest are features
            all_pos.append(pc[:, :3])
            all_x.append(pc[:, 3:])
            
            # Create batch indices for this sample
            all_batch.append(torch.full((num_points,), i, dtype=torch.long))
        
        pos = torch.cat(all_pos, dim=0)
        x = torch.cat(all_x, dim=0)
        batch = torch.cat(all_batch, dim=0)
        
        # Consistency check
        assert pos.shape[0] == x.shape[0] == batch.shape[0], \
            f"Inconsistent tensor sizes: pos={pos.shape[0]}, x={x.shape[0]}, batch={batch.shape[0]}"
        
        return {
            'pos': pos,
            'x': x,
            'batch': batch
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
        assert set(datapoints_dict.keys()) == {'inputs', 'labels', 'meta_info'}, f"Expected keys 'inputs', 'labels', 'meta_info', got {datapoints_dict.keys()}"
        
        # Extract and transpose 'inputs' to get {'pc_0': [...], 'pc_1': [...]}
        inputs_list = datapoints_dict["inputs"]
        inputs_dict = transpose_buffer(inputs_list)
        assert set(inputs_dict.keys()) >= {'pc_0', 'pc_1'}, f"Inputs must contain at least 'pc_0' and 'pc_1', got {inputs_dict.keys()}"
        
        # Create batched point clouds for pc_0 and pc_1
        batched_inputs = {
            'pc_0': self._batch_point_cloud_data(inputs_dict['pc_0']),
            'pc_1': self._batch_point_cloud_data(inputs_dict['pc_1']),
        }
        
        # Process change maps
        labels_list = datapoints_dict["labels"]
        labels_dict = transpose_buffer(labels_list)
        assert 'change_map' in labels_dict, f"Labels must contain 'change_map', got {labels_dict.keys()}"
        
        # Simply concatenate the change maps
        batched_labels = {
            'change': torch.cat(labels_dict['change_map'], dim=0)
        }
        
        # Verify the total number of points matches between batched pc_1 and concatenated change maps
        assert batched_inputs['pc_1']['pos'].shape[0] == batched_labels['change'].shape[0], \
            f"Batched pc_1 has {batched_inputs['pc_1']['pos'].shape[0]} points, but batched change map has {batched_labels['change'].shape[0]} points"
        
        # Process meta information
        meta_info_list = datapoints_dict["meta_info"]
        meta_info_dict = transpose_buffer(meta_info_list)
        batched_meta_info = {}
        for key, values in batched_meta_info.items():
            batched_meta_info[key] = self._default_collate(values, "meta_info", key)
        
        return {
            "inputs": batched_inputs,
            "labels": batched_labels,
            "meta_info": batched_meta_info
        }
