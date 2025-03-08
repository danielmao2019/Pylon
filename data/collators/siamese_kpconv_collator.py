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
    
    METHOD_OPTIONS = ['train', 'eval']
    
    def __init__(self, method: str = 'eval'):
        """
        Args:
            method: Either 'train' or 'eval' specifying the collation mode
        """
        super(SiameseKPConvCollator, self).__init__()
        assert method in self.METHOD_OPTIONS, f"Invalid method: {method}"
        self.method = method
    
    def __call__(self, datapoints: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """
        Process a batch of datapoints for the SiameseKPConv model.
        
        Args:
            datapoints: List of individual sample dictionaries, each containing:
                - inputs: Dict with 'pc_0' and 'pc_1' (point clouds)
                - labels: Dict with 'change' (change labels)
                - meta_info: Dict with metadata
                
        Returns:
            Batched data with proper batch indices
        """
        # Transpose to get inputs, labels, meta_info at the top level
        datapoints = transpose_buffer(datapoints)
        
        # Process inputs (point clouds)
        inputs = transpose_buffer(datapoints["inputs"])
        assert set(inputs.keys()) == set(['pc_0', 'pc_1']), f"Expected keys 'pc_0' and 'pc_1', got {inputs.keys()}"
        
        # For each point cloud (pc_0 and pc_1)
        for pc_key in ['pc_0', 'pc_1']:
            # Transpose to get pos, x, etc. at the top level
            pc_batch = transpose_buffer(inputs[pc_key])
            
            # Get batch size
            batch_size = len(pc_batch['pos'])
            
            # Process positions, features, and other attributes
            all_pos = []
            all_x = []
            all_batch = []
            
            # For each sample in the batch
            for i in range(batch_size):
                # Get number of points in this sample
                num_points = pc_batch['pos'][i].shape[0]
                
                # Concatenate positions and features
                all_pos.append(pc_batch['pos'][i])
                all_x.append(pc_batch['x'][i])
                
                # Create batch indices for this sample (all points get the same batch index)
                all_batch.append(torch.full((num_points,), i, dtype=torch.long))
            
            # Concatenate all samples
            inputs[pc_key] = {
                'pos': torch.cat(all_pos, dim=0),
                'x': torch.cat(all_x, dim=0),
                'batch': torch.cat(all_batch, dim=0)
            }
        
        # Process labels
        labels = transpose_buffer(datapoints["labels"])
        
        # Process change labels
        if 'change' in labels:
            # Ensure change labels are tensors
            change_labels = []
            for i in range(len(labels['change'])):
                # Get number of points
                num_points = inputs['pc_1']['pos'].shape[0] if self.method == 'eval' else len(labels['change'][i])
                
                # If change is a single value, expand it to all points
                if isinstance(labels['change'][i], (int, float)) or (isinstance(labels['change'][i], torch.Tensor) and labels['change'][i].numel() == 1):
                    change_value = labels['change'][i].item() if isinstance(labels['change'][i], torch.Tensor) else labels['change'][i]
                    change_labels.append(torch.full((num_points,), change_value, dtype=torch.long))
                else:
                    # If already a tensor of labels for each point
                    change_labels.append(labels['change'][i])
            
            # In train mode, concat all change labels
            if self.method == 'train':
                labels['change'] = torch.cat(change_labels, dim=0)
            else:
                # In eval mode, keep as a list
                labels['change'] = change_labels
        
        # Process meta information
        meta_info = {}
        if "meta_info" in datapoints:
            meta_info = transpose_buffer(datapoints["meta_info"])
            for key, values in meta_info.items():
                meta_info[key] = self._default_collate(values, "meta_info", key)
        
        return {
            "inputs": inputs,
            "labels": labels,
            "meta_info": meta_info
        } 