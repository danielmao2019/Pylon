import pytest
import torch
from utils.builders.builder import build_from_config
from configs.common.models.point_cloud_registration.overlappredator_cfg import model_cfg


@pytest.fixture
def dummy_data():
    """Create dummy data matching the expected structure from overlappredator_collate_fn.py."""
    # Create dummy point clouds with multiple layers
    num_points = 256  # Points per cloud in first layer
    num_stages = 4    # Number of layers in the network
    
    # Create source and target point clouds for each layer
    # Points get downsampled by half at each layer
    points_list = []
    neighbors_list = []
    pools_list = []
    upsamples_list = []
    lengths_list = []
    
    # Create data for each layer
    for i in range(num_stages):
        # Number of points at this layer
        num_points_at_layer = num_points // (2**i)
        
        # Create points for this layer
        points = torch.randn(num_points_at_layer * 2, 3)  # *2 for src and tgt
        
        # Create neighbors (random indices for demonstration)
        if i < num_stages - 1:  # Not the last layer
            num_neighbors = 16
            # Create neighbors indices within current layer
            neighbors = torch.randint(0, num_points_at_layer * 2, (num_points_at_layer * 2, num_neighbors))
            
            # Create pooling indices (for next layer)
            next_layer_points = num_points_at_layer // 2
            # Create indices from current layer to next layer
            pools = torch.randint(0, num_points_at_layer * 2, (next_layer_points * 2, num_neighbors))
            
            # Create upsampling indices (from next layer back to current)
            upsamples = torch.randint(0, next_layer_points * 2, (num_points_at_layer * 2, 1))
        else:
            # Last layer has no neighbors, pools, or upsamples
            neighbors = torch.zeros((num_points_at_layer * 2, 1), dtype=torch.int64)
            pools = torch.zeros((num_points_at_layer, 1), dtype=torch.int64)
            upsamples = torch.zeros((num_points_at_layer * 2, 1), dtype=torch.int64)
        
        # Create batch lengths (equal split between src and tgt)
        lengths = torch.tensor([num_points_at_layer, num_points_at_layer])
        
        points_list.append(points)
        neighbors_list.append(neighbors)
        pools_list.append(pools)
        upsamples_list.append(upsamples)
        lengths_list.append(lengths)
    
    # Create features for the first layer
    features = torch.randn(num_points * 2, 1)  # *2 for src and tgt
    
    # Create data dictionary with the expected structure
    data_dict = {
        'points': points_list,
        'neighbors': neighbors_list,
        'pools': pools_list,
        'upsamples': upsamples_list,
        'features': features,
        'stack_lengths': lengths_list,
        'rot': torch.eye(3),  # Identity rotation
        'trans': torch.zeros(3),  # Zero translation
        'correspondences': torch.zeros((0, 2), dtype=torch.int64),  # Empty correspondences
        'src_pcd_raw': torch.randn(num_points, 3),  # Original source points
        'tgt_pcd_raw': torch.randn(num_points, 3),  # Original target points
        'sample': None  # Sample info not needed for testing
    }
    
    # Move all tensors to CUDA
    def move_to_cuda(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cuda()
        elif isinstance(obj, dict):
            return {k: move_to_cuda(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_to_cuda(item) for item in obj]
        return obj
    
    data_dict = move_to_cuda(data_dict)
    return data_dict


def test_overlappredator_forward(dummy_data):
    """Test the forward pass of the OverlapPredator model."""
    # Initialize model using the builder
    model = build_from_config(model_cfg)
    model = model.cuda()  # Move model to CUDA
    model.eval()  # Set to evaluation mode
    
    # Run forward pass
    with torch.no_grad():
        feats_f, scores_overlap, scores_saliency = model(dummy_data)
    
    # Validate output structure and shapes
    # 1. Check feature outputs
    assert feats_f.shape[1] == model_cfg['args']['final_feats_dim']
    assert feats_f.shape[0] == dummy_data['points'][0].shape[0]  # Should match first layer points
    
    # 2. Check score outputs
    assert scores_overlap.shape[0] == dummy_data['points'][0].shape[0]
    assert scores_saliency.shape[0] == dummy_data['points'][0].shape[0]
    
    # 3. Check value ranges
    assert torch.all(scores_overlap >= 0) and torch.all(scores_overlap <= 1)
    assert torch.all(scores_saliency >= 0) and torch.all(scores_saliency <= 1)
    
    # 4. Check feature normalization
    assert torch.allclose(torch.norm(feats_f, p=2, dim=1), torch.ones_like(torch.norm(feats_f, p=2, dim=1)))
