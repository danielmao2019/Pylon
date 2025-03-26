import pytest
import torch
from utils.builders.builder import build_from_config
from configs.common.models.point_cloud_registration.geotransformer import model_cfg
from easydict import EasyDict


@pytest.fixture
def model_config():
    """Get the GeoTransformer model configuration."""
    # Convert nested dictionaries to EasyDict
    def to_easydict(d):
        if isinstance(d, dict):
            return EasyDict({k: to_easydict(v) for k, v in d.items()})
        return d
    
    model_cfg['args']['cfg'] = to_easydict(model_cfg['args']['cfg'])
    return model_cfg 


@pytest.fixture
def dummy_data():
    """Create dummy data matching the expected structure."""
    # Create dummy point clouds - using more points to match model requirements
    num_points = 256  # Points per cloud (increased from 100)
    num_stages = 4  # Match backbone num_stages
    num_neighbors = 16  # Number of neighbors per point
    
    # Create dummy multi-resolution data
    def create_dummy_multi_res_data(points, feats):
        pos_list = []
        lengths_list = []
        neighbors_list = []
        subsampling_list = []
        upsampling_list = []
        
        for i in range(num_stages):
            # Points and lengths for each stage
            num_points_at_stage = num_points // (2**i)
            pos_list.append(points[:num_points_at_stage])
            lengths_list.append(torch.tensor([num_points_at_stage]))
            
            # Create neighbor indices - ensure 2D shape [num_points_at_stage, num_neighbors]
            neighbors = torch.randint(0, num_points_at_stage, (num_points_at_stage, num_neighbors))
            neighbors_list.append(neighbors)
            
            # Create subsampling and upsampling indices
            if i < num_stages - 1:
                next_stage_points = num_points_at_stage // 2
                # For each point in the next stage, create neighbor indices from current stage
                subsampling = torch.randint(0, num_points_at_stage, (next_stage_points, num_neighbors))
                # For each point in current stage, store its nearest neighbor in next stage
                upsampling = torch.randint(0, next_stage_points, (num_points_at_stage, 1))
                subsampling_list.append(subsampling)
                upsampling_list.append(upsampling)
        
        return {
            'pos': pos_list,
            'feat': feats,
            'lengths': lengths_list,
            'neighbors': neighbors_list,
            'subsampling': subsampling_list,
            'upsampling': upsampling_list
        }
    
    # Create source and target point clouds
    src_points = torch.randn(num_points, 3)
    tgt_points = torch.randn(num_points, 3)
    src_feats = torch.randn(num_points, 1)  # Match backbone input_dim
    tgt_feats = torch.randn(num_points, 1)  # Match backbone input_dim
    
    # Create data dictionary with the expected structure
    data_dict = {
        'inputs': {
            'pc_1': create_dummy_multi_res_data(src_points, src_feats),
            'pc_2': create_dummy_multi_res_data(tgt_points, tgt_feats)
        },
        'labels': {
            'transform': torch.eye(4).unsqueeze(0),  # Identity transform
            'src_points': src_points,
            'tgt_points': tgt_points,
            'src_feats': src_feats,
            'tgt_feats': tgt_feats
        }
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


def test_geotransformer_forward(model_config, dummy_data):
    """Test the forward pass of the GeoTransformer model."""
    # Initialize model using the builder
    model = build_from_config(model_config)
    model = model.cuda()  # Move model to CUDA
    model.eval()  # Set to evaluation mode
    
    # Run forward pass
    with torch.no_grad():
        output_dict = model(dummy_data['inputs'], dummy_data['labels'])
    
    # Validate output structure
    # 1. Check point cloud outputs
    assert 'ref_points_c' in output_dict
    assert 'src_points_c' in output_dict
    assert 'ref_points_f' in output_dict
    assert 'src_points_f' in output_dict
    assert 'ref_points' in output_dict
    assert 'src_points' in output_dict
    
    # 2. Check feature outputs
    assert 'ref_feats_c' in output_dict
    assert 'src_feats_c' in output_dict
    assert 'ref_feats_f' in output_dict
    assert 'src_feats_f' in output_dict
    
    # 3. Check correspondence outputs
    assert 'gt_node_corr_indices' in output_dict
    assert 'gt_node_corr_overlaps' in output_dict
    assert 'ref_node_corr_indices' in output_dict
    assert 'src_node_corr_indices' in output_dict
    
    # 4. Check node correspondence outputs
    assert 'ref_node_corr_knn_points' in output_dict
    assert 'src_node_corr_knn_points' in output_dict
    assert 'ref_node_corr_knn_masks' in output_dict
    assert 'src_node_corr_knn_masks' in output_dict
    
    # 5. Check matching outputs
    assert 'matching_scores' in output_dict
    
    # 6. Check final outputs
    assert 'ref_corr_points' in output_dict
    assert 'src_corr_points' in output_dict
    assert 'corr_scores' in output_dict
    assert 'estimated_transform' in output_dict
    
    # Validate shapes
    # 1. Check point cloud shapes
    assert output_dict['ref_points_c'].shape[1] == 3
    assert output_dict['src_points_c'].shape[1] == 3
    assert output_dict['ref_points_f'].shape[1] == 3
    assert output_dict['src_points_f'].shape[1] == 3
    assert output_dict['ref_points'].shape[1] == 3
    assert output_dict['src_points'].shape[1] == 3
    
    # 2. Check feature shapes
    assert output_dict['ref_feats_c'].shape[1] == model_config['args']['cfg'].backbone.output_dim
    assert output_dict['src_feats_c'].shape[1] == model_config['args']['cfg'].backbone.output_dim
    assert output_dict['ref_feats_f'].shape[1] == model_config['args']['cfg'].backbone.output_dim
    assert output_dict['src_feats_f'].shape[1] == model_config['args']['cfg'].backbone.output_dim
    
    # 3. Check transform shape
    assert output_dict['estimated_transform'].shape == (1, 4, 4)
