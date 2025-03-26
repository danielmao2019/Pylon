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
    # Create dummy point clouds
    num_points = 100
    num_stages = 3
    
    # Create dummy multi-resolution data
    def create_dummy_multi_res_data(points, feats):
        points_list = []
        lengths_list = []
        neighbors_list = []
        subsampling_list = []
        upsampling_list = []
        
        for i in range(num_stages):
            # Points and lengths
            points_list.append(points)
            lengths_list.append(torch.tensor([num_points // (2**i)]))
            
            # Neighbors
            neighbors = torch.randint(0, num_points // (2**i), (num_points // (2**i), 16))
            neighbors_list.append(neighbors)
            
            if i < num_stages - 1:
                # Subsampling
                subsampling = torch.randint(0, num_points // (2**i), (num_points // (2**(i+1)), 16))
                subsampling_list.append(subsampling)
                
                # Upsampling
                upsampling = torch.randint(0, num_points // (2**(i+1)), (num_points // (2**i), 16))
                upsampling_list.append(upsampling)
        
        return {
            'pos': points_list,
            'lengths': lengths_list,
            'neighbors': neighbors_list,
            'subsampling': subsampling_list,
            'upsampling': upsampling_list,
        }
    
    # Create source and target point clouds
    src_points = torch.randn(num_points, 3)
    tgt_points = torch.randn(num_points, 3)
    src_feats = torch.ones(num_points, 1)  # Dummy features
    tgt_feats = torch.ones(num_points, 1)  # Dummy features
    
    # Create data dictionary
    data_dict = {
        'inputs': {
            'pc_1': {
                'feat': src_feats,
                **create_dummy_multi_res_data(src_points, src_feats)
            },
            'pc_2': {
                'feat': tgt_feats,
                **create_dummy_multi_res_data(tgt_points, tgt_feats)
            }
        },
        'labels': {
            'transform': torch.eye(4).unsqueeze(0)  # Dummy transform
        },
        'meta_info': {
            'idx': torch.tensor([0]),
            'point_indices': [torch.arange(num_points)],
            'filepath': ['dummy.ply'],
            'batch_size': 1
        }
    }
    
    return data_dict


def test_geotransformer_forward(model_config, dummy_data):
    """Test the forward pass of the GeoTransformer model."""
    # Initialize model using the builder
    model = build_from_config(model_config)
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
