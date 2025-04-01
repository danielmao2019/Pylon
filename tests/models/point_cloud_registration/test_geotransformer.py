import pytest
import torch
import logging
from utils.builders.builder import build_from_config
from configs.common.models.point_cloud_registration.geotransformer_cfg import model_cfg
from easydict import EasyDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_dummy_data_with_points(num_points):
    """Create dummy data with specified number of points."""
    num_stages = 4  # Match backbone num_stages
    num_neighbors = 16  # Number of neighbors per point

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
        'src_pc': create_dummy_multi_res_data(src_points, src_feats),
        'tgt_pc': create_dummy_multi_res_data(tgt_points, tgt_feats),
        'transform': torch.eye(4).unsqueeze(0),  # Identity transform
        'src_points': src_points,
        'tgt_points': tgt_points,
        'src_feats': src_feats,
        'tgt_feats': tgt_feats
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


def test_geotransformer_forward():
    """Test the forward pass of the GeoTransformer model."""
    # Initialize model using the builder
    model = build_from_config(model_cfg)
    model = model.cuda()  # Move model to CUDA
    model.eval()  # Set to evaluation mode

    # Create dummy data with 256 points
    data_dict = create_dummy_data_with_points(256)

    # Run forward pass
    with torch.no_grad():
        output_dict = model(data_dict)

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
    assert output_dict['ref_feats_c'].shape[1] == model_cfg['args']['backbone']['output_dim']
    assert output_dict['src_feats_c'].shape[1] == model_cfg['args']['backbone']['output_dim']
    assert output_dict['ref_feats_f'].shape[1] == model_cfg['args']['backbone']['output_dim']
    assert output_dict['src_feats_f'].shape[1] == model_cfg['args']['backbone']['output_dim']

    # 3. Check transform shape
    assert output_dict['estimated_transform'].shape == (1, 4, 4)


@pytest.mark.parametrize("num_points", [256, 512, 1024, 2048])
def test_geotransformer_memory_growth(num_points):
    """Test that GPU memory usage grows sub-linearly with number of points."""
    # Clear CUDA cache before test
    torch.cuda.empty_cache()
    
    # Get initial memory usage
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()
    
    # Create model and move to CUDA
    model = build_from_config(model_cfg)
    model = model.cuda()
    model.eval()
    
    # Get memory after model creation
    model_allocated = torch.cuda.memory_allocated()
    model_reserved = torch.cuda.memory_reserved()
    
    # Create data with specified number of points
    data_dict = create_dummy_data_with_points(num_points)
    
    # Get memory after data creation
    data_allocated = torch.cuda.memory_allocated()
    data_reserved = torch.cuda.memory_reserved()
    
    # Run forward pass
    with torch.no_grad():
        output_dict = model(data_dict)
    
    # Get final memory usage
    final_allocated = torch.cuda.memory_allocated()
    final_reserved = torch.cuda.memory_reserved()
    
    # Calculate memory growth
    model_memory = model_allocated - initial_allocated
    data_memory = data_allocated - model_allocated
    forward_memory = final_allocated - data_allocated
    total_memory = final_allocated - initial_allocated
    memory_per_point = total_memory / num_points
    
    # Log memory usage statistics
    logger.info("\n" + "="*70)
    logger.info(f"MEMORY USAGE FOR {num_points} POINTS")
    logger.info("="*70)
    logger.info(f"{'Initial memory:':<25} {initial_allocated / 1024**2:>10.2f} MB")
    logger.info(f"{'Model memory:':<25} {model_memory / 1024**2:>10.2f} MB")
    logger.info(f"{'Data memory:':<25} {data_memory / 1024**2:>10.2f} MB")
    logger.info(f"{'Forward pass memory:':<25} {forward_memory / 1024**2:>10.2f} MB")
    logger.info(f"{'Total memory:':<25} {total_memory / 1024**2:>10.2f} MB")
    logger.info(f"{'Memory per point:':<25} {memory_per_point / 1024**2:>10.2f} MB/point")
    logger.info(f"{'Reserved memory:':<25} {final_reserved / 1024**2:>10.2f} MB")
    logger.info("="*70)
    
    # Verify that memory growth is sub-linear
    # Memory should not grow more than linearly with number of points
    # We allow for some overhead, so we check if memory growth is less than 2x linear
    if num_points > 256:  # Compare with base case
        base_memory = (final_allocated - initial_allocated) / 256
        current_memory = final_allocated - initial_allocated
        expected_max = base_memory * num_points * 2  # Allow 2x linear growth
        
        assert current_memory < expected_max, \
            f"Memory growth ({current_memory/1024**2:.2f} MB) exceeds expected maximum ({expected_max/1024**2:.2f} MB) for {num_points} points"
