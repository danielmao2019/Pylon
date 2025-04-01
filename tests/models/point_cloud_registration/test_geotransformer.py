import pytest
import torch
import logging
from utils.builders.builder import build_from_config
from configs.common.models.point_cloud_registration.geotransformer_cfg import model_cfg
from easydict import EasyDict
from utils.ops.apply import apply_tensor_op

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_dummy_data_with_points(num_points):
    """Create dummy data with specified number of points."""
    num_stages = 4  # Match backbone num_stages
    num_neighbors = 16  # Number of neighbors per point

    # Create base points and features
    points = torch.randn(num_points, 3)
    features = torch.randn(num_points, 1)  # Match backbone input_dim

    # Create multi-resolution data
    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    for i in range(num_stages):
        # Points and lengths for each stage
        num_points_at_stage = num_points // (2**i)
        points_list.append(points[:num_points_at_stage])
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

    # Create transform (identity matrix)
    transform = torch.eye(4)

    # Create data dictionary with the expected structure
    data_dict = {
        'points': points_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
        'features': features,
        'transform': transform,
    }

    # Move all tensors to CUDA using apply_tensor_op
    data_dict = apply_tensor_op(lambda x: x.cuda(), data_dict)
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


@pytest.mark.parametrize("num_points,bounds", [
    (256, {
        'total': 60,     # Actual ~50.74MB
        'model': 40,     # Actual ~37.51MB
        'data': 1,       # Actual ~0.20MB
        'forward': 15    # Actual ~13.03MB
    }),
    (512, {
        'total': 50,     # Actual ~43.13MB
        'model': 40,     # Actual ~37.51MB
        'data': 1,       # Actual ~0.40MB
        'forward': 10    # Actual ~5.21MB
    }),
    (1024, {
        'total': 50,     # Actual ~44.16MB
        'model': 40,     # Actual ~37.51MB
        'data': 1,       # Actual ~0.80MB
        'forward': 10    # Actual ~5.84MB
    }),
    (2048, {
        'total': 50,     # Actual ~46.20MB
        'model': 40,     # Actual ~37.51MB
        'data': 2,       # Actual ~1.60MB
        'forward': 10    # Actual ~7.09MB
    })
])
def test_geotransformer_memory_growth(num_points, bounds):
    """Test that GPU memory usage stays within expected bounds for different point cloud sizes.
    
    Args:
        num_points: Number of points in the point cloud
        bounds: Dictionary containing memory thresholds in MB for:
            - total: Maximum total memory usage
            - model: Maximum model memory
            - data: Maximum data memory
            - forward: Maximum forward pass memory
    """
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
    
    # Convert all memory values to MB for logging and comparison
    memory_stats = {
        'initial': initial_allocated / 1024**2,
        'model': model_memory / 1024**2,
        'data': data_memory / 1024**2,
        'forward': forward_memory / 1024**2,
        'total': total_memory / 1024**2,
        'per_point': memory_per_point / 1024**2,
        'reserved': final_reserved / 1024**2
    }
    
    # Log memory usage statistics with thresholds
    logger.info("\n" + "="*70)
    logger.info(f"MEMORY USAGE FOR {num_points} POINTS")
    logger.info("="*70)
    logger.info(f"{'Initial memory:':<25} {memory_stats['initial']:>10.2f} MB")
    logger.info(f"{'Model memory:':<25} {memory_stats['model']:>10.2f} MB (threshold: {bounds['model']} MB)")
    logger.info(f"{'Data memory:':<25} {memory_stats['data']:>10.2f} MB (threshold: {bounds['data']} MB)")
    logger.info(f"{'Forward pass memory:':<25} {memory_stats['forward']:>10.2f} MB (threshold: {bounds['forward']} MB)")
    logger.info(f"{'Total memory:':<25} {memory_stats['total']:>10.2f} MB (threshold: {bounds['total']} MB)")
    logger.info(f"{'Memory per point:':<25} {memory_stats['per_point']:>10.2f} MB/point")
    logger.info(f"{'Reserved memory:':<25} {memory_stats['reserved']:>10.2f} MB")
    logger.info("="*70)
    
    # Log memory usage percentages relative to thresholds
    logger.info("\nMEMORY USAGE RELATIVE TO THRESHOLDS:")
    logger.info("-"*70)
    for component in ['model', 'data', 'forward', 'total']:
        usage_percent = (memory_stats[component] / bounds[component]) * 100
        logger.info(f"{component.capitalize():<10} memory: {usage_percent:>6.1f}% of threshold")
    logger.info("="*70)
    
    # Assert memory usage is within thresholds
    assert memory_stats['total'] <= bounds['total'], \
        f"Total memory usage ({memory_stats['total']:.2f} MB) exceeds threshold ({bounds['total']} MB) for {num_points} points"
    assert memory_stats['model'] <= bounds['model'], \
        f"Model memory usage ({memory_stats['model']:.2f} MB) exceeds threshold ({bounds['model']} MB) for {num_points} points"
    assert memory_stats['data'] <= bounds['data'], \
        f"Data memory usage ({memory_stats['data']:.2f} MB) exceeds threshold ({bounds['data']} MB) for {num_points} points"
    assert memory_stats['forward'] <= bounds['forward'], \
        f"Forward pass memory usage ({memory_stats['forward']:.2f} MB) exceeds threshold ({bounds['forward']} MB) for {num_points} points"
