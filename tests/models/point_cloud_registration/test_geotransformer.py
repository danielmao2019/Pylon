from typing import Dict, Any, Tuple
import pytest
from easydict import EasyDict
import torch
from configs.common.models.point_cloud_registration.geotransformer_cfg import model_cfg
from data.dataloaders.geotransformer_dataloader import GeoTransformerDataloader
from data.datasets.base_dataset import BaseDataset
from utils.builders.builder import build_from_config
from utils.ops.apply import apply_tensor_op
from data.dataloaders.geotransformer_dataloader import GeoTransformerDataloader
from tests.data.dataloaders.test_geotransformer_dataloader import DummyPCRDataset

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DummyPCRDataset(BaseDataset):
    """Dummy dataset that mimics SynthPCRDataset's data structure with random data."""
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {'train': 10, 'val': 10, 'test': 10}  # Fixed size for all splits
    INPUT_NAMES = ['src_pc', 'tgt_pc', 'correspondences']
    LABEL_NAMES = ['transform']
    SHA1SUM = None

    def __init__(
        self,
        num_points: int = 1024,
        **kwargs,
    ) -> None:
        self.num_points = num_points
        super(DummyPCRDataset, self).__init__(**kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _init_annotations(self):
        """Initialize dataset with dummy data."""
        # Create a fixed number of dummy samples
        self.annotations = list(range(10))  # 10 dummy samples

    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Generate dummy data with uniformly distributed points."""
        # Generate random points and features using uniform distribution
        # Points are generated in a unit cube [-1, 1]^3
        src_points = 2 * torch.rand(self.num_points, 3, device=self.device) - 1
        tgt_points = 2 * torch.rand(self.num_points, 3, device=self.device) - 1
        src_feats = torch.rand(self.num_points, 1, device=self.device)
        tgt_feats = torch.rand(self.num_points, 1, device=self.device)

        # Generate random correspondences (just random pairs of indices)
        correspondences = torch.randint(0, self.num_points, (2, self.num_points), device=self.device)

        # Generate random transform (just a random 4x4 matrix)
        transform = torch.randn(4, 4, device=self.device)

        inputs = {
            'src_pc': {
                'pos': src_points,
                'feat': src_feats,
            },
            'tgt_pc': {
                'pos': tgt_points,
                'feat': tgt_feats,
            },
            'correspondences': correspondences,
        }

        labels = {
            'transform': transform,
        }

        meta_info = {
            'idx': idx,
            'point_indices': torch.arange(self.num_points, device=self.device),
            'filepath': f'dummy_{idx}.ply',
        }

        return inputs, labels, meta_info


def test_geotransformer_forward():
    """Test the forward pass of the GeoTransformer model."""
    # Initialize model using the builder
    model = build_from_config(model_cfg)
    model = model.cuda()  # Move model to CUDA
    model.eval()  # Set to evaluation mode

    # Create dataset and dataloader
    dataset = DummyPCRDataset(num_points=1024, split='train')
    dataloader = GeoTransformerDataloader(
        dataset=dataset,
        num_stages=4,
        voxel_size=0.025,  # Fixed value
        search_radius=0.0625,  # Fixed value
        batch_size=1,
        num_workers=0,
        keep_ratio=0.8  # Fixed value
    )

    # Get one batch
    batch = next(iter(dataloader))

    # Run forward pass
    with torch.no_grad():
        output_dict = model(batch['inputs'])

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


@pytest.mark.parametrize('num_points,bounds', [
    (256, {
        'total': 48,     # Actual ~43.27MB + 10% buffer
        'model': 41.5,   # Actual ~37.51MB + 10% buffer
        'data': 0.31,    # Actual ~0.28MB + 10% buffer
        'forward': 6.1   # Actual ~5.48MB + 10% buffer
    }),
    (512, {
        'total': 49,     # Actual ~44.53MB + 10% buffer
        'model': 41.5,   # Actual ~37.51MB + 10% buffer
        'data': 0.76,    # Actual ~0.69MB + 10% buffer
        'forward': 7     # Actual ~6.33MB + 10% buffer
    }),
    (1024, {
        'total': 52.5,   # Actual ~47.70MB + 10% buffer
        'model': 41.5,   # Actual ~37.51MB + 10% buffer
        'data': 1.96,    # Actual ~1.78MB + 10% buffer
        'forward': 9.3   # Actual ~8.41MB + 10% buffer
    }),
    (2048, {
        'total': 58,     # Actual ~52.70MB + 10% buffer
        'model': 41.5,   # Actual ~37.51MB + 10% buffer
        'data': 4.83,    # Actual ~4.39MB + 10% buffer
        'forward': 11.9  # Actual ~10.80MB + 10% buffer
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
    
    # Create dataset and dataloader
    dataset = DummyPCRDataset(num_points=num_points, split='train')
    dataloader = GeoTransformerDataloader(
        dataset=dataset,
        num_stages=4,
        voxel_size=0.025,  # Fixed value
        search_radius=0.0625,  # Fixed value
        batch_size=1,
        num_workers=0,
        keep_ratio=0.8  # Fixed value
    )
    
    # Get one batch from dataloader
    batch = next(iter(dataloader))
    
    # Get memory after data creation
    data_allocated = torch.cuda.memory_allocated()
    data_reserved = torch.cuda.memory_reserved()
    
    # Run forward pass
    with torch.no_grad():
        output_dict = model(batch['inputs'])
    
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
    
    # Log suggested new bounds based on actual usage
    logger.info("\nSUGGESTED NEW BOUNDS:")
    logger.info("-"*70)
    logger.info(f"{'Model:':<10} {max(memory_stats['model'] * 1.2, bounds['model']):>6.1f} MB")
    logger.info(f"{'Data:':<10} {max(memory_stats['data'] * 1.2, bounds['data']):>6.1f} MB")
    logger.info(f"{'Forward:':<10} {max(memory_stats['forward'] * 1.2, bounds['forward']):>6.1f} MB")
    logger.info(f"{'Total:':<10} {max(memory_stats['total'] * 1.2, bounds['total']):>6.1f} MB")
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
