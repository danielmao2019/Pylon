import pytest
import torch
import logging
from typing import Dict, Any, Tuple
from data.dataloaders.geotransformer_dataloader import GeoTransformerDataloader
from data.collators.geotransformer.geotransformer_collate_fn import geotransformer_collate_fn
from data.datasets.base_dataset import BaseDataset

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
        """Generate completely random dummy data."""
        # Generate random points and features
        src_points = torch.randn(self.num_points, 3, device=self.device)
        tgt_points = torch.randn(self.num_points, 3, device=self.device)
        src_feats = torch.randn(self.num_points, 1, device=self.device)
        tgt_feats = torch.randn(self.num_points, 1, device=self.device)
        
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


def log_memory_stats(stage: str, memory_stats: Dict[str, float]):
    """Log memory statistics in a formatted way."""
    logger.info(f"\n{stage} Memory Statistics:")
    logger.info("-" * 50)
    for key, value in memory_stats.items():
        logger.info(f"{key:<20}: {value:>10.2f} MB")


def log_data_structure(stage: str, batch: Dict[str, Any]):
    """Log data structure information."""
    logger.info(f"\n{stage} Data Structure:")
    logger.info("-" * 50)
    
    # Source point cloud structure
    logger.info("Source Point Cloud:")
    for i, pos in enumerate(batch['inputs']['src_pc']['pos']):
        logger.info(f"  Stage {i} points shape: {pos.shape}")
    for i, length in enumerate(batch['inputs']['src_pc']['lengths']):
        logger.info(f"  Stage {i} lengths: {length.tolist()}")
    for i, neighbors in enumerate(batch['inputs']['src_pc']['neighbors']):
        logger.info(f"  Stage {i} neighbors shape: {neighbors.shape}")
    if 'subsampling' in batch['inputs']['src_pc']:
        for i, sub in enumerate(batch['inputs']['src_pc']['subsampling']):
            logger.info(f"  Stage {i} subsampling shape: {sub.shape}")
    if 'upsampling' in batch['inputs']['src_pc']:
        for i, up in enumerate(batch['inputs']['src_pc']['upsampling']):
            logger.info(f"  Stage {i} upsampling shape: {up.shape}")
    
    # Target point cloud structure (similar to source)
    logger.info("\nTarget Point Cloud:")
    for i, pos in enumerate(batch['inputs']['tgt_pc']['pos']):
        logger.info(f"  Stage {i} points shape: {pos.shape}")
    for i, length in enumerate(batch['inputs']['tgt_pc']['lengths']):
        logger.info(f"  Stage {i} lengths: {length.tolist()}")
    for i, neighbors in enumerate(batch['inputs']['tgt_pc']['neighbors']):
        logger.info(f"  Stage {i} neighbors shape: {neighbors.shape}")
    if 'subsampling' in batch['inputs']['tgt_pc']:
        for i, sub in enumerate(batch['inputs']['tgt_pc']['subsampling']):
            logger.info(f"  Stage {i} subsampling shape: {sub.shape}")
    if 'upsampling' in batch['inputs']['tgt_pc']:
        for i, up in enumerate(batch['inputs']['tgt_pc']['upsampling']):
            logger.info(f"  Stage {i} upsampling shape: {up.shape}")
    
    # Other data
    logger.info("\nOther Data:")
    logger.info(f"  Transform shape: {batch['inputs']['transform'].shape}")
    if 'correspondences' in batch['inputs']:
        logger.info(f"  Correspondences shape: {batch['inputs']['correspondences'].shape}")


@pytest.mark.parametrize("num_points", [256, 512, 1024, 2048])
def test_memory_vs_num_points(num_points):
    """Test memory consumption and data structure as number of points changes."""
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Initial memory
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()
    
    # Create dataset and dataloader
    dataset = DummyPCRDataset(num_points=num_points, split='train')
    dataloader = GeoTransformerDataloader(
        dataset=dataset,
        num_stages=4,
        voxel_size=0.025,
        search_radius=2.5 * 0.025,
        batch_size=1,
        num_workers=0
    )
    
    # Memory after dataset/dataloader creation
    setup_allocated = torch.cuda.memory_allocated()
    setup_reserved = torch.cuda.memory_reserved()
    
    # Get one batch
    batch = next(iter(dataloader))
    
    # Memory after batch creation
    batch_allocated = torch.cuda.memory_allocated()
    batch_reserved = torch.cuda.memory_reserved()
    
    # Calculate memory statistics
    memory_stats = {
        'initial': initial_allocated / 1024**2,
        'setup': (setup_allocated - initial_allocated) / 1024**2,
        'batch': (batch_allocated - setup_allocated) / 1024**2,
        'total': (batch_allocated - initial_allocated) / 1024**2,
        'per_point': (batch_allocated - initial_allocated) / (num_points * 1024**2),
        'reserved': batch_reserved / 1024**2
    }
    
    log_memory_stats(f"Memory vs Num Points ({num_points} points)", memory_stats)
    log_data_structure(f"Data Structure for {num_points} points", batch)
    
    # Basic assertions
    assert memory_stats['total'] > 0, "Total memory should be positive"
    assert memory_stats['per_point'] > 0, "Memory per point should be positive"


@pytest.mark.parametrize("voxel_size", [0.01, 0.025, 0.05, 0.1])
def test_memory_vs_voxel_size(voxel_size):
    """Test memory consumption and data structure as voxel size changes."""
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Initial memory
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()
    
    # Create dataset and dataloader
    dataset = DummyPCRDataset(num_points=1024, split='train')
    dataloader = GeoTransformerDataloader(
        dataset=dataset,
        num_stages=4,
        voxel_size=voxel_size,
        search_radius=2.5 * voxel_size,
        batch_size=1,
        num_workers=0
    )
    
    # Memory after dataset/dataloader creation
    setup_allocated = torch.cuda.memory_allocated()
    setup_reserved = torch.cuda.memory_reserved()
    
    # Get one batch
    batch = next(iter(dataloader))
    
    # Memory after batch creation
    batch_allocated = torch.cuda.memory_allocated()
    batch_reserved = torch.cuda.memory_reserved()
    
    # Calculate memory statistics
    memory_stats = {
        'initial': initial_allocated / 1024**2,
        'setup': (setup_allocated - initial_allocated) / 1024**2,
        'batch': (batch_allocated - setup_allocated) / 1024**2,
        'total': (batch_allocated - initial_allocated) / 1024**2,
        'per_point': (batch_allocated - initial_allocated) / (1024 * 1024**2),
        'reserved': batch_reserved / 1024**2
    }
    
    log_memory_stats(f"Memory vs Voxel Size ({voxel_size})", memory_stats)
    log_data_structure(f"Data Structure for voxel size {voxel_size}", batch)
    
    # Basic assertions
    assert memory_stats['total'] > 0, "Total memory should be positive"
    assert memory_stats['per_point'] > 0, "Memory per point should be positive"


@pytest.mark.parametrize("search_radius_multiplier", [1.0, 2.0, 3.0, 4.0])
def test_memory_vs_search_radius(search_radius_multiplier):
    """Test memory consumption and data structure as search radius changes."""
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Initial memory
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()
    
    # Create dataset and dataloader
    voxel_size = 0.025
    dataset = DummyPCRDataset(num_points=1024, split='train')
    dataloader = GeoTransformerDataloader(
        dataset=dataset,
        num_stages=4,
        voxel_size=voxel_size,
        search_radius=search_radius_multiplier * voxel_size,
        batch_size=1,
        num_workers=0
    )
    
    # Memory after dataset/dataloader creation
    setup_allocated = torch.cuda.memory_allocated()
    setup_reserved = torch.cuda.memory_reserved()
    
    # Get one batch
    batch = next(iter(dataloader))
    
    # Memory after batch creation
    batch_allocated = torch.cuda.memory_allocated()
    batch_reserved = torch.cuda.memory_reserved()
    
    # Calculate memory statistics
    memory_stats = {
        'initial': initial_allocated / 1024**2,
        'setup': (setup_allocated - initial_allocated) / 1024**2,
        'batch': (batch_allocated - setup_allocated) / 1024**2,
        'total': (batch_allocated - initial_allocated) / 1024**2,
        'per_point': (batch_allocated - initial_allocated) / (1024 * 1024**2),
        'reserved': batch_reserved / 1024**2
    }
    
    log_memory_stats(f"Memory vs Search Radius (multiplier: {search_radius_multiplier})", memory_stats)
    log_data_structure(f"Data Structure for search radius {search_radius_multiplier * voxel_size}", batch)
    
    # Basic assertions
    assert memory_stats['total'] > 0, "Total memory should be positive"
    assert memory_stats['per_point'] > 0, "Memory per point should be positive"
