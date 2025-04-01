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

    # Points structure for each stage
    logger.info("Points Structure:")
    for i, points in enumerate(batch['inputs']['points']):
        logger.info(f"  Stage {i} points shape: {points.shape}")
    
    # Lengths for each stage
    logger.info("\nLengths Structure:")
    for i, lengths in enumerate(batch['inputs']['lengths']):
        logger.info(f"  Stage {i} lengths: {lengths.tolist()}")
    
    # Neighbors for each stage
    logger.info("\nNeighbors Structure:")
    for i, neighbors in enumerate(batch['inputs']['neighbors']):
        logger.info(f"  Stage {i} neighbors shape: {neighbors.shape}")
    
    # Subsampling for each stage (except last)
    if 'subsampling' in batch['inputs']:
        logger.info("\nSubsampling Structure:")
        for i, sub in enumerate(batch['inputs']['subsampling']):
            logger.info(f"  Stage {i} subsampling shape: {sub.shape}")
    
    # Upsampling for each stage (except first)
    if 'upsampling' in batch['inputs']:
        logger.info("\nUpsampling Structure:")
        for i, up in enumerate(batch['inputs']['upsampling']):
            logger.info(f"  Stage {i} upsampling shape: {up.shape}")

    # Features and transform
    logger.info("\nOther Data:")
    logger.info(f"  Features shape: {batch['inputs']['features'].shape}")
    logger.info(f"  Transform shape: {batch['inputs']['transform'].shape}")


def log_neighbor_limits(dataloader: GeoTransformerDataloader):
    """Log neighbor limits analysis in a formatted way."""
    logger.info("\nNeighbor Limits Analysis:")
    logger.info("-" * 50)

    # Log configured neighbor limits
    logger.info("Configured Neighbor Limits:")
    for i, limit in enumerate(dataloader.neighbor_limits):
        logger.info(f"  Stage {i}: {limit}")

    logger.info(f"  Number of stages: {len(dataloader.neighbor_limits)}")


def log_neighbor_limits(dataloader: GeoTransformerDataloader):
    """Log neighbor limits analysis in a formatted way."""
    logger.info("\nNeighbor Limits Analysis:")
    logger.info("-" * 50)
    
    # Log configured neighbor limits
    logger.info("Configured Neighbor Limits:")
    for i, limit in enumerate(dataloader.neighbor_limits):
        logger.info(f"  Stage {i}: {limit}")
    
    logger.info(f"  Number of stages: {len(dataloader.neighbor_limits)}")


@pytest.mark.parametrize("num_points", [256, 512, 1024, 2048])
def test_num_points_impact(num_points):
    """Test how number of points affects memory, data structure, and neighbor limits."""
    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Initial memory
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()

    # Create dataset and dataloader with fixed parameters
    dataset = DummyPCRDataset(num_points=num_points, split='train')
    dataloader = GeoTransformerDataloader(
        dataset=dataset,
        num_stages=4,
        voxel_size=0.025,  # Fixed value
        search_radius=0.0625,  # Fixed value
        batch_size=1,
        num_workers=0,
        keep_ratio=0.8,  # Fixed value
        sample_threshold=2000  # Fixed value
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

    # Log results
    logger.info(f"\nNumber of Points Impact Analysis (num_points={num_points}):")
    logger.info("-" * 50)

    # Log memory statistics
    log_memory_stats("Memory Statistics", memory_stats)

    # Log data structure
    log_data_structure("Data Structure", batch)

    # Log neighbor limits analysis
    log_neighbor_limits(dataloader)

    # Basic assertions
    assert memory_stats['total'] > 0, "Total memory should be positive"
    assert memory_stats['per_point'] > 0, "Memory per point should be positive"


@pytest.mark.parametrize("voxel_size", [0.01, 0.025, 0.05, 0.1])
def test_voxel_size_impact(voxel_size):
    """Test how voxel size affects memory, data structure, and neighbor limits."""
    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Initial memory
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()

    # Create dataset and dataloader with fixed parameters
    dataset = DummyPCRDataset(num_points=1024, split='train')
    dataloader = GeoTransformerDataloader(
        dataset=dataset,
        num_stages=4,
        voxel_size=voxel_size,
        search_radius=0.0625,  # Fixed value
        batch_size=1,
        num_workers=0,
        keep_ratio=0.8  # Fixed value
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

    # Log results
    logger.info(f"\nVoxel Size Impact Analysis (voxel_size={voxel_size}):")
    logger.info("-" * 50)

    # Log memory statistics
    log_memory_stats("Memory Statistics", memory_stats)

    # Log data structure
    log_data_structure("Data Structure", batch)

    # Log neighbor limits analysis
    log_neighbor_limits(dataloader)

    # Basic assertions
    assert memory_stats['total'] > 0, "Total memory should be positive"
    assert memory_stats['per_point'] > 0, "Memory per point should be positive"


@pytest.mark.parametrize("search_radius", [0.025, 0.0625, 0.125, 0.25])
def test_search_radius_impact(search_radius):
    """Test how search radius affects memory, data structure, and neighbor limits."""
    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Initial memory
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()

    # Create dataset and dataloader with fixed parameters
    dataset = DummyPCRDataset(num_points=1024, split='train')
    dataloader = GeoTransformerDataloader(
        dataset=dataset,
        num_stages=4,
        voxel_size=0.025,  # Fixed value
        search_radius=search_radius,
        batch_size=1,
        num_workers=0,
        keep_ratio=0.8  # Fixed value
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

    # Log results
    logger.info(f"\nSearch Radius Impact Analysis (search_radius={search_radius}):")
    logger.info("-" * 50)

    # Log memory statistics
    log_memory_stats("Memory Statistics", memory_stats)

    # Log data structure
    log_data_structure("Data Structure", batch)

    # Log neighbor limits analysis
    log_neighbor_limits(dataloader)

    # Basic assertions
    assert memory_stats['total'] > 0, "Total memory should be positive"
    assert memory_stats['per_point'] > 0, "Memory per point should be positive"


@pytest.mark.parametrize("keep_ratio", [0.6, 0.8, 0.9])
def test_keep_ratio_impact(keep_ratio):
    """Test how keep ratio affects memory, data structure, and neighbor limits."""
    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Initial memory
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()

    # Create dataset and dataloader with fixed parameters
    dataset = DummyPCRDataset(num_points=1024, split='train')
    dataloader = GeoTransformerDataloader(
        dataset=dataset,
        num_stages=4,
        voxel_size=0.025,  # Fixed value
        search_radius=0.0625,  # Fixed value
        batch_size=1,
        num_workers=0,
        keep_ratio=keep_ratio
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

    # Log results
    logger.info(f"\nKeep Ratio Impact Analysis (keep_ratio={keep_ratio}):")
    logger.info("-" * 50)

    # Log memory statistics
    log_memory_stats("Memory Statistics", memory_stats)

    # Log data structure
    log_data_structure("Data Structure", batch)

    # Log neighbor limits analysis
    log_neighbor_limits(dataloader)

    # Basic assertions
    assert memory_stats['total'] > 0, "Total memory should be positive"
    assert memory_stats['per_point'] > 0, "Memory per point should be positive"


@pytest.mark.parametrize("sample_threshold", [1000, 2000, 4000])
def test_sample_threshold_impact(sample_threshold):
    """Test how sample_threshold affects memory, data structure, and neighbor limits."""
    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Initial memory
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()

    # Create dataset and dataloader with fixed parameters
    dataset = DummyPCRDataset(num_points=1024, split='train')
    dataloader = GeoTransformerDataloader(
        dataset=dataset,
        num_stages=4,
        voxel_size=0.025,  # Fixed value
        search_radius=0.0625,  # Fixed value
        batch_size=1,
        num_workers=0,
        keep_ratio=0.8,  # Fixed value
        sample_threshold=sample_threshold
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

    # Log results
    logger.info(f"\nSample Threshold Impact Analysis (sample_threshold={sample_threshold}):")
    logger.info("-" * 50)

    # Log memory statistics
    log_memory_stats("Memory Statistics", memory_stats)

    # Log data structure
    log_data_structure("Data Structure", batch)

    # Log neighbor limits analysis
    log_neighbor_limits(dataloader)

    # Basic assertions
    assert memory_stats['total'] > 0, "Total memory should be positive"
    assert memory_stats['per_point'] > 0, "Memory per point should be positive"
