import pytest
import torch
import logging
from typing import Dict, Any, Tuple, List
from data.dataloaders.geotransformer_dataloader import GeoTransformerDataloader
from data.datasets.pcr_datasets.synth_pcr_dataset import SynthPCRDataset
from models.point_cloud_registration.geotransformer.geotransformer import GeoTransformer
from utils.builders.builder import build_from_config
from configs.common.models.point_cloud_registration.geotransformer_cfg import model_cfg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Define parameter ranges for grid search
dataset_voxel_sizes = [10.0]  #, 15.0, 20.0, 25.0, 30.0]
dataloader_voxel_sizes = [0.1]  #, 0.5, 1.0, 1.5, 2.0, 2.5]

# Create full parameter grid
search_configs = []
for dataset_voxel_size in dataset_voxel_sizes:
    for dataloader_voxel_size in dataloader_voxel_sizes:
        # Search radius is fixed to 2.5x dataloader_voxel_size
        search_radius = 2.5 * dataloader_voxel_size
        search_configs.append({
            'dataset_voxel_size': dataset_voxel_size,
            'dataloader_voxel_size': dataloader_voxel_size,
            'search_radius': search_radius
        })

@pytest.mark.parametrize("config", search_configs)
@pytest.mark.parametrize("split", ['train', 'val'])
def test_configuration(config: Dict[str, float], split: str):
    """Test a configuration on a specific split."""
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Fixed parameters
    batch_size = 1
    num_workers = 0
    num_stages = 4
    
    logger.info(f"\nTesting configuration:")
    logger.info(f"Dataset voxel_size: {config['dataset_voxel_size']}")
    logger.info(f"Dataloader voxel_size: {config['dataloader_voxel_size']}")
    logger.info(f"Search radius: {config['search_radius']}")
    logger.info(f"Split: {split}")
    
    # Create dataset
    dataset = SynthPCRDataset(
        data_root='./data/datasets/soft_links/ivision-pcr-data',
        split=split,
        rot_mag=45.0,
        trans_mag=0.5,
        voxel_size=config['dataset_voxel_size']
    )
    
    # Create dataloader
    dataloader = GeoTransformerDataloader(
        dataset=dataset,
        num_stages=num_stages,
        voxel_size=config['dataloader_voxel_size'],
        search_radius=config['search_radius'],
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Create model
    model = build_from_config(model_cfg).cuda()
    num_points_in_patch = model.num_points_in_patch
    
    # Set model mode
    if split == 'train':
        model.train()
    else:
        model.eval()
    
    # Test all batches
    for batch_idx, batch in enumerate(dataloader):
        # Get points from the batch
        points = batch['inputs']['points']
        lengths = batch['inputs']['lengths']
        
        # Log point counts
        logger.info(f"\nBatch {batch_idx} point counts:")
        for i, (p, l) in enumerate(zip(points, lengths)):
            logger.info(f"Stage {i} points: {p.shape[0]}, length: {l[0].item()}")
        
        # # Check if we have enough points for the patch size
        # assert all(p.shape[0] >= num_points_in_patch for p in points), f"Not enough points for patch size in batch {batch_idx}"
        
        # Run model forward pass
        with torch.set_grad_enabled(split == 'train'):
            outputs = model(batch['inputs'])
        
        # Log model outputs
        logger.info(f"Model outputs: {outputs.keys()}")
    
    logger.info(f"Configuration works for {split} split!")
