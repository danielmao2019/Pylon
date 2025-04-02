import pytest
import torch
import logging
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
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
dataloader_voxel_sizes = [0.05]  #, 0.5, 1.0, 1.5, 2.0, 2.5]

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


def process_batch(args: Tuple[int, Dict[str, Any], int]) -> Tuple[Optional[Tuple[int, int, int]], Dict[str, List]]:
    """Process a single batch and collect statistics.
    Returns (None, stats) if test passes, or (error_info, stats) if test fails."""
    batch_idx, batch, num_points_in_patch = args
    points = batch['inputs']['points']
    lengths = batch['inputs']['lengths']
    neighbors = batch['inputs']['neighbors']
    subsampling = batch['inputs']['subsampling']
    upsampling = batch['inputs']['upsampling']
    assert len(points) == len(lengths) == len(neighbors) == len(subsampling) == len(upsampling), \
        f"{len(points)=}, {len(lengths)=}, {len(neighbors)=}, {len(subsampling)=}, {len(upsampling)=}"
    num_stages = len(points)

    # Initialize statistics dictionary with single lists
    stats = {
        'points_shape_0': [],
        'src_lengths': [],
        'tgt_lengths': [],
        'neighbor_counts': [],
        'subsampling_shape_0': [],
        'upsampling_shape_0': [],
    }

    # Check points_f (index 1) which is what's used in the model's assertions
    points_f = points[1]
    lengths_f = lengths[1]

    # Get lengths for this stage
    assert len(lengths_f) == 2
    ref_length_f = lengths_f[0].item()

    # Split points into ref and src
    ref_points_f = points_f[:ref_length_f]
    src_points_f = points_f[ref_length_f:]

    # Check if we have enough points for the patch size
    if ref_points_f.shape[0] < num_points_in_patch or src_points_f.shape[0] < num_points_in_patch:
        return (batch_idx, ref_points_f.shape[0], src_points_f.shape[0]), None

    # Collect statistics for each stage
    for stage in range(num_stages):
        # Points shape
        stats['points_shape_0'].append(points[stage].shape[0])
        # Source and target lengths
        src_length = lengths[stage][1].item()
        tgt_length = lengths[stage][0].item()
        stats['src_lengths'].append(src_length)
        stats['tgt_lengths'].append(tgt_length)

        # Neighbor counts
        neighbors = batch['inputs']['neighbors'][stage]
        assert isinstance(neighbors, torch.Tensor)
        assert neighbors.shape[0] == points[stage].shape[0], f"{neighbors.shape[0]=}, {points[stage].shape[0]=}"
        stats['neighbor_counts'].append(neighbors.shape[1])

        # Subsampling and upsampling ratios
        if stage < num_stages - 1:
            subsampling = batch['inputs']['subsampling'][stage]
            upsampling = batch['inputs']['upsampling'][stage]
            stats['subsampling_shape_0'].append(subsampling.shape[0])
            stats['upsampling_shape_0'].append(upsampling.shape[0])

    return None, stats


def plot_distributions(stats: Dict[str, List], config: Dict[str, float], split: str):
    """Create distribution plots for various data structures across the entire dataset."""
    # Set matplotlib style to a clean, modern look
    plt.style.use('default')

    # Create separate plots for each metric
    metrics = {
        'points_shape_0': 'Total Points',
        'src_lengths': 'Source Points',
        'tgt_lengths': 'Target Points',
        'neighbor_counts': 'Neighbor Counts',
        'subsampling_shape_0': 'Subsampling Shape',
        'upsampling_shape_0': 'Upsampling Shape'
    }

    for metric, title in metrics.items():
        fig = plt.figure(figsize=(15, 10))

        # Get the number of stages from the data
        num_stages = len(stats[metric]) // len(set(stats[metric]))

        # Create subplots for each stage
        for stage in range(num_stages):
            plt.subplot(2, (num_stages + 1) // 2, stage + 1)

            # Extract values for this stage
            stage_values = stats[metric][stage::num_stages]

            # Create histogram with a clean look
            plt.hist(stage_values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            plt.xlabel(title)
            plt.ylabel('Count')
            plt.title(f'Stage {stage} {title} Distribution')
            plt.grid(True, linestyle='--', alpha=0.7)

            # Add mean and std to the plot
            mean_val = np.mean(stage_values)
            std_val = np.std(stage_values)
            plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            plt.axvline(mean_val + std_val, color='green', linestyle=':', label=f'Std: {std_val:.2f}')
            plt.axvline(mean_val - std_val, color='green', linestyle=':')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f'{metric}_{split}_voxel{config["dataloader_voxel_size"]}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Print summary statistics for each stage
    logger.info("\nDataset Statistics Summary:")
    for metric, title in metrics.items():
        logger.info(f"\n{title}:")
        num_stages = len(stats[metric]) // len(set(stats[metric]))
        for stage in range(num_stages):
            stage_values = stats[metric][stage::num_stages]
            logger.info(f"Stage {stage}: mean={np.mean(stage_values):.2f}, std={np.std(stage_values):.2f}")


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
        voxel_size=config['dataset_voxel_size'],
        min_points=256,
        device='cpu',
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

    # Get num_points_in_patch from model config
    num_points_in_patch = model_cfg['args']['model']['num_points_in_patch']
    logger.info(f"Required points per patch: {num_points_in_patch}")

    # Create a pool of workers
    num_cpus = multiprocessing.cpu_count()
    num_workers = max(1, num_cpus - 1)

    # Prepare arguments for parallel processing
    batch_args = [(batch_idx, batch, num_points_in_patch)
                  for batch_idx, batch in enumerate(dataloader)]

    # Initialize statistics dictionary with single lists
    all_stats = {
        'points_shape_0': [],
        'src_lengths': [],
        'tgt_lengths': [],
        'neighbor_counts': [],
        'subsampling_shape_0': [],
        'upsampling_shape_0': [],
    }

    with multiprocessing.Pool(processes=num_workers) as pool:
        # Run all batch tests in parallel using map
        results = pool.map(process_batch, batch_args)

        # Process results
        failed_batches = []
        for error_info, batch_stats in results:
            if error_info is not None:
                failed_batches.append(error_info)
            elif batch_stats is not None:  # Only aggregate if batch_stats is not None
                # Aggregate statistics
                for key in all_stats:
                    all_stats[key].extend(batch_stats[key])
            else:
                assert 0, "Should not reach here."

    # Check if any batches failed
    if failed_batches:
        logger.error("\nFailed batches:")
        for batch_idx, ref_points, src_points in failed_batches:
            if ref_points == -1 and src_points == -1:
                logger.error(f"Batch {batch_idx}: Invalid lengths array")
            else:
                logger.error(f"Batch {batch_idx}: ref_points={ref_points}, src_points={src_points}")
        assert False, f"Found {len(failed_batches)} batches that failed the point count check"
    else:
        logger.info(f"Configuration works for {split} split!")

        # Generate distribution plots
        plot_distributions(all_stats, config, split)
