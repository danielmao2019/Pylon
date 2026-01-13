"""Test that datasets produce same cache version hash when accessed through different soft links.

This ensures that:
- The same dataset in different locations has the same hash (relocatable)
- Soft links pointing to the same data have the same hash
- Cache is stable regardless of where dataset is stored on disk
"""

import pytest
import tempfile
import os
import shutil

# Import all bi-temporal change detection datasets
from data.datasets.change_detection_datasets.bi_temporal.air_change_dataset import AirChangeDataset
from data.datasets.change_detection_datasets.bi_temporal.cdd_dataset import CDDDataset
from data.datasets.change_detection_datasets.bi_temporal.kc_3d_dataset import KC3DDataset
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset
from data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset import SLPCCDDataset
from data.datasets.change_detection_datasets.bi_temporal.sysu_cd_dataset import SYSU_CD_Dataset
from data.datasets.change_detection_datasets.bi_temporal.urb3dcd_dataset import Urb3DCDDataset
from data.datasets.change_detection_datasets.bi_temporal.xview2_dataset import xView2Dataset


# Define test cases with dataset class and soft link name
BI_TEMPORAL_DATASETS = [
    (AirChangeDataset, "AirChange", {'split': 'train'}),
    (CDDDataset, "CDD", {'split': 'train'}),
    (KC3DDataset, "KC3D", {'split': 'train'}),
    (LevirCdDataset, "LEVIR-CD", {'split': 'train'}),
    (OSCDDataset, "OSCD", {'split': 'train'}),
    (SLPCCDDataset, "SLPCCD", {'split': 'train', 'num_points': 8192}),
    (SYSU_CD_Dataset, "SYSU-CD", {'split': 'train'}),
    (Urb3DCDDataset, "Urb3DCD", {'sample_per_epoch': 100, 'radius': 100}),
    (xView2Dataset, "xView2", {'split': 'train'}),
]


def create_soft_link_to_dataset(original_path: str, link_name: str, temp_dir: str) -> str:
    """Create a soft link in temp directory pointing to the original dataset."""
    link_path = os.path.join(temp_dir, link_name)

    # Resolve the original path in case it's already a symlink
    real_path = os.path.realpath(original_path)

    # Create the symlink
    os.symlink(real_path, link_path)

    return link_path


@pytest.mark.parametrize("dataset_class,soft_link_name,extra_args", BI_TEMPORAL_DATASETS)
def test_bi_temporal_dataset_soft_link_hash_stability(dataset_class, soft_link_name, extra_args):
    """Test that bi-temporal datasets produce same hash when accessed through different soft links."""

    # Original dataset path
    original_path = f"./data/datasets/soft_links/{soft_link_name}"

    # Create dataset with original path
    dataset1 = dataset_class(data_root=original_path, **extra_args)
    hash1 = dataset1.get_cache_version_hash()

    # Create two different soft links pointing to the same data
    with tempfile.TemporaryDirectory() as temp_dir:
        # First alternative soft link
        link_path_1 = create_soft_link_to_dataset(original_path, f"{soft_link_name}_alt1", temp_dir)
        dataset2 = dataset_class(data_root=link_path_1, **extra_args)
        hash2 = dataset2.get_cache_version_hash()

        # Second alternative soft link (simulating dataset moved to different location)
        link_path_2 = create_soft_link_to_dataset(original_path, f"{soft_link_name}_moved", temp_dir)
        dataset3 = dataset_class(data_root=link_path_2, **extra_args)
        hash3 = dataset3.get_cache_version_hash()

        # All hashes should be identical
        assert hash1 == hash2, (
            f"{dataset_class.__name__}: Hash differs between original path and soft link 1\n"
            f"Original ({original_path}): {hash1}\n"
            f"Link 1 ({link_path_1}): {hash2}"
        )

        assert hash1 == hash3, (
            f"{dataset_class.__name__}: Hash differs between original path and soft link 2\n"
            f"Original ({original_path}): {hash1}\n"
            f"Link 2 ({link_path_2}): {hash3}"
        )

        assert hash2 == hash3, (
            f"{dataset_class.__name__}: Hash differs between two soft links\n"
            f"Link 1 ({link_path_1}): {hash2}\n"
            f"Link 2 ({link_path_2}): {hash3}"
        )


def test_comprehensive_soft_link_scenario():
    """Test a complex scenario simulating real-world dataset relocation."""

    # Use LEVIR-CD as an example
    original_path = "./data/datasets/soft_links/LEVIR-CD"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Simulate different scenarios:
        # 1. Dataset on local disk
        local_link = create_soft_link_to_dataset(original_path, "local_datasets/levir", temp_dir)

        # 2. Dataset on network mount
        network_link = create_soft_link_to_dataset(original_path, "mnt/nas/datasets/levir", temp_dir)

        # 3. Dataset on external drive
        external_link = create_soft_link_to_dataset(original_path, "media/usb/ml_data/levir", temp_dir)

        # Create datasets from all paths
        dataset_original = LevirCdDataset(data_root=original_path, split='train')
        dataset_local = LevirCdDataset(data_root=local_link, split='train')
        dataset_network = LevirCdDataset(data_root=network_link, split='train')
        dataset_external = LevirCdDataset(data_root=external_link, split='train')

        # Get hashes
        hash_original = dataset_original.get_cache_version_hash()
        hash_local = dataset_local.get_cache_version_hash()
        hash_network = dataset_network.get_cache_version_hash()
        hash_external = dataset_external.get_cache_version_hash()

        # All should be identical
        assert hash_original == hash_local == hash_network == hash_external, (
            f"Hashes differ across different mount points:\n"
            f"Original: {hash_original}\n"
            f"Local: {hash_local}\n"
            f"Network: {hash_network}\n"
            f"External: {hash_external}"
        )


def test_nested_soft_links():
    """Test that nested soft links (link to a link) also produce same hash."""

    original_path = "./data/datasets/soft_links/OSCD"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a chain of soft links
        link1 = create_soft_link_to_dataset(original_path, "link1", temp_dir)
        link2 = create_soft_link_to_dataset(link1, "link2", temp_dir)
        link3 = create_soft_link_to_dataset(link2, "link3", temp_dir)

        # Create datasets from original and nested links
        dataset_original = OSCDDataset(data_root=original_path, split='train')
        dataset_link1 = OSCDDataset(data_root=link1, split='train')
        dataset_link2 = OSCDDataset(data_root=link2, split='train')
        dataset_link3 = OSCDDataset(data_root=link3, split='train')

        # All should have same hash
        hash_original = dataset_original.get_cache_version_hash()
        hash_link1 = dataset_link1.get_cache_version_hash()
        hash_link2 = dataset_link2.get_cache_version_hash()
        hash_link3 = dataset_link3.get_cache_version_hash()

        assert hash_original == hash_link1 == hash_link2 == hash_link3, (
            f"Nested soft links produce different hashes:\n"
            f"Original: {hash_original}\n"
            f"Link 1: {hash_link1}\n"
            f"Link 2: {hash_link2}\n"
            f"Link 3: {hash_link3}"
        )
