"""Conftest file for dataloader tests."""

import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Tuple

import pytest
import torch
import xxhash

from data.collators.base_collator import BaseCollator
from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset
from data.structures.three_d.point_cloud.point_cloud import PointCloud


class DummyPCRDataset(BasePCRDataset):
    """Dummy PCR dataset for testing."""

    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {'train': 10, 'val': 5, 'test': 5}
    SHA1SUM = None

    def __init__(self, data_root: str, split: str = 'train', **kwargs):
        self.split = split
        super().__init__(data_root=data_root, split=split, **kwargs)

    def _init_annotations(self) -> None:
        """Initialize dummy annotations."""
        # DATASET_SIZE is normalized to int by BaseDataset._init_split()
        if isinstance(self.DATASET_SIZE, dict):
            dataset_size = self.DATASET_SIZE[self.split]
        else:
            dataset_size = self.DATASET_SIZE
        self.annotations = [{'idx': i} for i in range(dataset_size)]

    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, PointCloud], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load dummy PCR datapoint with deterministic random generation."""
        # Use deterministic seeding based on idx
        generator = torch.Generator()
        generator.manual_seed((self.base_seed or 0) + idx)

        # Generate random source and target point clouds
        src_points = torch.randn(1000, 3, generator=generator)
        tgt_points = torch.randn(800, 3, generator=generator)

        # Generate random transform matrix
        # Create a random rotation matrix and translation vector
        rotation_angles = torch.randn(3, generator=generator) * 0.1  # Small rotations
        translation = torch.randn(3, generator=generator) * 0.5      # Small translations

        # Simple rotation matrix around z-axis for simplicity
        angle = rotation_angles[2]
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ], dtype=torch.float32)

        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation

        inputs = {
            'src_pc': PointCloud(xyz=src_points),
            'tgt_pc': PointCloud(xyz=tgt_points),
        }

        labels = {
            'transform': transform
        }

        # meta_info is empty as required - base dataset will add 'idx'
        meta_info = {}

        return inputs, labels, meta_info

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Get cache version dict for dummy dataset."""
        base_dict = super()._get_cache_version_dict()
        base_dict['dummy_version'] = 'v1'
        return base_dict


class DummyPCRCollator(BaseCollator):
    """Dummy PCR collator that performs deterministic downsampling."""

    def __init__(self, downsample_factor: float = 0.5, seed: int = 42):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.seed = seed

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch with deterministic downsampling."""
        assert len(batch) == 1, "PCR collator expects batch_size=1"

        datapoint = batch[0]
        inputs = datapoint['inputs']
        labels = datapoint['labels']
        meta_info = datapoint['meta_info']

        # Deterministic downsampling using fixed seed
        generator = torch.Generator()
        generator.manual_seed(self.seed + meta_info['idx'])

        # Downsample source and target point clouds
        src_pc = inputs['src_pc']
        tgt_pc = inputs['tgt_pc']
        assert isinstance(src_pc, PointCloud), f"{type(src_pc)=}"
        assert isinstance(tgt_pc, PointCloud), f"{type(tgt_pc)=}"
        src_xyz = src_pc.xyz
        tgt_xyz = tgt_pc.xyz

        src_n_points = int(len(src_xyz) * self.downsample_factor)
        tgt_n_points = int(len(tgt_xyz) * self.downsample_factor)

        src_indices = torch.randperm(len(src_xyz), generator=generator)[:src_n_points]
        tgt_indices = torch.randperm(len(tgt_xyz), generator=generator)[:tgt_n_points]

        downsampled_inputs = {
            'src_pc': PointCloud(xyz=src_xyz[src_indices]),
            'tgt_pc': PointCloud(xyz=tgt_xyz[tgt_indices]),
        }

        return {
            'inputs': downsampled_inputs,
            'labels': labels,
            'meta_info': [meta_info]  # Collator wraps meta_info in list
        }

    def get_cache_version_hash(self) -> str:
        """Get cache version hash for dummy collator."""
        version_dict = self._get_cache_version_dict()
        hash_str = json.dumps(version_dict, sort_keys=True)
        return xxhash.xxh64(hash_str.encode()).hexdigest()[:16]

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Get cache version dict for dummy collator."""
        return {
            'collator_class': self.__class__.__name__,
            'downsample_factor': self.downsample_factor,
            'seed': self.seed,
            'version': 'v1'
        }


@pytest.fixture
def temp_data_root():
    """Create temporary data root directory."""
    temp_dir = tempfile.mkdtemp(prefix="test_pcr_dataset_")
    try:
        yield temp_dir
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.fixture
def dummy_pcr_dataset(temp_data_root):
    """Create dummy PCR dataset."""
    return DummyPCRDataset(
        data_root=temp_data_root,
        split='train',
        base_seed=123,
        use_cpu_cache=False,
        use_disk_cache=False
    )


@pytest.fixture
def dummy_pcr_collator():
    """Create dummy PCR collator."""
    return DummyPCRCollator(downsample_factor=0.5, seed=42)
