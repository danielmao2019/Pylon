import torch
import tempfile
import os
from typing import Dict, Any, Tuple
from data.datasets.base_dataset import BaseDataset
from data.transforms.compose import Compose
from data.transforms.vision_2d.random_rotation import RandomRotation
from data.transforms.vision_2d.crop.random_crop import RandomCrop
from data.transforms.vision_3d.random_rigid_transform import RandomRigidTransform
from data.transforms.vision_3d.shuffle import Shuffle
from data.transforms.vision_3d.uniform_pos_noise import UniformPosNoise
from data.transforms.vision_3d.scale import Scale


class DummyDataset(BaseDataset):
    """A dummy dataset for testing transform determinism."""

    SPLIT_OPTIONS = ['train']
    DATASET_SIZE = {'train': 4}  # Small dataset size for testing
    INPUT_NAMES = ['image', 'point_cloud']
    LABEL_NAMES = ['label']
    SHA1SUM = None

    def _init_annotations(self) -> None:
        """Initialize dummy annotations."""
        self.annotations = list(range(self.DATASET_SIZE[self.split]))
        # Create a generator for deterministic raw datapoint generation
        self.generator = torch.Generator()

    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load a dummy datapoint with deterministic generation based on idx."""
        # Set generator seed based on idx for deterministic generation
        self.generator.manual_seed(idx)

        inputs = {
            'image': torch.randn(3, 64, 64, generator=self.generator),
            'point_cloud': {
                'pos': torch.randn(1000, 3, generator=self.generator),
                'features': torch.randn(1000, 3, generator=self.generator),
            },
        }

        labels = {
            'label': torch.randint(0, 10, (1,), generator=self.generator),
        }

        meta_info = {
            'idx': idx,
        }

        return inputs, labels, meta_info


def test_transform_determinism():
    """Test that transforms produce identical results across epochs with batched data."""
    # Create dataset with stochastic transforms
    transforms_cfg = {
        'class': Compose,
        'args': {
            'transforms': [
                # 2D transforms
                {
                    'op': RandomRotation(range=(-45, 45)),
                    'input_names': ['image'],
                },
                {
                    'op': RandomCrop(size=(32, 32)),
                    'input_names': ['image'],
                },
                # 3D transforms
                {
                    'op': RandomRigidTransform(rot_mag=45.0, trans_mag=0.5),
                    'input_names': ['point_cloud'],
                },
                {
                    'op': Shuffle(),
                    'input_names': ['point_cloud'],
                },
                {
                    'op': UniformPosNoise(min_noise=-0.01, max_noise=0.01),
                    'input_names': ['point_cloud'],
                },
                {
                    'op': Scale(scale_factor=0.5),
                    'input_names': ['point_cloud'],
                },
            ],
        },
    }

    dataset = DummyDataset(
        split='train',
        transforms_cfg=transforms_cfg,
    )

    # Create dataloader
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling to maintain order
        num_workers=0,  # Single process for determinism
    )

    # Define training seeds for each epoch
    train_seeds = [42, 43]  # Two different seeds for two epochs
    num_epochs = len(train_seeds)

    # Create temporary directories for saving results
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        # First run
        for epoch, seed in enumerate(train_seeds):
            dataset.set_base_seed(seed)
            for batch_idx, batch in enumerate(dataloader):
                # Save batch to first temp directory
                torch.save(batch, os.path.join(tmpdir1, f'epoch_{epoch}_batch_{batch_idx}.pt'))

        # Second run
        for epoch, seed in enumerate(train_seeds):
            dataset.set_base_seed(seed)
            for batch_idx, batch in enumerate(dataloader):
                # Save batch to second temp directory
                torch.save(batch, os.path.join(tmpdir2, f'epoch_{epoch}_batch_{batch_idx}.pt'))

        # Compare results
        for epoch in range(num_epochs):
            for batch_idx in range(len(dataloader)):
                # Load batches from both runs
                batch1 = torch.load(os.path.join(tmpdir1, f'epoch_{epoch}_batch_{batch_idx}.pt'))
                batch2 = torch.load(os.path.join(tmpdir2, f'epoch_{epoch}_batch_{batch_idx}.pt'))

                # Compare all tensors in the batch
                for key in batch1.keys():
                    if isinstance(batch1[key], dict):
                        for subkey in batch1[key].keys():
                            if isinstance(batch1[key][subkey], torch.Tensor):
                                assert torch.allclose(
                                    batch1[key][subkey],
                                    batch2[key][subkey],
                                    rtol=1e-5,
                                    atol=1e-5,
                                ), f"Batches differ in {key}.{subkey} at epoch {epoch}, batch {batch_idx}"
                    elif isinstance(batch1[key], torch.Tensor):
                        assert torch.allclose(
                            batch1[key],
                            batch2[key],
                            rtol=1e-5,
                            atol=1e-5,
                        ), f"Batches differ in {key} at epoch {epoch}, batch {batch_idx}"
