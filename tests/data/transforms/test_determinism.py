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


def run() -> list:
    dataset = DummyDataset(
        split='train',
        transforms_cfg={
            'class': Compose,
            'args': {
                'transforms': [
                    # 2D transforms
                    {
                        'op': {
                            'class': RandomRotation,
                            'args': {'range': (-45, 45)},
                        },
                        'input_names': [('inputs', 'image')],
                    },
                    {
                        'op': {
                            'class': RandomCrop,
                            'args': {'size': (32, 32)},
                        },
                        'input_names': [('inputs', 'image')],
                    },
                    # 3D transforms
                    {
                        'op': {
                            'class': Shuffle,
                            'args': {},
                        },
                        'input_names': [('inputs', 'point_cloud')],
                    },
                    {
                        'op': {
                            'class': UniformPosNoise,
                            'args': {'min': -0.01, 'max': 0.01},
                        },
                        'input_names': [('inputs', 'point_cloud')],
                    },
                    {
                        'op': {
                            'class': Scale,
                            'args': {'scale_factor': 0.5},
                        },
                        'input_names': [('inputs', 'point_cloud')],
                    },
                ],
            },
        },
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    train_seeds = [42, 43]
    results = []
    for idx in range(2):
        dataloader.dataset.set_base_seed(train_seeds[idx])
        for batch in dataloader:
            results.append(batch)
    return results


def test_transform_determinism():
    """Test that transforms produce identical results across epochs with batched data."""
    # Run twice and compare results
    first_run_results = run()
    second_run_results = run()

    # Compare results
    assert len(first_run_results) == len(second_run_results)
    for batch1, batch2 in zip(first_run_results, second_run_results):
        assert batch1.keys() == batch2.keys() == {'inputs', 'labels', 'meta_info'}
        assert batch1['inputs'].keys() == batch2['inputs'].keys() == {'image', 'point_cloud'}
        assert batch1['labels'].keys() == batch2['labels'].keys() == {'label'}
        assert batch1['meta_info'].keys() == batch2['meta_info'].keys() == {'idx'}
        assert torch.allclose(batch1['inputs']['image'], batch2['inputs']['image'])
        assert torch.allclose(batch1['inputs']['point_cloud'], batch2['inputs']['point_cloud'])
        assert torch.allclose(batch1['labels']['label'], batch2['labels']['label'])
        assert torch.allclose(batch1['meta_info']['idx'], batch2['meta_info']['idx'])
