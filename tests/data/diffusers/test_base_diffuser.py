import pytest
from typing import Tuple, List
from data.diffusers.base_diffuser import BaseDiffuser
import torch
from data.datasets.multi_task_datasets.city_scapes_dataset import CityScapesDataset


@pytest.mark.parametrize("dataset, num_steps, keys", [
    (
        {
            'class': CityScapesDataset,
            'args': {
                'data_root': "./data/datasets/soft_links/city-scapes",
                'split': 'train',
                'indices': None,
                'transforms': None,
            },
        },
        100, [('labels', 'semantic_segmentation')],
    ),
])
def test_base_diffuser(dataset: torch.utils.data.Dataset, num_steps: int, keys: List[Tuple[str, str]]):
    diffuser = BaseDiffuser(dataset=dataset, num_steps=num_steps, keys=keys)
    for idx in range(10):
        diffuser[idx]
