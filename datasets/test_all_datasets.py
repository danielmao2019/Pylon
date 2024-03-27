import pytest
from datasets import CelebA
from datasets import MultiTaskFacialLandmark
from datasets import CityScapes
from datasets import NYUD_MT
import torch


@pytest.mark.parametrize("dataset", [
    (CityScapes(data_root="./datasets/data/city-scapes", split='train')),
    (CityScapes(data_root="./datasets/data/city-scapes", split='train', indices=[0, 2, 4, 6, 8])),
    (NYUD_MT(data_root="./datasets/data/NYUD_MT", split='train')),
    (NYUD_MT(data_root="./datasets/data/NYUD_MT", split='train', indices=[0, 2, 4, 6, 8])),
    (CelebA(data_root="./datasets/data/celeb-a", split='train')),
    (CelebA(data_root="./datasets/data/celeb-a", split='train', indices=[0, 2, 4, 6, 8])),
    (MultiTaskFacialLandmark(data_root="./datasets/data/multi-task-facial-landmark", split='train')),
    (MultiTaskFacialLandmark(data_root="./datasets/data/multi-task-facial-landmark", split='train', indices=[0, 2, 4, 6, 8])),
])
def test_dataset(dataset):
    print(f"Testing {dataset.__class__.__name__} dataset...")
    for i in range(min(len(dataset), 3)):
        example = dataset[i]
        assert type(example) == dict
        image = example['image']
        assert type(image) == torch.Tensor
        assert len(image.shape) == 3, f"{image.shape=}"
        assert -1 <= image.min() <= image.max() <= +1, f"{image.min()=}, {image.max()=}"
        del example['image']
        labels = example
        assert type(labels) == dict
        assert set(labels.keys()) == set(dataset.TASK_NAMES) or set(labels.keys()) == set(dataset.TASK_NAMES+['meta'])
