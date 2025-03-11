import pytest
from data.datasets.random_datasets.classification_random_dataset import ClassificationRandomDataset
from utils.input_checks import check_image


@pytest.mark.parametrize("num_classes, num_examples, image_res, initial_seed", [
    (10, 1000, (512, 512), None),
    (10, 1000, (512, 512), 0),
])
def test_classification_random_dataset(num_classes, num_examples, image_res, initial_seed) -> None:
    dataset = ClassificationRandomDataset(num_classes, num_examples, image_res, initial_seed)
    for idx in range(3):
        example = dataset[idx]
        assert set(example.keys()) == set(['inputs', 'labels', 'meta_info']), f"{example.keys()=}"
        assert set(example['inputs'].keys()) == set(['image']), f"{example['inputs'].keys()=}"
        check_image(example['inputs']['image'], batched=False)
        assert set(example['labels'].keys()) == set(['target']), f"{example['labels'].keys()=}"
        assert example['labels']['target'].shape == (), f"{example['labels']['target'].shape=}"
        assert set(example['meta_info'].keys()) == set(['seed']), f"{example['meta_info'].keys()=}"
        assert type(example['meta_info']['seed']) == int, f"{type(example['meta_info']['seed'])=}"
