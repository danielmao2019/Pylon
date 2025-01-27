import pytest
from .cdd_dataset import CDDDataset
import torch


@pytest.mark.parametrize("dataset", [
    (CDDDataset(data_root="./data/datasets/soft_links/CDD", split='train')),
    (CDDDataset(data_root="./data/datasets/soft_links/CDD", split='val')),
    (CDDDataset(data_root="./data/datasets/soft_links/CDD", split='test')),
])
def test_cdd_dataset(dataset: torch.utils.data.Dataset) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset), "Dataset is not a valid PyTorch dataset instance."
    
    for idx in range(len(dataset)):
        datapoint = dataset[idx]
        
        # Check the structure of the datapoint
        assert isinstance(datapoint, dict), f"Datapoint at index {idx} is not a dictionary."
        assert set(datapoint.keys()) == {'inputs', 'labels', 'meta_info'}, \
            f"Unexpected keys in datapoint at index {idx}: {datapoint.keys()}"

        # Validate inputs
        inputs = datapoint['inputs']
        assert isinstance(inputs, dict), f"Inputs at index {idx} are not a dictionary."
        assert set(inputs.keys()) == set(CDDDataset.INPUT_NAMES), \
            f"Unexpected input keys at index {idx}: {inputs.keys()}"
        
        img_1 = inputs['img_1']
        img_2 = inputs['img_2']
        assert isinstance(img_1, torch.Tensor), f"img_1 at index {idx} is not a Tensor."
        assert isinstance(img_2, torch.Tensor), f"img_2 at index {idx} is not a Tensor."
        assert img_1.ndim == 3 and img_2.ndim == 3, f"Input images at index {idx} must be 3D tensors."
        assert img_1.dtype == torch.float32 and img_2.dtype == torch.float32, \
            f"Input images at index {idx} must have dtype torch.float32."
        assert img_1.shape == img_2.shape, \
            f"Shape mismatch between img_1 and img_2 at index {idx}: {img_1.shape} vs {img_2.shape}"

        # Validate labels
        labels = datapoint['labels']
        assert isinstance(labels, dict), f"Labels at index {idx} are not a dictionary."
        assert set(labels.keys()) == set(CDDDataset.LABEL_NAMES), \
            f"Unexpected label keys at index {idx}: {labels.keys()}"

        change_map = labels['change_map']
        assert isinstance(change_map, torch.Tensor), f"Change map at index {idx} is not a Tensor."
        assert change_map.ndim == 2, f"Change map at index {idx} must be a 2D tensor."
        assert change_map.dtype == torch.int64, f"Change map at index {idx} must have dtype torch.int64."
        unique_values = set(torch.unique(change_map).tolist())
        assert unique_values.issubset({0, 1}), \
            f"Unexpected values in change map at index {idx}: {unique_values}"

        # Validate meta_info
        meta_info = datapoint['meta_info']
        assert isinstance(meta_info, dict), f"Meta info at index {idx} is not a dictionary."
        assert 'image_resolution' in meta_info, f"Missing 'image_resolution' in meta info at index {idx}."
