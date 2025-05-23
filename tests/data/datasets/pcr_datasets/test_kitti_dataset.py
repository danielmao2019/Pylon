import pytest
import torch
from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset


@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_kitti_dataset(split: str):
    """Test the structure and content of dataset outputs."""
    # Initialize dataset
    dataset = KITTIDataset(
        data_root='./data/datasets/soft_links/KITTI',
        split=split,
    )

    # Test all samples
    for idx in range(len(dataset))[:5]:
        print(f"Testing sample {idx}")
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"datapoint is not a dict in sample {idx}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}, f"datapoint keys incorrect in sample {idx}"

        # Check inputs
        inputs = datapoint['inputs']
        assert isinstance(inputs, dict), f"inputs is not a dict in sample {idx}"
        assert inputs.keys() == {'src_pc', 'tgt_pc'}, f"inputs keys incorrect in sample {idx}"
        src_pc = inputs['src_pc']
        tgt_pc = inputs['tgt_pc']
        assert isinstance(src_pc, dict), f"src_pc is not a dict in sample {idx}"
        assert isinstance(tgt_pc, dict), f"tgt_pc is not a dict in sample {idx}"
        assert src_pc.keys() == {'pos', 'reflectance'}, f"src_pc keys incorrect in sample {idx}"
        assert tgt_pc.keys() == {'pos', 'reflectance'}, f"tgt_pc keys incorrect in sample {idx}"

        # Check labels
        labels = datapoint['labels']
        assert isinstance(labels, dict), f"labels is not a dict in sample {idx}"
        assert labels.keys() == {'transform'}, f"labels keys incorrect in sample {idx}"
        assert isinstance(labels['transform'], torch.Tensor), f"transform is not torch.Tensor in sample {idx}"
        assert labels['transform'].shape == (4, 4), f"transform shape incorrect in sample {idx}"

        # Check meta_info
        meta_info = datapoint['meta_info']
        assert isinstance(meta_info, dict), f"meta_info is not a dict in sample {idx}"
        assert meta_info.keys() == {'seq', 't0', 't1'}, f"meta_info keys incorrect in sample {idx}"
        assert isinstance(meta_info['seq'], str), f"seq is not str in sample {idx}"
        assert isinstance(meta_info['t0'], int), f"{type(meta_info['t0'])=}"
        assert isinstance(meta_info['t1'], int), f"{type(meta_info['t1'])=}"
