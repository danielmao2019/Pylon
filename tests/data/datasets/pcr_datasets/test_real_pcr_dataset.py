import pytest
import torch
from data.datasets.pcr_datasets.real_pcr_dataset import RealPCRDataset


@pytest.mark.parametrize('dataset_params', [
    {
        'data_root': './data/datasets/soft_links/ivision-pcr-data',
        'cache_dirname': 'real_pcr_cache',
        'gt_transforms_filepath': './data/datasets/soft_links/ivision-pcr-data/transforms.json',
        'split': 'train',
        'voxel_size': 10.0,
        'min_points': 256,
        'max_points': 8192,
    },
    {
        'data_root': './data/datasets/soft_links/ivision-pcr-data',
        'cache_dirname': 'real_pcr_cache',
        'gt_transforms_filepath': './data/datasets/soft_links/ivision-pcr-data/transforms.json',
        'split': 'val',
        'voxel_size': 10.0,
        'min_points': 256,
        'max_points': 8192,
    },
])
def test_real_pcr_dataset(dataset_params):
    """Test basic functionality of RealPCRDataset."""
    # Initialize dataset
    dataset = RealPCRDataset(**dataset_params)

    # Basic dataset checks
    assert len(dataset) > 0, "Dataset should not be empty"
    assert hasattr(dataset, 'annotations'), "Dataset should have annotations"

    # Iterate through dataset
    for idx in range(len(dataset)):
        datapoint = dataset[idx]
        inputs, labels, meta_info = datapoint['inputs'], datapoint['labels'], datapoint['meta_info']

        # Check inputs structure
        assert isinstance(inputs, dict), "inputs should be a dictionary"
        assert inputs.keys() == {'src_pc', 'tgt_pc'}, "inputs should contain exactly src_pc and tgt_pc"

        # Check point cloud structure
        for pc_name in ['src_pc', 'tgt_pc']:
            pc = inputs[pc_name]
            assert isinstance(pc, dict), f"{pc_name} should be a dictionary"
            assert pc.keys() == {'pos', 'feat'}, f"{pc_name} should contain exactly pos and feat"

            # Check shapes and types
            assert isinstance(pc['pos'], torch.Tensor), f"{pc_name}['pos'] should be a torch.Tensor"
            assert isinstance(pc['feat'], torch.Tensor), f"{pc_name}['feat'] should be a torch.Tensor"
            assert pc['pos'].dim() == 2, f"{pc_name}['pos'] should be 2-dimensional"
            assert pc['pos'].size(1) == 3, f"{pc_name}['pos'] should have 3 coordinates"
            assert pc['feat'].dim() == 2, f"{pc_name}['feat'] should be 2-dimensional"
            assert pc['feat'].size(1) == 1, f"{pc_name}['feat'] should have 1 feature"

            # Check for NaN values
            assert not torch.isnan(pc['pos']).any(), f"{pc_name}['pos'] contains NaN values"
            assert not torch.isnan(pc['feat']).any(), f"{pc_name}['feat'] contains NaN values"

        # Check labels structure
        assert isinstance(labels, dict), "labels should be a dictionary"
        assert labels.keys() == {'transform'}, "labels should contain exactly transform"
        assert isinstance(labels['transform'], torch.Tensor), "transform should be a torch.Tensor"
        assert labels['transform'].shape == (4, 4), "transform should be a 4x4 matrix"
        assert not torch.isnan(labels['transform']).any(), "transform contains NaN values"

        # Validate transformation matrix
        transform = labels['transform']
        R = transform[:3, :3]
        t = transform[:3, 3]

        # Check rotation matrix properties
        assert torch.allclose(R @ R.T, torch.eye(3, device=R.device), atol=1e-6), \
            "Invalid rotation matrix: not orthogonal"
        assert torch.abs(torch.det(R) - 1.0) < 1e-6, \
            "Invalid rotation matrix: determinant not 1"

        # Check meta_info structure
        assert isinstance(meta_info, dict), "meta_info should be a dictionary"
        assert meta_info.keys() >= {'idx', 'src_path', 'tgt_path'}, \
            "meta_info missing required keys"

        # Check that source and target have same number of points
        assert inputs['src_pc']['pos'].shape[0] == inputs['src_pc']['feat'].shape[0], \
            "Source positions and features should have same number of points"
        assert inputs['tgt_pc']['pos'].shape[0] == inputs['tgt_pc']['feat'].shape[0], \
            "Target positions and features should have same number of points"
