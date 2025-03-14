import pytest
from data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset import SLPCCDDataset
import os
import torch


@pytest.mark.parametrize("split", ['train', 'val', 'test'])
def test_load_real_dataset(split):
    """Test loading the actual SLPCCD dataset."""
    # Set the data root path
    data_root = "./data/datasets/soft_links/SLPCCD"
    
    if not os.path.isdir(data_root):
        pytest.skip("SLPCCD dataset not found in the expected location")
    
    # Load the dataset with minimal preprocessing for faster test
    dataset = SLPCCDDataset(
        data_root=data_root,
        split=split,
        num_points=256,  # Use fewer points for faster testing
        use_hierarchy=False,  # Disable hierarchy for faster testing
        random_subsample=True
    )
    
    # Verify dataset has expected number of samples
    assert len(dataset) > 0, f"No data found in {split} split"
    
    # Test loading each datapoint
    for idx in range(len(dataset)):
        datapoint = dataset[idx]
        
        # Verify basic structure
        assert isinstance(datapoint, dict)
        assert set(datapoint.keys()) == {'inputs', 'labels', 'meta_info'}
        
        # Check inputs
        inputs = datapoint['inputs']
        assert set(inputs.keys()) == set(SLPCCDDataset.INPUT_NAMES)
        
        # Check point cloud data for both time points
        for pc_key in ['pc_1', 'pc_2']:
            assert pc_key in inputs
            assert 'xyz' in inputs[pc_key]
            assert isinstance(inputs[pc_key]['xyz'], torch.Tensor)
            assert inputs[pc_key]['xyz'].shape[1] == 3  # xyz coordinates
            
        # Check labels
        labels = datapoint['labels']
        assert set(labels.keys()) == set(SLPCCDDataset.LABEL_NAMES)
        assert isinstance(labels['change_map'], torch.Tensor)
        
        # Check meta info
        meta = datapoint['meta_info']
        assert 'idx' in meta
        assert 'pc_1_filepath' in meta
        assert 'pc_2_filepath' in meta
        
    print(f"Successfully verified all datapoints in {split} dataset")
