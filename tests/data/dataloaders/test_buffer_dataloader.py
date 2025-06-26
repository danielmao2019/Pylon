import pytest
import os
from configs.common.datasets.point_cloud_registration.train.buffer_data_cfg import data_cfg
from utils.builders import build_from_config


def test_buffer_dataloader() -> None:
    from configs.common.datasets.point_cloud_registration.train.buffer_data_cfg import get_transforms_cfg
    
    dataset_cfg = data_cfg['train_dataset']

    # Check if data directory exists, skip if not
    data_root = dataset_cfg['args']['data_root']
    if not os.path.exists(data_root):
        pytest.skip(f"KITTI dataset not found at {data_root}")

    # Add the transforms that BUFFER expects for the Ref stage
    dataset_cfg['args']['transforms_cfg'] = get_transforms_cfg('Euler', 3)
    dataset = build_from_config(dataset_cfg)
    dataloader_cfg = data_cfg['train_dataloader']
    dataloader = build_from_config(dataloader_cfg, dataset=dataset)
    print(f"Total batches: {len(dataloader)}. Checking first 2 batches...")
    idx = 0
    for dp in dataloader:
        print(f"Validating batch {idx}...")
        if idx >= 2:  # Only test 2 batches to keep test fast
            break
        assert isinstance(dp, dict), f"dp is not a dict"
        assert dp.keys() == {'inputs', 'labels', 'meta_info'}, f"dp keys incorrect"
        assert dp['inputs'].keys() == {'points', 'neighbors', 'pools', 'upsamples', 'features', 'stack_lengths', 'src_pcd_raw', 'tgt_pcd_raw', 'src_pcd', 'tgt_pcd', 'relt_pose'}, f"dp['inputs'] keys incorrect"
        assert dp['labels'].keys() == {'transform'}, f"dp['labels'] keys incorrect"
        assert dp['meta_info'].keys() == {'seq', 't0', 't1'}, f"dp['meta_info'] keys incorrect"
        idx += 1
