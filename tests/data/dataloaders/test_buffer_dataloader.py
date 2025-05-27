from configs.common.datasets.point_cloud_registration.train.buffer_data_cfg import data_cfg
from utils.builders import build_from_config


def test_buffer_dataloader() -> None:
    dataset_cfg = data_cfg['train_dataset']
    dataset = build_from_config(dataset_cfg)
    dataloader_cfg = data_cfg['train_dataloader']
    dataloader = build_from_config(dataloader_cfg, dataset=dataset)
    idx = 0
    for dp in dataloader:
        if idx >= 10:
            break
        assert isinstance(dp, dict), f"dp is not a dict"
        assert dp.keys() == {'inputs', 'labels', 'meta_info'}, f"dp keys incorrect"
        assert dp['inputs'].keys() == {'points', 'neighbors', 'pools', 'upsamples', 'features', 'stack_lengths', 'src_pcd_raw', 'tgt_pcd_raw', 'src_pcd', 'tgt_pcd', 'relt_pose'}, f"dp['inputs'] keys incorrect"
        assert dp['labels'].keys() == {'transform'}, f"dp['labels'] keys incorrect"
        assert dp['meta_info'].keys() == {'seq', 't0', 't1'}, f"dp['meta_info'] keys incorrect"
        idx += 1
