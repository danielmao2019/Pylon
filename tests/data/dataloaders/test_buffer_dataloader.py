import easydict as edict
from data.datasets import KITTIDataset
from data.dataloaders import BufferDataloader


def test_buffer_dataloader() -> None:
    dataset = KITTIDataset(
        data_root='./data/datasets/soft_links/KITTI',
        split='train',
    )

    dataloader = BufferDataloader(
        dataset=dataset,
        config=edict.EasyDict({
            'point': {
                'conv_radius': 2.0,
            },
            'data': {
                'voxel_size_0': 0.30,
            },
        }),
        batch_size=1,
        num_workers=16,
        shuffle=True,
        drop_last=True,
    )

    for dp in dataloader:
        assert isinstance(dp, dict), f"dp is not a dict"
        assert dp.keys() == {'inputs', 'labels', 'meta_info'}, f"dp keys incorrect"
        assert dp['inputs'].keys() == {'points', 'neighbors', 'pools', 'upsamples', 'features', 'stack_lengths', 'src_pcd_raw', 'tgt_pcd_raw', 'src_pcd', 'tgt_pcd', 'relt_pose'}, f"dp['inputs'] keys incorrect"
        assert dp['labels'].keys() == {'transform'}, f"dp['labels'] keys incorrect"
        assert dp['meta_info'].keys() == {'seq', 't0', 't1'}, f"dp['meta_info'] keys incorrect"
