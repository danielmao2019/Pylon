from configs.common.models.point_cloud_registration.buffer_cfg import model_cfg
from configs.common.datasets.point_cloud_registration.train.buffer_data_cfg import data_cfg
from utils.builders import build_from_config


def test_buffer() -> None:
    model_cfg['args']['config']['data']['dataset'] = 'KITTI'
    model_cfg['args']['config']['stage'] = 'Ref'
    model = build_from_config(model_cfg).cuda()
    dataset_cfg = data_cfg['train_dataset']
    dataset = build_from_config(dataset_cfg)
    dataloader_cfg = data_cfg['train_dataloader']
    dataloader = build_from_config(dataloader_cfg, dataset=dataset)
    idx = 0
    for dp in dataloader:
        print(f"Forward pass on batch {idx}...")
        if idx >= 10:
            break
        model(dp['inputs'])
        idx += 1
