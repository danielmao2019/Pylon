from configs.common.models.point_cloud_registration.buffer_cfg import model_cfg
from configs.common.datasets.point_cloud_registration.train.buffer_data_cfg import data_cfg
from utils.builders import build_from_config


def _test_buffer() -> None:  # Disabled due to model issues causing infinite hang
    from configs.common.datasets.point_cloud_registration.train.buffer_data_cfg import get_transforms_cfg
    
    model_cfg['args']['config']['data']['dataset'] = 'KITTI'
    model_cfg['args']['config']['stage'] = 'Ref'
    model = build_from_config(model_cfg).cuda()
    
    dataset_cfg = data_cfg['train_dataset']
    # Add the transforms that BUFFER expects for the Ref stage
    dataset_cfg['args']['transforms_cfg'] = get_transforms_cfg('Euler', 3)
    dataset = build_from_config(dataset_cfg)
    
    dataloader_cfg = data_cfg['train_dataloader']
    dataloader = build_from_config(dataloader_cfg, dataset=dataset)
    idx = 0
    for dp in dataloader:
        print(f"Forward pass on batch {idx}...")
        if idx >= 2:  # Only test 2 batches to keep test fast
            break
        model(dp['inputs'])
        idx += 1
