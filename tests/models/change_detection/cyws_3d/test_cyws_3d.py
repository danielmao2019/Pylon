from models.change_detection.cyws_3d.cyws_3d import CYWS3D
import torch
import torchvision
import data
import yaml
from easydict import EasyDict


def get_easy_dict_from_yaml_file(path_to_yaml_file):
    """
    Reads a yaml and returns it as an easy dict.
    """
    with open(path_to_yaml_file, "r") as stream:
        yaml_file = yaml.safe_load(stream)
    return EasyDict(yaml_file)


def test_cyws_3d() -> None:
    configs = get_easy_dict_from_yaml_file("/home/d6mao/repos/Pylon/models/change_detection/cyws_3d/config.yml")
    model = CYWS3D(configs).to('cuda')
    dataset = data.datasets.KC3DDataset(
        data_root="./data/datasets/soft_links/KC3D",
        split='train', use_ground_truth_registration=True,
        transforms_cfg={
            'class': data.transforms.Compose,
            'args': {
                'transforms': [
                    (
                        torchvision.transforms.Resize(size=(224, 224), antialias=True),
                        ('inputs', 'img_1'),
                    ),
                    (
                        torchvision.transforms.Resize(size=(256, 256), antialias=True),
                        ('inputs', 'img_2'),
                    ),
                ],
            },
        },
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
    )
    for idx, datapoint in enumerate(dataloader):
        if idx > 3:
            break
        _ = model(datapoint['inputs'])
