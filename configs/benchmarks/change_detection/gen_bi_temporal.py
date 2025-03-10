from typing import List
import os
import sys
sys.path.append("../../..")
import utils
os.chdir("../../..")


def add_heading(config: str) -> str:
    heading = ""
    heading += "# This file is automatically generated by `./configs/benchmarks/change_detection/gen_bi_temporal.py`.\n"
    heading += "# Please do not attempt to modify manually.\n"
    config = heading + config
    return config


def main(dataset: str, model: str) -> None:
    with open(f"./configs/benchmarks/change_detection/template.py", mode='r') as f:
        config = f.read() + '\n'
    config = add_heading(config)
    # add runner
    config += f"from runners import SupervisedSingleTaskTrainer\n"
    config += f"config['runner'] = SupervisedSingleTaskTrainer\n"
    config += '\n'
    # add dataset config
    config += f"# dataset config\n"
    config += f"from configs.common.datasets.change_detection.train.{dataset} import config as train_dataset_config\n" 
    config += f"config.update(train_dataset_config)\n"
    config += f"from configs.common.datasets.change_detection.val.{dataset} import config as val_dataset_config\n" 
    config += f"config.update(val_dataset_config)\n"
    config += '\n'
    # add model config
    config += f"# model config\n"
    if model.startswith("FC-"):
        config += f"from configs.common.models.change_detection.fc_siam import model_config\n"
        config += f"config['model'] = model_config\n"
        config += f"config['model']['args']['arch'] = \"{model}\"\n"
        config += f"config['model']['args']['in_channels'] = {6 if model == 'FC-EF' else 3}\n"
        config += '\n'
    elif model == "SNUNet_ECAM":
        config += f"import models\n"
        config += f"config['model'] = {{'class': models.change_detection.SNUNet_ECAM, 'args': {{}}}}\n"
        config += '\n'
        config += f"# criterion config\n"
        config += f"import criteria\n"
        config += f"config['criterion'] = {{'class': criteria.vision_2d.change_detection.SNUNetCDCriterion, 'args': {{}}}}\n"
        config += '\n'
    elif model == "DSIFN":
        config += f"import models\n"
        config += f"config['model'] = {{'class': models.change_detection.DSIFN, 'args': {{}}}}\n"
        config += '\n'
        config += f"# criterion config\n"
        config += f"import criteria\n"
        config += f"config['criterion'] = {{'class': criteria.vision_2d.change_detection.DSIFNCriterion, 'args': {{}}}}\n"
        config += '\n'
    elif model == "TinyCD":
        config += f"import models\n"
        config += f"config['model'] = {{'class': models.change_detection.TinyCD, 'args': {{}}}}\n"
        config += '\n'
    elif model.startswith("Changer"):
        config += f"from configs.common.models.change_detection.changer import changer_{model[len('Changer-'):].replace('-', '_').lower()}_cfg as model_cfg\n"
        config += f"config['model'] = model_cfg\n"
        config += '\n'
        config += f"from configs.common.datasets.change_detection.train._transforms_cfg import transforms_cfg\n"
        if dataset == "air_change":
            config += f"config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(first='ResizeMaps', size=(256, 256))\n"
            config += f"config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(first='RandomCrop', size=(112, 112), resize=(256, 256))\n"
        elif dataset == "oscd":
            config += f"config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(first='RandomCrop', size=(224, 224), resize=(256, 256))\n"
            config += f"config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(first='RandomCrop', size=(224, 224), resize=(256, 256))\n"
        else:
            config += f"config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(size=(256, 256))\n"
            config += f"config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(size=(256, 256))\n"
        config += '\n'
    elif model.startswith("ChangeFormer"):
        config += f"import models\n"
        config += f"config['model'] = {{'class': models.change_detection.{model}, 'args': {{}}}}\n"
        config += '\n'
        config += f"from configs.common.datasets.change_detection.train._transforms_cfg import transforms_cfg\n"
        if dataset == "air_change":
            config += f"config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(first='ResizeMaps', size=(256, 256))\n"
            config += f"config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(first='RandomCrop', size=(112, 112), resize=(256, 256))\n"
        elif dataset == "oscd":
            config += f"config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(first='RandomCrop', size=(224, 224), resize=(256, 256))\n"
            config += f"config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(first='RandomCrop', size=(224, 224), resize=(256, 256))\n"
        else:
            config += f"config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(size=(256, 256))\n"
            config += f"config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(size=(256, 256))\n"
        config += '\n'
        if int(model[-1]) in {4, 5, 6}:
            config += f"# criterion config\n"
            config += f"import criteria\n"
            config += f"""config['criterion'] = {{
    'class': criteria.wrappers.AuxiliaryOutputsCriterion,
    'args': {{
        'criterion_cfg': config['criterion'],
        'reduction': 'mean',
    }},
}}\n"""
            config += '\n'
    elif model.startswith("ChangeNext"):
        config += f"import models\n"
        config += f"config['model'] = {{'class': models.change_detection.{model}, 'args': {{}}}}\n"
        config += '\n'
        config += f"from configs.common.datasets.change_detection.train._transforms_cfg import transforms_cfg\n"
        if dataset == "air_change":
            config += f"config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(first='ResizeMaps', size=(256, 256))\n"
            config += f"config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(first='RandomCrop', size=(112, 112), resize=(256, 256))\n"
        elif dataset == "oscd":
            config += f"config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(first='RandomCrop', size=(224, 224), resize=(256, 256))\n"
            config += f"config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(first='RandomCrop', size=(224, 224), resize=(256, 256))\n"
        else:
            config += f"config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(size=(256, 256))\n"
            config += f"config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(size=(256, 256))\n"
        config += '\n'
        config += f"# criterion config\n"
        config += f"import criteria\n"
        config += f"""config['criterion'] = {{
    'class': criteria.wrappers.AuxiliaryOutputsCriterion,
    'args': {{
        'criterion_cfg': config['criterion'],
        'reduction': 'mean',
    }},
}}\n"""
        config += '\n'
    elif model == "FTN":
        config += f"import models\n"
        config += f"config['model'] = {{'class': models.change_detection.FTN, 'args': {{}}}}\n"
        config += '\n'
        config += f"# criterion config\n"
        config += f"import criteria\n"
        config += f"config['criterion'] = {{'class': criteria.vision_2d.change_detection.FTNCriterion, 'args': {{}}}}\n"
        config += '\n'
    elif model == "SRCNet":
        config += f"import models\n"
        config += f"config['model'] = {{'class': models.change_detection.SRCNet, 'args': {{}}}}\n"
        config += '\n'
        config += f"# criterion config\n"
        config += f"import criteria\n"
        config += f"config['criterion'] = {{'class': criteria.vision_2d.change_detection.SRCNetCriterion, 'args': {{}}}}\n"
        config += '\n'
    elif model == 'BiFA':
        config += f"import models\n"
        config += f"config['model'] = {{'class': models.change_detection.BiFA, 'args': {{}}}}\n"
        config += '\n'
        config += f"import criteria\n"
        config += f"config['criteria'] = {{'class': criteria.vision_2d.CEDiceLoss, 'args': {{}}}}\n"
        config += '\n'
    elif model == "CDXFormer":
        config += f"import models\n"
        config += f"config['model'] = {{'class': models.change_detection.CDXFormer, 'args': {{}}}}\n"
        config += '\n'
        config += f"import criteria\n"
        config += f"config['criteria'] = {{'class': criteria.vision_2d.CEDiceLoss, 'args': {{}}}}\n"
        config += '\n'
    elif model == "CSA_CDGAN":
        config += f"from configs.common.models.change_detection.csa_cdgan import model_config\n"
        config += f"config['model'] = model_config\n"
        config += '\n'
        config += f"# criterion config\n"
        config += f"from configs.common.criteria.change_detection.csa_cdgan import criterion_cfg\n"
        config += f"config['criterion'] = criterion_cfg\n"
        config += '\n'
        config += f"# optimizer config\n"
        config += f"from configs.common.optimizers.gans.csa_cdgan import optimizer_config\n"
        config += f"config['optimizer'] = optimizer_config\n"
        config += '\n'
        config += f"# scheduler config\n"
        config += f"from configs.common.schedulers.gans.gan import scheduler_cfg\n"
        config += f"config['scheduler'] = scheduler_cfg\n"
        config += '\n'
        config += f"from runners.gan_trainers import CSA_CDGAN_Trainer\n"
        config += f"config['runner'] = CSA_CDGAN_Trainer\n"
        config += '\n'
    elif model.startswith("ChangeMamba"):
        config += f"from configs.common.models.change_detection.change_mamba import model_{model.split('-')[1].lower()}_cfg as model_cfg\n"
        config += f"config['model'] = model_cfg\n"
        config += '\n'
        config += f"# criterion config\n"
        config += f"import criteria\n"
        config += f"config['criterion'] = {{'class': criteria.vision_2d.change_detection.STMambaBCDCriterion, 'args': {{}}}}\n"
        config += '\n'
    else:
        raise NotImplementedError
    # add seeds
    relpath = os.path.join("benchmarks", "change_detection", dataset)
    seeded_configs: List[str] = utils.automation.configs.generate_seeds(
        template_config=config, base_seed=relpath,
    )
    # save to disk
    os.makedirs(os.path.join("./configs", relpath), exist_ok=True)
    for idx, seeded_config in enumerate(seeded_configs):
        seeded_config += f"# work dir\n"
        seeded_config += f"config['work_dir'] = \"" + os.path.join("./logs", relpath, f"{model}_run_{idx}") + "\"\n"
        with open(os.path.join("./configs", relpath, f"{model}_run_{idx}.py"), mode='w') as f:
            f.write(seeded_config)


if __name__ == "__main__":
    import itertools
    for dataset, model in itertools.product(
        ['air_change', 'cdd', 'levir_cd', 'oscd', 'sysu_cd', 'whu_cd'],
        [
            'FC-EF', 'FC-Siam-conc', 'FC-Siam-diff', 'SNUNet_ECAM', 'DSIFN', 'TinyCD',
            'Changer-mit-b0', 'Changer-mit-b1', 'Changer-r18', 'Changer-s50', 'Changer-s101',
            'ChangeFormerV1', 'ChangeFormerV2', 'ChangeFormerV3', 'ChangeFormerV4', 'ChangeFormerV5', 'ChangeFormerV6',
            'ChangeNextV1', 'ChangeNextV2', 'ChangeNextV3',
            'FTN', 'SRCNet', 'BiFA', 'CDXFormer',
            'CSA_CDGAN',
            'ChangeMamba-Base', 'ChangeMamba-Small', 'ChangeMamba-Tiny',
        ],
    ):
        main(dataset, model)
