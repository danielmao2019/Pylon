import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
os.chdir(REPO_ROOT)

import criteria
import models
from configs.benchmarks.change_detection.template import config as template_config
from runners.trainers import SupervisedSingleTaskTrainer
from runners.trainers.gan_trainers import CSA_CDGAN_Trainer
from utils.automation.config_seeding import generate_seeded_configs
from utils.automation.config_to_file import add_heading
from utils.builders.builder import semideepcopy


def build_config(dataset: str, model: str):
    """
    Build config for change detection models.
    """
    # Start with template
    config = semideepcopy(template_config)

    # Set runner
    if model == "CSA_CDGAN":
        config['runner'] = CSA_CDGAN_Trainer
    else:
        config['runner'] = SupervisedSingleTaskTrainer

    # Load dataset-specific configs
    if dataset == "air_change":
        from configs.common.datasets.change_detection.train.air_change_data_cfg import (
            data_cfg as train_data_cfg,
        )
        from configs.common.datasets.change_detection.val.air_change_data_cfg import (
            data_cfg as val_data_cfg,
        )
    elif dataset == "cdd":
        from configs.common.datasets.change_detection.train.cdd_data_cfg import (
            data_cfg as train_data_cfg,
        )
        from configs.common.datasets.change_detection.val.cdd_data_cfg import (
            data_cfg as val_data_cfg,
        )
    elif dataset == "levir_cd":
        from configs.common.datasets.change_detection.train.levir_cd_data_cfg import (
            data_cfg as train_data_cfg,
        )
        from configs.common.datasets.change_detection.val.levir_cd_data_cfg import (
            data_cfg as val_data_cfg,
        )
    elif dataset == "oscd":
        from configs.common.datasets.change_detection.train.oscd_data_cfg import (
            data_cfg as train_data_cfg,
        )
        from configs.common.datasets.change_detection.val.oscd_data_cfg import (
            data_cfg as val_data_cfg,
        )
    elif dataset == "sysu_cd":
        from configs.common.datasets.change_detection.train.sysu_cd_data_cfg import (
            data_cfg as train_data_cfg,
        )
        from configs.common.datasets.change_detection.val.sysu_cd_data_cfg import (
            data_cfg as val_data_cfg,
        )
    elif dataset == "ivision_2dcd":
        from configs.common.datasets.change_detection.train.ivision_2dcd_synthetic_data_cfg import (
            data_cfg as train_data_cfg,
        )
        from configs.common.datasets.change_detection.val.ivision_2dcd_synthetic_data_cfg import (
            data_cfg as val_data_cfg,
        )
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")

    # Update template with dataset configs (deep copy to avoid contamination)
    config.update(semideepcopy(train_data_cfg))
    config.update(semideepcopy(val_data_cfg))

    # Set checkpoint frequency for ivision_2dcd experiments
    if dataset == "ivision_2dcd":
        config['checkpoint_method'] = 5  # Save checkpoint every 5 epochs

    # Model-specific configurations
    if model.startswith("FC-"):
        from configs.common.models.change_detection.fc_siam import model_config

        config['model'] = semideepcopy(model_config)
        config['model']['args']['arch'] = model
        config['model']['args']['in_channels'] = 6 if model == 'FC-EF' else 3

    elif model == "SNUNet_ECAM":
        config['model'] = {'class': models.change_detection.SNUNet_ECAM, 'args': {}}
        config['criterion'] = {
            'class': criteria.vision_2d.change_detection.SNUNetCDCriterion,
            'args': {},
        }

    elif model == "RFL_CDNet":
        config['model'] = {'class': models.change_detection.RFL_CDNet, 'args': {}}
        # Modify criterion for auxiliary outputs
        original_criterion = config['criterion']
        original_criterion['args']['use_buffer'] = False
        config['criterion'] = {
            'class': criteria.wrappers.AuxiliaryOutputsCriterion,
            'args': {'criterion_cfg': original_criterion, 'reduction': 'sum'},
        }

    elif model == "DSIFN":
        config['model'] = {'class': models.change_detection.DSIFN, 'args': {}}
        config['criterion'] = {
            'class': criteria.vision_2d.change_detection.DSIFNCriterion,
            'args': {},
        }

    elif model == "TinyCD":
        config['model'] = {'class': models.change_detection.TinyCD, 'args': {}}

    elif model == "HCGMNet":
        config['model'] = {'class': models.change_detection.HCGMNet, 'args': {}}
        # Modify criterion for auxiliary outputs
        original_criterion = config['criterion']
        original_criterion['args']['use_buffer'] = False
        config['criterion'] = {
            'class': criteria.wrappers.AuxiliaryOutputsCriterion,
            'args': {'criterion_cfg': original_criterion, 'reduction': 'sum'},
        }

    elif model == "HANet":
        config['model'] = {'class': models.change_detection.HANet, 'args': {}}
        config['criterion']['class'] = criteria.vision_2d.FocalDiceLoss

        # Add transforms config
        from configs.common.datasets.change_detection.train._transforms_cfg import (
            transforms_cfg,
        )

        if dataset == "air_change":
            config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='ResizeMaps', size=(256, 256)
            )
            config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='RandomCrop', size=(112, 112), resize=(256, 256)
            )
        elif dataset == "oscd":
            config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='RandomCrop', size=(224, 224), resize=(256, 256)
            )
            config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='RandomCrop', size=(224, 224), resize=(256, 256)
            )
        else:
            config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(
                size=(256, 256)
            )
            config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(
                size=(256, 256)
            )

    elif model == 'DsferNet':
        config['model'] = {'class': models.change_detection.DsferNet, 'args': {}}
        config['criterion'] = {
            'class': criteria.vision_2d.change_detection.DsferNetCriterion,
            'args': {},
        }

        # Add transforms config
        from configs.common.datasets.change_detection.train._transforms_cfg import (
            transforms_cfg,
        )

        if dataset == "air_change":
            config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='ResizeMaps', size=(256, 256)
            )
            config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='RandomCrop', size=(112, 112), resize=(256, 256)
            )
        elif dataset == "oscd":
            config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='RandomCrop', size=(224, 224), resize=(256, 256)
            )
            config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='RandomCrop', size=(224, 224), resize=(256, 256)
            )
        else:
            config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(
                size=(256, 256)
            )
            config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(
                size=(256, 256)
            )

    elif model == "DSAMNet":
        config['model'] = {'class': models.change_detection.DSAMNet, 'args': {}}
        config['criterion'] = {
            'class': criteria.vision_2d.change_detection.DSAMNetCriterion,
            'args': {'dice_weight': 0.1},
        }

    elif model.startswith("Changer"):
        # Import appropriate changer config
        changer_variant = model[len('Changer-') :].replace('-', '_').lower()
        if changer_variant == "mit_b0":
            from configs.common.models.change_detection.changer import (
                changer_mit_b0_cfg as model_cfg,
            )
        elif changer_variant == "mit_b1":
            from configs.common.models.change_detection.changer import (
                changer_mit_b1_cfg as model_cfg,
            )
        elif changer_variant == "r18":
            from configs.common.models.change_detection.changer import (
                changer_r18_cfg as model_cfg,
            )
        elif changer_variant == "s50":
            from configs.common.models.change_detection.changer import (
                changer_s50_cfg as model_cfg,
            )
        elif changer_variant == "s101":
            from configs.common.models.change_detection.changer import (
                changer_s101_cfg as model_cfg,
            )
        else:
            raise NotImplementedError(
                f"Changer variant {changer_variant} not implemented"
            )

        config['model'] = model_cfg

        # Add transforms config
        from configs.common.datasets.change_detection.train._transforms_cfg import (
            transforms_cfg,
        )

        if dataset == "air_change":
            config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='ResizeMaps', size=(256, 256)
            )
            config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='RandomCrop', size=(112, 112), resize=(256, 256)
            )
        elif dataset == "oscd":
            config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='RandomCrop', size=(224, 224), resize=(256, 256)
            )
            config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='RandomCrop', size=(224, 224), resize=(256, 256)
            )
        else:
            config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(
                size=(256, 256)
            )
            config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(
                size=(256, 256)
            )

    elif model.startswith("ChangeFormer"):
        # Get the model class
        model_class = getattr(models.change_detection, model)
        config['model'] = {'class': model_class, 'args': {}}

        # Add transforms config
        from configs.common.datasets.change_detection.train._transforms_cfg import (
            transforms_cfg,
        )

        if dataset == "air_change":
            config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='ResizeMaps', size=(256, 256)
            )
            config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='RandomCrop', size=(112, 112), resize=(256, 256)
            )
        elif dataset == "oscd":
            config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='RandomCrop', size=(224, 224), resize=(256, 256)
            )
            config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='RandomCrop', size=(224, 224), resize=(256, 256)
            )
        else:
            config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(
                size=(256, 256)
            )
            config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(
                size=(256, 256)
            )

        # Special handling for versions 4, 5, 6
        if int(model[-1]) in {4, 5, 6}:
            original_criterion = config['criterion']
            original_criterion['args']['use_buffer'] = False
            config['criterion'] = {
                'class': criteria.wrappers.AuxiliaryOutputsCriterion,
                'args': {'criterion_cfg': original_criterion, 'reduction': 'mean'},
            }

    elif model.startswith("ChangeNext"):
        # Get the model class
        model_class = getattr(models.change_detection, model)
        config['model'] = {'class': model_class, 'args': {}}

        # Add transforms config
        from configs.common.datasets.change_detection.train._transforms_cfg import (
            transforms_cfg,
        )

        if dataset == "air_change":
            config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='ResizeMaps', size=(256, 256)
            )
            config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='RandomCrop', size=(112, 112), resize=(256, 256)
            )
        elif dataset == "oscd":
            config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='RandomCrop', size=(224, 224), resize=(256, 256)
            )
            config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(
                first='RandomCrop', size=(224, 224), resize=(256, 256)
            )
        else:
            config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(
                size=(256, 256)
            )
            config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(
                size=(256, 256)
            )

        # Modify criterion for auxiliary outputs
        original_criterion = config['criterion']
        original_criterion['args']['use_buffer'] = False
        config['criterion'] = {
            'class': criteria.wrappers.AuxiliaryOutputsCriterion,
            'args': {'criterion_cfg': original_criterion, 'reduction': 'mean'},
        }

    elif model == "FTN":
        config['model'] = {'class': models.change_detection.FTN, 'args': {}}
        config['criterion'] = {
            'class': criteria.vision_2d.change_detection.FTNCriterion,
            'args': {},
        }

    elif model == "SRCNet":
        config['model'] = {'class': models.change_detection.SRCNet, 'args': {}}
        config['criterion'] = {
            'class': criteria.vision_2d.change_detection.SRCNetCriterion,
            'args': {},
        }

    elif model == 'BiFA':
        config['model'] = {'class': models.change_detection.BiFA, 'args': {}}
        config['criterion'] = {'class': criteria.vision_2d.CEDiceLoss, 'args': {}}

    elif model == "CDXFormer":
        config['model'] = {'class': models.change_detection.CDXFormer, 'args': {}}
        config['criterion'] = {'class': criteria.vision_2d.CEDiceLoss, 'args': {}}

    elif model == "CDMaskFormer":
        from configs.common.models.change_detection.cdmaskformer import model_cfg

        config['model'] = model_cfg
        config['criterion'] = {
            'class': criteria.vision_2d.change_detection.CDMaskFormerCriterion,
            'args': {},
        }

    elif model == "CSA_CDGAN":
        from configs.common.criteria.change_detection.csa_cdgan import criterion_cfg
        from configs.common.models.change_detection.csa_cdgan import model_config
        from configs.common.optimizers.gans.csa_cdgan import optimizer_config
        from configs.common.schedulers.gans.gan import scheduler_cfg

        config['model'] = model_config
        config['criterion'] = criterion_cfg
        config['optimizer'] = optimizer_config
        config['scheduler'] = scheduler_cfg

    elif model.startswith("ChangeMamba"):
        # Import appropriate change mamba config
        mamba_variant = model.split('-')[1].lower()
        if mamba_variant == "base":
            from configs.common.models.change_detection.change_mamba import (
                model_base_cfg as model_cfg,
            )
        elif mamba_variant == "small":
            from configs.common.models.change_detection.change_mamba import (
                model_small_cfg as model_cfg,
            )
        elif mamba_variant == "tiny":
            from configs.common.models.change_detection.change_mamba import (
                model_tiny_cfg as model_cfg,
            )
        else:
            raise NotImplementedError(
                f"ChangeMamba variant {mamba_variant} not implemented"
            )

        config['model'] = model_cfg
        config['criterion'] = {
            'class': criteria.vision_2d.change_detection.STMambaBCDCriterion,
            'args': {},
        }

    else:
        raise NotImplementedError(f"Model {model} not implemented")

    return config


def generate_configs(dataset: str, model: str) -> None:
    """Generate config files for a specific dataset and model combination."""

    # Build config
    config = build_config(dataset, model)

    # Generate seeded configs
    relpath = os.path.join("benchmarks", "change_detection", dataset)
    work_dir = os.path.join("./logs", relpath, model)

    # Generate seeded configs using the new dictionary-based approach
    seeded_configs = generate_seeded_configs(
        base_config=config, base_seed=relpath, base_work_dir=work_dir
    )

    # Add heading and save to disk
    generator_path = "./configs/benchmarks/change_detection/gen_bi_temporal.py"
    os.makedirs(os.path.join("./configs", relpath), exist_ok=True)
    for idx, seeded_config in enumerate(seeded_configs):
        # Add auto-generated header
        final_config = add_heading(seeded_config, generator_path)

        output_path = os.path.join("./configs", relpath, f"{model}_run_{idx}.py")
        with open(output_path, mode='w') as f:
            f.write(final_config)


def main(dataset: str, model: str) -> None:
    """Generate config file for a specific dataset and model combination."""
    generate_configs(dataset, model)


if __name__ == "__main__":
    import itertools

    for dataset, model in itertools.product(
        ['ivision_2dcd', 'air_change', 'cdd', 'levir_cd', 'oscd', 'sysu_cd'],
        [
            'FC-EF',
            'FC-Siam-conc',
            'FC-Siam-diff',
            'SNUNet_ECAM',
            'RFL_CDNet',
            'DSIFN',
            'TinyCD',
            'HCGMNet',
            'HANet',
            'DsferNet',
            'DSAMNet',
            'Changer-mit-b0',
            'Changer-mit-b1',
            'Changer-r18',
            'Changer-s50',
            'Changer-s101',
            'ChangeFormerV1',
            'ChangeFormerV2',
            'ChangeFormerV3',
            'ChangeFormerV4',
            'ChangeFormerV5',
            'ChangeFormerV6',
            'ChangeNextV1',
            'ChangeNextV2',
            'ChangeNextV3',
            'FTN',
            'SRCNet',
            'BiFA',
            'CDXFormer',
            'CDMaskFormer',
            'CSA_CDGAN',
            'ChangeMamba-Base',
            'ChangeMamba-Small',
            'ChangeMamba-Tiny',
        ],
    ):
        main(dataset, model)
