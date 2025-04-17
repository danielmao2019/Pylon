from typing import List
import os
import sys
sys.path.append("../../..")
import utils
os.chdir("../../..")


def add_heading(config: str) -> str:
    heading = ""
    heading += "# This file is automatically generated by `./configs/benchmarks/point_cloud_registration/gen.py`.\n"
    heading += "# Please do not attempt to modify manually.\n"
    config = heading + config
    return config


def main(dataset: str, overlap: float, model: str) -> None:
    template_name = "template_eval.py" if model in ['ICP', 'RANSAC_FPFH', 'TeaserPlusPlus'] else "template_train.py"
    with open(f"./configs/benchmarks/point_cloud_registration/{template_name}", mode='r') as f:
        config = f.read() + '\n'
    config = add_heading(config)
    # add runner
    if model in ['ICP', 'RANSAC_FPFH', 'TeaserPlusPlus']:
        config += f"from runners import BaseEvaluator\n"
        config += f"config['runner'] = BaseEvaluator\n"
        config += '\n'
    else:
        config += f"from runners import SupervisedSingleTaskTrainer\n"
        config += f"config['runner'] = SupervisedSingleTaskTrainer\n"
        config += '\n'
    # add model config
    if model in ['ICP', 'RANSAC_FPFH', 'TeaserPlusPlus']:
        config += f"# data config\n"
        config += f"from configs.common.datasets.point_cloud_registration.val.classic_{dataset}_data_cfg import data_cfg as eval_data_cfg\n"
        config += f"eval_data_cfg['eval_dataset']['args']['overlap'] = {overlap}\n"
        config += f"config.update(eval_data_cfg)\n"
        config += '\n'
        if model == 'TeaserPlusPlus':
            config += f"# model config\n"
            config += f"from configs.common.models.point_cloud_registration.teaserplusplus_cfg import model_cfg\n"
            config += f"config['model'] = model_cfg\n"
            config += '\n'
        else:
            config += f"# model config\n"
            config += f"from models.point_cloud_registration.classic import {model}\n"
            config += f"config['model'] = {{'class': {model}, 'args': {{}}}}\n"
            config += '\n'
        if model == 'TeaserPlusPlus':
            config += f"config['eval_n_jobs'] = 1\n"
            config += '\n'
    elif model == 'GeoTransformer':
        config += f"# data config\n"
        config += f"from configs.common.datasets.point_cloud_registration.train.geotransformer_{dataset}_data_cfg import data_cfg as train_data_cfg\n"
        config += f"train_data_cfg['train_dataset']['args']['overlap'] = {overlap}\n"
        if dataset == 'real_pcr' and overlap == 0.4:
            config += f"train_data_cfg['train_dataset']['args']['indices'] = list(range(7000))\n"
        config += f"config.update(train_data_cfg)\n"
        config += f"from configs.common.datasets.point_cloud_registration.val.geotransformer_{dataset}_data_cfg import data_cfg as val_data_cfg\n"
        config += f"val_data_cfg['val_dataset']['args']['overlap'] = {overlap}\n"
        config += f"config.update(val_data_cfg)\n"
        config += '\n'
        config += f"# model config\n"
        config += f"from configs.common.models.point_cloud_registration.geotransformer_cfg import model_cfg\n"
        config += f"config['model'] = model_cfg\n"
        config += '\n'
        config += f"from configs.common.criteria.point_cloud_registration.geotransformer_criterion_cfg import criterion_cfg\n"
        config += f"config['criterion'] = criterion_cfg\n"
        config += '\n'
        config += f"from configs.common.metrics.point_cloud_registration.geotransformer_metric_cfg import metric_cfg\n"
        config += f"config['metric'] = metric_cfg\n"
        config += '\n'
    elif model == 'OverlapPredator':
        config += f"# data config\n"
        config += f"from configs.common.datasets.point_cloud_registration.train.overlappredator_{dataset}_data_cfg import data_cfg as train_data_cfg\n"
        config += f"train_data_cfg['train_dataset']['args']['overlap'] = {overlap}\n"
        config += f"config.update(train_data_cfg)\n"
        config += f"from configs.common.datasets.point_cloud_registration.val.overlappredator_{dataset}_data_cfg import data_cfg as val_data_cfg\n"
        config += f"val_data_cfg['val_dataset']['args']['overlap'] = {overlap}\n"
        config += f"config.update(val_data_cfg)\n"
        config += '\n'
        config += f"# model config\n"
        config += f"from configs.common.models.point_cloud_registration.overlappredator_cfg import model_cfg\n"
        config += f"config['model'] = model_cfg\n"
        config += '\n'
        config += f"from configs.common.criteria.point_cloud_registration.overlappredator_criterion_cfg import criterion_cfg\n"
        config += f"config['criterion'] = criterion_cfg\n"
        config += '\n'
        config += f"from configs.common.metrics.point_cloud_registration.overlappredator_metric_cfg import metric_cfg\n"
        config += f"config['metric'] = metric_cfg\n"
        config += '\n'
    else:
        raise NotImplementedError
    # add seeds
    relpath = os.path.join("benchmarks", "point_cloud_registration", dataset, f"overlap_{overlap}")
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
    for dataset, overlap, model in itertools.product(
        ['synth_pcr', 'real_pcr'],
        [1.0, 0.5, 0.4],
        [
            'ICP', 'RANSAC_FPFH', 'TeaserPlusPlus',
            'GeoTransformer', 'OverlapPredator',
        ],
    ):
        main(dataset, overlap, model)
