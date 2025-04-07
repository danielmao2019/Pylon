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


def main(dataset: str, model: str) -> None:
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
        config += "from configs.common.datasets.point_cloud_registration.val.classic_data_cfg import data_cfg as eval_data_cfg\n"
        config += "config.update(eval_data_cfg)\n"
        config += '\n'
        config += f"# model config\n"
        config += f"from models.point_cloud_registration.classic import {model}\n"
        config += f"config['model'] = {{'class': {model}, 'args': {{}}}}\n"
        config += '\n'
    elif model == 'GeoTransformer':
        config += f"# data config\n"
        config += f"from configs.common.datasets.point_cloud_registration.train.geotransformer_data_cfg import data_cfg as train_data_cfg\n" 
        config += f"config.update(train_data_cfg)\n"
        config += f"from configs.common.datasets.point_cloud_registration.val.geotransformer_data_cfg import data_cfg as val_data_cfg\n" 
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
    else:
        raise NotImplementedError
    # add seeds
    relpath = os.path.join("benchmarks", "point_cloud_registration", dataset)
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
        ['synth_pcr_dataset'],
        [
            'ICP', 'RANSAC_FPFH', 'TeaserPlusPlus',
            'GeoTransformer',
        ],
    ):
        main(dataset, model)
