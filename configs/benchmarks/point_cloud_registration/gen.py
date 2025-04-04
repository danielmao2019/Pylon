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
    with open(f"./configs/benchmarks/point_cloud_registration/template.py", mode='r') as f:
        config = f.read() + '\n'
    config = add_heading(config)
    # add runner
    config += f"from runners import SupervisedSingleTaskTrainer\n"
    config += f"config['runner'] = SupervisedSingleTaskTrainer\n"
    config += '\n'
    # add dataset config
    config += f"# dataset config\n"
    config += f"from configs.common.datasets.point_cloud_registration.train.{dataset}_cfg import config as train_dataset_config\n" 
    config += f"config.update(train_dataset_config)\n"
    config += f"from configs.common.datasets.point_cloud_registration.val.{dataset}_cfg import config as val_dataset_config\n" 
    config += f"config.update(val_dataset_config)\n"
    config += '\n'
    # add model config
    config += f"# model config\n"
    if model == 'GeoTransformer':
        config += f"from configs.common.models.point_cloud_registration.geotransformer_cfg import model_cfg\n"
        config += f"config['model'] = model_cfg\n"
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
            'GeoTransformer',
        ],
    ):
        main(dataset, model)
