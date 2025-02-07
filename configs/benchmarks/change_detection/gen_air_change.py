from typing import List
import os
import sys
sys.path.append("../../..")
import utils
os.chdir("../../..")


def add_heading(config: str) -> str:
    heading = ""
    heading += "# This file is automatically generated by `./configs/benchmarks/change_detection/gen_air_change.py`.\n"
    heading += "# Please do not attempt to modify manually.\n"
    config = heading + config
    return config


def main(arch: str) -> None:
    with open(f"./configs/common/template.py", mode='r') as f:
        config = f.read() + '\n'
    config = add_heading(config)
    # add runner
    config += f"from runners import SupervisedSingleTaskTrainer\n"
    config += f"config['runner'] = SupervisedSingleTaskTrainer\n"
    config += '\n'
    # add dataset config
    config += f"# dataset config\n"
    config += f"from configs.common.datasets.change_detection.train.air_change import config as train_dataset_config\n" 
    config += f"config.update(train_dataset_config)\n"
    config += f"from configs.common.datasets.change_detection.val.air_change import config as val_dataset_config\n" 
    config += f"config.update(val_dataset_config)\n"
    config += '\n'
    # add model config
    config += f"# model config\n"
    config += f"from configs.common.models.change_detection.fc_siam import model_config\n"
    config += f"config['model'] = model_config\n"
    config += f"config['model']['args']['arch'] = \"{arch}\"\n"
    config += f"config['model']['args']['in_channels'] = {6 if arch == 'FC-EF' else 3}\n"
    config += '\n'
    # add optimizer config
    config += f"# optimizer config\n"
    config += f"from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config\n"
    config += f"from configs.common.optimizers.standard import adam_optimizer_config\n"
    config += f"optimizer_config['args']['optimizer_config'] = adam_optimizer_config\n"
    config += f"config['optimizer'] = optimizer_config\n"
    config += '\n'
    # add seeds
    relpath = os.path.join("benchmarks", "change_detection", "air_change")
    seeded_configs: List[str] = utils.configs.generate_seeds(
        template_config=config, base_seed=relpath,
    )
    # save to disk
    os.makedirs(os.path.join("./configs", relpath), exist_ok=True)
    for idx, seeded_config in enumerate(seeded_configs):
        seeded_config += f"# work dir\n"
        seeded_config += f"config['work_dir'] = \"" + os.path.join("./logs", relpath, f"{arch}_run_{idx}") + "\"\n"
        with open(os.path.join("./configs", relpath, f"{arch}_run_{idx}.py"), mode='w') as f:
            f.write(seeded_config)


if __name__ == "__main__":
    for arch in ['FC-EF', 'FC-Siam-conc', 'FC-Siam-diff']:
        main(arch)
