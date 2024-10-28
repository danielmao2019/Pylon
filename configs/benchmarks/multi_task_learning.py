from typing import List
import itertools
import os
import sys
sys.path.append("../..")
import data
import utils
os.chdir("../..")


def add_heading(config: str) -> str:
    heading = ""
    heading += "# This file is automatically generated by `./configs/benchmarks/multi_task_learning.py`.\n"
    heading += "# Please do not attempt to modify manually.\n"
    config = heading + config
    return config


def gen_single_task_configs(dataset_name: str, model_name: str, task_name: str) -> None:
    # initializations
    with open(f"./configs/common/template.py", mode='r') as f:
        config = f.read() + '\n'
    config = add_heading(config)
    relpath: str = os.path.join(dataset_name, model_name)
    # add runner
    config += f"from runners import SupervisedSingleTaskTrainer\n"
    config += f"config['runner'] = SupervisedSingleTaskTrainer\n"
    config += '\n'
    # add dataset config
    config += f"# dataset config\n"
    config += f"import data\n"
    config += f"from configs.common.datasets.{dataset_name} import config as dataset_config\n"
    config += f"for key in ['train_dataset', 'val_dataset', 'test_dataset']:\n"
    config += f"    dataset_config[key] = {{\n"
    config += f"        'class': data.datasets.ProjectionDatasetWrapper,\n"
    config += f"        'args': {{\n"
    config += f"            'dataset_config': dataset_config[key],\n"
    config += f"            'mapping': {{\n"
    config += f"                'inputs': ['image'],\n"
    config += f"                'labels': ['{task_name}'],\n"
    config += f"                'meta_info': ['image_filepath', 'image_resolution'],\n"
    config += f"            }},\n"
    config += f"        }},\n"
    config += f"    }}\n"
    config += f"dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['{task_name}']\n"
    config += f"dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['{task_name}']\n"
    config += f"config.update(dataset_config)\n"
    config += '\n'
    # add model config
    config += f"# model config\n"
    config += f"from configs.common.models.{dataset_name}.{model_name} import model_config_{task_name} as model_config\n"
    config += f"config['model'] = model_config\n"
    config += '\n'
    # add optimizer config
    config += f"# optimizer config\n"
    config += f"from configs.common.optimizers._core_ import adam_optimizer_config\n"
    config += f"config['optimizer'] = adam_optimizer_config\n"
    config += '\n'
    # add seeds
    seeded_configs: List[str] = utils.configs.generate_seeds(template_config=config)
    # save to disk
    os.makedirs(os.path.join("./configs/benchmarks", relpath), exist_ok=True)
    for idx, seeded_config in enumerate(seeded_configs):
        seeded_config += f"# work dir\n"
        seeded_config += f"config['work_dir'] = \"" + os.path.join("./logs/benchmarks", relpath, f"single_task_{task_name}_run_{idx}") + "\"\n"
        with open(os.path.join("./configs/benchmarks", relpath, f"single_task_{task_name}_run_{idx}.py"), mode='w') as f:
            f.write(seeded_config)


def gen_method_configs(dataset_name: str, model_name: str, method_name: str) -> None:
    # initializations
    with open(f"./configs/common/template.py", mode='r') as f:
        config = f.read() + '\n'
    config = add_heading(config)
    relpath: str = os.path.join(dataset_name, model_name)
    # add runner
    config += f"from runners import SupervisedMultiTaskTrainer\n"
    config += f"config['runner'] = SupervisedMultiTaskTrainer\n"
    config += '\n'
    # add dataset config
    config += f"# dataset config\n"
    config += f"from configs.common.datasets.{dataset_name} import config as dataset_config\n"
    config += f"config.update(dataset_config)\n"
    config += '\n'
    # add model config
    config += f"# model config\n"
    config += f"from configs.common.models.{dataset_name}.{model_name} import model_config_all_tasks as model_config\n"
    config += f"config['model'] = model_config\n"
    config += '\n'
    # add optimizer config
    config += f"# optimizer config\n"
    config += f"from configs.common.optimizers.{method_name} import optimizer_config\n"
    config += f"from configs.common.optimizers._core_ import adam_optimizer_config\n"
    config += f"optimizer_config['args']['optimizer_config'] = adam_optimizer_config\n"
    config += f"optimizer_config['args']['per_layer'] = False\n"
    config += f"config['optimizer'] = optimizer_config\n"
    config += '\n'
    # add seeds
    seeded_configs: List[str] = utils.configs.generate_seeds(template_config=config)
    # save to disk
    os.makedirs(os.path.join("./configs/benchmarks", relpath), exist_ok=True)
    for idx, seeded_config in enumerate(seeded_configs):
        seeded_config += f"# work dir\n"
        seeded_config += f"config['work_dir'] = \"" + os.path.join("./logs/benchmarks", relpath, f"{method_name}_run_{idx}") + "\"\n"
        with open(os.path.join("./configs/benchmarks", relpath, f"{method_name}_run_{idx}.py"), mode='w') as f:
            f.write(seeded_config)


def main(
    dataset_names: List[str],
    model_names: List[str],
    method_names: List[str],
    task_names: List[str],
) -> None:
    for dataset_name, model_name, task_name in itertools.product(dataset_names, model_names, task_names):
        gen_single_task_configs(dataset_name, model_name, task_name)
    for dataset_name, model_name, method_name in itertools.product(dataset_names, model_names, method_names):
        gen_method_configs(dataset_name, model_name, method_name)


if __name__ == "__main__":
    model_names = ['pspnet_resnet50']
    method_names = ['baseline', 'rgw', 'pcgrad', 'gradvac', 'mgda', 'mgda_ub', 'cagrad', 'graddrop', 'alignedmtl', 'alignedmtl_ub', 'imtl', 'cosreg']
    # CityScapes
    dataset_names = ['city_scapes_c', 'city_scapes_f']
    task_names = data.datasets.CityScapesDataset.LABEL_NAMES
    main(dataset_names, model_names, method_names, task_names)
    # NYUD-MT
    dataset_names = ['nyu_v2_c', 'nyu_v2_f']
    task_names = data.datasets.NYUv2Dataset.LABEL_NAMES
    main(dataset_names, model_names, method_names, task_names)
