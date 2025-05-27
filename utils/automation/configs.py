from typing import List, Optional
import random
import re


def generate_seeds(
    template_config: str,
    base_seed: str,
    num_repetitions: Optional[int] = 3,
    ub: Optional[int] = 10**8-1,
) -> None:
    # determine number of epochs
    epochs = re.findall(pattern="'epochs': (\d+),", string=template_config)
    if len(epochs) == 0:
        return _generate_eval_seeds(template_config, base_seed, ub)
    else:
        return _generate_train_seeds(template_config, base_seed, num_repetitions, ub)


def _generate_train_seeds(
    template_config: str,
    base_seed: str,
    num_repetitions: Optional[int] = 3,
    ub: Optional[int] = 10**8-1,
) -> None:
    # determine number of epochs
    epochs = re.findall(pattern="'epochs': (\d+),", string=template_config)
    epochs = list(map(int, epochs))
    # generate seeds
    seeded_configs: List[str] = []
    for idx in range(num_repetitions):
        random.seed(base_seed + str(idx))
        config = template_config
        # generate seeds
        if len(epochs) == 1:
            init_seed: int = random.randint(0, ub)
            train_seeds: List[int] = [random.randint(0, ub) for _ in range(epochs[0])]
            config += f"# seeds\n"
            config += f"config['init_seed'] = {init_seed}\n"
            config += f"config['train_seeds'] = [" + ", ".join(list(map(str, train_seeds))) + "]\n"
            config += '\n'
        else:
            init_seed_multi_stage: List[int] = [random.randint(0, ub) for _ in range(len(epochs))]
            train_seeds_multi_stage: List[List[int]] = [
                [random.randint(0, ub) for _ in range(num_epochs)]
                for num_epochs in epochs
            ]
            config += f"# seeds\n"
            for idx in range(len(epochs)):
                config += f"config[{idx}]['init_seed'] = {init_seed_multi_stage[idx]}\n"
                config += f"config[{idx}]['train_seeds'] = [" + ", ".join(list(map(str, train_seeds_multi_stage[idx]))) + "]\n"
            config += '\n'
        # append
        seeded_configs.append(config)
    return seeded_configs


def _generate_eval_seeds(
    template_config: str,
    base_seed: str,
    ub: Optional[int] = 10**8-1,
) -> None:
    # generate seeds
    seeded_configs: List[str] = []
    random.seed(base_seed)
    config = template_config
    # generate seeds
    seed: int = random.randint(0, ub)
    # write down seeds
    config += f"# seeds\n"
    config += f"config['seed'] = {seed}\n"
    config += '\n'
    # append
    seeded_configs.append(config)
    return seeded_configs
