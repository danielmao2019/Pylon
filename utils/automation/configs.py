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
    if len(epochs) == 1:
        return _generate_train_seeds(template_config, base_seed, num_repetitions, ub)
    elif len(epochs) == 0:
        return _generate_eval_seeds(template_config, base_seed, ub)
    else:
        raise ValueError(f"{epochs=}")


def _generate_train_seeds(
    template_config: str,
    base_seed: str,
    num_repetitions: Optional[int] = 3,
    ub: Optional[int] = 10**8-1,
) -> None:
    # determine number of epochs
    epochs = re.findall(pattern="'epochs': (\d+),", string=template_config)
    assert len(epochs) == 1, f"{epochs=}"
    epochs = int(epochs[0])
    # generate seeds
    seeded_configs: List[str] = []
    for idx in range(num_repetitions):
        random.seed(base_seed + str(idx))
        config = template_config
        # generate seeds
        init_seed: int = random.randint(0, ub)
        train_seeds: List[int] = [random.randint(0, ub) for _ in range(epochs)]
        # write down seeds
        config += f"# seeds\n"
        config += f"config['init_seed'] = {init_seed}\n"
        config += f"config['train_seeds'] = [" + ", ".join(list(map(str, train_seeds))) + "]\n"
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
