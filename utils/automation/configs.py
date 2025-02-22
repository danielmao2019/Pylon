from typing import List, Optional
import os
import random
import re


bool_pattern = "(True|False)"
int_pattern = "(?:\+|-)?\d+"
float_pattern = "(?:(?:(?:\+|-)?\d+\.\d+)|(?:\d+\.?\d*e(?:\+|-)?\d+))"
str_pattern = "\"[^\"]*\""
list_int_pattern = f"\[(?:{int_pattern}(?:, )?)*\]"
list_float_pattern = f"\[(?:{float_pattern}(?:, )?)*\]"


def update_config(config, key, val, type_pattern) -> str:
    pattern = f"'{key}': {type_pattern},"
    assert len(re.findall(pattern=pattern, string=config)) == 1, f"{pattern=}, {config=}"
    config = re.sub(
        pattern=pattern, string=config,
        repl=f"'{key}': {str(val)},",
    )
    return config


def generate_seeds(
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
