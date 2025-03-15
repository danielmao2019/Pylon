import os
from pathlib import Path


def get_work_dir(config_file: str) -> str:
    config_path = Path(config_file).resolve()  # Get absolute path
    configs_dir = Path('./configs').resolve()  # Get absolute path
    rel_path = Path(str(config_path.relative_to(configs_dir)).replace('.py', ''))
    return str(Path('./logs') / rel_path)


def get_config(work_dir: str) -> str:
    work_path = Path(work_dir).resolve()  # Get absolute path
    logs_dir = Path('./logs').resolve()  # Get absolute path
    rel_path = work_path.relative_to(logs_dir)
    return str(Path('./configs') / rel_path) + '.py'
