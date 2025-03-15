import os
from pathlib import Path


def get_work_dir(config_file: str) -> str:
    config_path = Path(config_file)
    configs_dir = Path('./configs')
    rel_path = Path(os.path.splitext(config_path.relative_to(configs_dir))[0])
    return str(Path('./logs') / rel_path)


def get_config(work_dir: str) -> str:
    work_path = Path(work_dir)
    logs_dir = Path('./logs')
    rel_path = work_path.relative_to(logs_dir)
    return str(Path('./configs') / rel_path) + '.py'
