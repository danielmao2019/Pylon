import os
from pathlib import Path


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent.parent


def get_work_dir(config_file: str) -> str:
    config_path = Path(config_file).resolve()  # Get absolute path
    configs_dir = (get_repo_root() / 'configs').resolve()  # Get absolute path from repo root
    rel_path = Path(str(config_path.relative_to(configs_dir)).replace('.py', ''))
    return str(get_repo_root() / 'logs' / rel_path)


def get_config(work_dir: str) -> str:
    work_path = Path(work_dir).resolve()  # Get absolute path
    logs_dir = (get_repo_root() / 'logs').resolve()  # Get absolute path from repo root
    rel_path = work_path.relative_to(logs_dir)
    return str(get_repo_root() / 'configs' / rel_path) + '.py'
