from pathlib import Path

import yaml
from nerfstudio.engine.trainer import TrainerConfig


def load_trainer_config(config_path: Path) -> TrainerConfig:
    # Input validations
    assert isinstance(config_path, Path), f"{type(config_path)=}"
    assert config_path.is_file(), f"{config_path=}"

    config_payload = yaml.load(
        config_path.read_text(encoding="utf-8"),
        Loader=yaml.Loader,
    )
    assert isinstance(
        config_payload, TrainerConfig
    ), f"Expected TrainerConfig, got {type(config_payload)}"
    return config_payload


def read_data_dir_from_config_path(config_path: Path) -> Path:
    # Input validations
    assert isinstance(config_path, Path), f"{type(config_path)=}"
    assert config_path.is_file(), f"{config_path=}"

    config = load_trainer_config(config_path=config_path)
    datamanager_data = config.pipeline.datamanager.data
    dataparser_data = config.pipeline.datamanager.dataparser.data

    datamanager_data_path = None
    if datamanager_data is not None and str(datamanager_data) not in ("", "."):
        datamanager_data_path = Path(str(datamanager_data)).expanduser()
        if not datamanager_data_path.is_absolute():
            datamanager_data_path = (Path.cwd() / datamanager_data_path).resolve()
        else:
            datamanager_data_path = datamanager_data_path.resolve()

    dataparser_data_path = None
    if dataparser_data is not None and str(dataparser_data) not in ("", "."):
        dataparser_data_path = Path(str(dataparser_data)).expanduser()
        if not dataparser_data_path.is_absolute():
            dataparser_data_path = (Path.cwd() / dataparser_data_path).resolve()
        else:
            dataparser_data_path = dataparser_data_path.resolve()

    has_datamanager_data = datamanager_data_path is not None
    has_dataparser_data = dataparser_data_path is not None
    assert has_datamanager_data or has_dataparser_data, (
        "Unable to resolve dataset directory from config. "
        f"{datamanager_data=}, {dataparser_data=}, {config_path=}"
    )

    if has_datamanager_data:
        data_dir_path = datamanager_data_path
    else:
        data_dir_path = dataparser_data_path
    assert data_dir_path is not None, f"{data_dir_path=}"

    assert data_dir_path.is_dir(), f"{data_dir_path=}"
    return data_dir_path
