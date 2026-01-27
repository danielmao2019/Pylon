from pathlib import Path
from typing import Any, Dict, Tuple

from utils.io.colmap.load_colmap import load_model, load_model_text
from utils.io.colmap.save_colmap import save_model_binary, save_model_text


def load_colmap_binary(model_dir: str | Path) -> Any:
    # Input validations
    assert isinstance(model_dir, (str, Path)), f"{type(model_dir)=}"

    from data.structures.colmap.colmap import COLMAP_Data

    return COLMAP_Data(model_dir=model_dir, file_format="binary")


def load_colmap_text(model_dir: str | Path) -> Any:
    # Input validations
    assert isinstance(model_dir, (str, Path)), f"{type(model_dir)=}"

    from data.structures.colmap.colmap import COLMAP_Data

    return COLMAP_Data(model_dir=model_dir, file_format="text")


def load_colmap_model(
    model_dir: str | Path,
    file_format: str,
) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
    # Input validations
    assert isinstance(model_dir, (str, Path)), f"{type(model_dir)=}"
    assert isinstance(file_format, str), f"{type(file_format)=}"
    assert file_format in ("binary", "text"), f"{file_format=}"

    path = Path(model_dir)
    if file_format == "binary":
        return load_model(str(path))
    return load_model_text(str(path))
