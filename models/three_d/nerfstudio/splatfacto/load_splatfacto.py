import json
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup

from models.three_d.nerfstudio.config_utils import read_data_dir_from_config_path
from models.three_d.nerfstudio.splatfacto.transform import apply_transform_to_splatfacto


def _load_applied_transform(
    data_dir: Path,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    # Input validations
    assert isinstance(data_dir, Path), f"{type(data_dir)=}"
    assert data_dir.is_dir(), f"{data_dir=}"
    assert isinstance(device, torch.device), f"{type(device)=}"

    transforms_path = data_dir / "transforms.json"
    assert transforms_path.is_file(), f"{transforms_path=}"
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    assert (
        "applied_transform" in payload
    ), f"{transforms_path=} missing applied_transform"
    applied_transform = torch.tensor(
        payload["applied_transform"],
        dtype=torch.float32,
        device=device,
    )
    assert applied_transform.shape == (3, 4), f"{applied_transform.shape=}"
    applied_rotation = applied_transform[:, :3]
    applied_translation = applied_transform[:, 3]
    if "applied_scale" in payload:
        applied_scale = float(payload["applied_scale"])
    else:
        applied_scale = 1.0
    assert applied_scale > 0.0, f"{applied_scale=}"
    return applied_rotation, applied_translation, applied_scale


def load_splatfacto_model(
    model_dir: str,
    device: Optional[torch.device] = None,
) -> Pipeline:
    """Load a Splatfacto model from a model directory.

    Args:
        model_dir: Path to the model directory containing config.yml and dataparser_transforms.json
        device: Target device for the model. If None, uses the pipeline's default device.

    Returns:
        Loaded Pipeline object containing the Splatfacto model
    """
    # Input validations
    assert isinstance(model_dir, str), f"{type(model_dir)=}"
    assert device is None or isinstance(device, torch.device), f"{type(device)=}"

    # Construct file paths
    splatfacto_config_path = os.path.join(model_dir, "config.yml")
    splatfacto_transforms_path = os.path.join(model_dir, "dataparser_transforms.json")
    config_path = Path(splatfacto_config_path)

    # Load the pipeline
    _, pipeline, _, _ = eval_setup(config_path, test_mode="test")

    # Use specified device or fallback to pipeline's device
    if device is None:
        device = pipeline.model.device
    else:
        # Move the model to the specified device
        pipeline.model = pipeline.model.to(device)

    # Load dataparser transform and invert to move out of Nerfstudio internal frame
    assert os.path.isfile(splatfacto_transforms_path), f"{splatfacto_transforms_path=}"
    with open(splatfacto_transforms_path, "r", encoding="utf-8") as file_handle:
        transforms = json.load(file_handle)
    assert "transform" in transforms, f"{transforms.keys()=}"
    assert "scale" in transforms, f"{transforms.keys()=}"
    dataparser_transform = torch.tensor(
        transforms["transform"],
        dtype=torch.float32,
        device=device,
    )
    assert dataparser_transform.shape == (3, 4), f"{dataparser_transform.shape=}"
    dataparser_rotation, dataparser_translation = torch.split(
        dataparser_transform,
        [3, 1],
        dim=-1,
    )
    inverse_rotation = dataparser_rotation.T
    inverse_translation = (-inverse_rotation @ dataparser_translation).view(3)
    inverse_scale = 1.0 / float(transforms["scale"])

    # Re-apply dataset applied_transform so rendering uses dataset frame (non-normalized frame in transforms.json)
    data_dir = read_data_dir_from_config_path(config_path=config_path)
    applied_rotation, applied_translation, applied_scale = _load_applied_transform(
        data_dir=data_dir, device=device
    )

    # Apply transforms to the model
    with torch.no_grad():
        apply_transform_to_splatfacto(
            pipeline,
            rotation=inverse_rotation,
            translation=inverse_translation,
            scale=inverse_scale,
        )
        apply_transform_to_splatfacto(
            pipeline,
            rotation=applied_rotation,
            translation=applied_translation,
            scale=None,
        )
        apply_transform_to_splatfacto(
            pipeline,
            rotation=None,
            translation=None,
            scale=applied_scale,
        )

    return pipeline
