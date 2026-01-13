"""LapisGS model loader.

LapisGS uses progressive training across 4 resolution levels (res8, res4, res2, res1).
The finest model (res1) contains all gaussians from all levels in first-slice ordering:
[res8_gaussians | res4_only_gaussians | res2_only_gaussians | res1_only_gaussians]
"""

from pathlib import Path
from typing import Union, Dict
import torch

from models.three_d.original_3dgs.loader import (
    GaussianModel,
    load_3dgs_model_original,
)


@torch.no_grad()
def load_lapis_gs(
    model_dir: Union[str, Path], device: Union[str, torch.device] = 'cuda'
) -> GaussianModel:
    """Load LapisGS model from parent directory containing res1/res2/res4/res8.

    This function loads all 4 resolution models to calculate split points, then
    returns the full res1 model with split_points attribute for layer filtering.

    Args:
        model_dir: Path to lapis parent directory (e.g., .../lapis/)
        device: Device to load model on

    Returns:
        GaussianModel with all layers (full res1) and split_points attribute.
        The split_points attribute is a list [0, res8_end, res4_end, res2_end, res1_end]
        indicating where each layer starts/ends in the gaussian array.
    """
    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(f"LapisGS directory does not exist: {model_dir}")

    # Find and validate 4 resolution subfolders
    res_dirs = _validate_lapis_structure(model_path)

    # Load all 4 models to get gaussian counts
    print(f"Loading LapisGS models from {model_path}")
    models = {}
    for res_name in ['res8', 'res4', 'res2', 'res1']:
        res_dir = res_dirs[res_name]
        print(f"  Loading {res_name} from {res_dir.name}...")
        load_device = device if res_name == 'res1' else 'cpu'
        # Lower resolutions stay on CPU to avoid unnecessary GPU usage.
        models[res_name] = load_3dgs_model_original(str(res_dir), device=load_device)

    # Get gaussian counts
    counts = {
        'res8': models['res8'].get_xyz.shape[0],
        'res4': models['res4'].get_xyz.shape[0],
        'res2': models['res2'].get_xyz.shape[0],
        'res1': models['res1'].get_xyz.shape[0],
    }

    print(f"  Gaussian counts: {counts}")

    # Verify subset chain (res8 ⊂ res4 ⊂ res2 ⊂ res1)
    assert (
        counts['res8'] <= counts['res4'] <= counts['res2'] <= counts['res1']
    ), f"Expected res8 <= res4 <= res2 <= res1, got {counts}"

    # Calculate split points [0, res8_end, res4_end, res2_end, res1_end]
    split_points = [0, counts['res8'], counts['res4'], counts['res2'], counts['res1']]

    print(f"  Split points: {split_points}")
    print(f"  Layer 0 (res8): indices {split_points[0]}:{split_points[1]}")
    print(f"  Layer 1 (res4_only): indices {split_points[1]}:{split_points[2]}")
    print(f"  Layer 2 (res2_only): indices {split_points[2]}:{split_points[3]}")
    print(f"  Layer 3 (res1_only): indices {split_points[3]}:{split_points[4]}")

    # Store split points as LapisGS-specific attributes in the res1 model
    full_model = models['res1']
    assert not hasattr(
        full_model, 'split_points'
    ), "GaussianModel should not have 'split_points' attribute - this is LapisGS-specific"
    assert not hasattr(
        full_model, 'layer_names'
    ), "GaussianModel should not have 'layer_names' attribute - this is LapisGS-specific"

    full_model.split_points = split_points
    full_model.layer_names = ['res8', 'res4', 'res2', 'res1']

    return full_model


def _validate_lapis_structure(model_path: Path) -> Dict[str, Path]:
    """Find and validate 4 resolution subfolders.

    Args:
        model_path: Path to lapis parent directory

    Returns:
        Dictionary mapping resolution names to directory paths

    Raises:
        ValueError: If structure is invalid (missing or duplicate subfolders)
    """
    res_dirs = {}

    for res_suffix in ['_res1', '_res2', '_res4', '_res8']:
        matching_dirs = list(model_path.glob(f'*{res_suffix}'))

        if len(matching_dirs) == 0:
            raise ValueError(
                f"No subfolder ending with '{res_suffix}' found in {model_path}"
            )
        if len(matching_dirs) > 1:
            raise ValueError(
                f"Multiple subfolders ending with '{res_suffix}' found in {model_path}: "
                f"{[d.name for d in matching_dirs]}"
            )

        res_name = res_suffix[1:]  # Remove leading underscore
        res_dirs[res_name] = matching_dirs[0]

    return res_dirs
