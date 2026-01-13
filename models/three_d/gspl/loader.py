import torch

from models.three_d.gspl.model import BaseGSPLModel, GaussianPlyUtils, VanillaGaussian


def load_gspl_model(
    scene_path: str, device: torch.device = torch.device("cuda")
) -> BaseGSPLModel:
    """
    Load only the Gaussian model from a ply scene file (no renderer).
    Mirrors the model-loading path of `initialize_model_and_renderer_from_ply_file`.
    """
    ply_utils = GaussianPlyUtils.load_from_ply(scene_path).to_parameter_structure()

    model_state_dict = {
        "_active_sh_degree": torch.tensor(
            ply_utils.sh_degrees, dtype=torch.int, device=device
        ),
        "gaussians.means": ply_utils.xyz.to(device),
        "gaussians.opacities": ply_utils.opacities.to(device),
        "gaussians.shs_dc": ply_utils.features_dc.to(device),
        "gaussians.shs_rest": ply_utils.features_rest.to(device),
        "gaussians.scales": ply_utils.scales.to(device),
        "gaussians.rotations": ply_utils.rotations.to(device),
    }

    model = VanillaGaussian(sh_degree=ply_utils.sh_degrees).instantiate()

    model.setup_from_number(ply_utils.xyz.shape[0])
    model.to(device)
    model.load_state_dict(model_state_dict, strict=False)

    model.eval()
    model.pre_activate_all_properties()

    return model
