import torch


def get_flattened_params(model: torch.nn.Module) -> torch.Tensor:
    params = torch.cat([
        p.data.flatten() for p in model.parameters() if p.requires_grad
    ], dim=0)
    params.requires_grad = True
    return params
