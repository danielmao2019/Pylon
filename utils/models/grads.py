from typing import List
import torch


def get_flattened_grads(model: torch.nn.Module) -> torch.Tensor:
    grads: List[torch.Tensor] = [
        p.grad if p.grad is not None else torch.zeros_like(p)
        for p in model.parameters() if p.requires_grad
    ]
    grads: torch.Tensor = torch.cat([g.flatten() for g in grads], dim=0)
    assert grads.ndim == 1, f"{grads.shape=}"
    return grads
