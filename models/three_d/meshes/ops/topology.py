"""
Mesh topology utilities.
"""

import torch


def build_topology_edges_from_faces(faces: torch.Tensor) -> torch.Tensor:
    # Input validations
    assert isinstance(faces, torch.Tensor)
    assert faces.ndim == 2
    assert faces.shape[1] == 3
    assert faces.dtype == torch.long
    assert int(faces.shape[0]) > 0
    assert int(faces.min().item()) >= 0

    topology_edges = torch.cat(
        [
            torch.stack([faces[:, 0], faces[:, 1]], dim=1),
            torch.stack([faces[:, 1], faces[:, 2]], dim=1),
            torch.stack([faces[:, 0], faces[:, 2]], dim=1),
        ],
        dim=0,
    )
    topology_edges, _ = torch.sort(topology_edges, dim=1)
    topology_edges = torch.unique(topology_edges, dim=0)
    return topology_edges
