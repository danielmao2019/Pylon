from typing import Tuple, Optional, Union
import torch
import numpy as np
from utils.input_checks.check_point_cloud import check_pc_xyz
from utils.point_cloud_ops.knn.knn_faiss import _knn_faiss
from utils.point_cloud_ops.knn.knn_pytorch3d import _knn_pytorch3d
from utils.point_cloud_ops.knn.knn_torch import _knn_torch
from utils.point_cloud_ops.knn.knn_scipy import _knn_scipy


def knn(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    k: Optional[int] = None,
    method: str = "pytorch3d",
    return_distances: bool = True,
    radius: Optional[float] = None,
    chunk_size: Optional[int] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Compute k-nearest neighbors using various backends.

    Args:
        query_points: Query point cloud [N, D] where D is dimension (usually 3)
        reference_points: Reference point cloud [M, D]
        k: Number of nearest neighbors to find (mutually exclusive with radius)
        method: Backend to use - "faiss", "pytorch3d", "torch", or "scipy".
            The winner was "pytorch3d", based on the benchmark results, so set as default.
        return_distances: If True, return (distances, indices). If False, return only indices
        radius: Radius for neighborhood search (mutually exclusive with k)

    Returns:
        If return_distances=True: (distances, indices)
        If return_distances=False: indices only

    Notes:
        - Exactly one of k or radius must be specified (not both, not neither)
        - If k is provided: return k nearest neighbors (inf distance and -1 index for no match)
        - If radius is provided: return all neighbors within radius
    """
    # Use check_pc_xyz for point cloud validation
    check_pc_xyz(query_points)
    check_pc_xyz(reference_points)

    assert (
        query_points.dtype == reference_points.dtype
    ), f"dtype mismatch: {query_points.dtype} vs {reference_points.dtype}"
    assert (
        query_points.device == reference_points.device
    ), f"device mismatch: {query_points.device} vs {reference_points.device}"
    assert k is None or isinstance(k, int), f"k must be None or int, got {type(k)}"
    assert k is None or k > 0, f"k must be positive if provided, got {k}"
    assert method in [
        "faiss",
        "pytorch3d",
        "torch",
        "scipy",
    ], f"Unknown method: {method}"
    assert isinstance(
        return_distances, bool
    ), f"return_distances must be bool, got {type(return_distances)}"
    assert radius is None or isinstance(
        radius, (int, float)
    ), f"radius must be None or numeric, got {type(radius)}"
    assert (
        radius is None or radius > 0
    ), f"radius must be positive if provided, got {radius}"

    # k and radius are mutually exclusive
    assert (k is None) != (
        radius is None
    ), "Exactly one of k or radius must be specified, not both or neither"

    device = query_points.device

    if method == "faiss":
        distances, indices = _knn_faiss(
            query_points=query_points,
            reference_points=reference_points,
            k=k,
            radius=radius,
        )
    elif method == "pytorch3d":
        distances, indices = _knn_pytorch3d(
            query_points=query_points,
            reference_points=reference_points,
            k=k,
            radius=radius,
            chunk_size=chunk_size,
        )
    elif method == "torch":
        distances, indices = _knn_torch(
            query_points=query_points,
            reference_points=reference_points,
            k=k,
            radius=radius,
        )
    elif method == "scipy":
        distances, indices = _knn_scipy(
            query_points=query_points,
            reference_points=reference_points,
            k=k,
            radius=radius,
        )

    # Convert numpy to torch if needed
    if not isinstance(distances, torch.Tensor):
        assert isinstance(
            distances, np.ndarray
        ), f"Expected np.ndarray, got {type(distances)}"
        distances = torch.from_numpy(distances).to(device)
    if not isinstance(indices, torch.Tensor):
        assert isinstance(
            indices, np.ndarray
        ), f"Expected np.ndarray, got {type(indices)}"
        indices = torch.from_numpy(indices).to(device)

    # Ensure correct types
    assert distances.dtype in [
        torch.float32,
        torch.float64,
    ], f"Distances have wrong dtype: {distances.dtype}"
    indices = indices.long()

    # Shape validation depends on whether using k-NN or radius search
    if k is not None:
        assert distances.shape == (
            query_points.shape[0],
            k,
        ), f"Wrong distances shape for k-NN: {distances.shape}"
        assert indices.shape == (
            query_points.shape[0],
            k,
        ), f"Wrong indices shape for k-NN: {indices.shape}"
    # For radius search, shape can vary per query point

    if return_distances:
        return distances, indices
    else:
        return indices
