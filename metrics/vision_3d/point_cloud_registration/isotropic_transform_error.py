from typing import Dict, Tuple
import numpy as np
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric


class IsotropicTransformError(SingleTaskMetric):
    """Metric for computing isotropic transform error between predicted and ground truth transformations.

    This metric computes:
    1. Relative Rotation Error (RRE) in degrees
    2. Relative Translation Error (RTE) in the same units as the input point clouds
    """

    def __init__(self, use_buffer: bool = True) -> None:
        """Initialize the IsotropicTransformError metric.

        Args:
            use_buffer: Whether to use buffer for storing results
        """
        super(IsotropicTransformError, self).__init__(use_buffer=use_buffer)

    @staticmethod
    def _get_rotation_translation(transform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Decompose transformation matrix into rotation matrix and translation vector.

        Args:
            transform (Tensor): (*, 4, 4)

        Returns:
            rotation (Tensor): (*, 3, 3)
            translation (Tensor): (*, 3)
        """
        assert transform.shape[-2:] == (4, 4), f"Expected transform shape (*, 4, 4), got {transform.shape}"
        rotation = transform[..., :3, :3]
        translation = transform[..., :3, 3]
        return rotation, translation

    def _compute_rotation_error(self, gt_rotations: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
        r"""Compute Relative Rotation Error (RRE).

        RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

        Args:
            gt_rotations: Ground truth rotation matrix (*, 3, 3)
            rotations: Estimated rotation matrix (*, 3, 3)

        Returns:
            Relative rotation errors in degrees (*)
        """
        assert gt_rotations.shape[-2:] == (3, 3), f"Expected rotation shape (*, 3, 3), got {gt_rotations.shape}"
        assert rotations.shape[-2:] == (3, 3), f"Expected rotation shape (*, 3, 3), got {rotations.shape}"

        # Compute relative rotation: R_rel = R_gt^T @ R_pred
        rel_rotations = torch.matmul(gt_rotations.transpose(-2, -1), rotations)

        # Extract rotation angle using trace
        traces = torch.diagonal(rel_rotations, dim1=-2, dim2=-1).sum(-1)

        # Clamp to valid range for arccos to handle numerical errors
        traces = torch.clamp(traces, -1.0, 3.0)

        # Compute rotation error in radians, then convert to degrees
        rotation_errors_rad = torch.acos((traces - 1.0) / 2.0)
        rotation_errors_deg = rotation_errors_rad * 180.0 / np.pi

        return rotation_errors_deg

    def _compute_translation_error(self, gt_translations: torch.Tensor, translations: torch.Tensor) -> torch.Tensor:
        r"""Compute Relative Translation Error (RTE).

        RTE = ||t_gt - t_pred||_2

        Args:
            gt_translations: Ground truth translation vector (*, 3)
            translations: Estimated translation vector (*, 3)

        Returns:
            Relative translation errors (*)
        """
        rte = torch.linalg.norm(gt_translations - translations, dim=-1)
        return rte

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute rotation and translation errors.

        Args:
            y_pred: Predicted transformation matrix (4x4 tensor)
            y_true: Ground truth transformation matrix (4x4 tensor)

        Returns:
            Dictionary containing rotation and translation errors
        """
        # Input validation
        assert isinstance(y_pred, torch.Tensor), f"Expected torch.Tensor for y_pred, got {type(y_pred)}"
        assert isinstance(y_true, torch.Tensor), f"Expected torch.Tensor for y_true, got {type(y_true)}"

        # Handle batch dimension - expect (1, 4, 4) from SingleTaskMetric
        if y_pred.ndim == 3:
            assert y_pred.shape[0] == 1, f"Expected batch size 1, got {y_pred.shape[0]}"
        elif y_pred.ndim == 2:
            y_pred = y_pred.unsqueeze(0)  # Add batch dimension
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {y_pred.ndim}D")

        if y_true.ndim == 3:
            assert y_true.shape[0] == 1, f"Expected batch size 1, got {y_true.shape[0]}"
        elif y_true.ndim == 2:
            y_true = y_true.unsqueeze(0)  # Add batch dimension
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {y_true.ndim}D")

        assert y_pred.shape == (1, 4, 4), f"Expected (1, 4, 4) transform, got {y_pred.shape}"
        assert y_true.shape == (1, 4, 4), f"Expected (1, 4, 4) transform, got {y_true.shape}"

        # Extract rotation and translation from transformation matrices
        gt_rotations, gt_translations = self._get_rotation_translation(y_true)
        rotations, translations = self._get_rotation_translation(y_pred)

        # Compute errors
        rotation_error = self._compute_rotation_error(gt_rotations, rotations)
        assert rotation_error.shape == (1,), f"{rotation_error.shape=}"
        translation_error = self._compute_translation_error(gt_translations, translations)
        assert translation_error.shape == (1,), f"{translation_error.shape=}"

        return {
            'rotation_error': rotation_error[0],
            'translation_error': translation_error[0],
        }
