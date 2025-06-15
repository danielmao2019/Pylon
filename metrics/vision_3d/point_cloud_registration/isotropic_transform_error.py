from typing import Dict
import torch
import numpy as np
from metrics.wrappers.single_task_metric import SingleTaskMetric


class IsotropicTransformError(SingleTaskMetric):
    """Metric for computing isotropic transform error between predicted and ground truth transformations.

    This metric computes:
    1. Relative Rotation Error (RRE) in degrees
    2. Relative Translation Error (RTE) in the same units as the input point clouds
    """

    @staticmethod
    def _get_rotation_translation(transform):
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
        mat = torch.matmul(rotations.transpose(-1, -2), gt_rotations)
        trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
        x = 0.5 * (trace - 1.0)
        x = x.clamp(min=-1.0, max=1.0)
        x = torch.arccos(x)
        rre = 180.0 * x / np.pi
        return rre

    def _compute_translation_error(self, gt_translations: torch.Tensor, translations: torch.Tensor) -> torch.Tensor:
        """Compute Relative Translation Error (RTE).

        RTE = ||t - \bar{t}||_2

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
            Dictionary containing:
                - 'RRE': Relative rotation error in degrees
                - 'RTE': Relative translation error
        """
        # Input validation
        assert isinstance(y_pred, torch.Tensor), f"Expected torch.Tensor for y_pred, got {type(y_pred)}"
        assert isinstance(y_true, torch.Tensor), f"Expected torch.Tensor for y_true, got {type(y_true)}"
        assert y_pred.shape[-2:] == (4, 4), f"Expected transform shape (*, 4, 4), got {y_pred.shape}"
        assert y_true.shape[-2:] == (4, 4), f"Expected transform shape (*, 4, 4), got {y_true.shape}"

        # Extract rotation and translation from transformation matrices
        gt_rotations, gt_translations = self._get_rotation_translation(y_true)
        rotations, translations = self._get_rotation_translation(y_pred)

        # Compute errors
        rotation_error = self._compute_rotation_error(gt_rotations, rotations)
        assert rotation_error.numel() == 1, f"Expected single value for RRE, got {rotation_error.numel()}"
        translation_error = self._compute_translation_error(gt_translations, translations)
        assert translation_error.numel() == 1, f"Expected single value for RTE, got {translation_error.numel()}"

        return {
            'RRE': rotation_error,
            'RTE': translation_error
        }
