import torch
from typing import Dict
from metrics.wrappers.single_task_metric import SingleTaskMetric


class CorrespondencePrecisionRecall(SingleTaskMetric):
    """
    Correspondence Precision and Recall metric for 3D point cloud registration.

    This metric computes precision, recall, and F1 score based on point correspondences
    after transformation, comparing the predicted correspondences against ground truth.
    """

    DIRECTION = 1  # Higher is better

    def __init__(self, threshold: float = 0.02) -> None:
        """
        Initialize the Correspondence Precision and Recall metric.

        Args:
            threshold: Distance threshold for considering a correspondence as valid (default: 0.02 units)
        """
        super(CorrespondencePrecisionRecall, self).__init__()
        self.threshold = threshold

    def _compute_score(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute precision and recall based on point correspondences.

        Args:
            y_pred: Predicted correspondences as a tensor of shape (N, 2, 3) where
                   each entry is a pair of (source, target) 3D point coordinates
            y_true: Ground truth correspondences as a tensor of shape (M, 2, 3)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing precision, recall, and F1 scores
        """
        # Input checks
        assert y_pred.dim() == 3 and y_pred.size(1) == 2 and y_pred.size(2) == 3, \
            f"Expected y_pred shape (N, 2, 3), got {y_pred.shape}"
        assert y_true.dim() == 3 and y_true.size(1) == 2 and y_true.size(2) == 3, \
            f"Expected y_true shape (M, 2, 3), got {y_true.shape}"

        # Extract source and target points from correspondences
        pred_source = y_pred[:, 0, :]  # (N, 3)
        pred_target = y_pred[:, 1, :]  # (N, 3)

        gt_source = y_true[:, 0, :]    # (M, 3)
        gt_target = y_true[:, 1, :]    # (M, 3)

        # Define a correspondence as a valid match if both endpoints are within threshold
        # For this, we need to find matches between predicted and ground truth correspondences

        # Check matches for source points
        # Expand dimensions for broadcasting
        pred_source_expanded = pred_source.unsqueeze(1)  # (N, 1, 3)
        gt_source_expanded = gt_source.unsqueeze(0)     # (1, M, 3)

        # Compute distances between all pairs of source points
        source_distances = torch.sqrt(((pred_source_expanded - gt_source_expanded) ** 2).sum(dim=2))  # (N, M)

        # Identify valid source matches (below threshold)
        source_matches = source_distances < self.threshold  # (N, M) boolean tensor

        # Check matches for target points
        pred_target_expanded = pred_target.unsqueeze(1)  # (N, 1, 3)
        gt_target_expanded = gt_target.unsqueeze(0)     # (1, M, 3)

        # Compute distances between all pairs of target points
        target_distances = torch.sqrt(((pred_target_expanded - gt_target_expanded) ** 2).sum(dim=2))  # (N, M)

        # Identify valid target matches (below threshold)
        target_matches = target_distances < self.threshold  # (N, M) boolean tensor

        # A correspondence is valid if both endpoints match
        valid_correspondences = source_matches & target_matches  # (N, M) boolean tensor

        # Count true positives (valid matches)
        tp = valid_correspondences.float().sum()

        # Compute precision and recall
        precision = tp / y_pred.size(0) if y_pred.size(0) > 0 else torch.tensor(0.0)
        recall = tp / y_true.size(0) if y_true.size(0) > 0 else torch.tensor(0.0)

        # Compute F1 score
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else torch.tensor(0.0)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        """
        Summarize accumulated scores.

        Args:
            output_path: Path to save the results, if provided.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of summarized metric scores.
        """
        # Use the parent class's summarize method
        result = super(CorrespondencePrecisionRecall, self).summarize(output_path)

        # If we already have the three main metrics in result, just return them
        if all(key in result for key in ["precision", "recall", "f1_score"]):
            return result

        # If f1_score is not already in the result, compute it from precision and recall
        if "precision" in result and "recall" in result and "f1_score" not in result:
            precision = result["precision"]
            recall = result["recall"]
            result["f1_score"] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else torch.tensor(0.0)

        return result
