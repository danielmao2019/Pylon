from typing import List, Dict
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric
from utils.input_checks import check_point_cloud_segmentation, check_write_file
from utils.ops import transpose_buffer
from utils.io import save_json


class PointCloudConfusionMatrix(SingleTaskMetric):
    """
    Confusion matrix metric for 3D point cloud segmentation tasks.

    This metric computes a confusion matrix and derived metrics (accuracy, precision, recall, F1)
    for point cloud segmentation results.

    Attributes:
        num_classes: Number of classes for segmentation.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Initialize the confusion matrix metric.

        Args:
            num_classes: Number of classes for segmentation.
        """
        super(PointCloudConfusionMatrix, self).__init__()
        assert type(num_classes) == int, f"{type(num_classes)=}"
        assert num_classes > 0, f"{num_classes=}"
        self.num_classes = num_classes

    @staticmethod
    def _get_bincount(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Compute bincount for confusion matrix from predictions and ground truth.

        Args:
            y_pred: Predicted class indices.
            y_true: Ground truth class indices.
            num_classes: Number of classes.

        Returns:
            Bincount tensor of shape (num_classes, num_classes).
        """
        # Input checks
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        assert 0 <= y_true.min() <= y_true.max() < num_classes, f"{y_true.min()=}, {y_true.max()=}, {num_classes=}"
        assert 0 <= y_pred.min() <= y_pred.max() < num_classes, f"{y_pred.min()=}, {y_pred.max()=}, {num_classes=}"

        # Compute bincount
        bincount = torch.bincount(
            y_true * num_classes + y_pred, minlength=num_classes**2,
        ).view((num_classes,) * 2)

        # Output checks
        assert not bincount.isnan().any(), f"{bincount=}"
        assert bincount.sum() == y_true.numel(), f"{bincount.sum()=}, {y_true.shape=}, {y_true.numel()=}"
        return bincount

    @staticmethod
    def _bincount2score(bincount: torch.Tensor, num_points: int) -> Dict[str, torch.Tensor]:
        """
        Convert bincount to metric scores.

        Args:
            bincount: Bincount tensor from _get_bincount.
            num_points: Number of points.

        Returns:
            Dictionary of metric scores.
        """
        score = {
            'tp': bincount.diag(),
            'tn': bincount.sum() - bincount.sum(dim=0) - bincount.sum(dim=1) + bincount.diag(),
            'fp': bincount.sum(dim=0) - bincount.diag(),
            'fn': bincount.sum(dim=1) - bincount.diag(),
        }
        assert torch.all(torch.stack(list(score.values()), dim=0).sum(dim=0) == num_points), \
            f"{torch.stack(list(score.values()), dim=0).sum(dim=0)=}"
        return score

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute confusion matrix scores for point cloud segmentation.

        Args:
            y_pred: Predicted logits tensor of shape [N, C], where N is the total number
                   of points (possibly from multiple samples) and C is the number of classes.
            y_true: Ground truth labels tensor of shape [N].

        Returns:
            Dictionary of metric scores.
        """
        # Input checks
        check_point_cloud_segmentation(y_pred=y_pred, y_true=y_true)
        # Convert logits to class indices
        y_pred = torch.argmax(y_pred, dim=1)
        bincount = self._get_bincount(y_pred=y_pred, y_true=y_true, num_classes=self.num_classes)

        return self._bincount2score(bincount, num_points=y_true.size(0))

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        """
        Summarize accumulated scores.

        Args:
            output_path: Path to save the results, if provided.

        Returns:
            Dictionary of summarized metric scores.
        """
        assert len(self.buffer) != 0
        buffer: Dict[str, List[torch.Tensor]] = transpose_buffer(self.buffer)

        # Summarize scores
        result: Dict[str, torch.Tensor] = {}
        confusion_matrix = {key: torch.stack(buffer[key], dim=0).sum(dim=0) for key in buffer}
        tp = confusion_matrix['tp']
        tn = confusion_matrix['tn']
        fp = confusion_matrix['fp']
        fn = confusion_matrix['fn']
        result.update(confusion_matrix)

        # Per-class metrics
        result['class_accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        result['class_precision'] = tp / (tp + fp + 1e-7)  # Add epsilon to avoid division by zero
        result['class_recall'] = tp / (tp + fn + 1e-7)     # Add epsilon to avoid division by zero
        result['class_f1'] = 2 * tp / (2 * tp + fp + fn + 1e-7)  # Add epsilon to avoid division by zero

        # Global metrics
        total = tp + tn + fp + fn
        assert torch.all(total == total[0]), "Inconsistent total counts across classes"
        result['accuracy'] = tp.sum() / total[0]
        result['mean_precision'] = result['class_precision'].mean()
        result['mean_recall'] = result['class_recall'].mean()
        result['mean_f1'] = result['class_f1'].mean()

        # Save to disk
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=result, filepath=output_path)

        return result
