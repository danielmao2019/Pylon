from typing import List, Dict
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric
from utils.input_checks import check_write_file
from utils.io import save_json
from utils.ops import transpose_buffer


class ConfusionMatrix(SingleTaskMetric):

    def __init__(self, num_classes: int) -> None:
        super(ConfusionMatrix, self).__init__()
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
    def _bincount2cm(bincount: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Convert bincount to confusion matrix.

        Args:
            bincount: Bincount tensor of shape (num_classes, num_classes).
            batch_size: Number of samples.

        Returns:
            Confusion matrix dictionary with keys 'class_tp', 'class_tn', 'class_fp', 'class_fn'.
        """
        cm = {
            'class_tp': bincount.diag(),
            'class_tn': bincount.sum() - bincount.sum(dim=0) - bincount.sum(dim=1) + bincount.diag(),
            'class_fp': bincount.sum(dim=0) - bincount.diag(),
            'class_fn': bincount.sum(dim=1) - bincount.diag(),
        }
        assert torch.all(torch.stack(list(cm.values()), dim=0).sum(dim=0) == batch_size), f"{torch.stack(list(cm.values()), dim=0).sum(dim=0)=}"
        return cm

    @staticmethod
    def _cm2score(cm: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert confusion matrix to scores.

        Args:
            cm: Confusion matrix dictionary with keys 'class_tp', 'class_tn', 'class_fp', 'class_fn'.

        Returns:
            Scores dictionary with keys 'class_accuracy', 'class_precision', 'class_recall', 'class_f1', 'accuracy', 'mean_precision', 'mean_recall', 'mean_f1'.
        """
        tp = cm['class_tp']
        tn = cm['class_tn']
        fp = cm['class_fp']
        fn = cm['class_fn']
        return {
            'class_accuracy': (tp + tn) / (tp + tn + fp + fn),
            'class_precision': tp / (tp + fp),
            'class_recall': tp / (tp + fn),
            'class_f1': 2 * tp / (2 * tp + fp + fn),
            'accuracy': tp.sum() / (tp + tn + fp + fn)[0],
            'mean_precision': (tp / (tp + fp)).mean(),
            'mean_recall': (tp / (tp + fn)).mean(),
            'mean_f1': (2 * tp / (2 * tp + fp + fn)).mean(),
        }

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        # input checks
        assert y_pred.ndim == 2, f"{y_pred.shape=}"
        assert y_true.ndim == 1, f"{y_true.shape=}"
        assert y_pred.shape[0] == y_true.shape[0], f"{y_pred.shape=}, {y_true.shape=}"
        # make prediction from output
        y_pred = torch.argmax(y_pred, dim=1).type(torch.int64)
        # compute confusion matrix
        bincount = self._get_bincount(y_pred=y_pred, y_true=y_true, num_classes=self.num_classes)
        scores = {}
        cm = self._bincount2cm(bincount, batch_size=y_true.size(0))
        cm_scores = self._cm2score(cm)
        scores.update(cm)
        scores.update(cm_scores)
        return scores

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        """Summarize the metric."""
        self._buffer_queue.join()  # Wait for all items to be processed
        assert self._buffer_queue.empty(), "Buffer queue is not empty when summarizing"
        assert len(self.buffer) != 0

        buffer: Dict[str, List[torch.Tensor]] = transpose_buffer(self.buffer)
        # summarize scores
        result: Dict[str, Dict[str, torch.Tensor]] = {
            "aggregated": {},
            "per_datapoint": buffer,
        }

        # Compute aggregated confusion matrix
        agg_cm = {
            key: torch.stack(buffer[key], dim=0).sum(dim=0)
            for key in ['class_tp', 'class_tn', 'class_fp', 'class_fn']
        }
        agg_scores = self._cm2score(agg_cm)
        result["aggregated"].update(agg_cm)
        result["aggregated"].update(agg_scores)
        # save to disk
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=result, filepath=output_path)
        return result
