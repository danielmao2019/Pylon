from typing import Dict
import torch
from metrics.base_metric import BaseMetric
from utils.input_checks import check_write_file
from utils.ops import transpose_buffer
from utils.io import save_json


class ConfusionMatrix(BaseMetric):

    def __init__(self, num_classes: int) -> None:
        super(ConfusionMatrix, self).__init__()
        assert type(num_classes) == int, f"{type(num_classes)=}"
        assert num_classes > 0, f"{num_classes=}"
        self.num_classes = num_classes

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        # input check
        assert y_pred.ndim == 2, f"{y_pred.shape=}"
        assert y_true.ndim == 1, f"{y_true.shape=}"
        assert y_pred.shape[0] == y_true.shape[0], f"{y_pred.shape=}, {y_true.shape=}"
        # make prediction from output
        y_pred = torch.argmax(y_pred, dim=1).type(torch.int64)
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        # compute confusion matrix
        assert 0 <= y_true.min() <= y_true.max() < self.num_classes, f"{y_true.min()=}, {y_true.max()=}, {self.num_classes=}"
        assert 0 <= y_pred.min() <= y_pred.max() < self.num_classes, f"{y_pred.min()=}, {y_pred.max()=}, {self.num_classes=}"
        count = torch.bincount(
            y_true * self.num_classes + y_pred, minlength=self.num_classes**2,
        ).view((self.num_classes,) * 2)
        assert not count.isnan().any(), f"{count=}"
        assert count.sum() == y_true.shape[0], f"{count.sum()=}, {y_true.shape=}"
        score = {
            'tp': count.diag(),
            'tn': count.sum() - count.sum(dim=0) - count.sum(dim=1) + count.diag(),
            'fp': count.sum(dim=0) - count.diag(),
            'fn': count.sum(dim=1) - count.diag(),
        }
        assert torch.all(torch.stack(list(score.values()), dim=0).sum(dim=0) == y_true.shape[0]), f"{torch.stack(list(score.values()), dim=0).sum(dim=0)=}"
        return score

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        if output_path is not None:
            check_write_file(path=output_path)
        result: Dict[str, torch.Tensor] = {}
        if len(self.buffer) != 0:
            buffer = transpose_buffer(self.buffer)
            confusion_matrix = {key: torch.stack(buffer[key], dim=0).sum(dim=0) for key in buffer}
            tp = confusion_matrix['tp']
            tn = confusion_matrix['tn']
            fp = confusion_matrix['fp']
            fn = confusion_matrix['fn']
            result.update(confusion_matrix)
            result['per_class_accuracy'] = (tp + tn) / (tp + tn + fp + fn)
            result['per_class_precision'] = tp / (tp + fp)
            result['per_class_recall'] = tp / (tp + fn)
            result['per_class_f1'] = 2 * tp / (2 * tp + fp + fn)
            total = tp + tn + fp + fn
            assert torch.all(total == total[0])
            result['accuracy'] = tp.sum() / total[0]
            result['reduced'] = result['accuracy'].clone()
        # save to disk
        if output_path is not None:
            save_json(obj=result, filepath=output_path)
        return result
