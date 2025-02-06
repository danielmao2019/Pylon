from typing import List, Dict
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric
from utils.input_checks import check_write_file
from utils.ops import transpose_buffer
from utils.io import save_json


class ConfusionMatrix(SingleTaskMetric):

    def __init__(self, num_classes: int) -> None:
        super(ConfusionMatrix, self).__init__()
        assert type(num_classes) == int, f"{type(num_classes)=}"
        assert num_classes > 0, f"{num_classes=}"
        self.num_classes = num_classes

    @staticmethod
    def _get_bincount(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
        # input checks
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        assert 0 <= y_true.min() <= y_true.max() < num_classes, f"{y_true.min()=}, {y_true.max()=}, {num_classes=}"
        assert 0 <= y_pred.min() <= y_pred.max() < num_classes, f"{y_pred.min()=}, {y_pred.max()=}, {num_classes=}"
        # compute bincount
        bincount = torch.bincount(
            y_true * num_classes + y_pred, minlength=num_classes**2,
        ).view((num_classes,) * 2)
        # output checks
        assert not bincount.isnan().any(), f"{bincount=}"
        assert bincount.sum() == y_true.numel(), f"{bincount.sum()=}, {y_true.shape=}, {y_true.numel()=}"
        return bincount

    @staticmethod
    def _bincount2score(bincount: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        score = {
            'tp': bincount.diag(),
            'tn': bincount.sum() - bincount.sum(dim=0) - bincount.sum(dim=1) + bincount.diag(),
            'fp': bincount.sum(dim=0) - bincount.diag(),
            'fn': bincount.sum(dim=1) - bincount.diag(),
        }
        assert torch.all(torch.stack(list(score.values()), dim=0).sum(dim=0) == batch_size), f"{torch.stack(list(score.values()), dim=0).sum(dim=0)=}"
        return score

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        # input checks
        assert y_pred.ndim == 2, f"{y_pred.shape=}"
        assert y_true.ndim == 1, f"{y_true.shape=}"
        assert y_pred.shape[0] == y_true.shape[0], f"{y_pred.shape=}, {y_true.shape=}"
        # make prediction from output
        y_pred = torch.argmax(y_pred, dim=1).type(torch.int64)
        # compute confusion matrix
        bincount = self._get_bincount(y_pred=y_pred, y_true=y_true, num_classes=self.num_classes)
        return self._bincount2score(bincount, batch_size=y_true.size(0))

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        assert len(self.buffer) != 0
        buffer: Dict[str, List[torch.Tensor]] = transpose_buffer(self.buffer)
        # summarize scores
        result: Dict[str, torch.Tensor] = {}
        confusion_matrix = {key: torch.stack(buffer[key], dim=0).sum(dim=0) for key in buffer}
        tp = confusion_matrix['tp']
        tn = confusion_matrix['tn']
        fp = confusion_matrix['fp']
        fn = confusion_matrix['fn']
        result.update(confusion_matrix)
        result['class_accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        result['class_precision'] = tp / (tp + fp)
        result['class_recall'] = tp / (tp + fn)
        result['class_f1'] = 2 * tp / (2 * tp + fp + fn)
        total = tp + tn + fp + fn
        assert torch.all(total == total[0])
        result['accuracy'] = tp.sum() / total[0]
        # save to disk
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=result, filepath=output_path)
        return result
