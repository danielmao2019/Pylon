from typing import List, Dict, Optional, Any
import torch
import torchvision
from metrics.common import ConfusionMatrix
from metrics.wrappers.single_task_metric import SingleTaskMetric
from utils.input_checks import check_write_file, check_semantic_segmentation
from utils.io import save_json
from utils.ops import transpose_buffer


class SemanticSegmentationMetric(SingleTaskMetric):

    DIRECTION = +1

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None) -> None:
        super(SemanticSegmentationMetric, self).__init__()
        assert type(num_classes) == int, f"{type(num_classes)=}"
        assert num_classes > 0, f"{num_classes=}"
        self.num_classes = num_classes
        if ignore_index is None:
            ignore_index = 255
        self.ignore_index = ignore_index

    @staticmethod
    def _bincount2score(bincount: torch.Tensor, nan_mask: torch.Tensor, num_classes: int) -> Dict[str, torch.Tensor]:
        # compute intersection over union
        intersection = bincount.diag()
        union = bincount.sum(dim=0, keepdim=False) + bincount.sum(dim=1, keepdim=False) - bincount.diag()
        iou: torch.Tensor = intersection / union
        # stabilize nan values
        assert torch.all(torch.logical_or(iou[nan_mask] == 0, torch.isnan(iou[nan_mask]))), \
            f"{iou.tolist()=}, {nan_mask.tolist()=}, {(iou[nan_mask] == 0)=}, {torch.isnan(iou[nan_mask])=}"
        iou[nan_mask] = float('nan')
        # output check
        assert iou.shape == (num_classes,), f"{iou.shape=}, {num_classes=}"
        assert iou.is_floating_point(), f"{iou.dtype=}"
        score = {'IoU': iou}
        return score

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""
        Args:
            y_pred (torch.Tensor): a float32 tensor of shape (N, C, H, W) for predicted logits.
            y_true (torch.Tensor): an int64 tensor of shape (N, H, W) for ground-truth mask.

        Return:
            score (Dict[str, torch.Tensor]): a dictionary with the following fields: [
                'IoU', 'class_tp', 'class_tn', 'class_fp', 'class_fn',
                'class_accuracy', 'class_precision', 'class_recall', 'class_f1',
                'accuracy', 'mean_precision', 'mean_recall', 'mean_f1',
            ]
        """
        # input checks
        check_semantic_segmentation(y_pred=y_pred, y_true=y_true)
        # match resolution
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_pred = torchvision.transforms.Resize(
                size=y_true.shape[-2:], interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST,
            )(y_pred)
        # make prediction from output
        y_pred = torch.argmax(y_pred, dim=1).type(torch.int64)
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        # apply valid mask
        valid_mask = y_true != self.ignore_index
        assert valid_mask.sum() >= 1
        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]
        # compute IoU
        bincount = ConfusionMatrix._get_bincount(y_pred=y_pred, y_true=y_true, num_classes=self.num_classes)
        nan_mask = torch.ones(size=(self.num_classes,), dtype=torch.bool, device=bincount.device)
        nan_mask[y_true.unique()] = False
        iou = self._bincount2score(bincount, nan_mask, self.num_classes)
        # compute confusion matrix
        cm = ConfusionMatrix._bincount2cm(bincount, batch_size=y_true.size(0))
        cm_scores = ConfusionMatrix._cm2score(cm)
        # prepare final output
        score = {}
        score.update(iou)
        score.update(cm)
        score.update(cm_scores)
        return score

    @staticmethod
    def _summarize(buffer: List[Dict[str, torch.Tensor]], num_classes) -> Dict[str, torch.Tensor]:
        num_datapoints = len(buffer)
        buffer: Dict[str, List[torch.Tensor]] = transpose_buffer(buffer)
        result: Dict[str, torch.Tensor] = {}
        # summarize IoU
        iou = torch.stack(buffer['IoU'], dim=0)
        assert iou.shape == (num_datapoints, num_classes), f"{iou.shape=}, {num_datapoints=}, {num_classes=}"
        class_iou = torch.nanmean(iou, dim=0)
        assert class_iou.shape == (num_classes,), f"{class_iou.shape=}"
        mean_iou = torch.nanmean(class_iou)
        assert mean_iou.ndim == 0, f"{mean_iou.shape=}"
        result['class_IoU'] = class_iou
        result['mean_IoU'] = mean_iou
        # summarize confusion matrix
        agg_cm = {
            key: torch.stack(buffer[key], dim=0).sum(dim=0)
            for key in ['class_tp', 'class_tn', 'class_fp', 'class_fn']
        }
        agg_scores = ConfusionMatrix._cm2score(agg_cm)
        result.update(agg_cm)
        result.update(agg_scores)
        return result

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        r"""This functions summarizes the semantic segmentation evaluation results on all examples
        seen so far into a single floating point number.
        """
        assert len(self.buffer) != 0
        buffer: Dict[str, List[torch.Tensor]] = transpose_buffer(self.buffer)
        
        # Initialize result structure
        result: Dict[str, Dict[str, torch.Tensor]] = {
            "aggregated": {},
            "per_datapoint": {},
        }

        # First compute per-datapoint scores
        for key in buffer:
            key_scores = torch.stack(buffer[key], dim=0)
            result["per_datapoint"][key] = key_scores

        # Compute aggregated metrics
        result["aggregated"] = self._summarize(self.buffer, self.num_classes)

        # save to disk
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=result, filepath=output_path)
        return result
