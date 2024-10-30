from typing import List, Dict
import torch
import torchvision
from metrics.wrappers.single_task_metric import SingleTaskMetric
from utils.input_checks import check_write_file, check_semantic_segmentation
from utils.io import save_json
from utils.ops import transpose_buffer


class SemanticSegmentationMetric(SingleTaskMetric):

    DIRECTION = +1

    def __init__(self, num_classes: int, ignore_index: int) -> None:
        super(SemanticSegmentationMetric, self).__init__()
        assert type(num_classes) == int, f"{type(num_classes)=}"
        assert num_classes > 0, f"{num_classes=}"
        self.num_classes = num_classes
        self.ignore_index= ignore_index

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""
        Args:
            y_pred (torch.Tensor): a float32 tensor of shape (N, C, H, W) for predicted logits.
            y_true (torch.Tensor): an int64 tensor of shape (N, H, W) for ground-truth mask.

        Return:
            score (Dict[str, torch.Tensor]): a dictionary with the following fields
            {
                'IoU': a 1D tensor of length `self.num_classes` representing the IoU scores for each class.
            }
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
        # compute confusion matrix
        valid_mask = y_true != self.ignore_index
        assert valid_mask.sum() >= 1
        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]
        assert 0 <= y_true.min() <= y_true.max() < self.num_classes, f"{y_true.min()=}, {y_true.max()=}, {self.num_classes=}"
        assert 0 <= y_pred.min() <= y_pred.max() < self.num_classes, f"{y_pred.min()=}, {y_pred.max()=}, {self.num_classes=}"
        count = torch.bincount(
            y_true * self.num_classes + y_pred, minlength=self.num_classes**2,
        ).view((self.num_classes,) * 2)
        # compute intersection over union
        intersection = count.diag()
        union = count.sum(dim=0, keepdim=False) + count.sum(dim=1, keepdim=False) - count.diag()
        score: torch.Tensor = intersection / union
        # stabilize nan values
        nan_mask = torch.ones(size=(self.num_classes,), dtype=torch.bool, device=count.device)
        nan_mask[y_true.unique()] = False
        assert torch.all(torch.logical_or(score[nan_mask] == 0, torch.isnan(score[nan_mask]))), \
            f"{score.tolist()=}, {nan_mask.tolist()=}, {(score[nan_mask] == 0)=}, {torch.isnan(score[nan_mask])=}"
        score[nan_mask] = float('nan')
        # output check
        assert score.shape == (self.num_classes,), f"{score.shape=}, {self.num_classes=}"
        assert score.is_floating_point(), f"{score.dtype=}"
        return {'IoU': score}

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        r"""This functions summarizes the semantic segmentation evaluation results on all examples
        seen so far into a single floating point number.
        """
        assert len(self.buffer) != 0
        buffer: Dict[str, List[torch.Tensor]] = transpose_buffer(self.buffer)
        # summarize scores
        result: Dict[str, torch.Tensor] = {}
        iou = torch.stack(buffer['IoU'], dim=0)
        assert iou.shape == (len(self.buffer), self.num_classes), f"{iou.shape=}"
        # log class IoU
        class_iou = torch.nanmean(iou, dim=0)
        assert class_iou.shape == (self.num_classes,), f"{class_iou.shape=}"
        result['class_IoU'] = class_iou
        # log mean IoU
        mean_iou = torch.nanmean(class_iou)
        assert mean_iou.ndim == 0, f"{mean_iou.shape=}"
        result['mean_IoU'] = mean_iou
        # save to disk
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=result, filepath=output_path)
        return result
