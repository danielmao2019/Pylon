from typing import Dict
import os
import torch
from .base_metric import BaseMetric
from utils.io import save_json


class SemanticSegmentationMetric(BaseMetric):

    def __init__(self, num_classes: int, ignore_index: int):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index= ignore_index

    def __call__(self, y_pred: torch.Tensor, y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""
        Return:
            score (torch.Tensor): 1D tensor of length num_classes representing the IoU scores for each class.
        """
        y_true = y_true['mask']
        # input checks
        assert type(y_pred) == torch.Tensor, f"{type(y_pred)=}"
        assert y_pred.dim() == 4, f"{y_pred.shape=}"
        assert y_pred.is_floating_point(), f"{y_pred.dtype=}"
        assert type(y_true) == torch.Tensor, f"{type(y_true)=}"
        assert y_true.dim() == 3, f"{y_true.shape=}"
        assert y_true.dtype == torch.int64, f"{y_true.dtype=}"
        assert y_pred.shape[0] == y_true.shape[0], f"{y_pred.shape=}, {y_true.shape=}"
        assert y_pred.shape[-2:] == y_true.shape[-2:], f"{y_pred.shape=}, {y_true.shape=}"
        # make prediction from output
        y_pred = torch.argmax(torch.nn.functional.softmax(y_pred, dim=1), dim=1)
        y_pred = y_pred.type(torch.int64)
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        # compute score
        mask = y_true != self.ignore_index
        assert mask.sum() >= 1
        count = torch.bincount(
            y_true[mask] * self.num_classes + y_pred[mask], minlength=self.num_classes**2,
        ).view((self.num_classes,) * 2)
        numerator = count.diag()
        denominator = count.sum(dim=0, keepdim=False) + count.sum(dim=1, keepdim=False) - count.diag()
        score: torch.Tensor = numerator / denominator
        assert score.shape == (self.num_classes,), f"{score.shape=}, {self.num_classes=}"
        # log score
        self.buffer.append(score)
        return score

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        r"""This functions summarizes the semantic segmentation evaluation results on all examples
        seen so far into a single floating point number.
        """
        if output_path is not None:
            assert type(output_path) == str, f"{type(output_path)=}"
            assert os.path.isdir(os.path.dirname(output_path)), f"{output_path=}"
        result: Dict[str, torch.Tensor] = {}
        score = torch.stack(self.buffer, dim=0)
        assert score.shape == (len(self.buffer), self.num_classes), f"{score.shape=}"
        score = torch.nanmean(score, dim=0)
        assert score.shape == (self.num_classes,), f"{score.shape=}"
        score = torch.nanmean(score)
        assert score.shape == (), f"{score.shape=}"
        result['score'] = score
        assert 'reduced' not in result, f"{result.keys()=}"
        result['reduced'] = self.reduce(result)
        if output_path is not None:
            save_json(obj=result, filepath=output_path)
        return result
