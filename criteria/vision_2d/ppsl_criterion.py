from typing import Dict
from criteria.vision_2d import SemanticSegmentationCriterion
import torch


class PPSLCriterion(SemanticSegmentationCriterion):

    def __init__(self, *args, **kwargs) -> None:
        super(PPSLCriterion, self).__init__(*args, **kwargs)
        self.change_criterion = torch.nn.CrossEntropyLoss()
        self.semantic_criterion = torch.nn.CrossEntropyLoss()

    @staticmethod
    def _metric_criterion(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_true.shape != y_pred.shape:
            y_true = torch.nn.functional.interpolate(y_true, y_pred.shape[-2:], mode='nearest')
        return torch.mean((1-y_true) * y_true**2 +  y_true * torch.clamp(2 - y_pred, min=0.0)**2)

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert type(y_pred) == dict and set(y_pred.keys()) == {'change_map', 'semantic_map', 'metrics'}
        assert type(y_true) == dict and set(y_true.keys()) == {'change_map', 'semantic_map'}
        change_loss = self.change_criterion(inputs=y_pred['change_map'], targets=y_true['change_map'])
        semantic_loss = self.semantic_criterion(
            inputs=torch.nn.functional.interpolate(y_pred['semantic_map'], y_true['semantic_map'].shape[-2:], mode='nearest'),
            targets=y_true['semantic_map'],
        )
        metrics_losses = [self._metric_criterion(metric, y_true['change_map']) for metric in y_pred['metrics']]
        total_loss = change_loss + semantic_loss + sum(metrics_losses) / len(metrics_losses)
        assert total_loss.ndim == 0, f"{total_loss.shape=}"
        self.buffer.append(total_loss.detach().cpu())
        return total_loss
