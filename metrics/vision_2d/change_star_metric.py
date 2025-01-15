from typing import Dict
import torch
import metrics
import metrics
from metrics.wrappers import SingleTaskMetric


class ChangeStarMetric(SingleTaskMetric):

    def __init__(self) -> None:
        self.change_metric = metrics.vision_2d.SemanticSegmentationMetric(num_classes=2)
        self.semantic_metric = metrics.vision_2d.SemanticSegmentationMetric(num_classes=5)

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Override parent class __call__ method.
        """
        assert set(y_pred.keys()) == set(['change', 'semantic_1', 'semantic_2'])
        assert set(y_true.keys()) == set(['change', 'semantic_1', 'semantic_2'])
        change_scores = self.metric(y_pred=y_pred['change'], y_true=y_true['change'])
        semantic_1_score = self.metric(y_pred=y_pred['semantic_1'], y_true=y_true['semantic_1'])
        semantic_2_score = self.metric(y_pred=y_pred['semantic_2'], y_true=y_true['semantic_2'])
        return {
            'change_': change_scores,
            'semantic_1': semantic_1_score,
            'semantic_2': semantic_2_score,
        }
