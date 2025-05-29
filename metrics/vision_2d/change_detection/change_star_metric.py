from typing import List, Dict
import torch
import metrics
import metrics
from metrics.wrappers import SingleTaskMetric
from utils.input_checks import check_write_file
from utils.io import save_json
from utils.ops import apply_tensor_op, transpose_buffer


class ChangeStarMetric(SingleTaskMetric):

    def __init__(self) -> None:
        super(ChangeStarMetric, self).__init__()
        self.change_metric = metrics.vision_2d.SemanticSegmentationMetric(num_classes=2)
        self.semantic_metric = metrics.vision_2d.SemanticSegmentationMetric(num_classes=5)

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Override parent class __call__ method.
        """
        assert set(y_pred.keys()) == set(['change', 'semantic_1', 'semantic_2'])
        if set(y_true.keys()) == {'change', 'semantic_1', 'semantic_2'}:
            scores = {
                'change': self.change_metric(y_pred=y_pred['change'], y_true=y_true['change']),
                'semantic_1': self.semantic_metric(y_pred=y_pred['semantic_1'], y_true=y_true['semantic_1']),
                'semantic_2': self.semantic_metric(y_pred=y_pred['semantic_2'], y_true=y_true['semantic_2']),
            }
        elif set(y_true.keys()) == {'change_map'}:
            scores = {
                'change': self.change_metric(y_pred=y_pred['change'], y_true=y_true['change_map']),
            }
        scores = apply_tensor_op(func=lambda x: x.detach().cpu(), inputs=scores)
        self.buffer.append(scores)
        return scores

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        assert len(self.buffer) != 0
        buffer: Dict[str, List[Dict[str, torch.Tensor]]] = transpose_buffer(self.buffer)

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
        result["aggregated"]['change'] = metrics.vision_2d.SemanticSegmentationMetric._summarize(
            buffer=buffer['change'], num_classes=2,
        )
        if len(buffer) == 3:
            result["aggregated"].update({
                'semantic_1': metrics.vision_2d.SemanticSegmentationMetric._summarize(
                    buffer=buffer['semantic_1'], num_classes=5,
                ),
                'semantic_2': metrics.vision_2d.SemanticSegmentationMetric._summarize(
                    buffer=buffer['semantic_2'], num_classes=5,
                ),
            })

        # save to disk
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=result, filepath=output_path)
        return result
