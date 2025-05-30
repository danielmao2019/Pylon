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

        # get aggregated summary
        buffer1: Dict[str, List[Dict[str, torch.Tensor]]] = transpose_buffer(self.buffer)
        aggregated_result: Dict[str, Dict[str, torch.Tensor]] = {
            'change': metrics.vision_2d.SemanticSegmentationMetric._summarize(
                buffer=buffer1['change'], num_classes=2,
        )}
        if len(buffer1) == 3:
            aggregated_result.update({
                'semantic_1': metrics.vision_2d.SemanticSegmentationMetric._summarize(
                    buffer=buffer1['semantic_1'], num_classes=5,
                ),
                'semantic_2': metrics.vision_2d.SemanticSegmentationMetric._summarize(
                    buffer=buffer1['semantic_2'], num_classes=5,
                ),
            })

        # get per-datapoint summary
        buffer2: Dict[str, Dict[str, List[torch.Tensor]]] = {
            key1: transpose_buffer(buffer1[key1])
            for key1 in buffer1.keys()
        }
        per_datapoint_result: Dict[str, Dict[str, torch.Tensor]] = {
            key1: {
                key2: torch.stack(buffer2[key1][key2], dim=0)
                for key2 in buffer2[key1].keys()
            } for key1 in buffer2.keys()
        }

        # define final result
        result: Dict[str, Dict[str, torch.Tensor]] = {
            "aggregated": aggregated_result,
            "per_datapoint": per_datapoint_result,
        }

        # save to disk
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=result, filepath=output_path)
        return result
