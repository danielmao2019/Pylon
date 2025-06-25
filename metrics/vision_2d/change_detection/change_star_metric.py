from typing import List, Dict
import torch
from metrics.vision_2d.semantic_segmentation_metric import SemanticSegmentationMetric
from metrics.wrappers import SingleTaskMetric
from utils.input_checks import check_write_file
from utils.io.json import save_json
from utils.ops.dict_as_tensor import transpose_buffer


class ChangeStarMetric(SingleTaskMetric):

    def __init__(self) -> None:
        super(ChangeStarMetric, self).__init__()
        self.change_metric = SemanticSegmentationMetric(num_classes=2, use_buffer=False)
        self.semantic_metric = SemanticSegmentationMetric(num_classes=5, use_buffer=False)

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor], idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """Override parent class __call__ method.
        """
        assert set(y_pred.keys()) == set(['change', 'semantic_1', 'semantic_2'])
        if set(y_true.keys()) == {'change', 'semantic_1', 'semantic_2'}:
            scores = {
                'change': self.change_metric(y_pred=y_pred['change'], y_true=y_true['change'], idx=idx),
                'semantic_1': self.semantic_metric(y_pred=y_pred['semantic_1'], y_true=y_true['semantic_1'], idx=idx),
                'semantic_2': self.semantic_metric(y_pred=y_pred['semantic_2'], y_true=y_true['semantic_2'], idx=idx),
            }
        elif set(y_true.keys()) == {'change_map'}:
            scores = {
                'change': self.change_metric(y_pred=y_pred['change'], y_true=y_true['change_map'], idx=idx),
            }
        self.add_to_buffer(scores, idx)
        return scores

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        """Summarize the metric."""
        self._buffer_queue.join()  # Wait for all items to be processed
        assert self._buffer_queue.empty(), "Buffer queue is not empty when summarizing"
        assert len(self.buffer) != 0

        buffer: Dict[str, List[Dict[str, torch.Tensor]]] = transpose_buffer(self.buffer)
        buffer: Dict[str, Dict[str, List[torch.Tensor]]] = {
            key1: transpose_buffer(buffer[key1])
            for key1 in buffer.keys()
        }

        # get aggregated summary
        aggregated_result: Dict[str, Dict[str, torch.Tensor]] = {
            'change': SemanticSegmentationMetric._summarize(
                buffer=buffer['change'], num_datapoints=len(self.buffer), num_classes=2,
        )}
        if len(buffer) == 3:
            aggregated_result.update({
                'semantic_1': SemanticSegmentationMetric._summarize(
                    buffer=buffer['semantic_1'], num_datapoints=len(self.buffer), num_classes=5,
                ),
                'semantic_2': SemanticSegmentationMetric._summarize(
                    buffer=buffer['semantic_2'], num_datapoints=len(self.buffer), num_classes=5,
                ),
            })

        # define final result
        result: Dict[str, Dict[str, torch.Tensor]] = {
            "aggregated": aggregated_result,
            "per_datapoint": buffer,
        }

        # save to disk
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=result, filepath=output_path)
        return result
