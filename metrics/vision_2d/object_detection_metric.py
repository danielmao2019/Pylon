"""Implementation largely based on https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/coco_evaluation.py.
"""
from typing import Tuple, List, Dict
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric
from utils.object_detection import pairwise_iou
from utils.input_checks import check_write_file
from utils.ops import transpose_buffer
from utils.io import save_json


class ObjectDetectionMetric(SingleTaskMetric):

    DIRECTION = +1

    AREA_RANGES = {
        "all": [0**2, 1e5**2],
        "small": [0**2, 32**2],
        "medium": [32**2, 96**2],
        "large": [96**2, 1e5**2],
        "96-128": [96**2, 128**2],
        "128-256": [128**2, 256**2],
        "256-512": [256**2, 512**2],
        "512-inf": [512**2, 1e5**2],
    }

    def __init__(self, areas: List[str], limits: List[int]):
        super(ObjectDetectionMetric, self).__init__()
        self.areas = areas
        assert set(areas).issubset(self.AREA_RANGES.keys()), f"Unknown area ranges: {set(areas) - set(self.AREA_RANGES.keys())}"
        self.limits = limits
        self.thresholds = torch.arange(0.5, 0.95 + 1e-5, 0.05, dtype=torch.float32)

    @staticmethod
    def _call_with_area_limit_(
        pred_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_areas: torch.Tensor,
        area_range: Tuple[int, int],
        limit: int,
    ) -> torch.Tensor:
        # initialization
        assert type(pred_bboxes) == torch.Tensor, f"{type(pred_bboxes)=}"
        assert len(pred_bboxes.shape) == 2 and pred_bboxes.shape[1] == 4, f"{pred_bboxes.shape=}"
        assert type(gt_bboxes) == torch.Tensor, f"{type(gt_bboxes)=}"
        assert len(gt_bboxes.shape) == 2 and gt_bboxes.shape[1] == 4, f"{gt_bboxes.shape=}"
        assert type(gt_areas) == torch.Tensor, f"{type(gt_areas)=}"
        assert len(gt_areas.shape) == 1 and len(gt_areas) == len(gt_bboxes)
        # filter ground truth bounding boxes based on given area range
        valid_indices = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_bboxes = gt_bboxes[valid_indices]
        # filter predicted bounding boxes based on given limit
        pred_bboxes = pred_bboxes[:limit]
        # compute score
        overlaps = pairwise_iou(pred_bboxes, gt_bboxes)
        assert overlaps.shape == (len(pred_bboxes), len(gt_bboxes)), f"{overlaps.shape=}"
        assert torch.all(overlaps >= 0), f"{overlaps.min()=}, {overlaps.max()=}"
        result = torch.zeros(len(gt_bboxes))
        for j in range(min(len(pred_bboxes), len(gt_bboxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)
            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, true_idx = max_overlaps.max(dim=0)
            # find the proposal box that covers the best covered gt box
            pred_idx = argmax_overlaps[true_idx]
            # record the iou coverage of this gt box
            result[j] = overlaps[pred_idx, true_idx]
            assert result[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[pred_idx, :] = -1
            overlaps[:, true_idx] = -1
        return result

    @staticmethod
    def _compute_recalls(overlaps: torch.Tensor, thresholds: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute recalls at different thresholds for a single datapoint."""
        recalls = torch.tensor([(overlaps >= t).type(torch.float32).mean() for t in thresholds])
        return {
            "AR": recalls.mean(),
            "recalls": recalls,
            "thresholds": thresholds,
        }

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        r"""
        Args:
            y_pred: {
                'labels' (torch.Tensor): int64 tensor of shape (B, N).
                'bboxes' (torch.Tensor): float32 tensor of shape (B, N, 4).
                'objectness' (torch.Tensor): float32 tensor of shape (B, N).
            }
            y_true: {
                'bboxes': (List[torch.Tensor]): list of float32 tensor, each of shape (N_i, 4).
                'areas': (List[torch.Tensor]): list of int tensor, each of shape (N_i,).
            }
        """
        # input checks
        assert type(y_pred) == dict, f"{type(y_pred)=}"
        assert set(['labels', 'bboxes', 'objectness']).issubset(set(y_pred.keys())), f"{y_pred.keys()=}"
        assert len(y_pred['labels'].shape) == 2, f"{y_pred['labels'].shape=}"
        assert len(y_pred['bboxes'].shape) == 3 and y_pred['bboxes'].shape[2] == 4, f"{y_pred['bboxes'].shape=}"
        assert len(y_pred['objectness'].shape) == 2, f"{y_pred['objectness'].shape=}"
        assert y_pred['labels'].shape == y_pred['bboxes'].shape[:2] == y_pred['objectness'].shape
        assert type(y_true) == dict, f"{type(y_true)=}"
        assert set(['bboxes', 'areas']).issubset(set(y_true.keys())), f"{y_true.keys()=}"
        assert type(y_true['bboxes']) == list, f"{type(y_true['bboxes'])=}"
        assert type(y_true['areas']) == list, f"{type(y_true['areas'])=}"
        # compute scores
        batch_size: int = len(y_pred['bboxes'])
        scores: List[Dict[str, Dict[str, torch.Tensor]]] = []
        for idx in range(batch_size):
            # sort predictions in descending order
            inds = torch.sort(y_pred['objectness'][idx], dim=0, descending=True)[1]
            pred_bboxes = y_pred['bboxes'][idx][inds]
            gt_bboxes = y_true['bboxes'][idx]
            gt_areas = y_true['areas'][idx]
            single_result: Dict[str, Dict[str, torch.Tensor]] = {}
            for area in self.areas:
                for limit in self.limits:
                    key = f"gt_overlaps_{area}@{limit}"
                    # Compute overlaps
                    overlaps = self._call_with_area_limit_(
                        pred_bboxes=pred_bboxes, gt_bboxes=gt_bboxes, gt_areas=gt_areas,
                        area_range=self.AREA_RANGES[area], limit=limit,
                    )
                    # Compute recalls
                    recalls_dict = self._compute_recalls(overlaps, self.thresholds)
                    # Store in nested structure
                    single_result[key] = {
                        'per_bbox': overlaps,
                        **recalls_dict,
                    }
            scores.append(single_result)
        self.buffer.extend(scores)
        return scores

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        """Summarize the metric."""
        self._buffer_queue.join()  # Wait for all items to be processed
        assert self._buffer_queue.empty(), "Buffer queue is not empty when summarizing"
        assert len(self.buffer) != 0

        buffer: Dict[str, List[Dict[str, torch.Tensor]]] = transpose_buffer(self.buffer)
        buffer: Dict[str, Dict[str, List[torch.Tensor]]] = {
            key1: transpose_buffer(buffer[key1]) for key1 in buffer.keys()
        }

        # summarize scores
        result: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {
            "aggregated": {
                key1: self._compute_recalls(torch.cat(buffer[key1]['per_bbox'], dim=0), self.thresholds)
                for key1 in buffer.keys()
            },
            "per_datapoint": {
                key1: {
                    key2: torch.stack(buffer[key1][key2], dim=0)
                    for key2 in buffer[key1].keys()
                    if key2 != 'per_bbox'
                } for key1 in buffer.keys()
            },
        }

        # save to disk
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=result, filepath=output_path)
        return result
