"""Implementation largely based on https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/coco_evaluation.py.
"""
from typing import Tuple, List, Dict, Any
import torch
from .base_metric import BaseMetric
from utils.object_detection import pairwise_iou
from utils.input_checks import check_write_file
from utils.ops import transpose_buffer
from utils.io import save_json


class ObjectDetectionMetric(BaseMetric):

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
        scores: List[Dict[str, torch.Tensor]] = []
        for idx in range(batch_size):
            # sort predictions in descending order
            inds = torch.sort(y_pred['objectness'][idx], dim=0, descending=True)[1]
            pred_bboxes = y_pred['bboxes'][idx][inds]
            gt_bboxes = y_true['bboxes'][idx]
            gt_areas = y_true['areas'][idx]
            single_result: Dict[str, torch.Tensor] = {}
            for area in self.areas:
                for limit in self.limits:
                    single_result[f"gt_overlaps_{area}@{limit}"] = self._call_with_area_limit_(
                        pred_bboxes=pred_bboxes, gt_bboxes=gt_bboxes, gt_areas=gt_areas,
                        area_range=self.AREA_RANGES[area], limit=limit,
                    )
            scores.append(single_result)
        scores = transpose_buffer(scores)
        for key in scores:
            scores[key] = torch.cat(scores[key], dim=0)
        self.buffer.append(scores)
        return scores

    @staticmethod
    def reduce(scores: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        return scores['gt_overlaps_all@1000']['AR']

    def summarize(self, output_path: str = None) -> Dict[str, Any]:
        if output_path is not None:
            check_write_file(path=output_path)
        result: Dict[str, Any] = {}
        if len(self.buffer) != 0:
            buffer: Dict[str, List[torch.Tensor]] = transpose_buffer(self.buffer)
            thresholds = torch.arange(0.5, 0.95 + 1e-5, 0.05, dtype=torch.float32)
            for key in buffer:
                key_scores = torch.cat(buffer[key], dim=0)
                assert key_scores.dim() == 1, f"{key_scores.shape=}"
                recalls = torch.tensor([(key_scores >= t).type(torch.float32).mean() for t in thresholds])
                result[key] =  {
                    "AR": recalls.mean(),
                    "recalls": recalls,
                    "thresholds": thresholds,
                }
            # log reduction
            assert 'reduced' not in result, f"{result.keys()=}"
            result['reduced'] = self.reduce(result)
        if output_path is not None:
            save_json(obj=result, filepath=output_path)
        return result
