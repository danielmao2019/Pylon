from typing import Tuple, List, Dict, Optional
import criteria.common
import torch
import criteria
from criteria.wrappers import SingleTaskCriterion


class CornerNetCriterion(SingleTaskCriterion):

    def __init__(self, scale_factor: Tuple[float, float]) -> None:
        self.scale_factor = scale_factor
        self.focal_loss_criterion = criteria.common.FocalLoss()

    def _draw_gaussian(heatmap: torch.Tensor, center: Tuple[int, int], radius: int) -> torch.Tensor:
        pass

    def _gen_heatmap_labels(
        bboxes_batch: List[torch.Tensor],
        labels_batch: List[torch.Tensor],
        num_classes: int,
        resolution: Tuple[int, int],
        radius: Optional[int] = 0,
    ) -> Dict[str, torch.Tensor]:
        labels: Dict[str, List[torch.Tensor]] = {'tl_true': [], 'br_true': []}
        device = list(bboxes_batch.values())[0].device
        for bboxes, labels in zip(bboxes_batch, labels_batch):
            tl_heatmaps = torch.zeros(size=(num_classes,)+resolution, dtype=torch.float32, device=device)
            br_heatmaps = torch.zeros(size=(num_classes,)+resolution, dtype=torch.float32, device=device)
            for bbox, label in zip(list(bboxes), list(labels)):
                tl_heatmaps[label, :, :] = CornerNetCriterion._draw_gaussian(tl_heatmaps[label], center=(bbox[0], bbox[1]), radius=radius)
                br_heatmaps[label, :, :] = CornerNetCriterion._draw_gaussian(br_heatmaps[label], center=(bbox[2], bbox[3]), radius=radius)
            labels['tl_true'].append(tl_heatmaps)
            labels['br_true'].append(br_heatmaps)
        labels['tl_true'] = torch.stack(labels['tl_true'], dim=0)
        labels['br_true'] = torch.stack(labels['br_true'], dim=0)
        return labels

    def _gen_offset_labels(bboxes_batch: List[torch.Tensor], scale_factor: Tuple[float, float]) -> List[torch.Tensor]:
        labels: List[torch.Tensor] = []
        for bboxes in bboxes_batch:
            assert bboxes.ndim == 2 and bboxes.size(1) == 4
            assert bboxes.dtype == torch.int64
            tl_x, tl_y, br_x, br_y = list(bboxes.t())
            tl_offsets = torch.stack([
                tl_x * scale_factor[1] - (tl_x * scale_factor[1]).floor(),
                tl_y * scale_factor[0] - (tl_y * scale_factor[0]).floor(),
            ], dim=1)
            br_offsets = torch.stack([
                br_x * scale_factor[1] - (br_x * scale_factor[1]).floor(),
                br_y * scale_factor[0] - (br_y * scale_factor[0]).floor(),
            ], dim=1)
            assert tl_offsets.shape == br_offsets.shape == (bboxes.size(0), 2), \
                f"{tl_offsets.shape=}, {br_offsets.shape=}, {bboxes.shape=}"
            assert torch.equal(tl_offsets, br_offsets)
            labels.append(tl_offsets)
        return labels

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        # check y_pred
        assert isinstance(y_pred, dict) and set(y_pred.keys()) == set(['tl_pred', 'br_pred'])
        tl_pred, br_pred = y_pred['tl_pred'], y_pred['br_pred']
        assert isinstance(tl_pred, dict) and set(tl_pred.keys()) == set(['heatmaps', 'embeddings', 'offsets'])
        assert isinstance(br_pred, dict) and set(br_pred.keys()) == set(['heatmaps', 'embeddings', 'offsets'])
        # check y_true
        assert isinstance(y_true, dict) and set(y_true.keys()) == set(['labels', 'bboxes'])
        assert isinstance(y_true['labels'], list) and all(isinstance(x, torch.Tensor) for x in y_true['labels'])
        assert isinstance(y_true['bboxes'], list) and all(isinstance(x, torch.Tensor) for x in y_true['bboxes'])
        assert all(b.shape == l.shape + (4,) for b, l in zip(y_true['labels'], y_true['bboxes']))
        heatmap_labels = self._gen_heatmap_labels(
            bboxes_batch=y_true['bboxes'], labels_batch=y_true['labels'],
            num_classes=self.num_classes, resolution=self.resolution, radius=self.radius,
        )
        offset_labels = self._gen_offset_labels(bboxes_batch=y_true['bboxes'], scale_factor=self.scale_factor)
        # Compute focal loss
        focal_loss: torch.Tensor = self.focal_loss_criterion(y_pred=y_pred['labels'], y_true=y_true['labels'])
        # Compute offset loss
        offset_loss = None
        # Compute pull loss
        pull_loss = None
        # Compute push loss
        push_loss = None
        # Compute total loss
        total_loss = focal_loss + offset_loss + pull_loss + push_loss
        assert total_loss.numel() == 1, f"{total_loss.shape=}"
        self.buffer.append(total_loss.detach().cpu())
        return total_loss
