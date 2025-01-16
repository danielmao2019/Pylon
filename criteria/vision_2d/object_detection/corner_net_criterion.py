"""
References:
    * https://github.com/princeton-vl/CornerNet
    * https://github.com/zzzxxxttt/pytorch_simple_CornerNet
"""
from typing import Tuple, List, Dict, Optional
import numpy
import torch
from criteria.wrappers import SingleTaskCriterion


class CornerNetCriterion(SingleTaskCriterion):

    def __init__(self, scale_factor: Tuple[float, float], num_classes: int, resolution: Tuple[int, int], radius: int) -> None:
        super(CornerNetCriterion, self).__init__()
        self.scale_factor = scale_factor
        self.num_classes = num_classes
        self.resolution = resolution
        self.radius = radius

    @staticmethod
    def _heatmap_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the heatmap loss for a single predicted tensor against the ground truth tensor.

        Args:
            y_pred (Tensor): The predicted heatmap tensor. Shape: (N, C, H, W), where
                            N is the batch size, C is the number of classes.
            y_true (Tensor): The ground truth heatmap tensor. Shape: (N, C, H, W).

        Returns:
            Tensor: The computed loss value as a scalar tensor.
        """
        # Identify positive and negative indices in the ground truth
        pos_inds = y_true.eq(1)  # Indices where the ground truth equals 1 (positive samples)
        neg_inds = y_true.lt(1)  # Indices where the ground truth is less than 1 (negative samples)

        # Compute weights for negative samples based on their ground truth values
        neg_weights = torch.pow(1 - y_true[neg_inds], 4)

        # Extract positive and negative predictions using the indices
        pos_pred = y_pred[pos_inds]  # Predictions corresponding to positive samples
        neg_pred = y_pred[neg_inds]  # Predictions corresponding to negative samples

        # Compute loss for positive predictions
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)

        # Compute loss for negative predictions
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        # Compute the number of positive samples
        num_pos = pos_inds.float().sum()

        # Sum the losses
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        # Handle cases where there are no positive samples
        if pos_pred.nelement() == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss

    @staticmethod
    def _embedding_loss(tag0, tag1, mask):
        num = mask.sum(dim=1, keepdim=True).float()
        tag0 = tag0.squeeze()
        tag1 = tag1.squeeze()

        tag_mean = (tag0 + tag1) / 2
        pull_loss = ((tag0 - tag_mean).pow(2) + (tag1 - tag_mean).pow(2)) / (num + 1e-4)
        pull_loss = pull_loss[mask].sum()

        mask = mask.unsqueeze(1) + mask.unsqueeze(2)
        mask = mask.eq(2)
        num = num.unsqueeze(2)
        num2 = (num - 1) * num
        dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
        dist = 1 - torch.abs(dist)
        dist = torch.nn.functional.relu(dist, inplace=True)
        dist -= 1 / (num + 1e-4)
        dist /= (num2 + 1e-4)
        dist = dist[mask]
        push_loss = dist.sum()
        return pull_loss, push_loss

    @staticmethod
    def _offset_loss(regr, gt_regr, mask):
        num = mask.float().sum()
        mask = mask.unsqueeze(2).expand_as(gt_regr)
        regr = regr[mask]
        gt_regr = gt_regr[mask]

        regr_loss = torch.nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
        regr_loss /= (num + 1e-4)
        return regr_loss

    @staticmethod
    def gaussian2D(shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = numpy.ogrid[-m:m + 1, -n:n + 1]
        h = numpy.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < numpy.finfo(h.dtype).eps * h.max()] = 0
        return h

    @staticmethod
    def draw_gaussian(heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = CornerNetCriterion.gaussian2D((diameter, diameter), sigma=diameter / 6)
        x, y = center
        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        numpy.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    @staticmethod
    def gaussian_radius(det_size, min_overlap):
        height, width = det_size
        a1, b1, c1 = 1, (height + width), width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = numpy.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 - sq1) / (2 * a1)

        a2, b2, c2 = 4, 2 * (height + width), (1 - min_overlap) * width * height
        sq2 = numpy.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 - sq2) / (2 * a2)

        a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (height + width), (min_overlap - 1) * width * height
        sq3 = numpy.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)
        return min(r1, r2, r3)

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
        # Compute loss
        focal_loss = self._heatmap_loss([y_pred['tl_preds']['heatmaps'], y_pred['br_preds']['heatmaps']],
                                    [y_true['tl_true']['heatmaps'], y_true['br_true']['heatmaps']])

        pull_loss, push_loss = self._embedding_loss(y_pred['tl_preds']['embeddings'], y_pred['br_preds']['embeddings'],
                                             y_true['tl_true']['offsets'])

        offset_loss = self._offset_loss(y_pred['tl_preds']['offsets'], y_true['tl_true']['offsets'], y_true['tl_true']['heatmaps'] > 0) + \
                      self._offset_loss(y_pred['br_preds']['offsets'], y_true['br_true']['offsets'], y_true['br_true']['heatmaps'] > 0)

        total_loss = focal_loss + pull_loss + push_loss + offset_loss
        assert total_loss.numel() == 1, f"{total_loss.shape=}"
        self.buffer.append(total_loss.detach().cpu())
        return total_loss
