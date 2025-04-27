from typing import Tuple, Dict
import torch
import torch.nn.functional as F
from criteria.wrappers import SingleTaskCriterion
from criteria.vision_2d.dense_prediction.dense_classification.semantic_segmentation import SemanticSegmentationCriterion
from criteria.vision_2d.dense_prediction.dense_classification.dice_loss import DiceLoss


class SRCNetCriterion(SingleTaskCriterion):

    def __init__(self):
        super(SRCNetCriterion, self).__init__()
        self.edge_loss = EdgeLoss(KSIZE=7)
        self.focal_loss = SemanticSegmentationCriterion()
        self.dice_loss = DiceLoss()

    def calloss(self, prediction, target, sigmas):
        focal = self.focal_loss(prediction, target)
        dice = self.dice_loss(prediction, target)
        edge = self.edge_loss(prediction, target)
        return focal / sigmas[0] + dice / sigmas[1] + edge / sigmas[2]

    def __call__(self, y_pred: Tuple[torch.Tensor, ...], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert isinstance(y_pred, tuple)
        assert len(y_pred) == 4
        assert all(isinstance(x, torch.Tensor) for x in y_pred)
        assert isinstance(y_true, dict)
        assert set(y_true.keys()) == {'change_map'}
        prediction, Dis, dif, sigma = y_pred
        target = y_true['change_map']
        loss = 0
        sigmas = sigma
        sigmas = sigmas * sigmas

        loss += self.calloss(prediction, target, sigmas)
        loss += self.calloss(Dis, target, sigmas)
        loss += dif
        loss += torch.sum(torch.log(sigmas)) / 2

        self.add_to_buffer(loss)
        return loss


class EdgeLoss(SingleTaskCriterion):

    def __init__(self, KSIZE=7) -> None:
        super(EdgeLoss, self).__init__()
        self.KSIZE = KSIZE
        self.MASK = torch.zeros([KSIZE, KSIZE])
        self.cal_mask(KSIZE)

    def cal_mask(self, ksize):
        num = 0
        MASK = self.MASK
        for x in range(0, ksize):
            for y in range(0, ksize):
                if (x + 0.5 - ksize / 2) ** 2 + (y + 0.5 - ksize / 2) ** 2 <= (
                    (ksize - 1) / 2
                ) ** 2:
                    MASK[x][y] = 1
                    num += 1
        MASK = MASK.reshape(1, 1, 1, 1, -1).float() / num
        MASK = MASK.to(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.MASK = MASK

    def tensor_average(self, bin_img, ksize):
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode="constant", value=0)

        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)

        eroded = torch.sum(patches.reshape(B, C, H, W, -1).float() * self.MASK, dim=-1)
        return eroded

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        targets = y_true.unsqueeze(dim=1)
        targetAve = self.tensor_average(targets, ksize=self.KSIZE)
        at = torch.abs(targets.float() - targetAve)
        # at[at == 0] = 0.2
        at = at.view(-1)

        if y_pred.dim() > 2:
            y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
            y_pred = y_pred.transpose(1, 2)
            y_pred = y_pred.contiguous().view(-1, y_pred.size(2))

        y_true = y_true.view(-1, 1)
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        logpt = torch.log(y_pred + 1e-10)
        logpt = logpt.gather(1, y_true)
        logpt = logpt.view(-1)

        loss = -1 * logpt * at
        return loss.mean()
