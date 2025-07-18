from typing import Dict, Any, Union, List, Optional
from itertools import filterfalse as ifilterfalse
import torch
from criteria.vision_2d import SemanticSegmentationCriterion
from criteria.wrappers import SingleTaskCriterion


def lovasz_softmax(probas: torch.Tensor, labels: torch.Tensor, classes: Union[str, List[int]] = 'present', per_image: bool = False, ignore: Optional[int] = None) -> torch.Tensor:
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, classes: Union[str, List[int]] = 'present') -> torch.Tensor:
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_probas(probas: torch.Tensor, labels: torch.Tensor, ignore: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def mean(l: Any, ignore_nan: bool = False, empty: Union[int, str] = 0) -> Union[float, int]:
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def isnan(x: Any) -> bool:
    return x != x


class STMambaBCDCriterion(SingleTaskCriterion):

    def __init__(self, **kwargs: Any) -> None:
        super(STMambaBCDCriterion, self).__init__(**kwargs)
        self.ce_loss = SemanticSegmentationCriterion(ignore_value=255)

    def __call__(self, y_pred: torch.Tensor, y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert isinstance(y_pred, torch.Tensor)
        assert isinstance(y_true, dict)
        assert set(y_true.keys()) == {'change_map'}
        ce_loss = self.ce_loss(y_pred, y_true['change_map'])
        lovasz_loss = lovasz_softmax(torch.nn.functional.softmax(y_pred, dim=1), y_true['change_map'], ignore=255)
        total_loss = ce_loss + 0.75 * lovasz_loss
        self.add_to_buffer(total_loss)
        return total_loss
