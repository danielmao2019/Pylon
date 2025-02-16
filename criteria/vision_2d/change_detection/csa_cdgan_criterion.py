from typing import Dict
import torch
from criteria.wrappers import SingleTaskCriterion, MultiTaskCriterion


class CSA_CDGAN_GeneratorCriterion(SingleTaskCriterion):

    def __init__(self) -> None:
        super(CSA_CDGAN_GeneratorCriterion, self).__init__()
        self.l_bce = torch.nn.BCELoss()
        self.l_con = torch.nn.L1Loss()

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        err_g_total = self.l_con(torch.nn.functional.softmax(y_pred['gen_image'], dim=1), y_true['change_map'])
        assert err_g_total.ndim == 0, f"{err_g_total.shape=}"
        # log loss
        self.buffer.append(err_g_total.detach().cpu())
        return err_g_total


class CSA_CDGAN_DiscriminatorCriterion(SingleTaskCriterion):

    def __init__(self) -> None:
        super(CSA_CDGAN_DiscriminatorCriterion, self).__init__()
        self.l_bce = torch.nn.BCELoss()

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        err_d_real = self.l_bce(y_pred['pred_real'], y_true['real_label'])
        err_d_fake = self.l_bce(y_pred['pred_fake'], y_true['fake_label'])
        err_d_total = (err_d_real + err_d_fake) * 0.5
        assert err_d_total.ndim == 0, f"{err_d_total.shape=}"
        # log loss
        self.buffer.append(err_d_total.detach().cpu())
        return err_d_total


class CSA_CDGAN_Criterion(MultiTaskCriterion, torch.nn.Module):

    def __init__(self, g_weight: float, d_weight: float) -> None:
        self.task_criteria = {
            'generator': CSA_CDGAN_GeneratorCriterion(g_weight, d_weight),
            'discriminator': CSA_CDGAN_DiscriminatorCriterion(),
        }
        self.task_names = set(self.task_criteria.keys())
        self.reset_buffer()

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def to(self, *args, **kwargs) -> None:
        return self
