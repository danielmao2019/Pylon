from typing import List, Dict, Optional
import random
import torch

from ._base_ import GradientManipulationBaseOptimizer


class GradVacOptimizer(GradientManipulationBaseOptimizer):
    __doc__ = r"""Paper: Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models (https://arxiv.org/abs/2010.05874)
    """

    def __init__(self, beta: Optional[float] = 0.1, **kwargs) -> None:
        super(GradVacOptimizer, self).__init__(**kwargs)
        self.phi: torch.Tensor = torch.zeros(size=(self.num_tasks, self.num_tasks), dtype=torch.float32, device=torch.device('cuda'))
        assert type(beta) in [float, int], f"{type(beta)=}"
        assert 0 <= beta <= 1, f"{beta=}"
        self.beta: float = beta

    def _gradient_manipulation_(
        self,
        grads_list: List[torch.Tensor],
        shared_rep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            grads_list (List[torch.Tensor]): the list of 1D gradient tensors of each objective.
            shared_rep (torch.Tensor): unused argument.
        Returns:
            result (torch.Tensor): the 1D manipulated gradient tensor.
        """
        # input checks
        assert len(grads_list) == self.num_tasks, f"{len(grads_list)=}, {self.num_tasks=}"
        # compute result
        result = torch.zeros_like(grads_list[0])
        for i in range(len(grads_list)):
            gi = grads_list[i].detach().clone()
            idx_j_order = list(range(len(grads_list)))
            idx_j_order.remove(i)
            random.shuffle(idx_j_order)
            for j in idx_j_order:
                gj = grads_list[j]
                phi_ij = torch.nn.CosineSimilarity(dim=0)(gi, gj)
                if phi_ij < self.phi[i, j]:
                    numerator = torch.linalg.vector_norm(gi) * (self.phi[i, j] * torch.sqrt(1-phi_ij**2) + phi_ij * torch.sqrt(1-self.phi[i, j]**2))
                    denominator = torch.linalg.vector_norm(gj) * torch.sqrt(1-self.phi[i, j]**2) + 1e-09
                    gi += (numerator / denominator) * gj
                self.phi[i, j] = (1-self.beta) * self.phi[i, j] + self.beta * phi_ij
            result += gi
        result /= len(grads_list)
        return result

    def state_dict(self) -> Dict[str, torch.Tensor]:
        result = super(GradVacOptimizer, self).state_dict()
        assert 'phi' not in result
        result['phi'] = self.phi.detach().clone()
        return result

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.phi = state_dict.pop('phi')
        super(GradVacOptimizer, self).load_state_dict(state_dict)
