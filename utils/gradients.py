from typing import List
import torch

import utils


NUMERICAL_STABILITY = 1.0e-07


def get_gram_matrix(
    grad_list: List[torch.Tensor],
    other: List[torch.Tensor] = None
) -> torch.Tensor:
    r"""This function computes a matrix whose (i, j)-th entry is torch.dot(grad_list[i], other[j]).
    other is default to grad_list so the output would be the true Gram matrix of grad_list.
    """
    # input checks
    assert type(grad_list) == list
    for g in grad_list:
        assert type(g) == torch.Tensor, f"{type(g)=}"
        assert len(g.shape) == 1
    if other is not None:
        assert type(other) == list and len(other) == len(grad_list)
    # initialization
    num_tasks = len(grad_list)
    # compute result
    result = torch.zeros(size=(num_tasks, num_tasks), dtype=torch.float32, device=torch.device('cuda'))
    for i in range(num_tasks):
        loop = range(i, num_tasks) if other is None else range(num_tasks)
        for j in loop:
            if other is None:
                dot_prod = torch.dot(grad_list[i], grad_list[j])
                result[i, j] = dot_prod
                result[j, i] = dot_prod
            else:
                dot_prod = torch.dot(grad_list[i], other[j])
                result[i, j] = dot_prod
    return result


def get_cosine_matrix(
    grad_list: List[torch.Tensor],
) -> torch.Tensor:
    r"""This function computes a matrix whose (i, j)-th entry is torch.nn.CosineSimilarity(dim=0)(grad_list[i], grad_list[j]).
    """
    assert type(grad_list) == list
    for g in grad_list:
        assert type(g) == torch.Tensor, f"{type(g)=}"
        assert len(g.shape) == 1
    return utils.apply_pairwise(lot=grad_list, func=torch.nn.CosineSimilarity(dim=0), symmetric=True)


def get_entropy(
    grads_list: List[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    # input checks
    assert type(grads_list) == list
    for g in grads_list:
        assert type(g) == torch.Tensor, f"{type(g)=}"
        assert len(g.shape) == 1
    # consider only gradient magnitudes
    stack = torch.stack([g.flatten() for g in grads_list])
    assert stack.shape[0] == len(grads_list)
    stack = torch.abs(stack)
    # normalize gradient magnitudes into probability distribution
    if temperature == 0.0:
        stack = stack / (torch.sum(stack, dim=0, keepdim=True) + NUMERICAL_STABILITY)
    else:
        stack = torch.nn.Softmax(dim=0)(stack / temperature)
    assert not torch.any(torch.isnan(stack))
    assert 0 <= stack.min() <= stack.max() <= 1, f"{stack.min()=}, {stack.max()=}"
    # compute entropy
    entropy = -torch.sum(stack * torch.log(stack + NUMERICAL_STABILITY), dim=0, keepdim=False)
    assert len(entropy.shape) == 1, f"{entropy.shape=}"
    assert not torch.any(torch.isnan(entropy))
    assert entropy.min() >= 0, f"{entropy.min()=}"
    return entropy
