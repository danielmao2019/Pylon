from typing import List, Dict, Any
from .pcgrad import PCGradOptimizer
from .test_common.initialize import initialize
import torch
from utils.models import get_flattened_params, get_flattened_grads
from utils.ops import apply_tensor_op, buffer_close


def test_pcgrad_optimizer():
    init = initialize(optimizer_class=PCGradOptimizer, wrt_rep=False, per_layer=False)
    model = init['model']
    dataloader = init['dataloader']
    criterion = init['criterion']
    optimizer = init['optimizer']
    trajectory: List[Dict[str, Any]] = [{'params': get_flattened_params(model).detach().clone()}]
    for dp in dataloader:
        outputs = model(dp['inputs'])
        losses = criterion(outputs, dp['labels'])
        optimizer.zero_grad()
        optimizer.backward(losses=losses, shared_rep=outputs['shared_rep'])
        optimizer.step()
        traj_item = {
            'inputs': dp['inputs'],
            'labels': dp['labels'],
            'activations': torch.cat([a.flatten() for a in model.activations.values()], dim=0),
            'outputs': outputs,
            'losses': losses,
            'grads': get_flattened_grads(model),
            'params': get_flattened_params(model),
        }
        traj_item = apply_tensor_op(lambda x: x.detach().clone(), traj_item)
        trajectory.append(traj_item)
        break # there is only one iteration hand-computed now
    from .test_pcgrad_ground_truth import ground_truth
    assert buffer_close(trajectory, ground_truth, rtol=0, atol=1.0e-08)
