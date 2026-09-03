# runners/trainers — code implementation structure

## Code implementation structure trees

`runners/trainers/supervised_single_task_trainer.py`

```text
supervised_single_task_trainer.py
├── from typing import Any, Dict
├── import torch
├── from optimizers.single_task_optimizer import SingleTaskOptimizer
├── from runners.trainers.base_trainer import BaseTrainer
├── from utils.builders import build_from_config, build_scheduler
└── class SupervisedSingleTaskTrainer(BaseTrainer)
    ├── # Supervised single-task trainer: one optimizer over the model params, single scalar-loss backprop, no-grad validation.
    ├── def _init_optimizer(self) -> None   [override]
    │   ├── # Builds the single-task optimizer over the model parameters, or skips when unconfigured.
    │   ├── if not self.config.get('optimizer', None)
    │   │   └── return
    │   ├── impls set optimizer_config's params to list(self.model.parameters())
    │   └── calls build_from_config  # the optimizer
    ├── def _init_scheduler(self)   [override]
    │   ├── # Builds the LR scheduler over the single-task optimizer, or skips when unconfigured.
    │   ├── if not self.config.get('scheduler', None)
    │   │   └── return
    │   ├── impls assert the optimizer is a SingleTaskOptimizer wrapping a torch.optim.Optimizer
    │   └── calls build_scheduler
    ├── def _set_gradients_(self, dp: Dict[str, Dict[str, Any]]) -> None   [override]
    │   ├── # Zeros grads and backprops the single scalar loss.
    │   ├── impls optimizer.zero_grad
    │   ├── impls assert dp['losses'] is a single-element torch.Tensor
    │   └── impls losses.backward
    └── @torch.no_grad() def _val_epoch_(self) -> None   [override]
        ├── # Runs the base validation epoch under torch.no_grad to prevent gradient computation.
        └── calls super()._val_epoch_
```
