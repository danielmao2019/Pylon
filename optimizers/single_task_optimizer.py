from typing import Any
from optimizers import BaseOptimizer
from utils.builders import build_from_config


class SingleTaskOptimizer(BaseOptimizer):

    def __init__(self, optimizer_config: dict) -> None:
        super(SingleTaskOptimizer, self).__init__()
        self.optimizer = build_from_config(config=optimizer_config)

    def backward(self, *args, **kwargs) -> Any:
        r"""Intentionally left blank. This is not needed as back-propagation is done in SupervisedSingleTaskTrainer._set_gradients_.
        """
