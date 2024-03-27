import os
import random
import numpy
import torch


def set_determinism():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_seed(seed: int):
    assert type(seed) == int, f"{type(seed)=}"
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
