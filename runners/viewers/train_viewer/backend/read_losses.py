from typing import List
import os

import torch

def read_losses(logs_dirpath: str, epochs: int) -> List[torch.Tensor]:
    results = []
    for idx in range(epochs):
        epoch_dir = os.path.join(logs_dirpath, f'epoch_{idx}')
        losses_file = os.path.join(epoch_dir, 'training_losses.pt')
        losses = torch.load(losses_file)
        results.append(losses)
    return results
