from typing import Tuple, Dict, Any
import torch
from data.datasets import BaseDataset, BaseSyntheticDataset


class GANDataset(BaseSyntheticDataset):

    def __init__(self, source: BaseDataset, latent_dim: int, **kwargs) -> None:
        assert type(latent_dim) == int, f"{type(latent_dim)=}"
        self.latent_dim = latent_dim
        super(GANDataset, self).__init__(source=source, dataset_size=len(source), **kwargs)

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        meta_info = {
            'cpu_rng_state': torch.get_rng_state(),
            'gpu_rng_state': torch.cuda.get_rng_state(),
        }
        inputs = {
            'z': torch.normal(mean=0, std=1, size=(self.latent_dim,)),
        }
        labels = {
            'image': self.source[idx]['inputs']['image'],
        }
        return inputs, labels, meta_info
