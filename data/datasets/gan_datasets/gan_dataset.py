from typing import Tuple, Dict, Any
import torch
from data.datasets import BaseDataset, BaseSyntheticDataset


class GANDataset(BaseSyntheticDataset):

    def __init__(self, source: BaseDataset, latent_dim: int, **kwargs) -> None:
        assert type(latent_dim) == int, f"{type(latent_dim)=}"
        self.latent_dim = latent_dim
        super(GANDataset, self).__init__(source=source, dataset_size=len(source), **kwargs)

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()
        # GANDataset has parameters that affect data generation
        version_dict.update({
            'synthetic_ratio': self.synthetic_ratio,
            'latent_dim': self.latent_dim,
            'base_dataset_config': self.base_dataset_config,
        })
        return version_dict

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        meta_info = {
            'cpu_rng_state': torch.get_rng_state(),
            'gpu_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        inputs = {
            'z': torch.normal(mean=0, std=1, size=(self.latent_dim,)),
        }
        labels = {
            'image': self.source[idx]['inputs']['image'],
        }
        return inputs, labels, meta_info
