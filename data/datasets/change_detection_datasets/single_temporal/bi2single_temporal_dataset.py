from typing import Tuple, Dict, Any, Optional, List
import torch
from data.datasets import BaseDataset, BaseSyntheticDataset


class Bi2SingleTemporal(BaseSyntheticDataset):

    def __init__(
        self,
        source: BaseDataset,
        transforms_cfg: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = True,
        device: Optional[torch.device] = torch.device('cuda'),
    ):
        assert source.INPUT_NAMES == ['img_1', 'img_2']
        super(Bi2SingleTemporal, self).__init__(
            source, 2*len(source), transforms_cfg, use_cache, device,
        )

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()
        # Bi2SingleTemporal depends on the source dataset's version
        version_dict.update({
            'source_version': self.source.get_cache_version_hash(),
        })
        return version_dict

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        source_idx = idx // 2
        input_idx = idx % 2 + 1
        inputs, _, _ = self.source._load_datapoint(source_idx)
        inputs = {
            'image': inputs[f"img_{input_idx}"],
        }
        return inputs, {}, {'input_idx': input_idx}

    def display_datapoint(
        self,
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None
    ) -> None:
        """Minimal display_datapoint implementation for synthetic datasets.

        Full visualization support for synthetic datasets is not yet implemented.
        """
        return None
