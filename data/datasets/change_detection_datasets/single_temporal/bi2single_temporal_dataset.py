from typing import Tuple, Dict, Any, Optional
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
        assert source.LABEL_NAMES == ['img_1', 'img_2']
        super(Bi2SingleTemporal, self).__init__(
            source, 2*len(source), transforms_cfg, use_cache, device,
        )

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        source_idx = idx // 2
        input_idx = idx % 2
        inputs, _, _ = super(Bi2SingleTemporal, self)._load_datapoint(source_idx)
        inputs = {
            'image': inputs[f"img_{input_idx}"],
        }
        return inputs, {}, {}
