from typing import List, Dict, Any, Optional
import random
import torch
from data.collators import BaseCollator
from utils.ops import transpose_buffer


class ChangeStarCollator(BaseCollator):

    def __init__(self, max_trails: Optional[int] = 50) -> None:
        if max_trails is not None:
            assert isinstance(max_trails, int)
        self.max_trails = max_trails

    @staticmethod
    def _shuffle(original: list, max_trails: int) -> list:
        t = 0
        while t < max_trails:
            t += 1
            proposal = original.copy()
            random.shuffle(proposal)
            accept = all(x != y for x, y in zip(original, proposal))
            if accept:
                break
        return proposal

    def __call__(self, datapoints: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        # initialization
        batch_size = len(datapoints)
        datapoints = transpose_buffer(datapoints)
        for key1, sub_dict in datapoints.items():
            datapoints[key1] = transpose_buffer(sub_dict)
        assert set(datapoints['inputs'].keys()) == set(['image']), f"{set(datapoints['inputs'].keys())=}"
        datapoints['inputs']['img_1'] = torch.stack(datapoints['inputs']['image'], dim=0)
        del datapoints['inputs']['image']
        assert set(datapoints['labels'].keys()) == set(['semantic_segmentation']), f"{set(datapoints['labels'].keys())=}"
        lbl_1 = torch.stack(datapoints['labels']['semantic_segmentation'], dim=0)
        del datapoints['labels']['semantic_segmentation']
        # shuffle
        original = list(range(batch_size))
        shuffled = self._shuffle(original)
        datapoints['inputs']['img_2'] = datapoints['inputs']['img_1'][shuffled, :, :, :]
        datapoints['labels']['change_map'] = torch.logical_xor(lbl_1, lbl_1[shuffled, :, :])
        # apply default collate function on meta info
        for key2 in datapoints['meta_info']:
            datapoints['meta_info'] = self._default_collate(
                values=datapoints['meta_info'][key2], key1='meta_info', key2=key2,
            )
