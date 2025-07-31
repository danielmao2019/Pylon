from typing import Tuple, List, Dict, Union, Any
import torch
from utils.builders import build_from_config


class ProjectionDatasetWrapper(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_config: dict,
        mapping: Dict[str, List[Union[str, Tuple[str, str]]]],
    ) -> None:
        super(ProjectionDatasetWrapper, self).__init__()
        self.dataset = build_from_config(dataset_config)
        self._init_mapping_(mapping)

    def _init_mapping_(self, mapping: Dict[str, List[Union[str, Tuple[str, str]]]]) -> None:
        assert type(mapping) == dict, f"{type(mapping)=}"
        assert set(mapping.keys()).issubset(['inputs', 'labels', 'meta_info']), f"{mapping.keys()=}"
        for group in mapping:
            assert type(mapping[group]) == list, f"{type(mapping[group])=}"
            for idx in range(len(mapping[group])):
                if type(mapping[group][idx]) == str:
                    mapping[group][idx] = (mapping[group][idx],) * 2
                assert type(mapping[group][idx]) == tuple, f"{type(mapping[group][idx])=}"
                assert len(mapping[group][idx]) == 2, f"{len(mapping[group][idx])=}"
                assert type(mapping[group][idx][0]) == type(mapping[group][idx][1]) == str, f"{tuple(type(elem) for elem in mapping[group][idx])=}"
        self.mapping: Dict[str, List[Tuple[str, str]]] = mapping

    def get_cache_version_hash(self) -> str:
        """Get cache version hash delegating to the wrapped dataset.
        
        Returns:
            str: 16-character hexadecimal hash string
        """
        # ProjectionDatasetWrapper delegates cache versioning to the wrapped dataset
        # The mapping doesn't affect cache content since it's just a projection
        return self.dataset.get_cache_version_hash()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Any]]:
        datapoint = self.dataset[idx]
        result = {}
        for group in self.mapping:
            result[group] = {}
            for src, tgt in self.mapping[group]:
                result[group][tgt] = datapoint[group][src]
        return result
