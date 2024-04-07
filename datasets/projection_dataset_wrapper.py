from typing import Tuple, List, Dict, Union, Any
import torch
from utils.builder import build_from_config


class ProjectionDatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset: dict, mapping: Dict[str, List[Union[str, Tuple[str, str]]]]):
        super(ProjectionDatasetWrapper, self).__init__()
        self.dataset = build_from_config(dataset)
        self._init_mapping_(mapping)

    def _init_mapping_(self, mapping: Dict[str, List[Union[str, Tuple[str, str]]]]) -> None:
        assert set(mapping.keys()).issubset(['inputs', 'labels', 'meta_info']), f"{mapping.keys()}"
        for key in mapping:
            for idx in range(len(mapping[key])):
                if type(mapping[key][idx]) == str:
                    mapping[key][idx] = (mapping[key][idx],) * 2
                else:
                    assert type(mapping[key][idx]) == tuple
                    assert len(mapping[key][idx]) == 2
                    assert type(mapping[key][idx][0]) == type(mapping[key][idx][1]) == str
        self.mapping: Dict[str, List[Tuple[str, str]]] = mapping

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Any]]:
        example = self.dataset[idx]
        result = {}
        for key in self.mapping:
            result[key] = {}
            for src, tgt in self.mapping[key]:
                result[key][tgt] = example[key][src]
        return result
