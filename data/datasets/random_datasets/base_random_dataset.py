from typing import Tuple, List, Dict, Callable, Any, Union, Optional
import torch
from data.datasets.base_dataset import BaseDataset


class BaseRandomDataset(BaseDataset):

    SPLIT_OPTIONS = ['all']
    INPUT_NAMES = None
    LABEL_NAMES = None

    def __init__(
        self,
        num_examples: int,
        gen_func_config: Dict[str, Dict[str, Tuple[Callable, dict]]],
        **kwargs,
    ) -> None:
        # init num examples
        assert isinstance(num_examples, int), f"{type(num_examples)=}"
        assert num_examples >= 0, f"{num_examples=}"
        self.num_examples = num_examples
        # init gen func config
        self._init_gen_func_config_(config=gen_func_config)
        # init transform
        super(BaseRandomDataset, self).__init__(**kwargs)

    def _init_gen_func_config_(self, config: Dict[str, Dict[str, Tuple[Callable, dict]]]) -> None:
        assert isinstance(config, dict), f"{type(config)=}"
        assert set(config.keys()) == set(['inputs', 'labels']), f"{config.keys()=}"
        self.INPUT_NAMES = list(config['inputs'].keys())
        self.LABEL_NAMES = list(config['labels'].keys())
        for key1 in config:
            assert isinstance(config[key1], dict), f"{type(config[key1])=}"
            for key2 in config[key1]:
                assert isinstance(config[key1][key2], tuple), f"{type(config[key1][key2])=}"
                assert len(config[key1][key2]) == 2, f"{len(config[key1][key2])=}"
                assert callable(config[key1][key2][0]), f"{type(config[key1][key2][0])=}"
                assert isinstance(config[key1][key2][1], dict), f"{type(config[key1][key2][1])=}"
        self.gen_func_config = config

    def _init_annotations_all_splits(self) -> None:
        r"""Override to skip annotation initialization for random datasets.
        
        BaseRandomDataset generates data on-the-fly and doesn't use annotations.
        """
        pass

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()
        version_dict.update({
            'num_examples': self.num_examples,
            'gen_func_config': str(self.gen_func_config),  # Convert to string for hashing
            'base_seed': self.base_seed,
        })
        return version_dict

    def __len__(self) -> int:
        return self.num_examples

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        # Create generator locally to avoid pickle issues with multiprocessing
        generator = torch.Generator()
        seed = (self.base_seed or 0) + idx
        generator.manual_seed(seed)

        # Always create tensors on CPU - BaseDataset handles device transfer
        inputs, labels = tuple({
            key2: self.gen_func_config[key1][key2][0](
                **self.gen_func_config[key1][key2][1],
                generator=generator,
            )
            for key2 in self.gen_func_config[key1]
        } for key1 in ['inputs', 'labels'])

        meta_info = {'seed': seed}
        return inputs, labels, meta_info

    def display_datapoint(
        self,
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None
    ) -> None:
        """Minimal display_datapoint implementation for random datasets.
        
        Random datasets do not support visualization.
        """
        return None
