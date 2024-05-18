from typing import Dict, Any
from .semantic_segmentation_diffuser import SemanticSegmentationDiffuser


class CCDMDiffuser(SemanticSegmentationDiffuser):

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Any]]:
        example = super().__getitem__(idx)
        assert set(example['inputs'].keys()) == set(['image', 'diffused_mask', 'time']), f"{example['inputs'].keys()=}"
        assert set(example['labels'].keys()) == set(['original_mask']), f"{example['labels'].keys()=}"
        example['labels']['time'] = example['inputs']['time'].clone()
        example['labels']['diffused_mask'] = example['inputs']['diffused_mask'].clone()
        return example
