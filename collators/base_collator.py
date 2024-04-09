from typing import List, Dict, Callable, Any, Optional
import torch
from utils.ops import transpose_buffer


class BaseCollator:

    def __init__(self, collators: Optional[Dict[str, Dict[str, Callable[[List[torch.Tensor]], Any]]]] = None) -> None:
        if collators is None:
            collators = {}
        self.collators = collators

    def __call__(self, examples: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        examples = transpose_buffer(examples)
        for key1 in examples:
            examples[key1] = transpose_buffer(examples[key1])
            for key2 in examples[key1]:
                if (key1 not in self.collators) or (key2 not in self.collators[key1]):
                    # apply default collate function: torch.stack
                    try:
                        examples[key1][key2] = torch.stack(examples[key1][key2], dim=0)
                    except Exception as e:
                        raise RuntimeError(f"[ERROR] Cannot stack tensors into batch at {key1=}, {key2=}: {e}")
                else:
                    examples[key1][key2] = self.collators[key1][key2](examples[key1][key2])
        return examples
