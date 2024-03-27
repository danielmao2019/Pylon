from typing import Tuple, List, Dict, Any
import os
import json


def build_from_config(config: Dict[str, Any], **kwargs) -> Any:
    assert type(config) == dict, f"{type(config)=}"
    assert 'class' in config and 'args' in config, f"{config.keys()}"
    return config['class'](**config['args'], **kwargs)


def find_best_checkpoint(checkpoints: List[str]) -> str:
    r"""
    Args:
        checkpoints (List[str]): a list of filepaths to checkpoints.
    Returns:
        best_checkpoint (str): the filepath to the checkpoint with the highest validation score.
    """
    assert type(checkpoints) == list, f"{type(checkpoints)=}"
    if len(checkpoints) == 1:
        return checkpoints[0]
    avg_scores: List[Tuple[str, float]] = []
    for fp in checkpoints:
        epoch_dir: str = os.path.dirname(fp)
        with open(os.path.join(epoch_dir, "validation_scores.json"), mode='r') as f:
            scores: Dict[str, float] = json.load(f)
        avg_scores.append((epoch_dir, scores['average']))
    best_checkpoint: str = os.path.join(max(avg_scores, key=lambda x: x[1])[0], "checkpoint.pt")
    assert best_checkpoint in checkpoints
    return best_checkpoint
