from typing import List, Optional
import os
import json
import torch


def _check_file_loadable(filepath: str) -> bool:
    assert os.path.isfile(filepath)
    assert filepath.endswith(".json") or filepath.endswith(".pt")
    result: bool = True
    if filepath.endswith(".json"):
        try:
            with open(filepath, mode='r') as f:
                _ = json.load(f)
        except:
            result = False
    elif filepath.endswith(".pt"):
        try:
            _ = torch.load(filepath)
        except:
            result = False
    else:
        assert 0
    return result


def check_epoch_finished(epoch_dir: str, expected_files: List[str], check_load: Optional[bool] = True) -> bool:
    r"""Three criteria:
        1. File exists.
        2. File non-empty.
        3. File load-able.
    """
    return all([
        os.path.isfile(os.path.join(epoch_dir, filename)) and
        os.path.getsize(os.path.join(epoch_dir, filename)) > 0 and
        ((not check_load) or _check_file_loadable(os.path.join(epoch_dir, filename)))
        for filename in expected_files
    ])
