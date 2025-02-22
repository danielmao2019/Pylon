from typing import List, Optional
import os
import glob
import json
import time
import torch


def is_running(work_dir: str, sleep_time) -> bool:
    # input checks
    assert os.path.isdir(work_dir), f"{work_dir=}"
    # determine if session is running
    logs = glob.glob(os.path.join(work_dir, "train_val*.log"))
    if len(logs) == 0:
        return False
    last_update = max([os.path.getmtime(fp) for fp in logs])
    return time.time() - last_update <= sleep_time


def get_session_progress(work_dir: str, expected_files: List[str], epochs: int) -> int:
    idx = 0
    while True:
        if idx >= epochs:
            break
        if not check_epoch_finished(
            epoch_dir=os.path.join(work_dir, f"epoch_{idx}"),
            expected_files=expected_files,
            check_load=False,
        ):
            break
        idx += 1
    return idx


def has_finished(work_dir: str, epochs: int) -> bool:
    assert os.path.isdir(work_dir), f"{work_dir=}"
    return get_session_progress(work_dir) == epochs


def has_failed(work_dir: str) -> bool:
    return not is_running(work_dir) and not has_finished(work_dir)


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
