import os
import glob
import time


def is_running(work_dir: str, wait_time: float) -> bool:
    logs = glob.glob(os.path.join(work_dir, "train_val*.log"))
    if len(logs) == 0:
        return False
    return time.time() - max([os.path.getmtime(fp) for fp in logs])<= wait_time


def has_finished(work_dir: str, expected_epochs: int) -> bool:
    for idx in range(expected_epochs):
        epoch_finished = all([
            os.path.isfile(os.path.join(work_dir, f"epoch_{idx}", filename))
            for filename in ["training_losses.pt", "validation_scores.json", "gradient_measurements.pt"]
        ])
        if not epoch_finished:
            return False
    return True


def has_failed(work_dir: str, expected_epochs: int, wait_time: float) -> bool:
    return not is_running(work_dir=work_dir, wait_time=wait_time) and \
           not has_finished(work_dir=work_dir, expected_epochs=expected_epochs)
