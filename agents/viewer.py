from typing import List, Dict
import os
import json
import torch
import tqdm
import matplotlib.pyplot as plt
from utils.ops import transpose_buffer
from agents import BaseAgent


class Viewer(BaseAgent):

    def _plot_training_losses_single(self, config_file: str) -> None:
        # load training losses
        work_dir = self._get_work_dir(config_file)
        logs: List[Dict[str, torch.Tensor]] = []
        idx = 0
        while True:
            epoch_dir = os.path.join(work_dir, f"epoch_{idx}")
            if not all([
                os.path.isfile(os.path.join(epoch_dir, filename))
                for filename in self.expected_files
            ]):
                break
            logs.append(torch.load(os.path.join(epoch_dir, "training_losses.pt")))
            idx += 1
        logs: Dict[str, List[torch.Tensor]] = transpose_buffer(logs)
        # plot training losses
        for key in logs:
            plt.figure()
            plt.plot(torch.stack(logs[key], dim=0).tolist())
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title(f"Training Losses: {key}")
            # save to disk
            output_dir = os.path.join(work_dir, "visualization")
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"training_losses_{key}.png"))
        return

    def _plot_validation_scores_single(self, config_file: str) -> None:
        # load validation scores
        work_dir = self._get_work_dir(config_file)
        logs: List[Dict[str, float]] = []
        idx = 0
        while True:
            epoch_dir = os.path.join(work_dir, f"epoch_{idx}")
            if not all([
                os.path.isfile(os.path.join(epoch_dir, filename))
                for filename in self.expected_files
            ]):
                break
            logs.append(json.load(os.path.join(epoch_dir, "validation_scores.json")))
            idx += 1
        logs: Dict[str, List[torch.Tensor]] = transpose_buffer(logs)
        # plot validation scores
        for key in logs:
            plt.figure()
            plt.plot(torch.stack(logs[key], dim=0).tolist())
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"Validation Scores: {key}")
            # save to disk
            output_dir = os.path.join(work_dir, "visualization")
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"validation_scores_{key}.png"))
        return

    def plot_training_losses_all(self) -> None:
        for config_file in tqdm.tqdm(self.config_files):
            self._plot_training_losses_single(config_file)

    def plot_validation_scores_all(self) -> None:
        for config_file in tqdm.tqdm(self.config_files):
            self._plot_validation_scores_single(config_file)
