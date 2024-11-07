from typing import List, Dict, Any, Optional
import os
import subprocess
import time
import glob
import json
import random
import torch
import matplotlib.pyplot as plt
import tqdm
import utils
from utils.ops import transpose_buffer


class Agent:

    def __init__(
        self,
        project_dir: str,
        conda_env: str,
        config_files: List[str],
        servers: List[str],
        expected_files: Dict[str, List[str]],
        epochs: int = 100,
        sleep_time: Optional[int] = 180,
        keep_tmux: Optional[bool] = False,
    ) -> None:
        r"""
        Args:
            config_files (List[str]): the set of experiments to take care of.
            servers (List[str]): a list of setup servers.
            expected_files (Dict[str, List[str]]): the expected files under a work dir to check for.
            sleep_time (int): the time in seconds to wait to determine if a sessions is still running.
        """
        self.project_dir = project_dir
        self.conda_env = conda_env
        self.config_files = config_files
        self.servers = servers
        self.expected_files = expected_files
        self.epochs = epochs
        self.sleep_time = sleep_time
        self.keep_tmux = keep_tmux
        self.logger = utils.logging.Logger(filepath="./project/run_agent.log")

    # ====================================================================================================
    # session status checking
    # ====================================================================================================

    def _is_running(self, work_dir: str) -> bool:
        # input checks
        assert os.path.isdir(work_dir), f"{work_dir=}"
        # determine if session is running
        logs = glob.glob(os.path.join(work_dir, "train_val*.log"))
        if len(logs) == 0:
            return False
        last_update = max([os.path.getmtime(fp) for fp in logs])
        return time.time() - last_update <= self.sleep_time

    def _get_session_progress(self, work_dir: str) -> int:
        idx = 0
        while True:
            if idx >= self.epochs:
                break
            epoch_finished: bool = all([
                os.path.isfile(os.path.join(work_dir, f"epoch_{idx}", filename))
                for filename in self.expected_files
            ])
            if not epoch_finished:
                break
            idx += 1
        return idx

    def _has_finished(self, work_dir: str) -> bool:
        assert os.path.isdir(work_dir), f"{work_dir=}"
        return self._get_session_progress(work_dir) == self.epochs

    def _has_failed(self, work_dir: str) -> bool:
        return not self._is_running(work_dir) and not self._has_finished(work_dir)

    # ====================================================================================================
    # GPU status checking
    # ====================================================================================================

    @staticmethod
    def _parse_csv(outputs: str) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        if outputs == "":
            return result
        outputs: List[str] = outputs.strip().split('\n')
        outputs: List[List[str]] = [line.strip().split(',') for line in outputs]
        outputs: List[List[str]] = [[cell.strip() for cell in line] for line in outputs]
        for line in outputs:
            assert len(line) == 2, f"{line=}"
            result[line[0]] = result.get(line[0], []) + [line[1]]
        return result

    @staticmethod
    def _get_gpu_pids(server: str) -> List[List[str]]:
        cmd = ' '.join([
            'nvidia-smi', '--query-gpu=index,gpu_uuid', '--format=csv,noheader',
            '&&',
            f"echo {'-'*10}",
            '&&',
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader',
        ])
        cmd = f"ssh {server} '" + cmd + "'"
        outputs = subprocess.check_output(cmd, shell=True, text=True).strip()
        outputs = outputs.split('-'*10)
        assert len(outputs) == 2, f"{outputs=}"
        idx2uuid: Dict[str, List[str]] = Agent._parse_csv(outputs[0])
        uuid2pid: Dict[str, List[str]] = Agent._parse_csv(outputs[1])
        num_gpus = len(idx2uuid)
        result: List[List[str]] = [None] * num_gpus
        for gpu_index in range(num_gpus):
            assert len(idx2uuid[str(gpu_index)]) == 1
            uuid = idx2uuid[str(gpu_index)][0]
            result[gpu_index] = uuid2pid.get(uuid, [])
        return result

    @staticmethod
    def _get_user_pids(server: str) -> List[str]:
        cmd = ' '.join([
            'ps', '-u', server.split('@')[0], '-o', 'pid=',
        ])
        cmd = f"ssh {server} '" + cmd + "'"
        outputs = subprocess.check_output(cmd, shell=True, text=True).strip()
        result: List[str] = list(map(lambda x: x.strip(), outputs.split('\n')))
        return result

    def _find_running(self) -> List[Dict[str, Any]]:
        r"""This function finds all GPU processes launched by the user.

        Returns:
            all_running (List[Dict[str, Any]]): a list of dictionaries with the following fields
            {
                server (str): a string in <user_name>@<server_ip> format.
                gpu_index (int): index of GPU on the server.
                command (str): the command this GPU is running by the user.
            }
        """
        all_running: List[Dict[str, Any]] = []
        for server in self.servers:
            gpu_pids = Agent._get_gpu_pids(server)
            user_pids = Agent._get_user_pids(server)
            for gpu_index in range(len(gpu_pids)):
                for pid in gpu_pids[gpu_index]:
                    if pid not in user_pids:
                        continue
                    cmd = ' '.join([
                        'ps', '-p', pid, '-o', 'cmd=',
                    ])
                    cmd = f"ssh {server} '" + cmd + "'"
                    running = subprocess.check_output(cmd, shell=True, text=True).strip()
                    all_running.append({
                        'server': server,
                        'gpu_index': gpu_index,
                        'command': running
                    })
        return all_running

    def _find_idle_gpus(self) -> List[Dict[str, Any]]:
        r"""
        Returns:
            all_idle_gpus (List[Dict[str, Any]]): a list of dictionaries with the following fields
            {
                server (str): a string in <user_name>@<server_ip> format.
                gpu_index (int): index of GPU on the server.
            }
        """
        all_idle_gpus: List[Dict[str, Any]] = []
        for server in self.servers:
            cmd = ' '.join([
                'nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader',
                '&&',
                'echo', '-'*10,
                '&&',
                'nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader',
            ])
            cmd = f"ssh {server} '" + cmd + "'"
            outputs = subprocess.check_output(cmd, shell=True, text=True).strip()
            outputs = outputs.split('-'*10)
            assert len(outputs) == 2, f"{outputs=}"
            gpu_util = Agent._parse_csv(outputs[0])
            gpu_fmem = Agent._parse_csv(outputs[1])
            assert set(gpu_util.keys()) == set(gpu_fmem.keys())
            gpu_pids = Agent._get_gpu_pids(server)
            for gpu_index in gpu_util:
                assert len(gpu_util[gpu_index]) == 1
                util = gpu_util[gpu_index][0]
                util = int(util.split('%')[0])
                assert len(gpu_fmem[gpu_index]) == 1
                fmem = gpu_fmem[gpu_index][0]
                fmem = int(fmem.split('MiB')[0])
                if util < 50 and fmem > 12 * 1024 and len(gpu_pids[int(gpu_index)]) < 1:
                    all_idle_gpus.append({
                        'server': server,
                        'gpu_index': gpu_index,
                    })
        return all_idle_gpus

    # ====================================================================================================
    # experiment management
    # ====================================================================================================

    def _find_missing_runs(self) -> List[str]:
        r"""
        Returns:
            result (List[str]): the config filepaths for the missing experiment runs.
        """
        result: List[str] = []
        for config_file in self.config_files:
            work_dir = Agent._get_work_dir(config_file)
            if not os.path.isdir(work_dir) or self._has_failed(work_dir):
                result.append(config_file)
        random.shuffle(result)
        return result

    def _launch_missing(self) -> bool:
        r"""
        Returns:
            done (bool): nothing more to launch.
        """
        missing_runs: List[str] = self._find_missing_runs()
        if len(missing_runs) == 0:
            return True
        gpu_pool: List[Dict[str, Any]] = self._find_idle_gpus()
        if len(gpu_pool) == 0:
            self.logger.info("Waiting for idle GPU...")
            return False
        num_launch = min(len(gpu_pool), len(missing_runs))
        gpu_pool = gpu_pool[:num_launch]
        missing_runs = missing_runs[:num_launch]
        for gpu, run in zip(gpu_pool, missing_runs):
            cmd = ' '.join([
                'ssh', gpu['server'],
                "'",
                    'tmux', 'new-session', '-d',
                    '"',
                        # navigate to project dir
                        'cd', self.project_dir, '&&',
                        # pull latest code
                        'git', 'checkout', 'main', '&&', 'git', 'pull', '&&',
                        # conda environment
                        'source', '~/miniconda3/bin/activate', self.conda_env, '&&',
                        # launch command
                        "MKL_SERVICE_FORCE_INTEL=1",
                        f"CUDA_VISIBLE_DEVICES={gpu['gpu_index']}",
                        'python', 'project/main.py', '--config-filepath', run,
                        *([';', 'exec', 'bash'] if self.keep_tmux else []),
                    '"',
                "'",
            ])
            self.logger.info(cmd)
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                # Check if the command was successful
                if result.returncode != 0:
                    print(result.stderr)
                    print(result.stdout)
                    self.logger.error(f"Command failed with error: {result.stderr}")
                    self.logger.error(f"Output:\n{result.stdout}")
            except Exception as e:
                self.logger.error(str(e))
        return False

    # ====================================================================================================
    # results reporting
    # ====================================================================================================

    def _plot_training_losses_single(self, config_file: str) -> None:
        # load training losses
        work_dir = Agent._get_work_dir(config_file)
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
        work_dir = Agent._get_work_dir(config_file)
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

    # ====================================================================================================
    # api
    # ====================================================================================================

    def plot_training_losses_all(self) -> None:
        for config_file in tqdm.tqdm(self.config_files):
            self._plot_training_losses_single(config_file)

    def plot_validation_scores_all(self) -> None:
        for config_file in tqdm.tqdm(self.config_files):
            self._plot_validation_scores_single(config_file)

    def spawn(self) -> None:
        while True:
            self.logger.info('='*50)
            done = self._launch_missing()
            if done:
                self.logger.info("All done.")
            self.logger.info("")
            time.sleep(self.sleep_time)

    def progress_report(self) -> str:
        result: List[str] = []
        total: int = 0
        for config_file in self.config_files:
            work_dir = self._get_work_dir(config_file)
            progress = self._get_session_progress(work_dir)
            percentage = int(progress / self.epochs * 100)
            total += percentage
            result.append(f"[{percentage*'#'}{(100-percentage)*'-'}] {percentage:02d}%")
        total = round(total / len(self.config_files), 2)
        result = f"Total progress: {total}%\n" + '\n'.join(result)
        return result

    # ====================================================================================================
    # ====================================================================================================

    @staticmethod
    def _get_work_dir(config_file: str) -> str:
        return os.path.splitext(os.path.join("./logs", os.path.relpath(config_file, start="./configs")))[0]
