from typing import List, Dict, Any, Optional, Tuple
import os
import time
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from agents import BaseAgent
from utils.automation.cfg_log_conversion import get_work_dir
from utils.automation.run_status import RunStatus, get_all_run_status, parse_config
from utils.monitor.gpu_status import GPUStatus
from utils.monitor.gpu_monitor import GPUMonitor
from utils.logging.text_logger import TextLogger


class Launcher(BaseAgent):

    def __init__(
        self,
        config_files: List[str],
        expected_files: List[str],
        project_dir: str,
        conda_env: str,
        gpu_pool: List[Tuple[str, List[int]]],
        log_path: str,
        epochs: int = 100,
        sleep_time: Optional[int] = 180,
        outdated_days: int = 120,
        keep_tmux: Optional[bool] = False,
    ) -> None:
        r"""
        Args:
            config_files (List[str]): the set of experiments to take care of.
            expected_files (List[str]): the expected files under a work dir to check for.
            project_dir (str): the project directory.
            conda_env (str): the conda environment to use.
            gpu_pool (List[Tuple[str, List[int]]]): list of (server, gpu_indices) tuples.
            log_path (str): the path to the log file.
            epochs (int): the number of epochs to run.
            sleep_time (int): the time in seconds to wait to determine if a sessions is still running.
            outdated_days (int): the number of days to wait to consider a run outdated.
            keep_tmux (Optional[bool]): whether to keep the tmux session alive.
        """
        super(Launcher, self).__init__(config_files=config_files, expected_files=expected_files)
        self.project_dir = project_dir
        self.conda_env = conda_env
        self.epochs = epochs
        self.sleep_time = sleep_time
        self.outdated_days = outdated_days
        self.keep_tmux = keep_tmux
        self.logger = TextLogger(filepath=log_path)

        # Initialize GPU objects from pool
        self.gpus = [
            GPUStatus(
                server=server,
                index=idx,
                max_memory=0,  # Will be populated by monitor
                processes=[],
                window_size=10,
                memory_window=[],
                util_window=[],
                memory_stats={'min': None, 'max': None, 'avg': None},
                util_stats={'min': None, 'max': None, 'avg': None}
            )
            for server, indices in gpu_pool
            for idx in indices
        ]

        # Initialize monitor
        self.monitor = GPUMonitor(self.gpus)
        self.monitor.start()

    # ====================================================================================================
    # experiment management
    # ====================================================================================================

    def _remove_stuck(self, all_running_status: List[RunStatus]) -> None:
        stuck_cfgs = [run.config for run in all_running_status if run.status == 'stuck']

        def process_gpu(gpu):
            gpu_stuck_info = {}
            for proc in gpu['processes']:
                if proc['user'] != gpu['server'].split('@')[0]:
                    continue
                cfg = parse_config(proc['cmd'])
                if cfg in stuck_cfgs:
                    gpu_stuck_info[cfg] = (gpu['server'], proc['pid'])
            return gpu_stuck_info

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_gpu, self.gpus))

        # Combine all GPU results into a single dictionary
        stuck_cfgs_info = {}
        for gpu_info in results:
            stuck_cfgs_info.update(gpu_info)

        self.logger.info(f"The following processes will be killed {stuck_cfgs_info}")
        for server, pid in stuck_cfgs_info.values():
            cmd = ['ssh', server, 'kill', '-9', pid]
            subprocess.check_output(cmd)

    def _remove_outdated(self, all_running_status: List[RunStatus]) -> None:
        outdated_runs = list(filter(lambda x: x.status == 'outdated', all_running_status))
        self.logger.info(f"The following runs has not been updated in the last {self.outdated_days} days and will be removed: {[run.work_dir for run in outdated_runs]}")
        with ThreadPoolExecutor() as executor:
            list(executor.map(lambda x: os.system(f"rm -rf {x.work_dir}"), outdated_runs))

    def _find_missing_runs(self, all_running_status: List[RunStatus]) -> List[str]:
        r"""
        Returns:
            result (List[str]): the config filepaths for the missing experiment runs.
        """
        return [
            run.config for run in all_running_status if run.status == 'failed'
        ]

    def _find_idle_gpus(self, num_jobs: int) -> List[Dict[str, Any]]:
        r"""
        Args:
            num_jobs (int): the maximum number of jobs allowed on a single GPU.
        Returns:
            all_idle_gpus (List[Dict[str, Any]]): a list of dictionaries with the following fields
            {
                server (str): a string in <user_name>@<server_ip> format.
                gpu_index (int): index of GPU on the server.
            }
        """
        idle_gpus = []
        for gpu in self.gpus:
            if (
                gpu['util_stats']['avg'] < 50 and
                gpu['memory_stats']['avg'] > 12 * 1024 and
                len(gpu['processes']) < num_jobs
            ):
                idle_gpus.append({
                    'server': gpu['server'],
                    'gpu_index': gpu['index'],
                })
        return idle_gpus

    def _launch_missing(self, all_running: List[Dict[str, Any]], num_jobs: int) -> bool:
        r"""
        Returns:
            done (bool): nothing more to launch.
        """
        missing_runs: List[str] = self._find_missing_runs(all_running)
        if len(missing_runs) == 0:
            return True
        gpu_pool: List[Dict[str, Any]] = self._find_idle_gpus(num_jobs)
        if len(gpu_pool) == 0:
            self.logger.info("Waiting for idle GPU...")
            return False
        random.shuffle(missing_runs)
        random.shuffle(gpu_pool)
        num_launch = min(len(gpu_pool), len(missing_runs))
        gpu_pool = gpu_pool[:num_launch]
        missing_runs = missing_runs[:num_launch]

        def launch_job(gpu, run):
            error_log = os.path.join(get_work_dir(run), "error.log")
            if os.path.isfile(error_log) and os.path.getsize(error_log) > 0:
                self.logger.error(f"Please fix {run}. {error_log=}.")
            cmd = ' && '.join([
                f"cd {self.project_dir}",
                "git checkout main",
                "git pull --rebase origin main",
                "source ~/.bashrc",
                f"source ~/miniconda3/bin/activate {self.conda_env}",
                f"mkdir -p {os.path.dirname(error_log)}",
                ' '.join([
                    "MKL_SERVICE_FORCE_INTEL=1",
                    f"CUDA_VISIBLE_DEVICES={gpu['gpu_index']}",
                    "python", "main.py", "--config-filepath", run,
                    # "2>", error_log,
                ]),
            ])
            cmd = cmd + "; exec bash" if self.keep_tmux else cmd
            cmd = f"tmux new-session -d \"{cmd}\""
            cmd = f"ssh {gpu['server']} '{cmd}'"
            self.logger.info(cmd)
            os.system(cmd)

        for gpu, run in zip(gpu_pool, missing_runs):
            launch_job(gpu, run)
        return False

    def spawn(self, num_jobs: Optional[int] = 1) -> None:
        while True:
            self.logger.info('='*50)

            self.logger.info("Collecting all running jobs...")
            all_running_status = get_all_run_status(
                config_files=self.config_files,
                expected_files=self.expected_files,
                epochs=self.epochs,
                servers=list(set([gpu['server'] for gpu in self.gpus])),
                sleep_time=self.sleep_time,
                outdated_days=self.outdated_days,
            )

            self.logger.info("Removing stuck jobs...")
            self._remove_stuck(all_running_status)

            self.logger.info("Removing outdated jobs...")
            self._remove_outdated(all_running_status)

            self.logger.info("Launching missing jobs...")
            done = self._launch_missing(all_running_status, num_jobs=num_jobs)

            if done:
                self.logger.info("All done.")

            self.logger.info(f"Sleeping for {self.sleep_time} seconds...")
            time.sleep(self.sleep_time)
