from typing import Any, Dict, List, Optional, Tuple
import os
import time
from concurrent.futures import ThreadPoolExecutor
import random
from agents.base_agent import BaseAgent
from utils.automation.cfg_log_conversion import get_work_dir
from utils.automation.run_status import RunStatus, get_all_run_status, parse_config
from utils.logging import TextLogger
from utils.ssh.pool import _ssh_pool


class Launcher(BaseAgent):

    def __init__(
        self,
        config_files: List[str],
        expected_files: List[str],
        epochs: int = 100,
        sleep_time: int = 180,
        outdated_days: int = 120,
        gpu_pool: List[Tuple[str, List[int]]] = [],
        user_names: Dict[str, str] = {},
        timeout: int = 5,
        log_path: str = "",
        project_dir: str = "",
        conda_env: str = "",
        keep_tmux: Optional[bool] = False,
    ) -> None:
        r"""
        Args:
            config_files (List[str]): the set of experiments to take care of.
            expected_files (List[str]): the expected files under a work dir to check for.
            epochs (int): the number of epochs to run.
            sleep_time (int): the time in seconds to wait to determine if a sessions is still running.
            outdated_days (int): the number of days to wait to consider a run outdated.
            gpu_pool (List[Tuple[str, List[int]]]): list of (server, gpu_indices) tuples.
            user_names (Dict[str, str]): the user names for the servers.
            timeout (int): the timeout for the GPU monitor.
            log_path (str): the path to the log file.
            project_dir (str): the project directory.
            conda_env (str): the conda environment to use.
            keep_tmux (Optional[bool]): whether to keep the tmux session alive.
        """
        super(Launcher, self).__init__(
            config_files=config_files,
            expected_files=expected_files,
            epochs=epochs,
            sleep_time=sleep_time,
            outdated_days=outdated_days,
            gpu_pool=gpu_pool,
            user_names=user_names,
            timeout=timeout,
        )
        self.project_dir = project_dir
        self.conda_env = conda_env
        self.keep_tmux = keep_tmux
        self.logger = TextLogger(filepath=log_path)
        self.ssh_pool = _ssh_pool

    # ====================================================================================================
    # experiment management
    # ====================================================================================================

    def _remove_stuck(self, all_running_status: List[RunStatus]) -> None:
        stuck_cfgs = [run.config for run in all_running_status if run.status == 'stuck']

        def process_gpu(gpu):
            gpu_stuck_info = {}
            if not gpu['connected']:
                return {}
            for proc in gpu['processes']:
                if proc['user'] != gpu['server'].split('@')[0]:
                    continue
                if 'python main.py --config-filepath' not in proc['cmd']:
                    continue
                cfg = parse_config(proc['cmd'])
                if cfg in stuck_cfgs:
                    gpu_stuck_info[cfg] = (gpu['server'], proc['pid'])
            return gpu_stuck_info

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_gpu, self.system_monitor.connected_gpus))

        # Combine all GPU results into a single dictionary
        stuck_cfgs_info = {}
        for gpu_info in results:
            stuck_cfgs_info.update(gpu_info)

        self.logger.info(f"The following processes will be killed {stuck_cfgs_info}")
        for server, pid in stuck_cfgs_info.values():
            self.ssh_pool.execute(server, ['kill', '-9', str(pid)])

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
        Find idle GPUs that meet both GPU and CPU resource constraints.

        Args:
            num_jobs (int): the maximum number of jobs allowed on a single GPU.
        Returns:
            idle_gpus (List[Dict[str, Any]]): a list of dictionaries with the following fields
            {
                server (str): a string in <user_name>@<server_ip> format.
                resource_id (int): GPU index
            }
        """
        idle_gpus = []

        # Build a map of server -> CPU status for quick lookup
        cpu_status_by_server = {}
        for cpu in self.system_monitor.connected_cpus:
            cpu_status_by_server[cpu['server']] = cpu

        # Find idle GPUs with CPU constraints
        for gpu in self.system_monitor.connected_gpus:
            # Check GPU constraints
            gpu_util_ok = gpu['util_stats']['avg'] < 50
            gpu_mem_ok = (gpu['max_memory'] - gpu['memory_stats']['avg']) > 12 * 1024
            gpu_jobs_ok = len([p for p in gpu['processes'] if 'python main.py --config-filepath' in p['cmd']]) < num_jobs

            # Check CPU constraints for the same server
            server = gpu['server']
            cpu_ok = False
            if server in cpu_status_by_server:
                cpu = cpu_status_by_server[server]
                if cpu['cpu_stats'] is not None and cpu['memory_stats'] is not None:
                    cpu_util_ok = cpu['cpu_stats']['avg'] < 80
                    cpu_mem_ok = (cpu['max_memory'] - cpu['memory_stats']['avg']) > 4 * 1024  # 4GB
                    cpu_load_ok = cpu['load_stats']['avg'] < 8  # Assume 8 cores
                    cpu_ok = cpu_util_ok and cpu_mem_ok and cpu_load_ok

            # GPU is only considered idle if both GPU and CPU resources are available
            if gpu_util_ok and gpu_mem_ok and gpu_jobs_ok and cpu_ok:
                idle_gpus.append({
                    'server': gpu['server'],
                    'resource_id': gpu['index'],
                })

        self.logger.warning(f"Disconnected GPUs: {self.system_monitor.disconnected_gpus}")
        self.logger.warning(f"Disconnected CPUs: {self.system_monitor.disconnected_cpus}")

        return idle_gpus

    def _launch_missing(self, all_running: List[Dict[str, Any]], num_jobs: int) -> bool:
        r"""
        Returns:
            done (bool): nothing more to launch.
        """
        missing_runs: List[str] = self._find_missing_runs(all_running)
        if len(missing_runs) == 0:
            return True
        idle_gpus: List[Dict[str, Any]] = self._find_idle_gpus(num_jobs)
        if len(idle_gpus) == 0:
            self.logger.info("Waiting for idle GPUs (with sufficient CPU resources)...")
            return False
        random.shuffle(missing_runs)
        random.shuffle(idle_gpus)
        num_launch = min(len(idle_gpus), len(missing_runs))
        idle_gpus = idle_gpus[:num_launch]
        missing_runs = missing_runs[:num_launch]

        def launch_job(resource, run):
            error_log = os.path.join(get_work_dir(run), "error.log")
            if os.path.isfile(error_log) and os.path.getsize(error_log) > 0:
                self.logger.error(f"Please fix {run}. {error_log=}.")

            # Set GPU environment variables (all jobs are GPU-based)
            device_env = f"CUDA_VISIBLE_DEVICES={resource['resource_id']}"
            resource_label = f"GPU-{resource['resource_id']}"

            cmd = ' && '.join([
                f"cd {self.project_dir}",
                "git checkout main",
                "git pull --rebase origin main",
                "source ~/.bashrc",
                f"source ~/miniconda3/bin/activate {self.conda_env}",
                f"mkdir -p {os.path.dirname(error_log)}",
                ' '.join([
                    "MKL_SERVICE_FORCE_INTEL=1",
                    device_env,
                    "python", "main.py", "--config-filepath", run,
                    # "2>", error_log,
                ]),
            ])
            cmd = cmd + "; exec bash" if self.keep_tmux else cmd
            session_name = f"{'/'.join(os.path.splitext(run)[0].split('/')[-2:])}-{resource_label}"
            tmux_cmd = f"tmux new-session -d -s {session_name} \"{cmd}\""

            self.logger.info(f"Executing command on {resource['server']} ({resource_label}): {tmux_cmd}")
            self.ssh_pool.execute(resource['server'], [tmux_cmd])

        for gpu, run in zip(idle_gpus, missing_runs):
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
                sleep_time=self.sleep_time,
                outdated_days=self.outdated_days,
                system_monitor=self.system_monitor,
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
