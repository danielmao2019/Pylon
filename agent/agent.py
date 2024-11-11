from typing import List, Dict, Any, Optional
import os
import subprocess
import threading
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import time
import glob
import json
import random
import torch
import matplotlib.pyplot as plt
import tqdm
import utils
from utils.ops import transpose_buffer
from utils.progress import check_epoch_finished


class Agent:

    def __init__(
        self,
        project_dir: str,
        conda_env: str,
        config_files: List[str],
        servers: List[str],
        expected_files: List[str],
        epochs: int = 100,
        sleep_time: Optional[int] = 180,
        keep_tmux: Optional[bool] = False,
    ) -> None:
        r"""
        Args:
            config_files (List[str]): the set of experiments to take care of.
            servers (List[str]): a list of setup servers.
            expected_files (List[str]): the expected files under a work dir to check for.
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
        self._init_status()

    def _init_status(self) -> None:
        self.status = {}
        threading.Thread(target=self._get_status, daemon=True).start()
        self.logger.info("Waiting for self.status initialization...")
        while set(self.status.keys()) != set(self.servers):
            time.sleep(1)
        self.logger.info("Done.")

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
            if not check_epoch_finished(
                epoch_dir=os.path.join(work_dir, f"epoch_{idx}"),
                expected_files=self.expected_files,
            ):
                break
            idx += 1
        return idx

    def _has_finished(self, work_dir: str) -> bool:
        assert os.path.isdir(work_dir), f"{work_dir=}"
        return self._get_session_progress(work_dir) == self.epochs

    def _has_failed(self, work_dir: str) -> bool:
        return not self._is_running(work_dir) and not self._has_finished(work_dir)

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

    # ====================================================================================================
    # GPU status checking
    # ====================================================================================================

    @staticmethod
    def _build_mapping(output: str) -> Dict[str, List[str]]:
        lines = output.decode().strip().splitlines()
        assert type(lines) == list
        assert all([type(elem) == str for elem in lines])
        lines = [line.strip().split(", ") for line in lines]
        assert all([len(line) == 2 for line in lines])
        result: Dict[str, List[str]] = {}
        for line in lines:
            result[line[0]] = result.get(line[0], []) + [line[1]]
        return result

    @staticmethod
    def _get_index2util(server: str) -> List[Dict[str, int]]:
        fmem_cmd = ['ssh', server, 'nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits']
        util_cmd = ['ssh', server, 'nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits']
        fmem_out: Dict[str, List[str]] = Agent._build_mapping(subprocess.check_output(fmem_cmd))
        util_out: Dict[str, List[str]] = Agent._build_mapping(subprocess.check_output(util_cmd))
        index2fmem: List[int] = []
        index2util: List[int] = []
        for index in fmem_out:
            assert len(fmem_out[index]) == 1
            index2fmem.append(int(fmem_out[index][0]))
        for index in util_out:
            assert len(util_out[index]) == 1
            index2util.append(int(util_out[index][0]))
        result: List[Dict[str, int]] = [{
            'fmem': fmem,
            'util': util,
        } for fmem, util in zip(index2fmem, index2util)]
        return result

    @staticmethod
    def _get_index2pids(server: str) -> List[List[str]]:
        index2gpu_uuid_cmd = ['ssh', server, 'nvidia-smi', '--query-gpu=index,gpu_uuid', '--format=csv,noheader']
        gpu_uuid2pids_cmd = ['ssh', server, 'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader']
        index2gpu_uuid: Dict[str, List[str]] = Agent._build_mapping(subprocess.check_output(index2gpu_uuid_cmd))
        gpu_uuid2pids: Dict[str, List[str]] = Agent._build_mapping(subprocess.check_output(gpu_uuid2pids_cmd))
        num_gpus = len(index2gpu_uuid)
        result: List[List[str]] = [None] * num_gpus
        for idx in range(num_gpus):
            assert len(index2gpu_uuid[str(idx)]) == 1
            gpu_uuid = index2gpu_uuid[str(idx)][0]
            result[idx] = gpu_uuid2pids.get(gpu_uuid, [])
        return result

    @staticmethod
    def _get_all_p(server: str) -> Dict[str, Dict[str, str]]:
        cmd = ['ssh', server, "ps", "-eo", "pid,user,lstart,cmd"]
        out = subprocess.check_output(cmd)
        lines = out.decode().strip().splitlines()[1:]
        result: Dict[str, Dict[str, str]] = {}
        for line in lines:
            if "from multiprocessing.spawn import spawn_main; spawn_main" in line:
                continue
            parts = line.split()
            result[parts[0]] = {'pid': parts[0], 'user': parts[1], 'start': ' '.join(parts[2:7]), 'cmd': ' '.join(parts[7:])}
        return result

    @staticmethod
    def _get_server_status(server: str) -> List[Dict[str, Any]]:
        index2pids = Agent._get_index2pids(server)
        all_p = Agent._get_all_p(server)
        index2util = Agent._get_index2util(server)
        result: List[Dict[str, Any]] = [{
            'processes': [all_p[pid] for pid in pids if pid in all_p],
            'util': util,
        } for pids, util in zip(index2pids, index2util)]
        return result

    def _get_status(self, interval: Optional[int] = 2, window_size: Optional[int] = 10) -> None:
        while True:
            for server in self.servers:
                # initialize
                if server not in self.status:
                    self.status[server] = []
                # retrieve
                try:
                    server_status = Agent._get_server_status(server)
                except Exception as e:
                    self.logger.error(f"{server=}, {e}")
                    continue
                # collect
                if not self.status[server]:
                    self.status[server] = [{
                        'processes': None,
                        'util': {
                            'fmem': [],
                            'fmem_avg': None,
                            'util': [],
                            'util_avg': None,
                        },
                    } for _ in range(len(server_status))]
                for idx, gpu_status in enumerate(server_status):
                    self.status[server][idx]['processes'] = gpu_status['processes']
                    self.status[server][idx]['util']['fmem'].append(gpu_status['util']['fmem'])
                    if len(self.status[server][idx]['util']['fmem']) > window_size:
                        self.status[server][idx]['util']['fmem'] = self.status[server][idx]['util']['fmem'][-window_size:]
                    self.status[server][idx]['util']['fmem_avg'] = sum(self.status[server][idx]['util']['fmem']) / len(self.status[server][idx]['util']['fmem'])
                    self.status[server][idx]['util']['util'].append(gpu_status['util']['util'])
                    if len(self.status[server][idx]['util']['util']) > window_size:
                        self.status[server][idx]['util']['util'] = self.status[server][idx]['util']['util'][-window_size:]
                    self.status[server][idx]['util']['util_avg'] = sum(self.status[server][idx]['util']['util']) / len(self.status[server][idx]['util']['util'])
            time.sleep(interval)

    # ====================================================================================================
    # dashboard
    # ====================================================================================================

    def _get_progress(self) -> float:
        result: int = 0
        for config_file in self.config_files:
            work_dir = self._get_work_dir(config_file)
            cur_epochs = self._get_session_progress(work_dir)
            percentage = int(cur_epochs / self.epochs * 100)
            result += percentage
        result: float = round(result / len(self.config_files), 2)
        return result

    def launch_dashboard(self, port: Optional[int] = 8050) -> None:
        """Launches the dashboard to display server status."""

        import datetime  # For displaying the last update timestamp
        from project.user_names import user_names

        # Initialize Dash app
        app = dash.Dash(__name__)

        # Helper function to generate table data from `self.status`
        def generate_table_data(status: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
            table_data = []
            for server, gpus in status.items():
                for idx, gpu_info in enumerate(gpus):
                    if not gpu_info['processes']:
                        table_data.append({
                            "Server": server,
                            "GPU Index": idx,
                            "GPU Utilization": f"{gpu_info['util']['util_avg']:.2f}%",
                            "Free Memory": f"{gpu_info['util']['fmem_avg']:.2f} MiB",
                            "User": None,
                            "PID": None,
                            "Start": None,
                            "CMD": None,
                        })
                    else:
                        for proc in sorted(gpu_info['processes'], key=lambda x: x['user']):
                            table_data.append({
                                "Server": server,
                                "GPU Index": idx,
                                "GPU Utilization": f"{gpu_info['util']['util_avg']:.2f}%",
                                "Free Memory": f"{gpu_info['util']['fmem_avg']:.2f} MiB",
                                "User": user_names.get(proc['user'], proc['user']),
                                "PID": proc['pid'],
                                "Start": proc['start'],
                                "CMD": proc['cmd'],
                            })
            return table_data

        # Generate alternating row styles based on server
        def generate_table_style(table_data):
            styles = []
            color_map = {'color_1': 'white', 'color_2': 'lightblue'}
            last_server = None
            current_color = 'color_2'

            for i, row in enumerate(table_data):
                if row['Server'] != last_server:
                    current_color = 'color_1' if current_color == 'color_2' else 'color_2'
                    last_server = row['Server']

                styles.append({
                    'if': {'row_index': i},
                    'backgroundColor': color_map[current_color]
                })

            return styles

        initial_last_update = f"Last Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        initial_progress = f"Progress: {self._get_progress()}%"
        initial_data = generate_table_data(self.status)
        initial_style = generate_table_style(initial_data)

        # Layout of the Dash app
        app.layout = html.Div([
            html.H1("Server GPU Status Dashboard"),
            html.Div(id='last-update', children=initial_last_update, style={'marginTop': '10px'}),  # Display the last update timestamp
            html.Div(id='progress', children=initial_progress, style={'marginTop': '10px'}),  # New Div for progress
            dcc.Interval(id='interval-component', interval=2*1000, n_intervals=0),  # Update every 2 seconds
            dash_table.DataTable(
                id='status-table',
                columns=[
                    {"name": "Server", "id": "Server"},
                    {"name": "GPU Index", "id": "GPU Index"},
                    {"name": "GPU Utilization", "id": "GPU Utilization"},
                    {"name": "Free Memory", "id": "Free Memory"},
                    {"name": "User", "id": "User"},
                    {"name": "PID", "id": "PID"},
                    {"name": "Start", "id": "Start"},
                    {"name": "CMD", "id": "CMD"},
                ],
                data=initial_data,
                merge_duplicate_headers=True,
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data={'whiteSpace': 'normal', 'height': 'auto'},
                style_data_conditional=initial_style,
            )
        ])

        # Callback to update table data, timestamp, and progress
        @app.callback(
            [
                Output('last-update', 'children'),
                Output('progress', 'children'),
                Output('status-table', 'data'),
                Output('status-table', 'style_data_conditional'),
            ],
            Input('interval-component', 'n_intervals')
        )
        def update_table(n_intervals):
            last_update = f"Last Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            progress = f"Progress: {self._get_progress()}%"  # Retrieve progress percentage
            table_data = generate_table_data(self.status)
            table_style = generate_table_style(table_data)
            return last_update, progress, table_data, table_style

        # Run app
        app.run_server(debug=True, port=port)

    # ====================================================================================================
    # GPU status checking - level 2
    # ====================================================================================================

    @staticmethod
    def _get_user_pids(server: str) -> List[str]:
        cmd = ['ssh', server, 'ps', '-u', server.split('@')[0], '-o', 'pid=']
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
            gpu_pids = Agent._get_index2pids(server)
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
            server_status = self.status[server]
            for idx, gpu_status in enumerate(server_status):
                if (
                    gpu_status['util']['util_avg'] < 50 and
                    gpu_status['util']['fmem_avg'] > 12 * 1024 and
                    len(gpu_status['processes']) < 1
                ):
                    all_idle_gpus.append({
                        'server': server,
                        'gpu_index': idx,
                    })
        return all_idle_gpus

    # ====================================================================================================
    # experiment management
    # ====================================================================================================

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
            error_log = os.path.join(self._get_work_dir(run), "error.log")
            if os.path.isfile(error_log) and os.path.getsize(error_log) > 0:
                self.logger.error(f"Please fix {run}. {error_log=}.")
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
                        '2>', error_log,
                        *([';', 'exec', 'bash'] if self.keep_tmux else []),
                    '"',
                "'",
            ])
            self.logger.info(cmd)
            os.system(cmd)
        return False

    def spawn(self) -> None:
        while True:
            self.logger.info('='*50)
            done = self._launch_missing()
            if done:
                self.logger.info("All done.")
            self.logger.info("")
            time.sleep(self.sleep_time)

    # ====================================================================================================
    # plotting
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

    def plot_training_losses_all(self) -> None:
        for config_file in tqdm.tqdm(self.config_files):
            self._plot_training_losses_single(config_file)

    def plot_validation_scores_all(self) -> None:
        for config_file in tqdm.tqdm(self.config_files):
            self._plot_validation_scores_single(config_file)

    # ====================================================================================================
    # ====================================================================================================

    @staticmethod
    def _get_work_dir(config_file: str) -> str:
        return os.path.splitext(os.path.join("./logs", os.path.relpath(config_file, start="./configs")))[0]
