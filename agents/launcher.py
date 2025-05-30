from typing import List, Dict, Any, Optional
import os
import time
import random
import threading
import subprocess
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from agents import BaseAgent
from utils.automation.cfg_log_conversion import get_work_dir
from utils.automation.run_status import get_session_progress, has_stuck, has_failed, parse_config
from utils.monitor.gpu_status import get_server_status, get_all_p, find_running
from utils.logging.text_logger import TextLogger
from concurrent.futures import ThreadPoolExecutor


class Launcher(BaseAgent):

    def __init__(
        self,
        config_files: List[str],
        expected_files: List[str],
        project_dir: str,
        conda_env: str,
        servers: List[str],
        log_path: str,
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
        super(Launcher, self).__init__(config_files=config_files, expected_files=expected_files)
        self.project_dir = project_dir
        self.conda_env = conda_env
        self.servers = servers
        self.epochs = epochs
        self.sleep_time = sleep_time
        self.keep_tmux = keep_tmux
        self.logger = TextLogger(filepath=log_path)
        self._init_status()

    def _init_status(self) -> None:
        self.status = {}
        threading.Thread(target=self._get_status, daemon=True).start()
        self.logger.info("Waiting for status initialization...")
        while set(self.status.keys()) != set(self.servers):
            time.sleep(1)
        self.logger.info("Status initialized.")

    def _get_status(self, interval: Optional[int] = 2, window_size: Optional[int] = 10) -> None:
        while True:
            def process_server(server):
                # initialize
                if server not in self.status:
                    self.status[server] = []
                # retrieve
                try:
                    server_status = get_server_status(server)
                except Exception as e:
                    self.logger.error(f"{server=}, {e}")
                    return
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

            with ThreadPoolExecutor() as executor:
                list(executor.map(process_server, self.servers))
            time.sleep(interval)

    # ====================================================================================================
    # dashboard
    # ====================================================================================================

    def _get_progress(self) -> float:
        result: int = 0
        for config_file in self.config_files:
            work_dir = get_work_dir(config_file)
            cur_epochs = get_session_progress(work_dir=work_dir, expected_files=self.expected_files, epochs=self.epochs)
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
        app.run(debug=True, port=port)

    # ====================================================================================================
    # experiment management
    # ====================================================================================================

    def _find_missing_runs(self, all_running: List[Dict[str, Any]]) -> List[str]:
        r"""
        Returns:
            result (List[str]): the config filepaths for the missing experiment runs.
        """
        def process_config(config_file):
            work_dir = get_work_dir(config_file)
            if not os.path.isdir(work_dir) or has_failed(
                work_dir, all_running=all_running, sleep_time=self.sleep_time, expected_files=self.expected_files, epochs=self.epochs,
            ):
                return config_file
            return None

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_config, self.config_files))
        return [r for r in results if r is not None]

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
        def process_server(server):
            idle_gpus = []
            server_status = self.status[server]
            for idx, gpu_status in enumerate(server_status):
                if (
                    gpu_status['util']['util_avg'] < 50 and
                    gpu_status['util']['fmem_avg'] > 12 * 1024 and
                    len(gpu_status['processes']) < num_jobs
                ):
                    idle_gpus.append({
                        'server': server,
                        'gpu_index': idx,
                    })
            return idle_gpus

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_server, self.servers))
        return [gpu for server_gpus in results for gpu in server_gpus]

    def _remove_stuck(self, all_running: List[Dict[str, Any]]) -> None:
        stuck_cfgs = list(filter(lambda x: has_stuck(get_work_dir(x), all_running), self.config_files))

        def process_server(server):
            server_stuck_info = {}
            server_pids = get_all_p(server)
            for pid in server_pids:
                try:
                    cfg = parse_config(server_pids[pid]['cmd'])
                    if cfg in stuck_cfgs:
                        server_stuck_info[cfg] = (server, pid)
                except:
                    pass
            return server_stuck_info

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_server, self.servers))

        # Combine all server results into a single dictionary
        stuck_cfgs_info = {}
        for server_info in results:
            stuck_cfgs_info.update(server_info)

        self.logger.info(f"The following processes will be killed {stuck_cfgs_info}")
        for server, pid in stuck_cfgs_info.values():
            cmd = ['ssh', server, 'kill', '-9', pid]
            subprocess.check_output(cmd)

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

        def launch_job(args):
            gpu, run = args
            error_log = os.path.join(get_work_dir(run), "error.log")
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
                        'mkdir', '-p', os.path.dirname(error_log), '&&',
                        # launch command
                        "MKL_SERVICE_FORCE_INTEL=1",
                        f"CUDA_VISIBLE_DEVICES={gpu['gpu_index']}",
                        'python', 'main.py', '--config-filepath', run,
                        '2>', error_log,
                        *([';', 'exec', 'bash'] if self.keep_tmux else []),
                    '"',
                "'",
            ])
            self.logger.info(cmd)
            os.system(cmd)

        with ThreadPoolExecutor() as executor:
            list(executor.map(launch_job, zip(gpu_pool, missing_runs)))
        return False

    def spawn(self, num_jobs: Optional[int] = 1) -> None:
        while True:
            self.logger.info('='*50)

            self.logger.info("Collecting all running jobs...")
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(find_running, self.servers))
            all_running = [process for server_running in results for process in server_running]

            self.logger.info("Removing stuck jobs...")
            self._remove_stuck(all_running)

            self.logger.info("Launching missing jobs...")
            done = self._launch_missing(all_running, num_jobs=num_jobs)

            if done:
                self.logger.info("All done.")

            self.logger.info("")

            time.sleep(self.sleep_time)
