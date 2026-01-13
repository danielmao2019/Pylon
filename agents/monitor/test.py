"""Minimal Dash app for monitoring resources across servers."""

import argparse
import datetime
from contextlib import ExitStack
from typing import Any, Dict, List, Optional

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

from agents.monitor.system_monitor import SystemMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dash dashboard for SystemMonitor")
    parser.add_argument(
        "--servers",
        nargs="+",
        required=True,
        help="List of servers to monitor (use user@host if needed).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=2000,
        help="Refresh interval in milliseconds (default: 2000).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="SSH command timeout in seconds (default: 5).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to serve the dashboard on (default: 8050).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Rolling window size for CPU/GPU statistics (default: 10 samples).",
    )
    return parser.parse_args()


def create_monitors(
    servers: List[str],
    timeout: int,
    window_size: int,
    stack: ExitStack,
) -> Dict[str, SystemMonitor]:
    monitors: Dict[str, SystemMonitor] = {}
    for server in servers:
        monitor = SystemMonitor(server=server, timeout=timeout, window_size=window_size)
        stack.callback(monitor.stop)
        monitors[server] = monitor
    return monitors


def build_table_rows(monitors: Dict[str, SystemMonitor]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for server, monitor in monitors.items():
        monitor.cpu_monitor._update_resource()
        for gpu_monitor in monitor.gpu_monitors.values():
            gpu_monitor._update_resource()

        status = monitor.get_system_status()
        cpu = status['cpu']
        cpu_memory_stats = (
            cpu['memory_stats']
            if cpu['memory_stats']
            else {'min': None, 'max': None, 'avg': None}
        )
        cpu_util_stats = (
            cpu['cpu_stats']
            if cpu['cpu_stats']
            else {'min': None, 'max': None, 'avg': None}
        )
        cpu_max_memory = cpu['max_memory']
        rows.append(
            {
                'Server': server,
                'Resource': 'CPU',
                'Connected': 'Yes' if cpu['connected'] else 'No',
                'Memory Min': (
                    f"{cpu_memory_stats['min']:.2f}"
                    if cpu_memory_stats['min'] is not None
                    else 'n/a'
                ),
                'Memory Max': (
                    f"{cpu_memory_stats['max']:.2f}"
                    if cpu_memory_stats['max'] is not None
                    else 'n/a'
                ),
                'Memory Avg': (
                    f"{cpu_memory_stats['avg']:.2f}"
                    if cpu_memory_stats['avg'] is not None
                    else 'n/a'
                ),
                'Max Memory': (
                    f"{cpu_max_memory:.2f}" if cpu_max_memory is not None else 'n/a'
                ),
                'Util Min': (
                    f"{cpu_util_stats['min']:.2f}"
                    if cpu_util_stats['min'] is not None
                    else 'n/a'
                ),
                'Util Max': (
                    f"{cpu_util_stats['max']:.2f}"
                    if cpu_util_stats['max'] is not None
                    else 'n/a'
                ),
                'Util Avg': (
                    f"{cpu_util_stats['avg']:.2f}"
                    if cpu_util_stats['avg'] is not None
                    else 'n/a'
                ),
            }
        )

        for gpu in status['gpus']:
            gpu_memory_stats = (
                gpu['memory_stats']
                if gpu['memory_stats']
                else {'min': None, 'max': None, 'avg': None}
            )
            gpu_util_stats = (
                gpu['util_stats']
                if gpu['util_stats']
                else {'min': None, 'max': None, 'avg': None}
            )
            gpu_max_memory = gpu['max_memory']
            rows.append(
                {
                    'Server': server,
                    'Resource': f"GPU-{gpu['index']}",
                    'Connected': 'Yes' if gpu['connected'] else 'No',
                    'Memory Min': (
                        f"{gpu_memory_stats['min']:.2f}"
                        if gpu_memory_stats['min'] is not None
                        else 'n/a'
                    ),
                    'Memory Max': (
                        f"{gpu_memory_stats['max']:.2f}"
                        if gpu_memory_stats['max'] is not None
                        else 'n/a'
                    ),
                    'Memory Avg': (
                        f"{gpu_memory_stats['avg']:.2f}"
                        if gpu_memory_stats['avg'] is not None
                        else 'n/a'
                    ),
                    'Max Memory': (
                        f"{gpu_max_memory:.2f}" if gpu_max_memory is not None else 'n/a'
                    ),
                    'Util Min': (
                        f"{gpu_util_stats['min']:.2f}"
                        if gpu_util_stats['min'] is not None
                        else 'n/a'
                    ),
                    'Util Max': (
                        f"{gpu_util_stats['max']:.2f}"
                        if gpu_util_stats['max'] is not None
                        else 'n/a'
                    ),
                    'Util Avg': (
                        f"{gpu_util_stats['avg']:.2f}"
                        if gpu_util_stats['avg'] is not None
                        else 'n/a'
                    ),
                }
            )
    return rows


def build_style(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    disconnected_query = '{Connected} = "No"'
    style_rules: List[Dict[str, Any]] = []
    if rows:
        style_rules.append(
            {
                'if': {'filter_query': disconnected_query},
                'backgroundColor': '#ffe5e5',
            }
        )
    return style_rules


def make_app(monitors: Dict[str, SystemMonitor], interval_ms: int) -> dash.Dash:
    app = dash.Dash(__name__)

    first_monitor = next(iter(monitors.values()), None)
    window_size: Optional[int] = first_monitor.window_size if first_monitor else None
    update_period_seconds = interval_ms / 1000.0
    meta_parts: List[str] = []
    if window_size is not None:
        meta_parts.append(f"Window size: {window_size} samples")
    meta_parts.append(f"Update period: {update_period_seconds:.2f}s")
    meta_text = ' · '.join(meta_parts)

    columns = [
        {'name': 'Server', 'id': 'Server'},
        {'name': 'Resource', 'id': 'Resource'},
        {'name': 'Connected', 'id': 'Connected'},
        {'name': 'Memory Min', 'id': 'Memory Min'},
        {'name': 'Memory Max', 'id': 'Memory Max'},
        {'name': 'Memory Avg', 'id': 'Memory Avg'},
        {'name': 'Max Memory', 'id': 'Max Memory'},
        {'name': 'Util Min', 'id': 'Util Min'},
        {'name': 'Util Max', 'id': 'Util Max'},
        {'name': 'Util Avg', 'id': 'Util Avg'},
    ]

    initial_rows = build_table_rows(monitors)

    app.layout = html.Div(
        [
            html.H2("Agents Monitor Dashboard"),
            html.Div(
                meta_text, id='monitor-meta', style={'margin': '8px 0', 'color': '#555'}
            ),
            html.Div(id='last-update', style={'margin': '10px 0'}),
            dash_table.DataTable(
                id='resource-table',
                columns=columns,
                data=initial_rows,
                style_cell={'textAlign': 'left', 'padding': '6px'},
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
                style_data={'whiteSpace': 'normal', 'height': 'auto'},
            ),
            dcc.Interval(id='refresh-interval', interval=interval_ms, n_intervals=0),
        ]
    )

    @app.callback(
        Output('resource-table', 'data'),
        Output('resource-table', 'style_data_conditional'),
        Output('last-update', 'children'),
        Input('refresh-interval', 'n_intervals'),
    )
    def update_table(_n: int):
        rows = build_table_rows(monitors)
        style = build_style(rows)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return rows, style, f"Last update: {timestamp}"

    return app


def main() -> None:
    args = parse_args()
    with ExitStack() as stack:
        monitors = create_monitors(
            args.servers,
            timeout=args.timeout,
            window_size=args.window_size,
            stack=stack,
        )
        app = make_app(monitors, args.interval)
        app.run(debug=False, port=args.port)


if __name__ == '__main__':
    main()
