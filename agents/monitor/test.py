"""Minimal Dash app for monitoring resources across servers."""
from __future__ import annotations

import argparse
import datetime
import atexit
from typing import Dict, List

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
    return parser.parse_args()


def create_monitors(servers: List[str], timeout: int) -> Dict[str, SystemMonitor]:
    monitors = {server: SystemMonitor(server=server, timeout=timeout) for server in servers}

    def _shutdown():
        for monitor in monitors.values():
            monitor.stop()

    atexit.register(_shutdown)
    return monitors


def build_table_rows(monitors: Dict[str, SystemMonitor]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for server, monitor in monitors.items():
        monitor.cpu_monitor._update_resource()
        for gpu_monitor in monitor.gpu_monitors.values():
            gpu_monitor._update_resource()

        status = monitor.get_system_status()
        cpu = status['cpu']
        cpu_mem_pct = (
            (cpu['memory_stats']['avg'] / cpu['max_memory']) * 100
            if cpu['max_memory'] and cpu['memory_stats'] and cpu['memory_stats']['avg'] is not None
            else None
        )
        cpu_util_pct = cpu['cpu_stats']['avg'] if cpu['cpu_stats'] else None
        rows.append(
            {
                'Server': server,
                'Resource': 'CPU',
                'Index': '-',
                'Connected': 'Yes' if cpu['connected'] else 'No',
                'Memory %': f"{cpu_mem_pct:.2f}" if cpu_mem_pct is not None else 'n/a',
                'Util %': f"{cpu_util_pct:.2f}" if cpu_util_pct is not None else 'n/a',
                'Processes': str(len(cpu['processes'])) if cpu['processes'] else '0',
            }
        )

        for gpu in status['gpus']:
            gpu_mem_pct = (
                (gpu['memory_stats']['avg'] / gpu['max_memory']) * 100
                if gpu['max_memory'] and gpu['memory_stats'] and gpu['memory_stats']['avg'] is not None
                else None
            )
            gpu_util_pct = gpu['util_stats']['avg'] if gpu['util_stats'] else None
            rows.append(
                {
                    'Server': server,
                    'Resource': 'GPU',
                    'Index': str(gpu['index']),
                    'Connected': 'Yes' if gpu['connected'] else 'No',
                    'Memory %': f"{gpu_mem_pct:.2f}" if gpu_mem_pct is not None else 'n/a',
                    'Util %': f"{gpu_util_pct:.2f}" if gpu_util_pct is not None else 'n/a',
                    'Processes': str(len(gpu['processes'])) if gpu['processes'] else '0',
                }
            )
    return rows


def build_style(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    disconnected_query = '{Connected} = "No"'
    return [
        {
            'if': {'filter_query': disconnected_query},
            'backgroundColor': '#ffe5e5',
        }
    ] if rows else []


def make_app(monitors: Dict[str, SystemMonitor], interval_ms: int) -> dash.Dash:
    app = dash.Dash(__name__)

    columns = [
        {'name': 'Server', 'id': 'Server'},
        {'name': 'Resource', 'id': 'Resource'},
        {'name': 'Index', 'id': 'Index'},
        {'name': 'Connected', 'id': 'Connected'},
        {'name': 'Memory %', 'id': 'Memory %'},
        {'name': 'Util %', 'id': 'Util %'},
        {'name': 'Processes', 'id': 'Processes'},
    ]

    initial_rows = build_table_rows(monitors)

    app.layout = html.Div(
        [
            html.H2("Agents Monitor Dashboard"),
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
    def update_table(_n: int):  # type: ignore
        rows = build_table_rows(monitors)
        style = build_style(rows)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return rows, style, f"Last update: {timestamp}"

    return app


def main() -> None:
    args = parse_args()
    monitors = create_monitors(args.servers, timeout=args.timeout)
    app = make_app(monitors, args.interval)
    app.run_server(debug=False)


if __name__ == '__main__':
    main()
