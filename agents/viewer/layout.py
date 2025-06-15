from typing import List, Dict, Any
from dash import dcc, html, dash_table
from utils.monitor.gpu_monitor import GPUMonitor
from agents.viewer.backend import get_progress


def generate_table_data(gpu_monitor: GPUMonitor, user_names: Dict[str, str]) -> List[Dict[str, Any]]:
    """Generate table data from the GPU monitor status."""
    table_data = []
    for gpu in gpu_monitor.gpus:
        # Skip disconnected GPUs
        if not gpu['connected']:
            continue
            
        if not gpu['processes']:
            table_data.append({
                "Server": gpu['server'],
                "GPU Index": gpu['index'],
                "GPU Utilization": f"{gpu['util_stats']['avg']:.2f}%" if gpu['util_stats']['avg'] is not None else "N/A",
                "Free Memory": f"{int(gpu['max_memory'] - gpu['memory_stats']['avg'])} MiB" if gpu['memory_stats']['avg'] is not None else f"{int(gpu['max_memory'])} MiB",
                "User": None,
                "PID": None,
                "Start": None,
                "CMD": None,
            })
        else:
            for proc in sorted(gpu['processes'], key=lambda x: x['user']):
                table_data.append({
                    "Server": gpu['server'],
                    "GPU Index": gpu['index'],
                    "GPU Utilization": f"{gpu['util_stats']['avg']:.2f}%" if gpu['util_stats']['avg'] is not None else "N/A",
                    "Free Memory": f"{int(gpu['max_memory'] - gpu['memory_stats']['avg'])} MiB" if gpu['memory_stats']['avg'] is not None else f"{int(gpu['max_memory'])} MiB",
                    "User": user_names.get(proc['user'], proc['user']),
                    "PID": proc['pid'],
                    "Start": proc['start_time'],
                    "CMD": proc['cmd'],
                })
    return table_data


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


def create_layout(
    config_files: List[str],
    expected_files: List[str],
    epochs: int,
    sleep_time: int,
    outdated_days: int,
    servers: List[str],
    gpu_monitor: GPUMonitor,
    user_names: Dict[str, str],
) -> html.Div:
    """Create the dashboard layout.
    
    Args:
        config_files: List of config file paths
        expected_files: List of expected file patterns
        epochs: Total number of epochs
        sleep_time: Time to wait for the status to update
        outdated_days: Number of days to consider a run outdated
        servers: List of servers
        gpu_monitor: GPUMonitor object
        user_names: Dict of user names
        
    Returns:
        html.Div: The dashboard layout
    """
    assert isinstance(config_files, list)
    assert isinstance(expected_files, list)
    assert isinstance(epochs, int)
    assert isinstance(sleep_time, int)
    assert isinstance(outdated_days, int)
    assert isinstance(servers, list)
    assert isinstance(gpu_monitor, GPUMonitor)
    assert isinstance(user_names, dict)

    import datetime
    
    initial_last_update = f"Last Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    initial_progress = f"Progress: {get_progress(config_files, expected_files, epochs, sleep_time, outdated_days, servers)}%"
    initial_data = generate_table_data(gpu_monitor, user_names)
    initial_style = generate_table_style(initial_data)

    return html.Div([
        html.H1("Server GPU Status Dashboard"),
        html.Div(id='last-update', children=initial_last_update, style={'marginTop': '10px'}),
        html.Div(id='progress', children=initial_progress, style={'marginTop': '10px'}),
        dcc.Interval(id='interval-component', interval=2*1000, n_intervals=0),
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
