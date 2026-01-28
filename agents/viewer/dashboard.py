"""Dashboard helpers for the Agents viewer Dash app."""

import datetime
from typing import Any, Dict, List

from agents.monitor.system_monitor import SystemMonitor


def generate_table_data(
    system_monitors: Dict[str, SystemMonitor], user_names: Dict[str, str]
) -> List[Dict[str, Any]]:
    # Input validations
    assert isinstance(
        system_monitors, dict
    ), f"system_monitors must be dict, got {type(system_monitors)}"
    assert system_monitors, "system_monitors must be non-empty"
    assert all(
        isinstance(monitor, SystemMonitor) for monitor in system_monitors.values()
    ), "system_monitors must contain SystemMonitor values"
    assert isinstance(
        user_names, dict
    ), f"user_names must be dict, got {type(user_names)}"

    table_data: List[Dict[str, Any]] = []

    # Add GPU data
    for monitor in system_monitors.values():
        for gpu in monitor.gpus:
            # Skip disconnected GPUs
            if not gpu.connected:
                continue

            if not gpu.processes:
                table_data.append(
                    {
                        "Server": gpu.server,
                        "Resource": f"GPU-{gpu.index}",
                        "Utilization": (
                            f"{gpu.util_stats['avg']:.2f}%"
                            if gpu.util_stats["avg"] is not None
                            else "N/A"
                        ),
                        "Free Memory": (
                            f"{int(gpu.max_memory - gpu.memory_stats['avg'])} MiB"
                            if gpu.memory_stats["avg"] is not None
                            else f"{int(gpu.max_memory)} MiB"
                        ),
                        "User": None,
                        "PID": None,
                        "Start": None,
                        "CMD": None,
                    }
                )
            else:
                for proc in sorted(gpu.processes, key=lambda x: x.user):
                    if (
                        "python -c from multiprocessing.spawn import spawn_main; spawn_main"
                        in proc.cmd
                    ):
                        continue
                    if proc.user in user_names:
                        user_name = user_names[proc.user]
                    else:
                        user_name = proc.user
                    table_data.append(
                        {
                            "Server": gpu.server,
                            "Resource": f"GPU-{gpu.index}",
                            "Utilization": (
                                f"{gpu.util_stats['avg']:.2f}%"
                                if gpu.util_stats["avg"] is not None
                                else "N/A"
                            ),
                            "Free Memory": (
                                f"{int(gpu.max_memory - gpu.memory_stats['avg'])} MiB"
                                if gpu.memory_stats["avg"] is not None
                                else f"{int(gpu.max_memory)} MiB"
                            ),
                            "User": user_name,
                            "PID": proc.pid,
                            "Start": proc.start_time,
                            "CMD": proc.cmd,
                        }
                    )

    # Add CPU data
    for monitor in system_monitors.values():
        cpu = monitor.cpu
        # Skip disconnected CPUs
        if not cpu.connected:
            continue

        # Filter for relevant processes (python main.py commands)
        relevant_processes = [
            p for p in cpu.processes if "python main.py --config-filepath" in p.cmd
        ]

        if not relevant_processes:
            table_data.append(
                {
                    "Server": cpu.server,
                    "Resource": "CPU",
                    "Utilization": (
                        f"{cpu.cpu_stats['avg']:.2f}%"
                        if cpu.cpu_stats["avg"] is not None
                        else "N/A"
                    ),
                    "Free Memory": (
                        f"{int(cpu.max_memory - cpu.memory_stats['avg'])} MiB"
                        if cpu.memory_stats["avg"] is not None
                        else f"{int(cpu.max_memory)} MiB"
                    ),
                    "User": None,
                    "PID": None,
                    "Start": None,
                    "CMD": None,
                }
            )
        else:
            for proc in sorted(relevant_processes, key=lambda x: x.user):
                if proc.user in user_names:
                    user_name = user_names[proc.user]
                else:
                    user_name = proc.user
                table_data.append(
                    {
                        "Server": cpu.server,
                        "Resource": "CPU",
                        "Utilization": (
                            f"{cpu.cpu_stats['avg']:.2f}%"
                            if cpu.cpu_stats["avg"] is not None
                            else "N/A"
                        ),
                        "Free Memory": (
                            f"{int(cpu.max_memory - cpu.memory_stats['avg'])} MiB"
                            if cpu.memory_stats["avg"] is not None
                            else f"{int(cpu.max_memory)} MiB"
                        ),
                        "User": user_name,
                        "PID": proc.pid,
                        "Start": proc.start_time,
                        "CMD": proc.cmd,
                    }
                )
    return table_data


def generate_table_style(table_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Input validations
    assert isinstance(
        table_data, list
    ), f"table_data must be list, got {type(table_data)}"

    styles: List[Dict[str, Any]] = []
    color_map = {"color_1": "white", "color_2": "lightblue"}
    last_server = None
    current_color = "color_2"

    for i, row in enumerate(table_data):
        if row["Server"] != last_server:
            current_color = "color_1" if current_color == "color_2" else "color_2"
            last_server = row["Server"]

        styles.append(
            {
                "if": {"row_index": i},
                "backgroundColor": color_map[current_color],
            }
        )

    return styles


def format_last_update() -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Last Update: {timestamp}"
