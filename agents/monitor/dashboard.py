"""Helper utilities for constructing the Agents monitor Dash dashboard."""

import datetime
from contextlib import ExitStack
from typing import Any, Dict, List, Optional

from agents.monitor.system_monitor import SystemMonitor


def create_monitors(
    servers: List[str],
    timeout: int,
    window_size: int,
    stack: ExitStack,
) -> Dict[str, SystemMonitor]:
    # Input validations
    assert isinstance(servers, list), f"servers must be list, got {type(servers)}"
    assert isinstance(timeout, int), f"timeout must be int, got {type(timeout)}"
    assert isinstance(
        window_size, int
    ), f"window_size must be int, got {type(window_size)}"
    assert isinstance(stack, ExitStack), f"stack must be ExitStack, got {type(stack)}"

    monitors: Dict[str, SystemMonitor] = {}
    for server in servers:
        monitor = SystemMonitor(server=server, timeout=timeout, window_size=window_size)
        stack.callback(monitor.stop)
        monitors[server] = monitor
    return monitors


def build_columns() -> List[Dict[str, str]]:
    return [
        {"name": "Server", "id": "Server"},
        {"name": "Resource", "id": "Resource"},
        {"name": "Connected", "id": "Connected"},
        {"name": "Memory Min", "id": "Memory Min"},
        {"name": "Memory Max", "id": "Memory Max"},
        {"name": "Memory Avg", "id": "Memory Avg"},
        {"name": "Max Memory", "id": "Max Memory"},
        {"name": "Util Min", "id": "Util Min"},
        {"name": "Util Max", "id": "Util Max"},
        {"name": "Util Avg", "id": "Util Avg"},
    ]


def build_table_rows(monitors: Dict[str, SystemMonitor]) -> List[Dict[str, Any]]:
    # Input validations
    assert isinstance(monitors, dict), f"monitors must be dict, got {type(monitors)}"

    rows: List[Dict[str, Any]] = []
    for server, monitor in monitors.items():
        monitor.cpu_monitor._update_resource()
        for gpu_monitor in monitor.gpu_monitors.values():
            gpu_monitor._update_resource()

        status = monitor.get_system_status()
        cpu = status["cpu"]
        cpu_memory_stats = (
            cpu["memory_stats"]
            if cpu["memory_stats"]
            else {"min": None, "max": None, "avg": None}
        )
        cpu_util_stats = (
            cpu["cpu_stats"]
            if cpu["cpu_stats"]
            else {"min": None, "max": None, "avg": None}
        )
        cpu_max_memory = cpu["max_memory"]
        rows.append(
            {
                "Server": server,
                "Resource": "CPU",
                "Connected": "Yes" if cpu["connected"] else "No",
                "Memory Min": (
                    f"{cpu_memory_stats['min']:.2f}"
                    if cpu_memory_stats["min"] is not None
                    else "n/a"
                ),
                "Memory Max": (
                    f"{cpu_memory_stats['max']:.2f}"
                    if cpu_memory_stats["max"] is not None
                    else "n/a"
                ),
                "Memory Avg": (
                    f"{cpu_memory_stats['avg']:.2f}"
                    if cpu_memory_stats["avg"] is not None
                    else "n/a"
                ),
                "Max Memory": (
                    f"{cpu_max_memory:.2f}" if cpu_max_memory is not None else "n/a"
                ),
                "Util Min": (
                    f"{cpu_util_stats['min']:.2f}"
                    if cpu_util_stats["min"] is not None
                    else "n/a"
                ),
                "Util Max": (
                    f"{cpu_util_stats['max']:.2f}"
                    if cpu_util_stats["max"] is not None
                    else "n/a"
                ),
                "Util Avg": (
                    f"{cpu_util_stats['avg']:.2f}"
                    if cpu_util_stats["avg"] is not None
                    else "n/a"
                ),
            }
        )

        for gpu in status["gpus"]:
            gpu_memory_stats = (
                gpu["memory_stats"]
                if gpu["memory_stats"]
                else {"min": None, "max": None, "avg": None}
            )
            gpu_util_stats = (
                gpu["util_stats"]
                if gpu["util_stats"]
                else {"min": None, "max": None, "avg": None}
            )
            gpu_max_memory = gpu["max_memory"]
            rows.append(
                {
                    "Server": server,
                    "Resource": f"GPU-{gpu['index']}",
                    "Connected": "Yes" if gpu["connected"] else "No",
                    "Memory Min": (
                        f"{gpu_memory_stats['min']:.2f}"
                        if gpu_memory_stats["min"] is not None
                        else "n/a"
                    ),
                    "Memory Max": (
                        f"{gpu_memory_stats['max']:.2f}"
                        if gpu_memory_stats["max"] is not None
                        else "n/a"
                    ),
                    "Memory Avg": (
                        f"{gpu_memory_stats['avg']:.2f}"
                        if gpu_memory_stats["avg"] is not None
                        else "n/a"
                    ),
                    "Max Memory": (
                        f"{gpu_max_memory:.2f}" if gpu_max_memory is not None else "n/a"
                    ),
                    "Util Min": (
                        f"{gpu_util_stats['min']:.2f}"
                        if gpu_util_stats["min"] is not None
                        else "n/a"
                    ),
                    "Util Max": (
                        f"{gpu_util_stats['max']:.2f}"
                        if gpu_util_stats["max"] is not None
                        else "n/a"
                    ),
                    "Util Avg": (
                        f"{gpu_util_stats['avg']:.2f}"
                        if gpu_util_stats["avg"] is not None
                        else "n/a"
                    ),
                }
            )
    return rows


def build_style(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Input validations
    assert isinstance(rows, list), f"rows must be list, got {type(rows)}"

    disconnected_query = '{Connected} = "No"'
    style_rules: List[Dict[str, Any]] = []
    if rows:
        style_rules.append(
            {"if": {"filter_query": disconnected_query}, "backgroundColor": "#ffe5e5"}
        )
    return style_rules


def build_meta_text(monitors: Dict[str, SystemMonitor], interval_ms: int) -> str:
    # Input validations
    assert isinstance(monitors, dict), f"monitors must be dict, got {type(monitors)}"
    assert isinstance(
        interval_ms, int
    ), f"interval_ms must be int, got {type(interval_ms)}"

    first_monitor: Optional[SystemMonitor] = next(iter(monitors.values()), None)
    update_period_seconds = interval_ms / 1000.0
    parts: List[str] = []
    if first_monitor is not None:
        parts.append(f"Window size: {first_monitor.window_size} samples")
    parts.append(f"Update period: {update_period_seconds:.2f}s")
    return " · ".join(parts)


def format_last_update() -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Last update: {timestamp}"
