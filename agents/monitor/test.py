"""Minimal Dash app for monitoring resources across servers."""

import argparse
from contextlib import ExitStack

from agents.monitor.app import create_app
from agents.monitor.dashboard import create_monitors


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


def main() -> None:
    args = parse_args()
    with ExitStack() as stack:
        monitors = create_monitors(
            servers=args.servers,
            timeout=args.timeout,
            window_size=args.window_size,
            stack=stack,
        )
        app = create_app(monitors=monitors, interval_ms=args.interval)
        app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
