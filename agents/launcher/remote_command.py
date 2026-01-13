import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from typing import List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command, stream output, and report failures to a collector."
    )
    parser.add_argument(
        "--command",
        required=True,
        type=str,
        help="The original command to execute.",
    )
    parser.add_argument(
        "--collector-url",
        required=True,
        type=str,
        help="HTTP endpoint for posting failure payloads.",
    )
    parser.add_argument(
        "--collector-token",
        required=True,
        type=str,
        help="Bearer token used for authorization.",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        type=str,
        help="Work directory associated with the job.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    command = args.command.strip()
    collector_url = args.collector_url.strip()
    collector_token = args.collector_token.strip()
    work_dir = os.path.normpath(args.work_dir)

    assert command, "command must be provided"
    assert collector_url, "collector_url must be provided"
    assert collector_token, "collector_token must be provided"
    assert work_dir, "work_dir must be provided"

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout_pipe = process.stdout
    assert stdout_pipe is not None

    output_lines: List[str] = []
    for line in stdout_pipe:
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)

    process.wait()
    status_code = process.returncode
    assert status_code is not None

    if status_code != 0:
        payload = json.dumps(
            {
                "host": socket.gethostname(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "command": command,
                "output_lines": output_lines,
                "work_dir": work_dir,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            collector_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {collector_token}",
            },
        )
        urllib.request.urlopen(request, timeout=10).read()

    sys.exit(status_code)


if __name__ == "__main__":
    main()
