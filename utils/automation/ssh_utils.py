from typing import List, Optional
import subprocess


def _safe_check_output(cmd: List[str], server: str, operation: str) -> Optional[str]:
    """Safely execute subprocess.check_output with error reporting

    Args:
        cmd: Command to execute
        server: Server being queried
        operation: Description of what operation is being performed

    Returns:
        Command output as string, or None if command fails
    """
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.PIPE, text=True)
        return result
    except subprocess.CalledProcessError as e:
        error_msg = f"Server {server} failed during {operation}: {e}"
        if e.stderr:
            error_msg += f"\nStderr: {e.stderr}"
        print(f"ERROR: {error_msg}")
        return None
