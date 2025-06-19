from typing import List
import subprocess


def get_ssh_cmd(server: str, command: List[str]) -> List[str]:
    """Helper function to build SSH command with stability options

    Args:
        server: The server to connect to (format: user@host or localhost)
        command: The command to execute on the server

    Returns:
        List of command arguments for subprocess
    """
    if server == 'localhost':
        return command
    else:
        # SSH options for better connection stability:
        ssh_options = [
            'ssh',
            '-o', 'ConnectTimeout=10',  # 10 second connection timeout
            '-o', 'ServerAliveInterval=60',  # Send keepalive every 60 seconds
            '-o', 'ServerAliveCountMax=3',  # Allow 3 missed keepalives before disconnecting
            '-o', 'BatchMode=yes',  # Don't prompt for password (use keys)
            '-o', 'StrictHostKeyChecking=no',  # Skip host key verification
            server
        ]
        return ssh_options + command


def safe_check_output(cmd: List[str], server: str, operation: str) -> str:
    """Safely execute subprocess.check_output with error reporting

    Args:
        cmd: Command to execute
        server: Server being queried
        operation: Description of what operation is being performed

    Returns:
        Command output as string

    Raises:
        subprocess.CalledProcessError: If command fails, with enhanced error message
    """
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.PIPE, text=True)
        return result
    except subprocess.CalledProcessError as e:
        error_msg = f"Server {server} failed during {operation}: {e}"
        if e.stderr:
            error_msg += f"\nStderr: {e.stderr}"

        # Create a new exception with the enhanced error message
        new_exception = subprocess.CalledProcessError(e.returncode, e.cmd, e.stdout, e.stderr)
        new_exception.args = (error_msg,)
        raise new_exception
