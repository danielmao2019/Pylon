from typing import List


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
        # -o ConnectTimeout=10: 10 second connection timeout
        # -o ServerAliveInterval=60: Send keepalive every 60 seconds
        # -o ServerAliveCountMax=3: Allow 3 missed keepalives before disconnecting
        # -o BatchMode=yes: Don't prompt for password (use keys)
        # -o StrictHostKeyChecking=no: Skip host key verification
        ssh_options = [
            'ssh',
            '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=60',
            '-o', 'ServerAliveCountMax=3',
            '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=no',
            server
        ]
        return ssh_options + command
