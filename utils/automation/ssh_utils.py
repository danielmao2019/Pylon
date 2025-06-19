from typing import List, Dict
import subprocess
import threading
from contextlib import contextmanager
import queue


class SSHConnectionPool:
    """Thread-safe SSH connection pool for reusing connections to reduce overhead."""

    def __init__(self, max_connections_per_server: int = 3, connection_timeout: int = 30, command_timeout: int = 30):
        self.max_connections_per_server = max_connections_per_server
        self.connection_timeout = connection_timeout
        self.command_timeout = command_timeout
        self._pools: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._active_connections: Dict[str, int] = {}
        self._connection_locks: Dict[str, threading.Lock] = {}

        # SSH options for better connection stability
        self.ssh_options = [
            '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=60',
            '-o', 'ServerAliveCountMax=3',
            '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null'
        ]

    def _get_pool(self, server: str) -> queue.Queue:
        """Get or create connection pool for a server."""
        with self._lock:
            if server not in self._pools:
                self._pools[server] = queue.Queue(maxsize=self.max_connections_per_server)
                self._active_connections[server] = 0
                self._connection_locks[server] = threading.Lock()
            return self._pools[server]

    def _get_connection_lock(self, server: str) -> threading.Lock:
        """Get the connection lock for a server."""
        with self._lock:
            if server not in self._connection_locks:
                self._connection_locks[server] = threading.Lock()
            return self._connection_locks[server]

    def execute(self, server: str, command: List[str]) -> str:
        """
        Execute a command on a remote server using the connection pool.

        Args:
            server: Server in format user@host
            command: Command to execute on the remote server

        Returns:
            Command output as string

        Raises:
            SSHCommandError: If command fails with detailed error information
        """
        if server == 'localhost':
            # For localhost, execute directly without SSH
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.command_timeout
                )
                if result.returncode != 0:
                    raise SSHCommandError(f"Local command failed: {result.stderr.strip()}")
                return result.stdout.strip()
            except subprocess.TimeoutExpired:
                raise SSHCommandError(f"Local command timed out after {self.command_timeout}s")
            except Exception as e:
                raise SSHCommandError(f"Local command failed: {str(e)}")

        # For remote servers, use connection pool for concurrency control
        with self._get_connection_slot(server):
            try:
                # Build full SSH command
                ssh_cmd = ['ssh'] + self.ssh_options + [server] + command

                result = subprocess.run(
                    ssh_cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.command_timeout
                )

                if result.returncode != 0:
                    error_msg = f"SSH command failed on {server}"
                    if result.stderr:
                        error_msg += f": {result.stderr.strip()}"
                    if result.stdout:
                        error_msg += f" (stdout: {result.stdout.strip()})"
                    raise SSHCommandError(error_msg)

                return result.stdout.strip()

            except subprocess.TimeoutExpired:
                raise SSHCommandError(f"SSH command timed out on {server} after {self.command_timeout}s")
            except Exception as e:
                if isinstance(e, SSHCommandError):
                    raise
                raise SSHCommandError(f"SSH command failed on {server}: {str(e)}")

    @contextmanager
    def _get_connection_slot(self, server: str):
        """Get a connection slot from the pool for concurrency control."""
        pool = self._get_pool(server)
        connection_lock = self._get_connection_lock(server)

        # Try to get a connection slot from the pool
        connection_slot = None
        try:
            connection_slot = pool.get_nowait()
        except queue.Empty:
            pass

        # If no slot in pool, create a new one if under limit
        if connection_slot is None:
            with connection_lock:
                if self._active_connections[server] < self.max_connections_per_server:
                    connection_slot = f"slot_{self._active_connections[server]}"
                    self._active_connections[server] += 1
                else:
                    # Wait for a connection slot to become available
                    try:
                        connection_slot = pool.get(timeout=self.connection_timeout)
                    except queue.Empty:
                        raise SSHCommandError(f"Connection pool timeout on {server}: no slots available within {self.connection_timeout}s")

        try:
            yield connection_slot
        finally:
            # Return connection slot to pool
            try:
                pool.put_nowait(connection_slot)
            except queue.Full:
                # Pool is full, just discard the slot
                with connection_lock:
                    self._active_connections[server] -= 1


class SSHCommandError(Exception):
    """Enhanced exception for SSH command failures with detailed error information."""
    pass


# Global connection pool instance
_ssh_pool = SSHConnectionPool()
