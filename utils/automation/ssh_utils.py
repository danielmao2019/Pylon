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
        
        # For remote servers, use connection pool
        with self._get_connection(server) as connection:
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
    def _get_connection(self, server: str):
        """Get an SSH connection from the pool or create a new one."""
        pool = self._get_pool(server)
        connection_lock = self._get_connection_lock(server)
        
        # Try to get an existing connection from the pool
        connection = None
        try:
            connection = pool.get_nowait()
        except queue.Empty:
            pass
        
        # If no connection in pool, create a new one if under limit
        if connection is None:
            with connection_lock:
                if self._active_connections[server] < self.max_connections_per_server:
                    connection = self._create_connection(server)
                    self._active_connections[server] += 1
                else:
                    # Wait for a connection to become available
                    connection = pool.get(timeout=self.connection_timeout)
        
        try:
            yield connection
        finally:
            # Return connection to pool if it's still valid
            if connection and self._is_connection_valid(connection, server):
                try:
                    pool.put_nowait(connection)
                except queue.Full:
                    # Pool is full, close the connection
                    self._close_connection(connection)
                    with connection_lock:
                        self._active_connections[server] -= 1
            else:
                # Connection is invalid, close it
                if connection:
                    self._close_connection(connection)
                with connection_lock:
                    self._active_connections[server] -= 1
    
    def _create_connection(self, server: str) -> subprocess.Popen:
        """Create a new SSH connection."""
        cmd = ['ssh'] + self.ssh_options + [server, 'bash', '-c', 'echo "connection_test"']
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    def _is_connection_valid(self, connection: subprocess.Popen, server: str) -> bool:
        """Check if an SSH connection is still valid."""
        if connection.poll() is not None:
            return False
        
        # Test the connection with a simple command
        try:
            test_cmd = ['ssh'] + self.ssh_options + [server, 'echo', 'test']
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
    
    def _close_connection(self, connection: subprocess.Popen):
        """Close an SSH connection."""
        try:
            connection.terminate()
            connection.wait(timeout=5)
        except (subprocess.TimeoutExpired, Exception):
            try:
                connection.kill()
            except Exception:
                pass
    
    def test_connection(self, server: str, timeout: int = 10) -> bool:
        """
        Test SSH connection to a server.
        
        Args:
            server: Server in format user@host
            timeout: Connection timeout in seconds
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.execute(server, ['echo', 'test'])
            return True
        except Exception:
            return False


class SSHCommandError(Exception):
    """Enhanced exception for SSH command failures with detailed error information."""
    pass


# Global connection pool instance
_ssh_pool = SSHConnectionPool()
