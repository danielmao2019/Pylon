from typing import List, Dict, Generator
import subprocess
import threading
import queue
import paramiko


class SSHResult:
    """Result object for SSH command execution, similar to database cursor."""
    
    def __init__(self, stdout_channel, stderr_channel, stdin_channel=None):
        self._stdout_channel = stdout_channel
        self._stderr_channel = stderr_channel
        self._stdin_channel = stdin_channel
        self._output = None
        self._error = None
        self._return_code = None
        
    def fetch(self) -> str:
        """Fetch command output."""
        if self._output is None:
            # Read all stdout and get return code
            self._output = self._stdout_channel.read().decode('utf-8').strip()
            self._return_code = self._stdout_channel.channel.recv_exit_status()
            
            # Check for command failure
            if self._return_code != 0:
                error = self.fetcherror()
                raise SSHCommandError(f"Command failed with exit code {self._return_code}: {error}")
            
        return self._output
        
    def fetcherror(self) -> str:
        """Fetch error output."""
        if self._error is None:
            self._error = self._stderr_channel.read().decode('utf-8').strip()
        return self._error
        
    def get_return_code(self) -> int:
        """Get command return code."""
        if self._return_code is None:
            # Ensure output is fetched first to get return code
            self.fetch()
        return self._return_code


class SSHConnector:
    """Persistent SSH connection similar to database connector."""
    
    def __init__(self, server: str, timeout: int = 30):
        self.server = server
        self.timeout = timeout
        self._client = None
        self._connected = False
        self._lock = threading.Lock()
        
        # Parse server string (user@host:port or user@host)
        if '@' in server:
            self.username, host_part = server.split('@', 1)
        else:
            self.username = None
            host_part = server
            
        if ':' in host_part:
            self.hostname, port_str = host_part.split(':', 1)
            self.port = int(port_str)
        else:
            self.hostname = host_part
            self.port = 22
    
    def connect(self):
        """Establish persistent SSH connection."""
        with self._lock:
            if self._connected:
                return
                
            self._client = paramiko.SSHClient()
            self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            try:
                # Connect with key-based auth first, then password if needed
                connect_kwargs = {
                    'hostname': self.hostname,
                    'port': self.port,
                    'timeout': self.timeout,
                    'banner_timeout': self.timeout,
                    'auth_timeout': self.timeout,
                }
                
                if self.username:
                    connect_kwargs['username'] = self.username
                    
                self._client.connect(**connect_kwargs)
                self._connected = True
                
            except Exception as e:
                if self._client:
                    self._client.close()
                    self._client = None
                raise SSHCommandError(f"Failed to connect to {self.server}: {str(e)}")
    
    def execute(self, command: List[str]) -> SSHResult:
        """Execute command and return result object."""
        if not self._connected:
            self.connect()
            
        cmd_str = ' '.join(command)
        
        try:
            with self._lock:
                stdin, stdout, stderr = self._client.exec_command(
                    cmd_str, 
                    timeout=self.timeout,
                    get_pty=False
                )
                
            return SSHResult(stdout, stderr, stdin)
            
        except Exception as e:
            # Connection might be broken, mark as disconnected
            self._connected = False
            raise SSHCommandError(f"SSH command failed on {self.server}: {str(e)}")
    
    def is_connected(self) -> bool:
        """Check if connection is still alive."""
        if not self._connected or not self._client:
            return False
            
        try:
            # Test connection with a lightweight command
            transport = self._client.get_transport()
            return transport and transport.is_active()
        except:
            self._connected = False
            return False
    
    def close(self):
        """Close the connection."""
        with self._lock:
            if self._client:
                try:
                    self._client.close()
                except:
                    pass
                self._client = None
            self._connected = False
    
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class LocalhostConnector:
    """Special connector for localhost that uses subprocess instead of SSH."""
    
    def __init__(self, timeout: int = 30):
        self.server = 'localhost'
        self.timeout = timeout
        self._connected = True
    
    def connect(self):
        """No-op for localhost."""
        pass
    
    def execute(self, command: List[str]) -> 'LocalhostResult':
        """Execute command locally using subprocess."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return LocalhostResult(result)
        except subprocess.TimeoutExpired:
            raise SSHCommandError(f"Local command timed out after {self.timeout}s")
        except Exception as e:
            raise SSHCommandError(f"Local command failed: {str(e)}")
    
    def is_connected(self) -> bool:
        """Always connected for localhost."""
        return True
    
    def close(self):
        """No-op for localhost."""
        pass
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class LocalhostResult:
    """Result object for localhost command execution."""
    
    def __init__(self, subprocess_result):
        self._result = subprocess_result
        
    def fetch(self) -> str:
        """Fetch command output."""
        if self._result.returncode != 0:
            error_msg = f"Local command failed: {self._result.stderr.strip()}"
            raise SSHCommandError(error_msg)
        return self._result.stdout.strip()
        
    def fetcherror(self) -> str:
        """Fetch error output."""
        return self._result.stderr.strip()
        
    def get_return_code(self) -> int:
        """Get command return code."""
        return self._result.returncode


class SSHConnectionPool:
    """Thread-safe SSH connection pool for reusing persistent connections."""

    def __init__(self, max_connections_per_server: int = 3, connection_timeout: int = 30, command_timeout: int = 30) -> None:
        self.max_connections_per_server = max_connections_per_server
        self.connection_timeout = connection_timeout
        self.command_timeout = command_timeout
        self._pools: Dict[str, queue.Queue] = {}  # Pools of SSHConnector objects
        self._lock = threading.Lock()
        self._active_connections: Dict[str, int] = {}
        self._connection_locks: Dict[str, threading.Lock] = {}

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

    def get_connector(self, server: str):
        """Get or create a persistent connector for the server."""
        if server == 'localhost':
            return LocalhostConnector(timeout=self.command_timeout)

        pool = self._get_pool(server)
        connection_lock = self._get_connection_lock(server)

        # Try to get an existing connector from the pool
        try:
            connector = pool.get_nowait()
            if connector.is_connected():
                return connector
            else:
                # Connection is dead, close and create new
                connector.close()
                with connection_lock:
                    self._active_connections[server] -= 1
        except queue.Empty:
            pass

        # Create new connector if under limit
        with connection_lock:
            if self._active_connections[server] < self.max_connections_per_server:
                connector = SSHConnector(server, timeout=self.connection_timeout)
                connector.connect()
                self._active_connections[server] += 1
                return connector

        # Wait for a connector to become available
        try:
            connector = pool.get(timeout=self.connection_timeout)
            if connector.is_connected():
                return connector
            else:
                # Dead connection, create new
                connector.close()
                connector = SSHConnector(server, timeout=self.connection_timeout)
                connector.connect()
                return connector
        except queue.Empty:
            raise SSHCommandError(f"Connection pool timeout on {server}: no connectors available within {self.connection_timeout}s")

    def return_connector(self, connector):
        """Return a connector to the pool for reuse."""
        if isinstance(connector, LocalhostConnector):
            return  # Localhost connectors don't need pooling

        if connector.is_connected():
            pool = self._get_pool(connector.server)
            try:
                pool.put_nowait(connector)
            except queue.Full:
                # Pool is full, close the connection and decrement count
                connector.close()
                with self._get_connection_lock(connector.server):
                    self._active_connections[connector.server] -= 1
        else:
            # Connection is dead, just decrement count
            connector.close()
            with self._get_connection_lock(connector.server):
                self._active_connections[connector.server] -= 1

    def execute(self, server: str, command: List[str]) -> str:
        """
        Execute a command on a remote server using persistent connections.
        
        Automatically manages connector lifecycle and returns output directly.

        Args:
            server: Server in format user@host
            command: Command to execute on the remote server

        Returns:
            Command output as string

        Raises:
            SSHCommandError: If command fails with detailed error information
            
        Example:
            output = _ssh_pool.execute('user@host', ['nvidia-smi'])
        """
<<<<<<< HEAD
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
    def _get_connection_slot(self, server: str) -> Generator[str, None, None]:
        """Get a connection slot from the pool for concurrency control."""
        pool = self._get_pool(server)
        connection_lock = self._get_connection_lock(server)

        # Try to get a connection slot from the pool
        connection_slot = None
=======
        connector = self.get_connector(server)
        
>>>>>>> f93c67eb (f)
        try:
            ssh_result = connector.execute(command)
            output = ssh_result.fetch()
            return output
        finally:
            # Always return connector to pool (except for localhost which handles itself)
            if server != 'localhost':
                self.return_connector(connector)

    def close_all_connections(self):
        """Close all connections in all pools. Useful for cleanup."""
        with self._lock:
            for server, pool in self._pools.items():
                # Close all connectors in the pool
                connectors_to_close = []
                while not pool.empty():
                    try:
                        connector = pool.get_nowait()
                        connectors_to_close.append(connector)
                    except queue.Empty:
                        break
                
                for connector in connectors_to_close:
                    connector.close()
                
                # Reset active connection count
                self._active_connections[server] = 0


class SSHCommandError(Exception):
    """Enhanced exception for SSH command failures with detailed error information."""
    pass


# Global connection pool instance
_ssh_pool = SSHConnectionPool()
