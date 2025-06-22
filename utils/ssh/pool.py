from typing import List, Dict
import threading
import queue
from utils.ssh.connector import SSHConnector, LocalhostConnector
from utils.ssh.error import SSHCommandError


class SSHConnectionPool:
    """Thread-safe SSH connection pool for reusing persistent connections."""

    def __init__(self, max_connections_per_server: int = 3, connection_timeout: int = 30, command_timeout: int = 30):
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
        connector = self.get_connector(server)

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


# Global connection pool instance
_ssh_pool = SSHConnectionPool()