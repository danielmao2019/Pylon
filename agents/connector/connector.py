from typing import List
import subprocess
import threading
import paramiko
from utils.ssh.error import SSHCommandError
from utils.ssh.result import SSHResult, LocalhostResult


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

    def execute(self, command: List[str]) -> LocalhostResult:
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
