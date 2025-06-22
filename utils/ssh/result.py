from utils.ssh.error import SSHCommandError


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