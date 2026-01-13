"""Agents connector module (moved from utils.ssh)."""

from agents.connector.connector import LocalhostConnector, SSHConnector
from agents.connector.error import SSHCommandError
from agents.connector.pool import SSHConnectionPool, _ssh_pool
from agents.connector.result import LocalhostResult, SSHResult

__all__ = [
    'LocalhostConnector',
    'SSHConnector',
    'SSHCommandError',
    'SSHConnectionPool',
    '_ssh_pool',
    'LocalhostResult',
    'SSHResult',
]
