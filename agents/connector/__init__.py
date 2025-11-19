"""Agents connector module (moved from utils.ssh)."""

from agents.connector.pool import _ssh_pool, SSHConnectionPool
from agents.connector.connector import SSHConnector, LocalhostConnector
from agents.connector.result import SSHResult, LocalhostResult
from agents.connector.error import SSHCommandError

__all__ = [
    '_ssh_pool',
    'SSHConnectionPool',
    'SSHConnector',
    'LocalhostConnector',
    'SSHResult',
    'LocalhostResult',
    'SSHCommandError',
]
