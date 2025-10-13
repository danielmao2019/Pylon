"""Pytest configuration for monitor integration checks."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--server",
        action="store",
        default="localhost",
        help="Server hostname (or user@host) to run monitor tests against. Defaults to localhost.",
    )
    parser.addoption(
        "--gpu-index",
        action="store",
        default=None,
        type=int,
        help="Optional GPU index to probe. If omitted, GPU-specific tests will be skipped.",
    )


@pytest.fixture
def monitor_server(request) -> str:
    return request.config.getoption("--server")


@pytest.fixture
def gpu_index(request) -> int | None:
    return request.config.getoption("--gpu-index")
