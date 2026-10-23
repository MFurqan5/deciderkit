import pytest

from deciderkit import DeciderClient
from deciderkit.testing.mock_server import MockDeciderServer


@pytest.fixture()
def server():
    with MockDeciderServer() as srv:
        yield srv


@pytest.fixture()
def client(server):
    return DeciderClient(server.url, timeout=3.0)


def convo(*pairs):
    """convo(("user", "text"), ("tool_result", "text"))"""
    return [{"role": r, "text": t} for r, t in pairs]
