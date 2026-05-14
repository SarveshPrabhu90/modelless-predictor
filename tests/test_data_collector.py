"""Unit tests for the data collector (mocked TCP)."""

import json
import socket
import threading

import numpy as np
import pytest

from src.data_collector import collect


def _fake_server(port: int, n: int, ready_event: threading.Event):
    """Minimal TCP server that responds with fake observations."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.settimeout(5)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    ready_event.set()
    conn, _ = srv.accept()
    with conn:
        data = b""
        while b"\n" not in data:
            data += conn.recv(4096)

        response = {
            "inputs": np.random.rand(n, 3).tolist(),
            "outputs": np.random.rand(n, 2).tolist(),
            "labels": {
                "inputs": ["temperature", "flow_rate", "concentration"],
                "outputs": ["yield", "purity"],
            },
        }
        conn.sendall((json.dumps(response) + "\n").encode())
    srv.close()


@pytest.fixture()
def fake_server_port():
    """Spin up a fake server on a random port; return the port number."""
    srv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv_socket.bind(("127.0.0.1", 0))
    port = srv_socket.getsockname()[1]
    srv_socket.close()

    ready = threading.Event()
    t = threading.Thread(target=_fake_server, args=(port, 10, ready), daemon=True)
    t.start()
    ready.wait(timeout=3)
    yield port
    t.join(timeout=2)


class TestCollect:
    def test_returns_numpy_arrays(self, fake_server_port):
        inputs, outputs = collect(10, port=fake_server_port)
        assert isinstance(inputs, np.ndarray)
        assert isinstance(outputs, np.ndarray)

    def test_correct_shapes(self, fake_server_port):
        inputs, outputs = collect(10, port=fake_server_port)
        assert inputs.shape == (10, 3)
        assert outputs.shape == (10, 2)
