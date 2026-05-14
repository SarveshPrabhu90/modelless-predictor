"""
Data Collector — TCP client that gathers observations from the plant.

Connects to a plant server via TCP, sends an ``observe`` request, and
returns the input-output pairs as NumPy arrays ready for learning.
"""

import json
import socket

import numpy as np

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9100


def collect(
    n: int,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect *n* observations from the plant server over TCP.

    Returns:
        (inputs, outputs) — arrays of shape (n, 3) and (n, 2)
    """
    request = json.dumps({"command": "observe", "n": n}) + "\n"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(10)
        sock.connect((host, port))
        sock.sendall(request.encode())

        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break

    response = json.loads(data.decode().strip())
    inputs = np.array(response["inputs"])
    outputs = np.array(response["outputs"])
    return inputs, outputs
