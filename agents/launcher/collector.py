import json
import os
import socket
import threading
from typing import Any, Dict, Tuple
from urllib.parse import urlparse


COLLECTOR_EVENTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "collector_events.json"
)
_FILE_LOCK = threading.Lock()


class Collector:

    def __init__(self, collector_url: str, collector_token: str) -> None:
        host, port, path = self._parse_collector_url(collector_url)
        self.host = host
        self.port = port
        self.path = path
        self.token = collector_token
        self._thread: threading.Thread | None = None
        self._started = False

    @staticmethod
    def _parse_collector_url(url: str) -> Tuple[str, int, str]:
        parsed = urlparse(url)
        assert parsed.scheme == "http", f"collector_url must use http, got {url}"
        host = parsed.hostname or "0.0.0.0"
        port = parsed.port or 80
        path = parsed.path or "/"
        return host, port, path

    @staticmethod
    def _write_event(event: Dict[str, Any]) -> None:
        dirname = os.path.dirname(COLLECTOR_EVENTS_FILE)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with _FILE_LOCK:
            events: list[Dict[str, Any]] = []
            if os.path.isfile(COLLECTOR_EVENTS_FILE):
                with open(COLLECTOR_EVENTS_FILE, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                assert isinstance(
                    loaded, list
                ), f"collector file must contain a list, got {type(loaded)}"
                events = loaded
            events.append(event)
            with open(COLLECTOR_EVENTS_FILE, "w", encoding="utf-8") as handle:
                json.dump(events, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _handle_client(conn: socket.socket, path: str, token: str) -> None:
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            chunk = conn.recv(4096)
            assert chunk, "Connection closed before headers were received"
            buffer += chunk
        header_end = buffer.index(b"\r\n\r\n")
        headers_bytes = buffer[:header_end]
        body_bytes = buffer[header_end + 4 :]
        header_lines = headers_bytes.decode("utf-8").split("\r\n")
        request_line = header_lines[0]
        method, req_path, _ = request_line.split()
        headers_dict: Dict[str, str] = {}
        for line in header_lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers_dict[key.strip().lower()] = value.strip()

        if method == "POST":
            assert req_path == path, f"Unexpected POST path {req_path}"
            auth_header = headers_dict.get("authorization")
            assert auth_header == f"Bearer {token}", "Invalid collector token"
            content_length = int(headers_dict.get("content-length", "0"))
            while len(body_bytes) < content_length:
                more = conn.recv(4096)
                assert more, "Connection closed mid-body"
                body_bytes += more
            event = json.loads(body_bytes.decode("utf-8"))
            required = ["host", "timestamp", "command", "output_lines", "work_dir"]
            for field in required:
                assert field in event and event[field], f"Missing field {field}"
            Collector._write_event(event)
            response = (
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
            )
            conn.sendall(response)
            return

        if method == "GET":
            assert (
                req_path == path or req_path == "/"
            ), f"Unexpected GET path {req_path}"
            body = b""
            if os.path.isfile(COLLECTOR_EVENTS_FILE):
                with open(COLLECTOR_EVENTS_FILE, "rb") as handle:
                    body = handle.read()
            response_headers = (
                f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n"
                "Content-Type: text/plain\r\nConnection: close\r\n\r\n"
            ).encode("utf-8")
            conn.sendall(response_headers + body)
            return

        conn.sendall(
            b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )

    @staticmethod
    def _collector_already_running(host: str, port: int, path: str) -> bool:
        connect_host = "127.0.0.1" if host == "0.0.0.0" else host
        probe_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe_socket.settimeout(1)
        connect_status = probe_socket.connect_ex((connect_host, port))
        if connect_status != 0:
            probe_socket.close()
            return False
        request = (
            f"GET {path} HTTP/1.1\r\nHost: {connect_host}\r\nConnection: close\r\n\r\n"
        ).encode("utf-8")
        probe_socket.sendall(request)
        response = probe_socket.recv(1024)
        probe_socket.close()
        assert response, f"Empty response from service on {connect_host}:{port}"
        status_line = response.split(b"\r\n", 1)[0].decode("utf-8")
        assert status_line.startswith(
            "HTTP/1.1"
        ), f"Unexpected response from {connect_host}:{port}: {status_line}"
        if status_line.startswith("HTTP/1.1 200"):
            return True
        assert (
            False
        ), f"Port {port} already in use by non-collector service: {status_line}"

    @staticmethod
    def _collector_server(host: str, port: int, path: str, token: str) -> None:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(5)
        while True:
            client_conn, _ = server_socket.accept()
            try:
                Collector._handle_client(client_conn, path, token)
            finally:
                client_conn.close()

    def start(self) -> None:
        if self._started:
            return
        if Collector._collector_already_running(self.host, self.port, self.path):
            self._started = True
            return
        thread = threading.Thread(
            target=Collector._collector_server,
            args=(self.host, self.port, self.path, self.token),
            daemon=True,
        )
        thread.start()
        self._thread = thread
        self._started = True
