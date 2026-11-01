import hashlib
import http.server
import threading
from functools import partial

import pytest

from ecgtrust.data.download import ChecksumError, download_file, parse_sha256sums

FILES = {"a.bin": bytes(range(256)) * 400, "truncate_once.bin": b"0123456789abcdef" * 8000}


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def do_GET(self):  # noqa: N802
        srv = self.server
        name = self.path.lstrip("/")
        body = FILES.get(name)
        if body is None:
            self.send_error(404)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng:
            srv.saw_range_request = True
            start = int(rng.split("=")[1].split("-")[0])
        payload = body[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(payload)))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        self.end_headers()
        if name == "truncate_once.bin" and not srv.truncated_once:
            srv.truncated_once = True
            self.wfile.write(payload[: len(payload) // 3])
            self.wfile.flush()
            self.connection.shutdown(2)  # simulate proxy cut-off mid-transfer
            return
        self.wfile.write(payload)


class _Server:
    def __init__(self):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.saw_range_request = False
        self.httpd.truncated_once = False
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def url(self, name):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/{name}"

    def content(self, name):
        return FILES[name]

    def sha(self, name):
        return hashlib.sha256(FILES[name]).hexdigest()

    @property
    def saw_range_request(self):
        return self.httpd.saw_range_request


@pytest.fixture()
def server(monkeypatch):
    # Local test server must bypass the environment's HTTPS/HTTP proxy.
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    s = _Server()
    yield s
    s.httpd.shutdown()


fast = partial(download_file, backoff=0.01)


def test_parse_sha256sums():
    assert parse_sha256sums("ab12 records100/x.dat\n") == {"records100/x.dat": "ab12"}


def test_download_verifies_checksum(server, tmp_path):
    p = fast(server.url("a.bin"), tmp_path / "a.bin", sha256=server.sha("a.bin"))
    assert p.read_bytes() == server.content("a.bin")


def test_checksum_mismatch_raises(server, tmp_path):
    with pytest.raises(ChecksumError):
        fast(server.url("a.bin"), tmp_path / "a.bin", sha256="0" * 64)
    assert not (tmp_path / "a.bin").exists()


def test_resumes_after_truncated_transfer(server, tmp_path):  # Review Focus 1
    p = fast(
        server.url("truncate_once.bin"), tmp_path / "t.bin", sha256=server.sha("truncate_once.bin")
    )
    assert p.read_bytes() == server.content("truncate_once.bin") and server.saw_range_request


def test_corrupt_cached_file_is_redownloaded(server, tmp_path):  # Review Focus 1
    (tmp_path / "a.bin").write_bytes(b"garbage")
    p = fast(server.url("a.bin"), tmp_path / "a.bin", sha256=server.sha("a.bin"))
    assert p.read_bytes() == server.content("a.bin")
