"""Check the public server's file boundary and browser download behavior."""

import gzip
import http.client
import importlib.util
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "public_demo", Path(__file__).resolve().parents[1] / "scripts/serve_pipette_demo_public.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def site(tmp_path):
    for name in set(module.FILES.values()):
        (tmp_path / name).write_bytes(b"0123456789")
    (tmp_path / "secret.txt").write_text("private")
    (tmp_path / "re_viewer.js.gz").write_bytes(gzip.compress(b"0123456789"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), module.handler_for(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, method="GET", headers=None):
        connection = http.client.HTTPConnection(*server.server_address)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    yield tmp_path, request
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize(
    "path",
    ["/secret.txt", "/../secret.txt", "/%2e%2e/secret.txt", "/proxy", "/source-manifest.json"],
)
def test_unpublished_paths_are_denied(site, path):
    assert site[1](path)[0] == 404


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_read_only(site, method):
    assert site[1]("/overview.rrd", method)[0] == 405


def test_symlink_not_served(site, tmp_path):
    root, request = site
    (root / "overview.rrd").unlink()
    (root / "overview.rrd").symlink_to(root / "secret.txt")
    assert request("/overview.rrd")[0] == 404


@pytest.mark.parametrize(
    "value,body", [("bytes=2-5", b"2345"), ("bytes=7-", b"789"), ("bytes=-3", b"789")]
)
def test_recording_ranges(site, value, body):
    status, headers, received = site[1]("/overview.rrd", headers={"Range": value})
    assert status == 206
    assert received == body
    assert int(headers["Content-Length"]) == len(body)
    assert headers["Content-Range"].endswith("/10")


@pytest.mark.parametrize("value", ["bytes=20-", "bytes=5-2", "bytes=-0", "bytes=1-2,5-6"])
def test_invalid_ranges(site, value):
    status, headers, body = site[1]("/overview.rrd", headers={"Range": value})
    assert status == 416
    assert headers["Content-Range"] == "bytes */10"
    assert body == b""


def test_compressed_viewer_and_head(site):
    request = site[1]
    status, headers, body = request("/re_viewer.js", headers={"Accept-Encoding": "gzip"})
    assert status == 200
    assert headers["Content-Encoding"] == "gzip"
    assert headers["rerun-final-length"] == "10"
    assert gzip.decompress(body) == b"0123456789"
    status, headers, body = request("/overview.rrd", "HEAD")
    assert status == 200
    assert headers["Content-Length"] == "10"
    assert body == b""
