"""Serve an allowlisted, read-only Rerun website on loopback for Tailscale Funnel."""

from __future__ import annotations

import argparse
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/re_viewer.js": "re_viewer.js",
    "/re_viewer_bg.wasm": "re_viewer_bg.wasm",
    "/favicon.ico": "favicon.ico",
    "/apple-touch-icon.png": "apple-touch-icon.png",
    "/overview.rrd": "overview.rrd",
    "/overview-demo.rbl": "overview-demo.rbl",
}
FILES.update(
    {
        f"/view-{view}-masks-{state}.rbl": f"view-{view}-masks-{state}.rbl"
        for view in ("orientation", "cameras", "evidence")
        for state in ("on", "off")
    }
)


def handler_for(directory: Path):
    root = directory.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond(send_body=True)

        def do_HEAD(self):
            self.respond(send_body=False)

        def do_POST(self):
            self.send_error(405, "Read-only demo")

        do_PUT = do_DELETE = do_PATCH = do_POST

        def respond(self, send_body: bool):
            filename = FILES.get(unquote(urlsplit(self.path).path))
            if filename is None:
                self.send_error(404)
                return
            source = root / filename
            if source.is_symlink() or not source.is_file():
                self.send_error(404)
                return
            original_size = source.stat().st_size
            compressed = False
            if (
                filename in ("re_viewer.js", "re_viewer_bg.wasm")
                and "gzip" in self.headers.get("Accept-Encoding", "")
                and not self.headers.get("Range")
                and source.with_suffix(source.suffix + ".gz").is_file()
            ):
                source = source.with_suffix(source.suffix + ".gz")
                if source.is_symlink():
                    self.send_error(404)
                    return
                compressed = True
            size = source.stat().st_size
            start, end, status = 0, size - 1, 200
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                if not match or not any(match.groups()):
                    self.range_error(size)
                    return
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = min(int(last), size - 1) if last else size - 1
                else:
                    length = int(last)
                    if length == 0:
                        self.range_error(size)
                        return
                    start = max(0, size - length)
                if start >= size or start > end:
                    self.range_error(size)
                    return
                status = 206
            content_type = {
                ".wasm": "application/wasm",
                ".js": "text/javascript",
                ".rrd": "application/octet-stream",
                ".rbl": "application/octet-stream",
            }.get(
                Path(filename).suffix,
                mimetypes.guess_type(filename)[0] or "application/octet-stream",
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-cache")
            if compressed:
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Vary", "Accept-Encoding")
                self.send_header("rerun-final-length", str(original_size))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if send_body:
                try:
                    with source.open("rb") as stream:
                        stream.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = stream.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def range_error(self, size: int):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--port", type=int, default=9093)
    args = parser.parse_args()
    for filename in set(FILES.values()):
        path = args.directory / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing public demo file: {filename}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(args.directory))
    print(f"Read-only demo listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
