"""Inspect a running native Rerun viewer through its MCP interface."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9980)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [str(Path(".venv/bin/rerun").resolve()), "viewer-mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    index = 0
    results = []

    def rpc(method, params):
        nonlocal index
        index += 1
        process.stdin.write(
            json.dumps({"jsonrpc": "2.0", "id": index, "method": method, "params": params}) + "\n"
        )
        process.stdin.flush()
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("Viewer MCP stopped")
            result = json.loads(line)
            if result.get("id") == index:
                return result

    def call(name, args):
        response = rpc("tools/call", {"name": name, "arguments": args})
        if response.get("error") or response.get("result", {}).get("isError"):
            raise RuntimeError(response)
        result = response["result"]
        result["content"] = [c for c in result.get("content", []) if c["type"] != "image"]
        results.append({"tool": name, "arguments": args, "result": result})
        print(name, json.dumps(result)[:300], flush=True)

    try:
        rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pipette-demo-review", "version": "1"},
            },
        )
        process.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        process.stdin.flush()
        call("connect", {"endpoint": f"http://127.0.0.1:{args.port}"})
        manifest = json.loads((args.bundle / "manifest.json").read_text())
        call("open_url", {"url": str((args.bundle / "overview.rrd").resolve())})
        call("open_url", {"url": str((args.bundle / "overview-demo.rbl").resolve())})
        time.sleep(3)
        call("viewer_state", {})
        frames = sorted(set([0, 100, 200, 599, 600, manifest["native_frames"] - 1]))
        for frame in frames:
            if frame >= manifest["native_frames"]:
                continue
            call("set_time", {"timeline": "raw_frame", "time": frame, "play": False})
            time.sleep(1)
            call(
                "screenshot",
                {"save_path": str((args.output / f"native-{frame:06d}.png").resolve())},
            )
        for layout in ("cameras", "evidence"):
            call("open_url", {"url": str((args.bundle / f"overview-{layout}.rbl").resolve())})
            time.sleep(1)
            call("screenshot", {"save_path": str((args.output / f"native-{layout}.png").resolve())})
            if layout == "cameras" and manifest["native_frames"] > 100:
                call("set_time", {"timeline": "raw_frame", "time": 100, "play": False})
                time.sleep(1)
                call(
                    "screenshot",
                    {"save_path": str((args.output / "native-cameras-100.png").resolve())},
                )
        call("open_url", {"url": str((args.bundle / "overview-demo.rbl").resolve())})
        call("set_time", {"timeline": "raw_frame", "time": 100, "play": False})
        (args.output / "native-viewer-check.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
