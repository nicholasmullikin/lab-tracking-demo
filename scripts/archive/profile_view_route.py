"""Time the display-only view-aid path the calibration browser actually exercises.

Reports per-operator cost, the end-to-end ``/api/view-filters/render`` route, and the
three interactions that dominate a tuning session: a cold render, a single slider tick
with everything on, and a repeat of an identical request.

This is a measurement helper, not part of the calibration pipeline. It serves a
throwaway workspace from a temporary directory and never reads or writes a real run.

    uv run python scripts/archive/profile_view_route.py --image path/to/frame.jpg
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_muggled_calibration_web import make_workspace  # noqa: E402

from battle import calibration_view_filters as view_filters  # noqa: E402
from battle.muggled_calibration_web import make_handler  # noqa: E402


def median_milliseconds(work: Callable[[], object], repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        work()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def post(base: str, body: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        f"{base}/api/view-filters/render",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(request).read())


def enable(operator_ids: Iterable[str]) -> dict[str, dict[str, object]]:
    settings = view_filters.default_settings()
    for operator_id in operator_ids:
        settings[operator_id] = {**settings[operator_id], "enabled": True}
    return settings


def report_operators(image_path: Path, repeats: int) -> None:
    with Image.open(image_path) as opened:
        source = np.asarray(opened.convert("RGB"))
    height, width = source.shape[:2]
    print(f"frame {image_path.name} {width}x{height} · VIGRA {view_filters.vigra_version()}")
    print(f"\n{'single operator':32s} {'ms':>8s}")
    for operator in view_filters.VIEW_FILTER_PIPELINE:
        settings = enable([operator.id])
        elapsed = median_milliseconds(
            lambda s=settings: view_filters.render_view(source, s), repeats
        )
        print(f"{operator.id:32s} {elapsed:8.1f}")


def report_route(image_path: Path, repeats: int, divisor: int) -> None:
    every = [operator.id for operator in view_filters.VIEW_FILTER_PIPELINE]
    with tempfile.TemporaryDirectory() as raw:
        workspace = make_workspace(Path(raw))
        frames = workspace.manifest_path.parent / "results/frames"
        frames.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_path, frames / "fixture.jpg")
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(workspace))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            print(f"\nroute at working resolution 1/{divisor}")
            print(f"{'interaction':40s} {'wall ms':>9s}")

            def request(settings: dict[str, object]) -> dict[str, object]:
                return post(
                    base,
                    {
                        "timestamp": 0,
                        "settings": settings,
                        "resolution_divisor": divisor,
                    },
                )

            for name, operator_ids in (
                ("cold: all twelve", every),
                ("cold: canny + corner response", ["canny", "corner_response"]),
                ("cold: adaptive threshold", ["adaptive_threshold"]),
            ):
                base_settings = enable(operator_ids)
                counter = iter(range(repeats))
                elapsed = median_milliseconds(
                    lambda s=base_settings, c=counter: request(
                        {**s, "brightness": {**s["brightness"], "amount": float(next(c))}}
                    ),
                    repeats,
                )
                print(f"{name:40s} {elapsed:9.1f}")

            settings = enable(every)
            request(settings)
            steps = iter(range(100))
            elapsed = median_milliseconds(
                lambda: request(
                    {
                        **settings,
                        "tensor_junction": {
                            **settings["tensor_junction"],
                            "scale": 1.0 + 0.1 * next(steps),
                        },
                    }
                ),
                repeats * 2,
            )
            print(f"{'slider tick, all twelve on':40s} {elapsed:9.1f}")

            toggles = iter(range(100))
            elapsed = median_milliseconds(
                lambda: request(
                    {
                        **settings,
                        "canny": {**settings["canny"], "enabled": next(toggles) % 2 == 0},
                    }
                ),
                repeats * 2,
            )
            print(f"{'toggle one aid, all twelve on':40s} {elapsed:9.1f}")

            payload = request(settings)
            elapsed = median_milliseconds(lambda: request(settings), repeats)
            size = len(str(payload.get("image_png_base64", ""))) / 1024
            print(f"{'identical repeat':40s} {elapsed:9.1f}")
            print(f"{'payload kB (base64)':40s} {size:9.1f}")
        finally:
            server.shutdown()
            server.server_close()
            workspace.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="a source frame JPEG to render")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--divisor", type=int, default=1, help="working resolution divisor to measure"
    )
    arguments = parser.parse_args()
    image_path = Path(arguments.image)
    report_operators(image_path, arguments.repeats)
    report_route(image_path, arguments.repeats, arguments.divisor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
