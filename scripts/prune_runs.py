#!/usr/bin/env python
"""List run directories that nothing committed refers to.

`runs/` grows with every experiment, and the ones that matter are the ones cited by the
README, the method ledger, a QA record, or a committed config. This reports the rest with
their sizes so a human can decide.

It never deletes anything. The last line is a `rm -rf` command to copy, edit, and run by
hand, because a run that is only referenced from an uncommitted note or from another run's
manifest is still evidence somebody may need.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def tracked_text(repository_root: Path) -> str:
    """Concatenate every tracked text file, which is where citations live."""
    listing = subprocess.run(
        ["git", "-C", str(repository_root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    chunks: list[str] = []
    for entry in listing:
        if not entry:
            continue
        path = repository_root / entry.decode()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4", ".rrd", ".pt", ".npz"}:
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(chunks)


def referenced_runs(text: str, names: set[str]) -> set[str]:
    return {name for name in names if name in text}


def referencing_manifests(runs_root: Path, names: set[str]) -> dict[str, set[str]]:
    """Map each run to the other runs whose manifests or indexes name it."""
    citations: dict[str, set[str]] = {name: set() for name in names}
    for run in sorted(runs_root.iterdir()):
        if not run.is_dir():
            continue
        for candidate in ("manifest.json", "interaction_review_index.json", "index.json"):
            path = run / candidate
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for name in names:
                if name != run.name and name in text:
                    citations[name].add(run.name)
    return citations


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--json", action="store_true", help="Emit the report as JSON instead of a table."
    )
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    runs_root = (repository_root / args.runs_root).resolve()
    if not runs_root.is_dir():
        parser.error(f"{runs_root} is not a directory")

    names = {path.name for path in runs_root.iterdir() if path.is_dir()}
    cited = referenced_runs(tracked_text(repository_root), names)
    manifest_citations = referencing_manifests(runs_root, names - cited)
    unreferenced = sorted(
        (name for name in names - cited if not manifest_citations[name]),
        key=lambda name: directory_bytes(runs_root / name),
        reverse=True,
    )
    only_manifests = sorted(name for name in names - cited if manifest_citations[name])

    if args.json:
        print(
            json.dumps(
                {
                    "runs": len(names),
                    "cited_by_tracked_files": sorted(cited),
                    "cited_only_by_another_run": {
                        name: sorted(manifest_citations[name]) for name in only_manifests
                    },
                    "unreferenced": [
                        {"run": name, "bytes": directory_bytes(runs_root / name)}
                        for name in unreferenced
                    ],
                },
                indent=2,
            )
        )
        return

    total = sum(directory_bytes(runs_root / name) for name in unreferenced)
    print(f"{len(names)} runs; {len(cited)} cited by tracked files")
    if only_manifests:
        print(f"\n{len(only_manifests)} cited only by another run's manifest (kept):")
        for name in only_manifests:
            print(f"  {name} <- {', '.join(sorted(manifest_citations[name]))}")
    print(f"\n{len(unreferenced)} unreferenced, {total / 1e9:.2f} GB:")
    for name in unreferenced:
        print(f"  {directory_bytes(runs_root / name) / 1e6:9.1f} MB  {name}")
    if unreferenced:
        print("\nReview this list, then delete by hand if you agree:")
        print(
            "  rm -rf " + " \\\n        ".join(str(args.runs_root / name) for name in unreferenced)
        )


if __name__ == "__main__":
    main()
