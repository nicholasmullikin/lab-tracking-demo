#!/usr/bin/env -S uv run python
"""Serial GPU job queue for the overnight pass; see `battle.overnight_queue` for the runner.

uv run scripts/overnight_queue.py runs/overnight-multicam-20260918/jobs.json
"""

from __future__ import annotations

import sys

from battle.overnight_queue import main

if __name__ == "__main__":
    sys.exit(main())
