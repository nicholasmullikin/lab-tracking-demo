#!/usr/bin/env bash
# Open the native Rerun viewer against the review server's gRPC proxy.
# Usage: connect.sh [server-host]   (default 100.64.0.7, the server's Tailscale IPv4)
set -euo pipefail
command -v rerun >/dev/null 2>&1 || { echo "rerun not on PATH; run $(dirname "$0")/install.sh first" >&2; exit 1; }
exec rerun --connect "rerun+http://${1:-100.64.0.7}:9876/proxy"
