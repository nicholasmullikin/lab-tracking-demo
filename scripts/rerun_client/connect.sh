#!/usr/bin/env bash
# Open the native Rerun viewer against the review server's gRPC proxy.
# Usage: connect.sh server-host
set -euo pipefail
if [[ $# -ne 1 || -z $1 ]]; then
  echo "Usage: connect.sh server-host" >&2
  exit 2
fi
command -v rerun >/dev/null 2>&1 || { echo "rerun not on PATH; run $(dirname "$0")/install.sh first" >&2; exit 1; }
exec rerun --connect "rerun+http://$1:9876/proxy"
