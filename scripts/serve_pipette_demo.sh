#!/usr/bin/env bash
# Serve an immutable demo without using the experiment's viewer or proxy ports.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundle="${repo_root}/runs/pipette-demo-20261002-final"
layout="demo"
dry_run=()
background=0
while [[ ${1:-} == --* ]]; do
  case "$1" in
    --dry-run) dry_run=(--dry-run);;
    --background) background=1;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
  shift
done
if [[ $# -gt 0 ]]; then bundle=$1; shift; fi
if [[ $# -gt 0 ]]; then layout=$1; shift; fi
[[ $# == 0 ]] || { echo "Usage: $0 [--dry-run] [--background] [bundle [demo|orientation|cameras|evidence]]" >&2; exit 2; }
case "$layout" in demo|orientation|cameras|evidence) ;; *) echo "Unknown layout: $layout" >&2; exit 2;; esac
[[ -f "$bundle/manifest.json" ]] || { echo "No completed demo manifest: $bundle" >&2; exit 1; }
bundle="$(cd "$bundle" && pwd)"
if ((background)) && [[ ${#dry_run[@]} == 0 ]]; then
  bash "$0" --dry-run "$bundle" "$layout"
  exec systemd-run --user --unit battle-pipette-demo --collect \
    --description "Saved pipette orientation demo" \
    --setenv "PATH=${PATH}" \
    --property "WorkingDirectory=${repo_root}" --property Restart=on-failure \
    --property RestartSec=3 \
    bash "${repo_root}/scripts/serve_pipette_demo.sh" "$bundle" "$layout"
fi
exec bash "${repo_root}/scripts/serve_review_over_tailscale.sh" \
  --https-port 8443 --web-viewer-port 9092 --grpc-port 9878 \
  --server-memory-limit 8GiB --viewer-memory-limit 8GiB \
  "${dry_run[@]}" "$bundle/overview.rrd" "$bundle/overview-${layout}.rbl"
