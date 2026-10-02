#!/usr/bin/env bash
# Serve a Rerun recording to other devices on your own tailnet.
#
# Runs `uv run rerun --serve-web <recording.rrd>`: a web viewer on :9090 and a
# gRPC proxy on :9876, bound to this machine's Tailscale IPv4 only (never a
# wildcard). Remote reviewers should connect with the NATIVE viewer
# (scripts/rerun_client/): the browser viewer loads over plain http, WebCodecs
# needs a secure context, so its video panes fail with "failed to create video
# decoder". For a browser-only client use --https-port (see below).
#
# Usage:
#   scripts/serve_review_over_tailscale.sh [options] [recording.rrd [blueprint.rbl]]
# With no paths it serves the v6 combined first-minute package with its segmentation preset;
# with paths it serves exactly what was given (a preset only fits the package it was built for).
# Options:
#   --server-memory-limit SIZE  gRPC server buffer (default 4GiB; 100 MB rrd needs > 1GiB default)
#   --https-port N              bind 127.0.0.1 and allow the https://<magicdns>:N origin so
#                               `tailscale serve --bg --https=N http://127.0.0.1:9090` can
#                               front the web viewer with HTTPS (the gRPC proxy is fronted on N+1)
#   --dry-run                   print the resolved rerun command and exit
#
# Requires: tailscale (tailscaled running), python3 (JSON parsing), uv.
set -euo pipefail

default_rrd="runs/interaction-review-first-minute-v6/interaction_review_combined.rrd"
default_rbl="runs/interaction-review-first-minute-v6/segmentation.rbl"
web_port=9090
grpc_port=9876
memory_limit="4GiB"
viewer_memory_limit="8GiB"
https_port=""
dry_run=0
rrd=""
rbl=""

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() {
  echo "error: $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: scripts/serve_review_over_tailscale.sh [options] [recording.rrd [blueprint.rbl]]
  default recording: ${default_rrd}
  default blueprint: ${default_rbl} (only when no paths are given)
Options:
  --server-memory-limit SIZE  gRPC server buffer (default ${memory_limit})
  --https-port N              bind 127.0.0.1 for 'tailscale serve --bg --https=N http://127.0.0.1:${web_port}'
  --web-viewer-port N         local web port (default ${web_port})
  --grpc-port N               local proxy port (default ${grpc_port})
  --viewer-memory-limit SIZE  viewer memory limit (default ${viewer_memory_limit})
  --dry-run                   print the resolved rerun command and exit
  -h, --help                  this text
EOF
}

while (($# > 0)); do
  case "$1" in
    --server-memory-limit | --memory-limit)
      [[ $# -ge 2 ]] || die "$1 needs a value"
      memory_limit=$2
      shift 2
      ;;
    --https-port)
      [[ $# -ge 2 ]] || die "$1 needs a value"
      [[ $2 =~ ^[0-9]+$ ]] || die "--https-port must be a port number, got '$2'"
      https_port=$2
      shift 2
      ;;
    --web-viewer-port | --grpc-port)
      [[ $# -ge 2 && $2 =~ ^[0-9]+$ ]] || die "$1 needs a port number"
      if [[ $1 == --web-viewer-port ]]; then web_port=$2; else grpc_port=$2; fi
      shift 2
      ;;
    --viewer-memory-limit)
      [[ $# -ge 2 ]] || die "$1 needs a value"
      viewer_memory_limit=$2
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    -*)
      usage >&2
      die "unknown option: $1"
      ;;
    *)
      if [[ -z ${rrd} ]]; then
        rrd=$1
      elif [[ -z ${rbl} ]]; then
        [[ $1 == *.rbl ]] || die "the second path must be a .rbl blueprint (got '$1')"
        rbl=$1
      else
        die "at most one recording and one blueprint are accepted (got '${rrd}', '${rbl}' and '$1')"
      fi
      shift
      ;;
  esac
done

if [[ -z ${rrd} ]]; then
  rrd="${repo_root}/${default_rrd}"
  rbl="${repo_root}/${default_rbl}"
fi
[[ -f ${rrd} ]] || die "recording not found: ${rrd}"
[[ -z ${rbl} || -f ${rbl} ]] || die "blueprint not found: ${rbl}"

command -v tailscale >/dev/null 2>&1 || die "tailscale CLI not found on PATH"
command -v python3 >/dev/null 2>&1 || die "python3 not found on PATH"
command -v uv >/dev/null 2>&1 || die "uv not found on PATH"

if ! status_json="$(tailscale status --self --json 2>/dev/null)"; then
  die "tailscale status failed: is tailscaled running? (systemctl status tailscaled)"
fi
backend_state="$(printf '%s' "${status_json}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("BackendState", ""))')"
[[ ${backend_state} == "Running" ]] || die "Tailscale backend state is '${backend_state:-unknown}', expected Running (try: tailscale up)"

if ! ts_ip="$(tailscale ip -4 2>/dev/null)" || [[ -z ${ts_ip} ]]; then
  die "tailscale ip -4 returned nothing: this machine has no Tailscale IPv4"
fi
ts_dns="$(printf '%s' "${status_json}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')"
[[ -n ${ts_dns} ]] || die "could not read Self.DNSName from tailscale status (MagicDNS disabled?)"

if [[ -n ${https_port} ]]; then
  bind_ip="127.0.0.1"
  https_grpc_port=$((https_port + 1))
  origins=("https://${ts_dns}:${https_port}")
else
  bind_ip="${ts_ip}"
  origins=("http://${ts_ip}:${web_port}" "http://${ts_dns}:${web_port}")
fi

cmd=(uv run --project "${repo_root}" rerun --serve-web
  --bind "${bind_ip}"
  --web-viewer-port "${web_port}"
  --port "${grpc_port}"
  --memory-limit "${viewer_memory_limit}"
  --server-memory-limit "${memory_limit}")
for origin in "${origins[@]}"; do
  cmd+=(--cors-allow-origin "${origin}")
done
cmd+=("${rrd}")
[[ -z ${rbl} ]] || cmd+=("${rbl}")

echo "Recording:        ${rrd}"
echo "Blueprint:        ${rbl:-(none)}"
echo "Tailscale IPv4:   ${ts_ip}"
echo "MagicDNS name:    ${ts_dns}"
echo "Bind address:     ${bind_ip}"
echo
if [[ -n ${https_port} ]]; then
  echo "HTTPS via Tailscale (run these once, in another shell; 'tailscale serve --https=${https_port} off' to remove):"
  echo "  tailscale serve --bg --https=${https_port} http://127.0.0.1:${web_port}"
  echo "  tailscale serve --bg --https=${https_grpc_port} http://127.0.0.1:${grpc_port}"
  echo "Web viewer (browser, secure context so H.264 can decode):"
  echo "  https://${ts_dns}:${https_port}/?renderer=webgl&persist=false&theme=dark&url=rerun%2Bhttps%3A%2F%2F${ts_dns}%3A${https_grpc_port}%2Fproxy"
  echo "Native viewer (from this machine only, loopback bind):"
  echo "  rerun --connect rerun+http://127.0.0.1:${grpc_port}/proxy"
else
  echo "Web viewer (browser; loads, but H.264 panes will NOT decode over plain http:"
  echo "WebCodecs requires a secure context; use the native viewer or --https-port):"
  echo "  http://${ts_ip}:${web_port}/?url=rerun+http://${ts_ip}:${grpc_port}/proxy"
  echo "  http://${ts_dns}:${web_port}/?url=rerun+http://${ts_dns}:${grpc_port}/proxy"
  echo "Native viewer (recommended; client needs rerun-sdk of the same version + ffmpeg >= 5.1,"
  echo "see scripts/rerun_client/README.md):"
  echo "  rerun --connect rerun+http://${ts_ip}:${grpc_port}/proxy"
  echo
  echo "Bound beyond loopback: no authentication, any client that reaches this address can read"
  echo "the recording. Reachable inside your tailnet only; stop with Ctrl-C when done."
fi
echo

if ((dry_run)); then
  echo "Dry run; would execute:"
  printf '  '
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi

exec "${cmd[@]}"
