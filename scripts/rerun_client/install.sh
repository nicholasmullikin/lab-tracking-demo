#!/usr/bin/env bash
# Install the Rerun NATIVE viewer on a macOS or Linux client for remote review.
#
# Installs uv (if missing), the pinned rerun-sdk as a uv tool (its wheel ships
# the `rerun` viewer binary), and ffmpeg. Idempotent: safe to re-run.
#
# RERUN_VERSION must equal the server's rerun-sdk (pyproject.toml pins
# rerun-sdk==0.37.1); the viewer refuses or misreads a proxy of another version.
#
# ffmpeg: Rerun's native viewer does not bundle a decoder. H.264/H.265/VP8/VP9
# are decoded through a separately installed `ffmpeg` executable found on PATH,
# minimum version 5.1 (https://rerun.io/docs/concepts/logging-and-ingestion/video).
# `rerun --version` must list "ffmpeg" under "Video features".
#
# Usage: install.sh [server-host]      (optional; prints a connection example)
# Env:   SKIP_FFMPEG_INSTALL=1         only check for ffmpeg, never install a package
#        UV_TOOL_DIR / UV_TOOL_BIN_DIR are honoured by uv (used by the self-test)
set -euo pipefail

RERUN_VERSION="0.37.1"
SERVER_HOST="${1:-host.example.ts.net}"
GRPC_PORT=9876
FFMPEG_MIN="5.1"

sudo_if_needed() {
  if [[ $(id -u) -eq 0 ]]; then "$@"; else sudo "$@"; fi
}

# 1. uv
if ! command -v uv >/dev/null 2>&1; then
  echo "== Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || { echo "uv still not on PATH; open a new shell and re-run" >&2; exit 1; }
fi

# 2. rerun viewer, pinned
echo "== Installing rerun-sdk==${RERUN_VERSION} (must match the server)"
uv tool install --force "rerun-sdk==${RERUN_VERSION}"
tool_bin="$(uv tool dir --bin)"
case ":${PATH}:" in
  *":${tool_bin}:"*) ;;
  *)
    export PATH="${tool_bin}:${PATH}"
    echo "note: ${tool_bin} is not on your PATH; run 'uv tool update-shell' once, then open a new shell"
    ;;
esac

# 3. ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
  if [[ ${SKIP_FFMPEG_INSTALL:-0} == 1 ]]; then
    echo "== ffmpeg not found; SKIP_FFMPEG_INSTALL=1 so not installing (video panes will not decode)"
  else
    echo "== Installing ffmpeg"
    case "$(uname -s)" in
      Darwin)
        if command -v brew >/dev/null 2>&1; then
          brew install ffmpeg
        else
          echo "Homebrew not found. Install it (https://brew.sh) and run 'brew install ffmpeg'," >&2
          echo "or download a static build from https://ffmpeg.org/download.html and put ffmpeg on PATH." >&2
        fi
        ;;
      Linux)
        if command -v apt-get >/dev/null 2>&1; then
          sudo_if_needed apt-get update && sudo_if_needed apt-get install -y ffmpeg
        elif command -v dnf >/dev/null 2>&1; then
          # Fedora's default `ffmpeg-free` omits the H.264 decoder; the full package needs RPM Fusion.
          sudo_if_needed dnf install -y ffmpeg
        elif command -v pacman >/dev/null 2>&1; then
          sudo_if_needed pacman -S --needed --noconfirm ffmpeg
        else
          echo "No apt-get/dnf/pacman found. Install ffmpeg >= ${FFMPEG_MIN} with your package manager" >&2
          echo "or a static build from https://ffmpeg.org/download.html, and put it on PATH." >&2
        fi
        ;;
      *)
        echo "Unsupported OS $(uname -s); install ffmpeg >= ${FFMPEG_MIN} manually and put it on PATH." >&2
        ;;
    esac
  fi
fi

# 4. verify
echo "== Versions"
rerun_version_text="$(rerun --version)"
echo "${rerun_version_text}"
rerun_reported="$(printf '%s\n' "${rerun_version_text}" | awk 'NR == 1 {print $2}')"
if [[ ${rerun_reported} != "${RERUN_VERSION}" ]]; then
  echo "error: rerun reports ${rerun_reported}, expected ${RERUN_VERSION} (another rerun earlier on PATH?)" >&2
  exit 1
fi
if [[ ${rerun_version_text} != *"Video features:"*ffmpeg* ]]; then
  echo "warning: this rerun build lists no ffmpeg video feature; H.264 will not decode" >&2
fi
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg_line="$(ffmpeg -version | awk 'NR == 1')"
  echo "${ffmpeg_line}"
  ffmpeg_ver="$(printf '%s' "${ffmpeg_line}" | sed -E 's/^ffmpeg version n?([0-9]+\.[0-9]+).*/\1/')"
  if [[ ${ffmpeg_ver} =~ ^[0-9]+\.[0-9]+$ ]] && [[ $(printf '%s\n%s\n' "${FFMPEG_MIN}" "${ffmpeg_ver}" | sort -V | head -1) != "${FFMPEG_MIN}" ]]; then
    echo "warning: ffmpeg ${ffmpeg_ver} is older than ${FFMPEG_MIN}; Rerun needs >= ${FFMPEG_MIN}" >&2
  fi
else
  echo "ffmpeg: not on PATH (Rerun will show 'failed to create video decoder' on H.264 panes)"
fi

echo
echo "== Ready. Connect to the server (it must be running scripts/serve_review_over_tailscale.sh):"
echo "  rerun --connect rerun+http://${SERVER_HOST}:${GRPC_PORT}/proxy"
echo "  (or: $(dirname "$0")/connect.sh ${SERVER_HOST})"
echo "If the server instead serves the .rrd file over plain http, open it directly:"
echo "  rerun http://${SERVER_HOST}:<port>/<recording>.rrd"
