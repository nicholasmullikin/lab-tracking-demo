#!/usr/bin/env bash
# Start the existing first-minute review with this machine's current tailnet address.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${repo_root}/scripts/serve_review_over_tailscale.sh" "$@"
