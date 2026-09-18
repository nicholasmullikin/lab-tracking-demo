#!/usr/bin/env bash
# Build the focused-window (source 294.000-386.700 s) 60 fps trims and 1280x720 CFR-30 proxies
# for every Assembly101 static view of the pinned recording, plus the 60 fps trims of the four
# ego cameras already on disk.  Remote views are read over HTTP Range from the pinned Hugging
# Face revision through `battle-fetch-assembly101-view`; the full 1.6-4 GB files are never
# downloaded.  Existing proxies (C10379, HMC_21110305) are kept; only their trims are added.
set -euo pipefail

remote_views=(C10095 C10115 C10118 C10119 C10390 C10395 C10404)
local_views=(C10379 HMC_21110305 HMC_21176623 HMC_21176875 HMC_21179183)
sequence="nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532"
records="data/raw/assembly101/${sequence}/static_views_focused_acquisition"

for view in "${remote_views[@]}"; do
  if [[ -f "${records}/${view}.json" ]]; then
    continue
  fi
  uv run battle-fetch-assembly101-view --view "${view}"
done

for view in "${local_views[@]}"; do
  if [[ -f "${records}/${view}.json" ]]; then
    continue
  fi
  uv run battle-fetch-assembly101-view --view "${view}" --local
done
uv run battle-fetch-assembly101-view --write-report
