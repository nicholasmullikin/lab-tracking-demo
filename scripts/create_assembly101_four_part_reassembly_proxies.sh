#!/usr/bin/env bash
# Build the approved four-part Assembly101 reassembly proxies from the raw recordings.
set -euo pipefail

sequence="nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532"
raw_dir="data/raw/assembly101/${sequence}/recordings/${sequence}"
output_dir="data/derived/assembly101/${sequence}"

# The source streams are CFR 60 fps. The half-open source interval
# [190.000, 386.700) maps to raw frames [11400, 23202), global 30-fps
# analysis/annotation frames [5700, 11601), and proxy frames [0, 5901).
# Source 190.000 was selected after direct human review showed cleaner separation
# of all four target surfaces; independent mask separability remains a G2 gate.
create_proxy() {
  local input_path=$1
  local video_filter=$2
  local output_path=$3

  ffmpeg -hide_banner -n -ss 190.000 -i "${input_path}" -map 0:v:0 -an \
    -vf "${video_filter}" -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
    -movflags +faststart -fps_mode cfr "${output_path}"
}

mkdir -p "${output_dir}"

create_proxy "${raw_dir}/C10379_rgb.mp4" \
  "trim=duration=196.700,setpts=PTS-STARTPTS,fps=30:round=near,scale=1280:720:flags=lanczos,setsar=1" \
  "${output_dir}/C10379_rgb_190.000-386.700_1280x720_30fps.mp4"

create_proxy "${raw_dir}/HMC_21110305_mono10bit.mp4" \
  "trim=duration=196.700,setpts=PTS-STARTPTS,fps=30:round=near,scale=954:720:flags=lanczos,setsar=1" \
  "${output_dir}/HMC_21110305_mono10bit_190.000-386.700_954x720_30fps.mp4"
