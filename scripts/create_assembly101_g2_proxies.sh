#!/usr/bin/env bash
# Build the approved Assembly101 G2 proxies without reading annotations or models.
set -euo pipefail

sequence="nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532"
raw_dir="data/raw/assembly101/${sequence}/recordings/${sequence}"
output_dir="data/derived/assembly101/${sequence}"

# The source streams are CFR 60 fps. -ss is an accurate timestamp seek while
# transcoding; trim therefore selects [215.000, 395.000), equivalent to raw
# frame indices [12900, 23700). The fps filter maps that interval to proxy
# frames [0, 5400), at globally indexed analysis frames [6450, 11850).
create_proxy() {
  local input_path=$1
  local video_filter=$2
  local output_path=$3

  ffmpeg -hide_banner -n -ss 215.000 -i "${input_path}" -map 0:v:0 -an \
    -vf "${video_filter}" -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
    -movflags +faststart -fps_mode cfr "${output_path}"
}

mkdir -p "${output_dir}"

create_proxy "${raw_dir}/C10379_rgb.mp4" \
  "trim=duration=180.000,setpts=PTS-STARTPTS,fps=30:round=near,scale=1280:720:flags=lanczos,setsar=1" \
  "${output_dir}/C10379_rgb_215.000-395.000_1280x720_30fps.mp4"

create_proxy "${raw_dir}/HMC_21110305_mono10bit.mp4" \
  "trim=duration=180.000,setpts=PTS-STARTPTS,fps=30:round=near,scale=954:720:flags=lanczos,setsar=1" \
  "${output_dir}/HMC_21110305_mono10bit_215.000-395.000_954x720_30fps.mp4"
