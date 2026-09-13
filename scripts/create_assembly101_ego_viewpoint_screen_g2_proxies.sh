#!/usr/bin/env bash
# Build only the three approved additional Assembly101 ego G2 proxies.
set -euo pipefail

sequence="nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532"
raw_dir="data/raw/assembly101/${sequence}/recordings/${sequence}"
output_dir="data/derived/assembly101/${sequence}"
filter="trim=duration=180.000,setpts=PTS-STARTPTS,fps=30:round=near,scale=954:720:flags=lanczos,setsar=1"

validate_proxy() {
  local path=$1
  local facts
  facts="$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames:format=duration \
    -of json "${path}")"
  python3 - "${facts}" <<'PY'
import json
import sys

facts = json.loads(sys.argv[1])
stream = facts["streams"][0]
assert stream["codec_name"] == "h264", stream
assert stream["pix_fmt"] == "yuv420p", stream
assert (stream["width"], stream["height"]) == (954, 720), stream
assert stream["r_frame_rate"] == "30/1", stream
assert int(stream["nb_frames"]) == 5400, stream
assert abs(float(facts["format"]["duration"]) - 180.0) < 0.01, facts["format"]
PY
  if ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "${path}" |
    rg -q .; then
    echo "proxy unexpectedly contains audio: ${path}" >&2
    exit 1
  fi
}

create_proxy() {
  local stem=$1
  local input_path="${raw_dir}/${stem}.mp4"
  local output_path="${output_dir}/${stem}_215.000-395.000_954x720_30fps.mp4"
  local temporary_path="${output_path}.partial.mp4"

  test -f "${input_path}"
  if test -f "${output_path}"; then
    validate_proxy "${output_path}"
    echo "Reused valid proxy: ${output_path}"
    return
  fi

  rm -f "${temporary_path}"
  ffmpeg -hide_banner -loglevel error -ss 215.000 -i "${input_path}" -map 0:v:0 -an \
    -vf "${filter}" -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
    -movflags +faststart -fps_mode cfr "${temporary_path}"
  validate_proxy "${temporary_path}"
  mv "${temporary_path}" "${output_path}"
  echo "Created valid proxy: ${output_path}"
}

mkdir -p "${output_dir}"
create_proxy "HMC_21176875_mono10bit"
create_proxy "HMC_21176623_mono10bit"
create_proxy "HMC_21179183_mono10bit"
