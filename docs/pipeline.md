# Pipeline

This page is how to run the lab: set up the three environments, run the FineBio pipeline one
stage at a time, run the pipettes-as-lines extension and its tip-click gate, open the recordings,
hold the two human gates, rebuild the Assembly101 review package, run the tests, queue GPU jobs
and prune old runs. I checked every flag below against the
tool's `--help`, and every tool answers `uv run battle-<name> --help` on the CPU. The numbers these
commands produced are in [`results.md`](results.md); the words are in the
[glossary](writing-style.md#glossary).

## Setup

**Three environments, one card.** The lab runs on CPython 3.12 through `uv` and must not run on the
system Python 3.14. The GPU is one RTX 5070 Ti with 16 GB. Every GPU tool checks the card through
`battle.gpu_guard` before it starts and refuses to share it with another Battle worker unless that
process is named with `--allow-gpu-neighbour PID`.

```bash
uv sync --python 3.12
uv run python --version
scripts/install_finebio_detector.sh --cuda     # the FineBio detector venv; --cpu for a CPU build
```

The SAM3 worker runs in its own interpreter, MuggledSAM at `/home/nick/src/muggled_sam` under
`/home/nick/.pyenv/versions/muggled_sam/bin/python` with the `sam3.1_multiplex.pt` weights; every
SAM3 tool takes `--external-python` and `--model` to point elsewhere. The FineBio detector
(MMDetection 3.3.0 with the authors' DINO and Deformable DETR configs and weights) lives in
`/home/nick/src/finebio-detector/.venv-cuda`: torch 2.13+cu130 and mmcv 2.1.0 compiled from source
for sm_120. CUDA 13.2's `nvcc` refuses GCC 16 as host compiler, so the script passes `CC=gcc-15
CXX=g++-15` to the mmcv build (`FINEBIO_HOST_CC` and `FINEBIO_HOST_CXX` override it). Torch 2.6 and
later load checkpoints weights-only and the authors' `.pth` files carry mmengine buffers, so the
detector worker sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for its own process. The script can be
run again safely, and `--skip-verify` and `--skip-weights` shorten a re-run.

Raw FineBio videos sit under `data/raw/finebio/`, the window proxies under
`data/derived/finebio/<trial>/600-4200/`, and every run under `runs/`. All three are ignored by Git.

## FineBio, one command per stage

**Six stages take a trial from raw video to a review recording, and trial 2 differs only in the
trial id.** The window is raw frames 600 to 4199, 120 s, six cameras. Trial 1 is `P03_03_01`; swap
in `P20_03_01` for trial 2, whose camera solve read markers at 30, 60 and 90 s instead.

```bash
T=P03_03_01
D=runs/finebio-detect-$T-600-4200/dino
uv run battle-finebio-cameras solve --trial $T --seconds 30,120,240       # configs/finebio/cameras/$T.json and the marker evidence
uv run battle-finebio-preprocess --trial $T --window-from configs/finebio/trials.json   # six proxies and configs/clips/finebio_${T}_600-4200.json
uv run battle-finebio-detect run --trial $T --views fpv,T1,T2,T3,T4,T5 --start 600 --count 3600 --model dino --device cuda --output $D   # every frame, GPU
uv run battle-finebio-rig --config configs/finebio/cameras/${T}_600-4200.json --detections $D --frames 600-4199 --output runs/finebio-rig-$T-600-4200 --negative-control
uv run battle-detector-seed run --trial $T --detections $D --start 600 --end 4200 --output runs/finebio-seeds-$T   # slots, seeds (GPU decode) and the gate 1 sheets
```

The camera solve keeps a shipped pose that fits the markers within 10 px and re-solves the rest
from the marker corners by PnP (perspective-n-point); no camera is ever faked. The rig writes its
gates as formulas into `rig.json`. The seed rule opens a slot for what moves or sits in a hand.
The arms ran on a second seed set, `with-plate/`, which adds the cell culture plate as a landmark
slot and caps each camera at 11 slots:

```bash
S=runs/finebio-seeds-$T/with-plate
cp -r runs/finebio-seeds-$T/instances $S/
uv run battle-detector-seed select --output $S --slot-cap 11 --landmark-classes centrifuge,vortex_mixer,pcr_machine,cell_culture_plate \
  --container-classes centrifuge,vortex_mixer,pcr_machine,magnetic_rack,micro_tube_rack,50ml_tube_rack,15ml_tube_rack,8_tube_stripes_rack,blue_tip_rack,yellow_tip_rack,red_tip_rack,8_channel_tip_rack,cell_culture_plate
uv run battle-detector-seed decode --output $S --trial $T && uv run battle-detector-seed sheets --output $S --trial $T
uv run battle-finebio-arms mark-plan-slots --seeds $S --classes cell_culture_plate --rule landmark_plan_shortlist --note "plan shortlist"
```

The mask arms are two SAM3 worker runs per camera, one camera at a time through the GPU queue below.
Per-frame box decode reads the detector's box stream and draws a fresh mask on every frame; video
memory reads the seed schedule once and tracks. The tracker then turns each arm's observations into
3D tracks, measures and an occlusion inventory, and `--ext` adds the four extensions (motion model,
containers, groups, held objects) into `tracks-ext/` beside the core `tracks/`.

```bash
A=runs/finebio-arms-$T
P=data/derived/finebio/$T/600-4200
for v in fpv T1 T2 T3 T4 T5; do
  uv run battle-muggled-arms box-decode --video $P/${T}_${v}_600-4200.mp4 --view-id $v --box-stream $S/box_streams/$v.jsonl --start-frame 0 --max-frames 3600 --max-side-length 1280 --run-root $A/b-box-decode/$v
  uv run battle-muggled-arms video-memory --video $P/${T}_${v}_600-4200.mp4 --view-id $v --schedule $S/schedules/$v.json --start-frame 0 --max-frames 3600 --max-side-length 1280 --prompt-memory-semantics append --checkpoint-every 600 --run-root $A/c-video-memory/$v
done
C=configs/clips/finebio_${T}_600-4200.json
G=runs/finebio-rig-$T-600-4200/rig.json
uv run battle-finebio-arms run --arm a --clip-config $C --detections $D --gates $G --seeds $S --output $A/a-boxes-only
uv run battle-finebio-arms run --arm b --clip-config $C --detections $D --gates $G --seeds $S --worker-root $A/b-box-decode --output $A/b-box-decode-arm
uv run battle-finebio-arms run --arm c --clip-config $C --detections $D --gates $G --seeds $S --worker-root $A/c-video-memory --output $A/c-video-memory-arm
uv run battle-finebio-arms run --arm b --clip-config $C --detections $D --gates $G --seeds $S --output $A/b-box-decode-arm --ext --reuse-observations   # same for a and c
uv run battle-finebio-arms decide --b $A/b-box-decode-arm --c $A/c-video-memory-arm --output $A/decision_c_vs_b.json   # the pre-registered rule for arm (d)
uv run battle-finebio-arms scoreboard --arm a=$A/a-boxes-only --arm b=$A/b-box-decode-arm --arm c=$A/c-video-memory-arm --output $A/scoreboard
```

Confidence, events and the recording are CPU passes over the tracks. Name the tracks directory,
because all three default to `tracks-ext/` when it exists. `--preset-dir ""` writes the three
presets beside the recording and leaves the committed `configs/rerun/finebio_*.rbl` alone.

```bash
for arm in a-boxes-only b-box-decode-arm c-video-memory-arm; do
  uv run battle-finebio-confidence --arm-dir $A/$arm --detections $D --tracks-dir tracks-ext --output $A/$arm/confidence-ext
  uv run battle-finebio-events --arm-dir $A/$arm --config $C --rig $G --tracks-dir tracks-ext --output $A/$arm/events-ext
done
uv run battle-finebio-viewer --clip-config $C --arm-dirs a=$A/a-boxes-only,b=$A/b-box-decode-arm,c=$A/c-video-memory-arm --rig $G --seeds $S --tracks-dir tracks-ext --preset-dir "" --output runs/finebio-review-$T-ext
uv run rerun rrd verify runs/finebio-review-$T-ext/review.rrd
```

The recording takes six to eight minutes to build and lands near 900 MB. One trial's two mask
arms cost about 2.3 h of GPU; the detector pass for both trials and both detectors cost 65 minutes.

## Pipettes as 3D lines

**The line extension runs on the CPU from the masks already on disk, and every step below is one
command.** It failed its pre-registered rule and is not adopted; the numbers are in
[`results.md`](results.md#pipettes-as-3d-lines). The rows first gain a mask axis per SAM3
observation (`remeasure`), the stand slice measures the pipettes at rest and writes the length
prior, the colour calibration finds the plunger colour, and then the tracker runs with `--lines`,
which is `--ext` plus `--line-classes pipette` into `tracks-lines/`. The scoreboard reads the
rule. `battle-finebio-tipseg` tries five ways to find the disposable tip in each camera (its
`sam3` step needs the GPU), and `battle-finebio-tips` is the tip-click gate: `prepare` cuts the
crops, `serve` shows the pages, `score` triangulates the clicks against any tracks files, `export`
writes the committed no-pixel record.

```bash
F=runs/finebio-arms-$T-filtered-20260927
L=runs/finebio-lines-$T-20260928
uv run battle-finebio-arms remeasure --observations $F/b-box-decode-arm/observations.jsonl --worker-root $A/b-box-decode --output $L/observations-b   # mask axes on every SAM3 row; --classes blue_pipette,yellow_pipette,red_pipette,8_channel_pipette re-reads those alone
uv run battle-finebio-stand report --observations $L/observations-b/observations.jsonl --cameras configs/finebio/cameras/${T}_600-4200.json --output $L/stand   # the pipettes at rest: lengths, flat-rest geometry
uv run battle-finebio-stand config --report $L/stand/stand_report.json --output configs/finebio/pipettes.json                                              # the length prior with provenance; repeat --report for trial 2
uv run battle-finebio-stand tips --observations $L/observations-b/observations.jsonl --cameras configs/finebio/cameras/${T}_600-4200.json --gate-px 30.083 --with-fpv --output $L/stand-tips   # bare and with-tip rest lengths
uv run battle-finebio-stand tips-config --config configs/finebio/pipettes.json --tips $L/stand-tips/tip_lengths.json                                       # writes bare_length_cm and tip_length_cm; repeat --tips for trial 2
uv run battle-finebio-colour sample --observations $L/observations-b --clip-config $C --output $L/colour && uv run battle-finebio-colour config --calibration $L/colour/colour_calibration.json --pipettes configs/finebio/pipettes.json
uv run battle-finebio-arms run --arm b --lines --clip-config $C --detections $D --gates $G --seeds $S/filtered --worker-root $A/b-box-decode --observations-dir $L/observations-b --output $L/arm-b-lines   # --ext alone for the point tracker on the same rows, into $L/arm-b-ext
uv run battle-finebio-colour plunger-ends --observations $L/observations-b-v4 --clip-config $C --output $L/observations-b-v4/plunger_ends.jsonl   # once per trial; reruns reuse the sidecar
uv run battle-finebio-arms run --arm b --lines --clip-config $C --detections $D --gates $G --seeds $S/filtered --worker-root $A/b-box-decode --observations-dir $L/observations-b-v4 --output $L/arm-b-lines-v5a --tracker-arg=--line-orientation --tracker-arg=vote   # writes tracks_oriented.jsonl beside tracks.jsonl; add --tracker-arg=--invert-colour-cue for the control in a fresh output
uv run battle-finebio-colour annotate --tracks $L/arm-b-lines/tracks-lines/tracks.jsonl --observations $L/observations-b --clip-config $C --every 3 --output $L/arm-b-lines/tracks-lines/tracks_colour.jsonl
uv run battle-finebio-arms lines-negative-controls --observations $L/observations-b --clip-config $C --gates $G --frames 1040-1339 --shipped-view T5 --shift-view T3,T4 --output $L/negative-controls-1040   # trial 1 only
uv run battle-finebio-arms lines-scoreboard --lines-dir $L/arm-b-lines --ext-dir $L/arm-b-ext --baseline-ext-dir $F/b-box-decode-arm/tracks-ext --observations $L/observations-b --clip-config $C --rig $G --prior configs/finebio/pipettes.json --stand-report $L/stand/stand_report.json --negative-controls $L/negative-controls-1040/negative_controls.json --output $L/scoreboard
uv run battle-finebio-events --arm-dir $L/arm-b-lines --tracks-dir tracks-lines --observations $L/observations-b/observations.jsonl --config $C --rig $G --tip-events --output $L/arm-b-lines/events-tips   # tip_picked / tip_ejected from the line rows
uv run battle-finebio-tipseg prepare --clip-config $C --observations $L/observations-b --tips-workspace runs/finebio-tips-$T-20260928 --output runs/finebio-tipseg-$T   # then `sam3 --output <dir>` (GPU) and `score --output <dir> --tips-workspace <workspace>`
W=runs/finebio-tips-$T-20260929
uv run battle-finebio-tips prepare --observations $L/observations-b --clip-config $C --frames 30 --no-marker --single-channel-only --output $W   # --event-frames 2180,2330 --event-class blue_pipette --event-quota 10 --states held,rest chose the second sitting's frames
uv run battle-finebio-tips serve --workspace $W --tailscale                                                                                        # click; Ctrl-C when done
uv run battle-finebio-tips score --workspace $W --tracks lines-v3=$L/arm-b-lines-v3/tracks-lines/tracks.jsonl --tracks points-ext=$F/b-box-decode-arm/tracks-ext/tracks.jsonl --clip-config $C
uv run battle-finebio-tips export --workspace $W --record $W/decisions.json --protocol "<how I chose between a click and h>" --output docs/qa/finebio-$T-tip-clicks-2.human-record.json
uv run battle-finebio-viewer --clip-config $C --arm-dirs b=$F/b-box-decode-arm,lines=$L/arm-b-lines-v3 --rig $G --seeds $S/filtered --tracks-dir b=tracks-ext,lines=tracks-lines --secondary-mask-arm "" --preset-dir "" --output runs/finebio-review-$T-lines-20260929   # the line arm's segments and tips beside the point arm
```

The viewer draws an arm whose rows carry `endpoints_cm` as segments under
`world/tracks/<arm>/<state>/lines`, its tip as a white point under `.../tips`, and the same
segment in every camera tile; `--tracks-dir` takes `label=name` pairs when the arms keep their
tracks in differently named directories. The lines recording for trial 1 took six minutes to
build and lands at 870 MB.

## Opening the recordings

**Three recordings are worth opening, each with a World, a Cameras and an Evidence preset.** Pass
the recording and one preset to the viewer; swap `world.rbl` for `cameras.rbl` or `evidence.rbl`, or
drag the other presets in once the viewer is open. The timeline is `frame`, the raw frame index;
`storyboard/marks` jumps between the bookmarked frames. The 20-minute route through both trials is
in [`review-guide-2026-09-25-finebio-3d.md`](review-guide-2026-09-25-finebio-3d.md). A fourth
recording shows the pipettes as lines beside the point tracker, with a "held pipette" storyboard
item at the longest held stretch.

```bash
uv run rerun runs/finebio-review-P03_03_01-filtered-20260927/review.rrd runs/finebio-review-P03_03_01-filtered-20260927/world.rbl   # trial 1 on human-filtered seeds, with the extensions (recommended)
uv run rerun runs/finebio-review-P03_03_01-20260925/review.rrd runs/finebio-review-P03_03_01-20260925/world.rbl                     # trial 1, core tracker, the recording the guide describes
uv run rerun runs/finebio-review-P20_03_01-20260925-ext/review.rrd runs/finebio-review-P20_03_01-20260925-ext/world.rbl             # trial 2, with the extensions
uv run rerun runs/finebio-review-P03_03_01-lines-20260929/review.rrd runs/finebio-review-P03_03_01-lines-20260929/world.rbl         # trial 1, the pipettes as 3D lines (v3) beside the point tracker
```

## The two human gates

**Gate 1 accepts or rejects each seed from a contact sheet; gate 2 picks the right mask on a sample
of frames.** Both are optional and both were held on trial 1. Nothing waits for them: the arms run
on the detector's own seeds and the scoreboard runs on an empty record.

Gate 1 reads the sheets under `runs/finebio-seeds-<trial>/with-plate/sheets/<view>.jpg`, one tile
per slot. Copy the template, write `accept`, `reject` or leave `null` per slot, then apply it; the
filter writes `filtered/` beside the seeds with the rejected slots gone and the rest renumbered.
Downstream, per-frame box decode is a row filter of its own observations, because every mask is
decoded from its own box; video memory needs the worker re-run per camera on
`filtered/schedules/<view>.json`, because one camera's slots share one memory.

```bash
S=runs/finebio-seeds-P03_03_01-20260925
cp $S/decisions.template.json $S/decisions.json           # then edit the decision per slot
uv run battle-detector-seed apply-decisions --output $S/with-plate --decisions $S/decisions.json
F=runs/finebio-arms-P03_03_01-filtered
uv run battle-finebio-arms filter-observations --observations $A/b-box-decode-arm/observations.jsonl --seeds $S/with-plate/filtered --output $F/b-box-decode-arm
uv run battle-finebio-arms run --arm b --clip-config $C --detections $D --gates $G --seeds $S/with-plate/filtered --worker-root $A/b-box-decode --reuse-observations --output $F/b-box-decode-arm
```

Gate 2 chooses the anchor frames from arm (b)'s output, decodes up to four candidate masks per cell
on the GPU, and serves the workspace as web pages: one keypress per row, the record saved on every
change. `--tailscale` binds the machine's Tailscale address so the sitting can happen from another
device; the default is loopback. The scoreboard and the committed no-pixel record follow, and the
records from both sittings are under [`qa/`](qa/README.md).

```bash
W=runs/finebio-anchors-P03_03_01-20260925
Q=configs/qa/finebio_P03_03_01_review_anchors.json
uv run battle-finebio-anchors select --clip $C --arm-b $A/b-box-decode-arm --seeds $S/with-plate/seeds.json --output $Q
uv run battle-finebio-anchors workspace --config $Q --arm-b $A/b-box-decode-arm --output $W          # GPU, about 30 s
uv run battle-finebio-anchors-web --workspace $W --tailscale                                            # label; Ctrl-C when done
uv run battle-finebio-anchors score --workspace $W --record $W/decisions.json --arms a=$A/a-boxes-only,b=$A/b-box-decode-arm,c=$A/c-video-memory-arm --tracks-dir tracks-ext --output $W/scoreboard/ext
uv run battle-finebio-anchors export --workspace $W --record $W/decisions.json --output docs/qa/finebio-P03_03_01-review-anchors.human-record.json
```

## Assembly101: the v6 review package

**One recording carries the whole Assembly101 phase, and three presets switch every panel at
once.** `segmentation.rbl` shows the reference beside the candidate arms with the anchor marks;
`hands.rbl` shows every hand source in 2D and world millimeters; `multiview.rbl` shows nine camera
tiles, the consensus markers, the hull and the 3D rig. The builder takes `--candidate-arm
NAME=RUN_DIR` once per arm; the nine-camera recording it merges with comes from
`battle-build-multiview-static-comparison` with the same application and recording id.

```bash
CUDA_VISIBLE_DEVICES="" uv run battle-build-interaction-review-v4 \
  --output-root runs/interaction-review-first-minute-v6 \
  --reference runs/ensemble-reference-first-minute-v2 \
  --multiview-consensus runs/multiview-part-consensus-first-minute-r1280-pm-append \
  --candidate-arm pm-append=runs/sam3-memory-arms-20260919/arms/pm-append/<run> \
  --candidate-arm dam4sam-large-1024-sched-60s=runs/dam4sam-arms-20260919/arms/dam4sam-large-1024-sched-60s \
  --confidence runs/detector-scorecard-20260920/pm-append \
  --application-id battle-interaction-review-v6 --recording-id interaction_review_first_minute_v6
uv run battle-review-presets --rrd runs/interaction-review-first-minute-v6/interaction_review_combined.rrd \
  --merge runs/interaction-review-first-minute-v6/interaction_review_first_minute_v4.rrd runs/multiview-static-comparison-first-minute-r1280-pm-append/multiview_static_comparison.rrd
uv run rerun runs/interaction-review-first-minute-v6/interaction_review_combined.rrd runs/interaction-review-first-minute-v6/segmentation.rbl   # or hands.rbl, multiview.rbl
uv run battle-anchor-iou --run pm-append=<run dir> --run dam4sam=<run dir> --output runs/<scoreboard>/anchor_iou.json   # rank arms on the anchors; --view for another camera
```

## Reviewing from another machine

**The recordings can be reviewed from any device on your own tailnet with the native viewer.**
`scripts/serve_review_over_tailscale.sh [recording.rrd [preset.rbl]]` binds this machine's Tailscale
address and prints the connect command; the client side is one installer and one connect script in
[`../scripts/rerun_client/README.md`](../scripts/rerun_client/README.md).

## Tests

**The default tier needs nothing outside the repository and runs in about 25 s.** Tests that read
`data/`, `runs/` or `models/` carry the `real_data` marker and are deselected by default; when the
artifact is absent they skip and name the missing path. `gpu` marks the checks that need a CUDA
device and local weights, `slow` the few that spend seconds in a subprocess. The markers and the
default deselection live in `pyproject.toml`, the skip rule in `tests/conftest.py`. Before claiming a
run's provenance, run the `real_data` tier and the relevant builder with `--verify-fingerprints`.

```bash
uv run pytest                                                # default tier
uv run pytest -m "not real_data and not gpu and not slow"    # the fast loop while editing
uv run pytest -n auto                                        # the default tier across cores
uv run pytest -m real_data                                   # the artifact checks
uv run pytest -m gpu                                         # the device-bound checks
uv run ruff check . && uv run ruff format --check .
```

## The GPU queue

**One process touches the GPU at a time, and only committed code reaches it.** `battle-code-snapshot`
archives `src/battle` at a commit into `<root>/code-snapshot-<sha>/` and prints the `PYTHONPATH` line
every job carries, so working-tree edits cannot reach a running job. The queue runs a JSON job list
one job at a time under `systemd-inhibit`, checks `nvidia-smi`, the kernel log and the card's headroom
before every job, kills a job past its timeout with its whole process group, and stops at the first
failure unless told otherwise. Events go to the log as JSON lines and each job's output to
`logs/<index>_<name>.log` beside it. A SAM3 worker whose run manifest records a failure exits 3, so
a dead worker stops the queue instead of counting as a success.

```bash
uv run battle-code-snapshot runs/<pass>                        # prints "# every job: env.PYTHONPATH=..."
cat > runs/<pass>/jobs.json <<'EOF'
{"jobs": [{"name": "b-fpv", "interpreter": ["uv", "run"], "argv": ["battle-muggled-arms", "box-decode", "..."],
           "cwd": "/home/nick/src/battle", "timeout_s": 1800, "env": {"PYTHONPATH": "runs/<pass>/code-snapshot-<sha>/src"}}]}
EOF
uv run scripts/overnight_queue.py runs/<pass>/jobs.json --dry-run
uv run scripts/overnight_queue.py runs/<pass>/jobs.json --log runs/<pass>/queue.log     # --continue-on-failure; --no-gpu-check for CPU lists
```

## Pruning runs

**`runs/` keeps every experiment, and the pruner reports what nothing committed cites without
deleting anything.** It lists the uncited directories largest first, separates the ones only another
run's manifest names, and ends with an `rm -rf` line to review and run by hand.

```bash
uv run python scripts/prune_runs.py            # --json for the report as JSON
```

## Story media

**The GIFs in [`story.md`](story.md) are rendered from the proxies and masks already on disk.**
`configs/story/media.json` names each GIF's sources and frame range; the renderer writes
`media/story/` with a `manifest.json` beside the GIFs and reports every source it had to fall back
from.

```bash
uv run battle-story-media check                                                              # list the manifest's entries
uv run battle-story-media render --manifest configs/story/media.json --output media/story    # --only <id> for one
```
