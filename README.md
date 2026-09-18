# Battle: video-understanding method lab

This is a no-training method lab for building a reviewable, machine-readable video
execution record. It includes versioned schemas, synthetic normalized fixtures, an
inference-free Rerun export, and a strictly bounded headless SAM3 smoke adapter—not
model accuracy claims.

## Where things stand

Start with [`docs/method-ledger.md`](docs/method-ledger.md). Its first half is the
short story: the original ask, a goals scorecard, a dated timeline from Sep 8 to the
present, and a plan-versus-actual list of where and why the work diverged. Its second
half holds the detailed per-run records with their claim boundaries. The plan the
project was measured against is preserved unedited in
[`docs/plan-2026-09-08-assembly-rerun-lab.md`](docs/plan-2026-09-08-assembly-rerun-lab.md),
and the research brief that preceded it, with the verbatim first request, is
[`battle_plan.agent.final.md`](battle_plan.agent.final.md).

In one paragraph, as of Sep 18: the repository has typed Pydantic manifests for clips,
runs, timing, coverage, and observations; an inference-free Rerun exporter; and nine
methods attempted on one Assembly101 recording. SAM3 via MuggledSAM is the only method
that completed both the RGB static and monochrome ego views, and only with human-seeded
masks and reviewed keyframe corrections from a browser calibration workspace; text
prompting failed on the ego view and the composite toy vocabulary failed everywhere. The
retained comparison unit is the first 60 seconds of a focused 92.7-second four-part
reassembly window, human-reviewed on Sep 17; later output is failure evidence. Over that
minute the review package (`battle-build-interaction-review-v4`) shows a per-target
ensemble segmentation reference (corrected SAM3 with provenance-tracked DAM4SAM fallback),
a WiLoR-primary stabilized hand layer with MediaPipe fallback, BoxMOT and Kineo body
context, label-free review metrics, and, since Sep 18, the dataset's own 60 fps hand poses
and 27 fine-grained action segments as external context. Grounded-SAM-2, SAMURAI, DAM4SAM,
CLIP + Drop-DTW and BoxMOT exist as smoke-tier arms; ATHENA's intrinsics blocker was
removed by fitting the dataset's own projection but no triangulation has been run. Human
QA dispositions are pending for every method, and no accuracy claim is made anywhere.

The rest of this file is the how-to: each section below gives the exact commands that
reproduce a stage. No recordings, annotations, or model weights are included. The SAM3
adapter executes only against the user-approved G2 proxies. `data/`, `runs/`, caches, and
Rerun outputs are ignored by Git.

## Hardware and claims policy

The intended future local target is a 16 GB NVIDIA RTX 5070 Ti. That is an operating
assumption, not a benchmark result. This repository makes no accuracy, throughput,
memory, dataset-coverage, or scientific-performance claim. Any future metric must name
its data split, protocol, environment, and uncertainty.

## Python environment

This project requires CPython 3.12 (`>=3.12,<3.13`) and uses `uv`; it must not run with
the system Python 3.14.

```bash
uv sync --python 3.12
uv run python --version
```

## MediaPipe Hands static-view baseline

`battle-mediapipe-hands` runs the official MediaPipe Tasks Hand Landmarker in VIDEO
mode against the approved focused RGB static proxy. The smoke contract is frames
`[0, 600)` / 20.0 seconds at 30 FPS. It emits one normalized observation per decoded
frame with 21 image-space landmarks, a landmark-derived box, handedness and its model
confidence. Public hand IDs are frame-local detector order, not identity tracks.
MediaPipe's selfie-oriented handedness is swapped for this unmirrored static camera,
then temporally voted over short internal tracks.

Download the pinned float16 v1 model bundle into ignored local storage and verify its
checksum:

```bash
mkdir -p models/mediapipe
curl -fL \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
  -o models/mediapipe/hand_landmarker_float16_v1.task
echo "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1  models/mediapipe/hand_landmarker_float16_v1.task" \
  | sha256sum --check
uv run battle-mediapipe-hands --seconds 20
```

The selected higher-recall smoke condition fuses the full frame with a fixed enlarged
workspace crop, maps crop landmarks back to full-frame coordinates, preserves primary
detections, removes near-identical duplicate skeletons, and fills missing hand slots
from the crop:

```bash
uv run battle-mediapipe-hands --seconds 20 \
  --roi 0.45,0.35,0.55,0.65 --roi-upscale 2 --include-full-frame \
  --min-detection-confidence 0.35 \
  --min-presence-confidence 0.35 \
  --min-tracking-confidence 0.35
```

Use `--seconds 60` with the same flags for the selected comparison run. That run
processed 1,800 frames in 30.668 seconds on CPU and produced at least one detection on
1,648 frames. Its 152 empty frames cluster under heavy hand/object occlusion, especially
in the final 20 seconds; they are retained as failure evidence rather than filled by
interpolation.

The run is headless. It writes `manifest.json`, `observations.jsonl`, a bounded input
video, `contact_sheet.png`, and `hands.rrd` under `runs/<run-id>/`. The Rerun recording
shows landmarks, hand skeletons, boxes, detection count, and mean handedness confidence.
Detection presence and handedness are model outputs without hand-pose ground truth, not
accuracy measurements.

## Audited exploratory queue

The classifications below are deliberately strict. An “integrated smoke” has Battle
normalization, a Rerun recording, and reproducible runner provenance. An “external partial”
may have native output or an inference-free Battle import, but it is not an end-to-end
Battle integration. None of these results supports an accuracy claim.

### WiLoR — external partial

`battle-wilor-hands` emitted 600 normalized rows (frames 0–599), 582 frames with a hand,
1,171 hand observations, 582 native JSON evidence files, and an inference-free `hands.rrd`.
Each emitted hand has 21 image-space joints and 21 non-metric camera-relative joints. The
worker measured 47.247 s (TTFU 5.885 s; 2,819,373,056 bytes peak allocated VRAM), correcting
the prior 57 s claim. The checkout used for inference is dirty, so its base revision alone
does not reproduce this run; it must not be promoted to an integrated smoke. WiLoR weights are
CC-BY-NC-ND and are not a multi-view reconstruction result.

### BoxMOT — integrated smoke

`battle-boxmot-track` invokes `YOLO(...)(frame)` per frame and passes those six-column
detections directly to `BotSort.update`; MuggledSAM IDs are not used. The audited run has 600
rows (frames 0–599), 557 frames/observations with tracks, IDs `boxmot-0` and `boxmot-1`, and
an inference-free `tracks.rrd`. Worker runtime is 7.483 s (TTFU 2.237 s; 74,796,544 bytes
allocated).

```bash
uv run battle-boxmot-track --seconds 20
```

Association remains conditional on the COCO `person` detector. Installed BoxMOT 25.0.0 and
Ultralytics 8.1.34 package metadata both declare AGPL-3.0; the local detector is
content-addressed but has no recorded upstream acquisition URL or separate model-license review.

### CLIP + Drop-DTW — integrated smoke

The worker directly calls OpenCLIP `encode_image`, `encode_text`, and
`dp.exact_dp.drop_dtw`; this is not synthetic alignment. It sampled 20 frames at 1 FPS, used
two Assembly101 coarse-label steps as declared weak supervision, and wrote cost 15.295 with
16/1 matched samples plus an inference-free scalar-only `alignment.rrd`. Worker runtime is
2.662 s. The rerun is offline and pins Drop-DTW `32ce9c8…` plus OpenCLIP 3.3.0,
`ViT-B-32`/`openai`, HF snapshot `a6f597a…`, and checkpoint SHA-256
`e6d1bd…2b7c31` (MIT package metadata). Its GT transcript remains weak supervision only.

```bash
uv run battle-drop-dtw-align --seconds 20
```

### Fine substep crop CLIP — agent-review experiment

Bounded follow-on for the focused static 20 s prefix: eleven substeps are checked in under
`configs/fine_substeps/assembly101_focused_static_first_20s_agent_labels.json` with provenance
tag `agent_authored_visual_review` (not Assembly101 GT). The runner samples 3 FPS, builds
WiLoR-primary hand/workspace crops with baseline SAM3 part boxes (MediaPipe fallback only on
WiLoR gaps), scores multi-prompt OpenCLIP ViT-B-32 crops, fuses weak motion/contact terms, and
aligns with both monotonic DP and Drop-DTW over eleven prototypes. Diagnostics compare against
agent labels only; coarse GT remains a separate weak anchor.

```bash
uv run battle-fine-substep-align --seconds 20
```

Preserved run `runs/fine-substep-static-20s-20260916t2255z/` writes `scores.json`,
`evaluation.json`, `boundary_contact_sheet.png`, `review_guide.md`, and inference-free
`fine_substep.rrd` with one embedded bounded video.

### Grounded-SAM-2 — bounded video smoke (`transformers_grounding_dino_plus_sam2_video_smoke`)

Battle now runs a checked-in bounded video smoke on the approved focused static proxy: frame 0
is initialized with the pinned Hugging Face `IDEA-Research/grounding-dino-tiny` revision
`a2bb814…` and prompt `hand.`, then SAM2.1 tiny (`configs/sam2.1/sam2.1_hiera_t.yaml`,
checkpoint SHA-256 `7402e0…be69`) propagates masks through the proxy prefix. The preserved
10-second / 300-frame run measures 15.9 s wall time and 6.37 GB peak VRAM; all 300 native
masks are nonempty with distinct content hashes. Normalized observations carry per-frame boxes,
scores, and `native/masks/*.png` references; `propagation.rrd` is inference-free.

The earlier one-frame external JSON (`score 0.953125` on an un-pinned HF revision) remains
importable via `battle-import-external-smoke grounded-sam2` and is not upgraded to a video run.

```bash
uv run battle-grounding-dino-sam2-video --seconds 10
```

### SAMURAI — bounded video smoke (`samurai_sam2_video_smoke`)

Battle runs a checked-in bounded video smoke on the approved focused static proxy with the same
deterministic frame-0 hand box `(881,446,152,129)` (xywh) used by Grounded-SAM-2 and DAM4SAM.
The worker selects `configs/samurai/sam2.1_hiera_t.yaml` (`samurai_mode: true`) and SAM2.1 tiny
(checkpoint SHA-256 `7402e0…be69`). Frame count is source-aligned: `round(seconds × 30 FPS)` —
300 frames for 10 s, 600 for 20 s — not the earlier unbounded 602-frame decode.

The preserved 10-second run measures 14.8 s wall time, 7.0 s TTFU, and 0.82 GB peak VRAM; all
300 native masks are nonempty with distinct content hashes. Normalized observations carry per-frame
boxes and `native/masks/*.png` references; `samurai.rrd` is inference-free.

```bash
uv run battle-samurai-video --seconds 10
```

### DAM4SAM — bounded video smoke (`dam4sam_video_smoke`)

Battle runs a checked-in headless `DAM4SAMTracker` smoke (`sam21pp-T`) on the same focused static
proxy and frame-0 hand box seed. Initialization bypasses the interactive `BoxSelector` and the VOT
mask prompt: frame 0 uses bbox→`estimate_mask_from_box`→`add_new_mask`; no extra initialization
frames are consumed. Native `return_all_masks` / `add_to_drm` logic remains DAM4SAM-specific.

Execution uses pyenv `samurai` (torch 2.11+cu128) because upstream torch 2.1+cu121 is incompatible
with sm_120 on this GPU. The preserved 10-second run measures 12.0 s wall time, 2.6 s TTFU, 0.81 GB
peak VRAM, and 22 DRM memory additions over 300 source-aligned frames. `dam4sam.rrd` is
inference-free. This does not prove distractor-scene behavior on identical parts.

The earlier 602-frame external mask import remains available but is not upgraded to the integrated
runner:

```bash
uv run battle-dam4sam-video --seconds 10
uv run battle-import-external-smoke dam4sam
```

### Kineo — `kineo_nlf_only_partial`

Checked-in config `configs/kineo_nlf_headless_only.yaml` runs the headless NLF-only path with
deterministic `best_bbox_only` person selection (highest detector confidence, not largest area).
`battle-kineo-nlf` records Kineo `HEAD` plus dirty diff fingerprint, model/checkpoint identities,
proxy checksum, PKL validation, native PKL hashes, normalized boxes, and the first 55 NLF body joints
in image-normalized coordinates. It is not SfM, metric 3D, BVH, or full Kineo.

```bash
uv run battle-kineo-nlf --seconds 20
```

Latest bounded rerun: `runs/kineo-nlf-headless-20s-20260916t0540z/` (456/600 frames with
NLF body joints) with inference-free `kineo_nlf_partial.rrd`. `battle-import-external-smoke`
and `battle-kineo-nlf` unpickle only caller-controlled local native artifacts; parsing is
structural and intended for trusted local evidence, not arbitrary uploads.

## Exploratory methods still blocked

- **ATHENA:** blocked for Assembly101 real data. HTTP Range inspection of official
  `cvml-nus/assembly101` `AssemblyPoses.zip` (72 GB, not downloaded whole) found extrinsics,
  positions, timestamps, and landmarks for the approved recording, but no intrinsics member.
  ATHENA requires per-camera intrinsics. Fixture-only smoke:
  `uv run battle-athena-fixture-smoke`. Evidence: `docs/athena_hf_calibration_probe.json`.
- **Deferred without integration:** LM-EEC, ObjectRelator, Qwen VLMs, supervised TAS, long-video
  VLMs.

## Fixed SAM3 core-method smoke

`battle-muggled-smoke` is a headless adapter based on MuggledSAM's
`simple_examples/video_segmentation_multiplexed.py`, not its interactive `run_video.py`.
It accepts only proxy frames `[0, 300)` (exactly 10.0 seconds at 30 FPS), processes one
view at a time, and uses a single CUDA-visible model with bounded continuous tracker
memory: one prompt-memory entry and four frame-memory entries. Object IDs are never
intentionally reset. Its fixed SAM3 text concepts are `hand`, `yellow toy body`, and
`toy wheel`.

Run either approved view separately:

```bash
uv run battle-muggled-smoke --view static-c10379
uv run battle-muggled-smoke --view ego-hmc21110305
```

### Aligned static hybrid candidate

The aligned four-label static candidate is explicitly hybrid, not pure zero-shot G3:
`left_hand`, `right_hand`, and `yellow_toy_top` use the three-target text config, while
`black_toy_top_base` uses one user-reviewed frame-zero mask. The versioned hybrid contract
pins both inputs, their fingerprints, and the canonical multiplex ordering. The user
approved smoke `muggledsam-sam3-smoke-hybrid-static-static-c10379-20260915t005256z`
with “Looks good!” at approximately 2026-09-14 20:53 EDT / 2026-09-15 00:53 UTC.
This followed rejection of all five black-base text-prompt variants: none visually selected
the intended base without unacceptable fixture/identity errors.

```bash
uv run battle-muggled-smoke \
  --view static-c10379 \
  --hybrid-config configs/muggledsam_static_aligned_hybrid.json
```

That approval produced full candidate
`muggledsam-sam3-g3-full-hybrid-static-static-c10379-20260915t005919z`. Its manifest
fingerprints the exact reviewed smoke and contact sheet. This route records initialization
provenance and `ground_truth_accuracy_claim: false`; continuity and tracker diagnostics are
not segmentation-accuracy evidence.

## G3 decision and candidate outcome

The original G3 review approved only the static RGB view and rejected the first ego
monochrome view as a likely grayscale-domain stress case. The historical static candidate is
`runs/muggledsam-sam3-g3-full-static-c10379-20260909t030710z/`; it covers all 5,400
analysis frames / 180.0 seconds, but uses the older three-target zero-shot contract and is
not the aligned canonical baseline. It remains historical evidence. The aligned canonical
static result is the four-target hybrid candidate named above; the selected alternate ego
view later received its own full four-target manual-seed baseline.

Re-running the aligned full candidate requires the explicit G3 flag and exact reviewed
smoke approval provenance:

```bash
uv run battle-muggled-smoke \
  --view static-c10379 \
  --hybrid-config configs/muggledsam_static_aligned_hybrid.json \
  --g3-full-static \
  --approved-smoke-manifest \
    runs/muggledsam-sam3-smoke-hybrid-static-static-c10379-20260915t005256z/manifest.json \
  --human-approved-at 2026-09-15T00:53:00+00:00 \
  --human-approved-by user \
  --human-approval-statement "Looks good!"
```

## Four-part static reassembly experiment

The current physical-part vocabulary supersedes the earlier composite
`yellow_toy_top` / `black_toy_top_base` interpretation for this experiment only:
`chassis`, `interior`, `rear_body`, and `cabin`, each meaning visible surface only.
The approved static proxy covers source `[190.0, 386.7)` seconds at 30 FPS.

The exploratory full run
`muggledsam-sam3-four-part-static-full-exploratory-static-c10379-20260916t012945z`
processed all 5,901 frames in 474.639 seconds, reached 2,209,013,760 bytes peak
allocated VRAM, and wrote 23,418 per-object masks. Its reviewed corrections were at
proxy frames 36, 65, and 162. An all-frame audit found mask IoU above 0.5 for
chassis/cabin on 77 frames, interior/cabin on 18, chassis/rear body on 9, and
interior/rear body on 7. Those overlaps, together with visual identity drift already
seen in the 20-second pilot, mean stable tracker IDs must not be interpreted as stable
physical identities.

Inspect the ignored local recording with:

```bash
uv run rerun \
  runs/muggledsam-sam3-four-part-static-full-exploratory-static-c10379-20260916t012945z/four_part_static_full_exploratory.rrd
```

The next experiment begins at old proxy frame 3120 (old proxy time 104.0 seconds;
source time 294.0 seconds). A new 2,781-frame / 92.7-second proxy maps that boundary to
frame 0, preserving the worker and calibration system's frame-0 initialization contract.
The exploratory run's masks are not transferable initialization evidence.

Build or verify the focused proxy:

```bash
bash scripts/create_assembly101_four_part_reassembly_proxies.sh
```

Create a fresh calibration at focused frame 0, with three provisional review points
across reassembly:

```bash
uv run battle-muggled-calibration-web \
  --config configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json \
  --view static-c10379 \
  --manual-seed-target-config \
    configs/muggledsam_static_four_part_reassembly_focused_manual_seed.json \
  --correction-policy \
    configs/muggledsam_static_four_part_reassembly_focused_correction_policy.json \
  --timestamps 0,30,60,90 \
  --output-dir "runs/muggledsam-sam3-four-part-focused-calibration-$(date -u +%Y%m%dt%H%M%Sz)"
```

After reviewing and finalizing that workspace, run the focused profile:

```bash
uv run battle-muggled-smoke \
  --config configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json \
  --view static-c10379 --four-part-static-focused --max-side-length 720 \
  --multi-keyframe-correction-schedule \
    runs/<focused-calibration>/multi_keyframe_correction_schedule.json
```

Completed focused run:
`muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260916t020716z`.
It processed 2,781/2,781 frames in 194.855 seconds, reached 2,209,960,448 bytes peak
allocated VRAM, and wrote 11,042 masks. Output coverage was 2,775 chassis, 2,774
interior, 2,751 rear-body, and 2,742 cabin frames. The all-frame geometric screen found
only one pair/frame above 0.5 mask IoU: chassis/interior at frame 2381 (0.6891).
Every other pair had zero frames above 0.5, including chassis/cabin. This is substantially
cleaner than the exploratory run, but still requires visual identity review.
For the follow-up at frames 327 and 1235, use
`configs/muggledsam_static_four_part_reassembly_focused_correction_policy_v2.json`;
it permits up to five later keyframes per target without changing the v1 policy
fingerprinted by the completed run.

The v2 rerun
`muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260916t023700z`
completed all 2,781 frames with the requested corrections. They improve their local
failure samples, but rear-body and cabin masks later merge around frame 2101 (maximum
pairwise IoU 0.9376), so this rerun is not yet an accepted tracking result.

```bash
uv run rerun \
  runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260916t023700z/four_part_static_focused_reassembly.rrd
```

Inspect the initial focused run separately:

```bash
uv run rerun \
  runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260916t020716z/four_part_static_focused_reassembly.rrd
```

### Focused monochrome ego counterpart

The aligned `HMC_21110305` run uses a separate focused G2 contract and entirely fresh
human masks; no static-view mask is transferred. Completed run
`muggledsam-sam3-four-part-ego-focused-reassembly-ego-hmc21110305-20260916t031515z`
processed 2,781 frames in 193.775 seconds with 2,139,766,784 bytes peak allocated VRAM
and 10,509 masks. Output coverage was 2,779 chassis, 2,712 interior, 2,645 rear-body,
and 2,373 cabin frames. No pair exceeded 0.5 mask IoU, but the large cabin/rear-body
output gaps and later assembled-object appearance still require visual identity review.

```bash
uv run rerun \
  runs/muggledsam-sam3-four-part-ego-focused-reassembly-ego-hmc21110305-20260916t031515z/four_part_ego_focused_reassembly.rrd
```

### Selected first-minute two-view comparison

Human review bounded the usable comparison to focused frames `[0,1800)` / 60.0 seconds.
The selected static input is the v2 correction run (`…20260916t023700z`); its known
rear-body/cabin merge begins after this window. The ego input is `…20260916t031515z`.
The builder truncates both run-local videos to exactly 1,800 CFR frames, fingerprints
those derived assets, merges the selected 60-second MediaPipe run into the static
observations, and packages existing masks, landmarks, skeletons, boxes, and confidence
traces without inference:

```bash
uv run battle-build-ego-static-comparison --focused-first-minute
uv run rerun \
  runs/four-part-focused-first-minute-comparison/four_part_focused_first_minute_ego_static_comparison.rrd
```

The default MediaPipe input is
`mediapipe-hands-static-60s-fused-dedup-th035-20260916t0430z`; override it with
`--hands-run runs/<run-id>`. The resulting 81.0 MB recording keeps the static RGB view
as the hand-pose claim and the monochrome ego view as the object-tracking stress test.

## Fixed human QA hard gate

`FixedTimestampHumanQARecord` formalizes the plan's pre-accuracy visual gate: exactly one
human-selected easy manipulation and one human-selected hard/occluded manipulation on the
source clock. Each checkpoint requires fingerprinted visual evidence. A non-pending
`pass`/`flag`/`fail` requires reviewer identity and a timezone-aware review time; the
overall status is validated as a conservative derivation, and
`ground_truth_accuracy_claim` is always `false`. Tracker IoU predictions and object scores
cannot populate this record.

The plan specified the easy/hard rule but the approved clip manifests did not preserve
literal checkpoint times. The human selected mask-bearing source frames 14,868 / 21,732
(analysis frames 984 / 4,416), and the two canonical SAM3 records are now tracked as
pending under
[`docs/qa/`](docs/qa/README.md). `battle-prepare-human-qa` validates only the explicitly
supplied completed runs, creates a lightweight two-checkpoint sheet without inference,
fingerprints it, and writes pending JSON records. It refuses to overwrite a reviewed record.

The candidate manifest records the source SHA-256
`450731ebbb50f46cf8279383e4737db6d76e967f23580d3555de1888b78a9db9`, proxy SHA-256
`ea9243591f6e8716ad7fe83f404e01ac45414071e9f06bc5ee68de29fcc543dc`, and config
SHA-256 `e6bb42376e72c8bdd58151afb748d4a4d0fbd3f6e097ae7190e70fe1a162fe36`.

## Bounded ego monochrome diagnostic

The original ego zero-shot smoke remains the baseline:
`runs/muggledsam-sam3-smoke-ego-hmc21110305-20260909t025910z/`. It uses the
unchanged three text concepts `hand`, `yellow toy body`, and `toy wheel`; its continuous
`hand` output is visually poor and the two object concepts had no initialization output.
It was not overwritten or reinterpreted.

The reproducible first-300-frame decode characterization is
`runs/muggledsam-sam3-ego-diagnostic-20260909t032450z/image_characterization.json`.
Every decoded B/G/R sample was equal (pairwise MAE 0, correlation 1.0), so the input has
no retained chroma. Aggregate luminance is p1/p50/p99 = 9/78/255, with 1.61% exact 255
values. This supports testing contrast representation, not pseudo-colorization.

`configs/muggledsam_ego_conditions.json` is schema-validated and declares two distinct
300-frame conditions, both with one continuous tracker stream:

- `CONTRAST-NORMALIZED`: exact text concept `hand`; each decoded frame is converted to
  gray, robustly rescaled from its p1–p99 range to 0–255, processed with CLAHE
  (`clipLimit=2.0`, 8×8 tiles), then replicated to BGR/RGB.
- `MANUAL-SEED`: original decoded BGR and a direct SAM3 first-frame tracking box prompt,
  not a detector result. The seed is frame 0 pixel box `(380,300)–(555,435)` (normalized
  `(0.3987408,0.4172462)–(0.5823715,0.6050070)`) over the central visible hand holding
  the small part. It is explicitly manual and uses `hand` only.

MuggledSAM SAM3.1 exposes direct normalized box/foreground/background prompts through
`tracking.encode_prompt_memory(...)`; the adapter uses that supported route for the
manual condition. The API does not expose a detector confidence for this path, so the
manual condition's initial observation stores `1.0` only as an initialization sentinel
and records that limitation in its worker settings.

Both runs are schema-valid and include separate `manifest.json`, normalized observations,
input-video Rerun exports, worker results, and 5-FPS external masks:

- `runs/muggledsam-sam3-smoke-contrast-normalized-hand-text-ego-hmc21110305-20260909t032357z/`
- `runs/muggledsam-sam3-smoke-manual-seed-hand-box-ego-hmc21110305-20260909t032421z/`

The factual 300-frame proxy measures and a single 0.000/5.000/9.967-second comparison
sheet are at
`runs/muggledsam-sam3-ego-diagnostic-20260909t032450z/proxy_metrics.json` and
`runs/muggledsam-sam3-ego-diagnostic-20260909t032450z/ego_comparison_contact_sheet.png`.
All three runs emitted one ID through 300/300 frames with no output gaps or ID restarts;
the transformed text condition did not improve initialization count or these continuity
proxies. Visual QA does not establish target accuracy without labels, and neither
condition is sufficiently better than the known poor zero-shot track to recommend a
60-second ego candidate. No 60- or 180-second ego inference was started.

Each attempt writes a gitignored `runs/<run-id>/manifest.json`, normalized
`observations.jsonl`, complete worker stdout/stderr logs, and runtime settings. On a
successful model run it also writes `smoke.rrd`: the input proxy is logged once,
detections are logged at every analysis frame, and masks are external PNG artifacts
written every analysis frame by default. This cadence is an internal worker setting
(`muggled_worker.py` defaults `--mask-period-frames` to 1), not a
`battle-muggled-smoke` command-line option; runs before Sep 13 used
`mask_period_frames=6`, or 5 FPS at the 30 FPS analysis clock. The RRD preserves those
references, embeds each object's mask as a translucent RGBA cut-out on every frame under
`views/<view>/masks/<object_id>`, and keeps a class-labelled `SegmentationImage` at 1 Hz
that is off in the default view. Boxes are derived from the mask's dominant connected
components (those at least 20% of the largest component's area), so stray mask speckle
does not inflate them; each run records the rule in its `runtime_settings`.

## Bounded ego viewpoint screen

The e1/e2/e4 screen adds three new monochrome ego recordings without changing the
original static/e3 G2 manifest, e3 baseline, or any approved full-duration decision.
`configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json` is the separate,
schema-valid G1/G2 contract. Its only permitted input videos are the three named raw
recordings and the exact `[215.000, 395.000)` proxy interval.

Create or verify just these proxies with:

```bash
bash scripts/create_assembly101_ego_viewpoint_screen_g2_proxies.sh
```

The script is idempotent and validates each existing or generated proxy as H.264,
`yuv420p`, 954×720, 30/1 FPS, 5,400 frames, 180.000 seconds, and audio-free. It never
downloads assets or runs a model.

Each screen run is exactly the original decoded BGR image representation, the three
fixed text concepts `hand`, `yellow toy body`, and `toy wheel`, and one bounded
continuous tracker stream. Run them sequentially:

```bash
for view in ego-hmc21176875 ego-hmc21176623 ego-hmc21179183; do
  uv run battle-muggled-smoke \
    --config configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json \
    --view "$view"
done
```

The completed screen reuses the preserved e3 original zero-shot smoke only for review
comparison. `docs/run-report-assembly101-ego-viewpoint-screen.md` records the
view-separated facts and limited human-review ranking. No aggregate ego claim,
ground-truth accuracy claim, or 60-/180-second ego run follows from this screen.

### Approved e4-only 60-second candidate

The approved continuation is limited to e4 / `ego-hmc21179183` proxy frames `[0,1800)`
(60.0 seconds). `--g4-e4-candidate` rejects every other view or frame count and keeps
the original decoded BGR input, exact three text concepts, CUDA-0 sequential setup, and
bounded continuous memory policy.

```bash
uv run battle-muggled-smoke \
  --config configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json \
  --view ego-hmc21179183 --g4-e4-candidate
```

The completed candidate is
`runs/muggledsam-sam3-g4-e4-candidate-ego-hmc21179183-20260909t033519z/`. Its
machine-readable continuity report and fixed 0.000/30.000/59.967-second QA sheet are
inside that directory. This is an e4-only monochrome candidate—not an accuracy result
or a general ego-performance claim—and it does not authorize the remaining 120 seconds.

The adapter invokes the explicitly configured MuggledSAM interpreter
`/home/nick/.pyenv/versions/muggled_sam/bin/python` and source
`/home/nick/src/muggled_sam`; it never edits that environment/repository or downloads
weights. Missing/incompatible runtime components or unavailable checkpoints are
recorded as blocked attempts, with unavailable measurements explicitly null.

## Interactive e4 box-prompt calibration

`battle-muggled-calibrate` is a local, user-operated calibration utility for the
currently selected monochrome e4 proxy (`HMC_21179183_mono10bit`). It is deliberately
separate from the zero-shot baseline and does not run the video tracker. It displays
the exact 954×720 decoded proxy frame at the default proxy timestamps 0, 10, 30, and
50 seconds; box coordinates are returned in that full-resolution pixel space.

Launch it from this repository only when a desktop session is available:

```bash
uv run battle-muggled-calibrate
```

At each timestamp, press Enter in the terminal to start or `s` to skip it; `q` ends the
session and opens the saved-review menu. The OpenCV window selects **one** rectangle:
drag it, then press Space or Enter to accept it, or `c` to cancel it. Once that window
closes, the terminal always asks what happens next—`a` adds another box, `d` decodes
the current boxes, `r` clears those unsaved boxes and starts again, and `s` skips that
timestamp. This deliberately does not use OpenCV's multi-ROI mode, whose completion
key differs from its per-rectangle accept key.

For every decoded box, enter `hand`, `yellow toy body`, `toy wheel`, or a custom label.
The review window saves its contact sheet **before** asking whether to accept it, so a
sheet remains available for a skipped or redrawn box. It uses a 2×2 candidate grid:
each candidate panel has a full-frame context view at left and a padded, aspect-preserved
prompt/mask ROI at right. Cyan is the translucent mask fill, magenta is its crisp
contour, and amber is the original prompt rectangle. The context view is the place to
spot a mask spilling across the arm or table; the ROI is for judging the local boundary.

Each panel also reports the candidate index, deterministic decoder tie-break status,
and that human selection is still pending. Its `IoU estimate` is a decoder output—not
measured accuracy. `mask` is the source-frame mask area and fraction; `in prompt` is
the count and fraction of all mask pixels that fall inside the prompt rectangle. These
are geometric diagnostics, not accuracy scores and they do not select a candidate.

After closing the image review window, enter `a` to accept and save the evidence, `r`
to redraw only that box, or `s` to skip it. Mark only deliberately chosen accepted boxes
as eligible for finalization. The terminal's saved-review menu can review, clear, or
toggle that eligibility; use `--resume --output-dir <existing-run-directory>` to
continue a session without removing already saved candidates. For example, resume an
interrupted run with:

```bash
uv run battle-muggled-calibrate --resume \
  --output-dir runs/muggledsam-sam3-e4-box-calibration-20260909t035704z
```

Use another ordered timestamp set if needed (all values are proxy-relative seconds):

```bash
uv run battle-muggled-calibrate --timestamps 0,12.5,30,55
```

The gitignored `runs/<calibration-id>/calibration_manifest.json` records the approved
G2 config and proxy hashes, exact pixel and normalized box, exact proxy/analysis/source
time, target label, tool version, all image-decoder candidates, deterministic
maximum-IoU (lowest-index tie-break) choice, and paths to source-sized PNG masks and
review contact sheets. Decoder logits are mapped from their square grid onto the complete
954×720 source frame with independent x/y source scaling and `align_corners=False`;
the review display never stretches an ROI. No image or mask payload is embedded in an
RRD.

Finalization requires explicit user-selected candidate IDs and writes a proposed,
non-authoritative configuration only; it neither claims ground-truth accuracy nor
starts tracking:

```bash
uv run battle-muggled-calibrate \
  --finalize runs/<calibration-id>/calibration_manifest.json \
  --candidate-id t000300-b01
```

The proposal names each selected target and reference frame, and records important
compatibility limits: MuggledSAM's direct box-prompt tracking memory is non-multiplexed,
and a box after frame 0 needs a future tracking initialization policy. The image API
returns mask candidates and IoU estimates, but not stability scores; its IoU estimates
are model outputs, not measured accuracy. This manual-seed condition cannot be compared
with the zero-shot condition until a later human G gate defines that comparison.

### Rapid browser workspace

`battle-muggled-calibration-web` replaces the repeated OpenCV/terminal ROI loop with a
loopback-only browser workspace. Start a new gitignored `runs/<calibration-id>/`
directory for each calibration; historical run directories are evidence and must not be
resumed or edited. The workspace samples only the selected timestamps (default
`0,10,30,50` seconds); it does not inspect all 1,800 candidate frames and it never
launches tracking.

```bash
uv run battle-muggled-calibration-web
# prints e.g. Rapid calibration workspace: http://127.0.0.1:8765/
```

The Battle `uv` process runs the HTTP server. It starts the configured isolated
`/home/nick/.pyenv/versions/muggled_sam/bin/python` worker once, keeps that image decoder
warm, serializes decode jobs, and records worker stderr at
`runs/<calibration-id>/worker.stderr.log`. The server binds only `127.0.0.1`; it does not
expose the raw video, model, or workspace to the network. Source-frame previews, masks,
per-candidate review panels, and contact sheets remain in that ignored run directory.

Usage:

1. Select a filmstrip timestamp or enter any proxy timestamp, then load its source
   frame. The slider spans every proxy frame and loads by exact frame index after a
   short debounce. Frames outside the configured calibration set open in **Browse only**
   mode: prompt, decode, and mask-review edits are disabled in both the browser and
   backend. Use **Enable labeling on this frame** to persist the exact browsed frame as
   a new calibration option. The canvas preserves the complete 954×720 aspect ratio. Use scroll to zoom,
   Shift/Alt/middle-button drag to pan, `B` to draw/edit a box, `F` to add a green
   foreground point, and `N` to add a pink background-exclusion point. Drag a marker to
   move it, right-click it to remove it, or use **Clear selected points**. Prompt edits
   autosave atomically.
2. Set the target label, draw and select its box, then add target-specific points. The
   stored source-pixel clicks and normalized MuggledSAM `boxes`, `fg_points`, and
   `bg_points` payload are retained with the pending prompt and decoded candidate.
   The right panel shows the resulting compact manifest difference.
3. Enable **Live decode** to preview the selected prompt after a configurable debounce,
   or disable it and use **Decode all pending on this frame** for a manual frame batch.
   The status table keeps every configured frame/target slot visible. Select a slot to
   show its mask on the canvas, then use keys `1`–`4` to accept a decoder option. `×`
   clears an accepted mask or preview; rejected candidates remain visible with `↶` so
   they can be restored. Model IoU and prompt-overlap values are diagnostics, not
   segmentation-accuracy measures.
4. Only an explicitly human-selected and accepted **frame-0** mask can be marked eligible
   and appear in the proposal list. Later timestamps are decoder checks only. Creating a
   proposal writes a non-authoritative JSON file and does not start tracking.

#### Display-only view aids (VIGRA)

The canvas has a **Display-only view aids** panel for reading the monochrome egocentric
frames. Every control in it is a view aid and nothing else: it never changes the pixels
sent to the SAM3 decoder, the extracted frame files, mask or review artifacts, prompts,
provenance hashes, or anything written into a calibration manifest, proposal, or
correction schedule. `tests/test_calibration_view_filters.py` pins that guarantee — one
test decodes the same prompts with every view aid on and with all of them off and
asserts the worker payload and the frame bytes are byte-identical, and another
fingerprints the whole run directory across repeated renders.

Rendering happens server-side from the already-extracted frame image, read-only, and
never writes anything. Nothing below trades away a VIGRA result for a cheaper
approximation; the operators and their arguments are unchanged.

- Results are memoized **per operator and parameter set**, not per whole stack, so
  toggling or re-tuning one aid leaves the other eleven untouched.
- Independent operators run concurrently — vigranumpy releases the GIL, so they really
  do overlap.
- The frame is decoded once per working resolution rather than on every request, and the
  cache is pinned to the frame file's size and modification time so a re-extracted frame
  is never served stale.
- Brightness and contrast alone are applied in the browser with the same formula, so
  they never touch the network. Enabling any threshold, edge, or corner aid moves the
  whole chain back to the server, because those detectors read the toned image.
- A **Working resolution** control runs the operators on a half, third, or quarter-size
  image and upscales unsmoothed for display. It is a named choice showing the pixel size,
  never an invisible speed-up, because the operators genuinely see less detail.

Measured on a 954×720 frame (median wall time for the browser's POST):

| Interaction | Before | Full resolution | Half resolution |
| --- | --- | --- | --- |
| Cold render, all twelve operators | 210 ms | 86 ms | 21 ms |
| Cold render, Canny + corner response | 70 ms | 19 ms | 6 ms |
| One slider tick with all twelve on | 200 ms | 50 ms | 14 ms |
| Toggling one aid with all twelve on | 200 ms | 1 ms | 1 ms |
| Brightness or contrast alone | 37 ms | no request (~3 ms in-browser) | — |

Slider edits are debounced and superseded requests are cancelled. View settings are
stored as a browser `localStorage` preference only.

Each operator is an independent on/off toggle with its own sliders, composed in one
fixed order — brightness → contrast → adaptive threshold → edge detectors → corner
detectors — so toggling controls in any sequence gives the same image. Edge results are
painted into the base image and corner results are drawn as rings, both underneath the
mask overlay, prompt box, and foreground/background markers so those stay legible.

Edge and corner operators call [VIGRA](https://ukoethe.github.io/vigra/) 1.12.4
directly. VIGRA is **not on PyPI** and is therefore absent from `uv.lock`; it is built
from source into `.venv` following [`docs/vigra-build.md`](docs/vigra-build.md), which
pins the tag, commit, Boost version, and CMake flags. When VIGRA is missing the
workspace still starts: `/api/view-filters` reports it, the VIGRA-backed controls render
disabled with the reason, and brightness, contrast, and adaptive threshold keep working.
Nothing is silently substituted for a VIGRA result.

| Control | Sliders | Backend |
| --- | --- | --- |
| Brightness | amount | NumPy — not a VIGRA operator |
| Contrast | amount | NumPy — not a VIGRA operator |
| Adaptive threshold | block size (odd), constant C, method (mean/gaussian) | NumPy — VIGRA has no adaptive threshold; the gaussian method takes its local background from `vigra.filters.gaussianSmoothing` |
| Canny edges | scale σ, gradient threshold, variant (thinned/plain) | `vigra.analysis.cannyEdgeImageWithThinning` / `cannyEdgeImage` |
| Zero crossings (LoG) | scale σ, jump threshold | `vigra.filters.laplacianOfGaussian` + NumPy crossing marker |
| Shen–Castan (ISEF) | scale σ, gradient threshold | `vigra.analysis.shenCastanEdgeImage` |
| Boundary tensor energy | scale σ, relative threshold | `vigra.filters.boundaryTensor2D` |
| Corner response function | scale σ, k-free relative threshold | `vigra.analysis.cornernessHarris` |
| Beaudet | scale σ, relative threshold | `vigra.analysis.cornernessBeaudet` |
| Rohr | scale σ, relative threshold | `vigra.analysis.cornernessRohr` |
| Förstner | scale σ, relative threshold | `vigra.analysis.cornernessFoerstner` |
| Tensor corner/junction | scale σ, relative threshold | `vigra.analysis.cornernessBoundaryTensor` |

Zero crossings is the one partial case: vigranumpy 1.12.4 does not export
`vigra::zeroCrossings`, so the Laplacian is VIGRA's but the crossings are marked at
whole-pixel positions in NumPy instead of VIGRA's sub-pixel ones. The panel says so.

Keyboard shortcuts, in addition to the existing `B`/`F`/`N` and `1`–`4`:

| Key | Action |
| --- | --- |
| `V` | Bypass all view aids (keeps their settings) |
| `R` | Reset view — zoom, pan, and every view aid back to defaults |
| `T` | Toggle adaptive threshold |
| `E` | Toggle Canny edges |
| `C` | Toggle the corner response function |

To re-measure any of the timings above:

```bash
uv run python scripts/profile_view_route.py \
  --image runs/<calibration-run>/results/frames/frame-000000.jpg [--divisor 2]
```

For a no-model static route check, pass `--no-worker` to the calibration web command:

```bash
uv run battle-muggled-calibration-web --no-worker
```

Frame decoding and masks will intentionally report that the decoder is offline. The
fixture tests exercise the persistent JSONL protocol, queue/persistence, and API/static
contract without CUDA, OpenCV, or a checkpoint. Actual image decoding and browser
interaction still require the local MuggledSAM checkpoint, CUDA runtime, and a desktop
browser; they are not model-accuracy or human-GUI test results.

### Fourth e4 object-semantic target set

Keep every existing three-target configuration, proposal, calibration manifest, and run
as a historical baseline. Do not resume or alter them. To create a separate fourth
target-set workspace on the same approved source/proxy, launch:

```bash
uv run battle-muggled-calibration-web \
  --config configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json \
  --manual-seed-target-config configs/muggledsam_e4_left_hand_right_hand_yellow_toy_top_black_toy_top_base_manual_seed.json \
  --output-dir "runs/muggledsam-sam3-e4-web-calibration-left-hand-right-hand-yellow-toy-top-black-toy-top-base-$(date -u +%Y%m%dt%H%M%Sz)"
```

This starts calibration only; it does not run tracking. At proxy frame 0, draw and
decode one prompt for each exact label: `left_hand`, `right_hand`, `yellow_toy_top`, and
`black_toy_top_base`. Inspect the source-context and padded-ROI card for each returned
mask, explicitly select and accept one mask per label, and mark each selected frame-0
mask eligible. Review `right_hand` anew in this workspace—do not reuse a right-hand or
any other candidate from an earlier calibration. Create a proposal only after all four
eligible candidates are checked. The policy rejects any other count, duplicate, missing,
or differently named target while allowing the four newly assigned candidate IDs to vary.

For a hand mask that includes a right-hand shadow: select `right_hand`, draw and select
its box, press `F` and click 1–3 unambiguous hand pixels, then press `N` and click 1–3
pixels in the unwanted shadow. The green `+` and pink `×` markers should appear on the
selected box before decoding. Decode that prompt again, inspect its candidates, and
explicitly accept only the corrected mask; reject the earlier candidate only after
unaccepting it.

After finalization, replace `<new-calibration-directory>` with the directory printed by
the workspace before starting the separate 10-second manual-seed smoke:

```bash
uv run battle-muggled-smoke \
  --config configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json \
  --view ego-hmc21179183 --start-frame 0 --max-frames 300 \
  --manual-seed-proposal runs/<new-calibration-directory>/proposed_tracking_prompt.json \
  --manual-seed-target-config configs/muggledsam_e4_left_hand_right_hand_yellow_toy_top_black_toy_top_base_manual_seed.json
```

The runner validates exactly these four human-selected frame-0 mask artifacts and uses
their policy order as multiplex slots 0–3 in one continuous
`encode_prompt_memory_from_mask` initialization. It does not reuse the three-target
proposal or create independent tracker streams.

### Four-target multi-keyframe human correction

Keep every historical one-frame calibration, proposal, configuration, and run unchanged.
For a new four-target correction schedule, select **2–4 representative frames** from the
first bounded 10 seconds: always include frame 0, then choose later frames at known
action/failure transitions. The later frames must map below analysis frame 300. Start a
new workspace (do not resume an earlier one):

```bash
uv run battle-muggled-calibration-web \
  --config configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json \
  --manual-seed-target-config configs/muggledsam_e4_left_hand_right_hand_yellow_toy_top_black_toy_top_base_manual_seed.json \
  --correction-policy configs/muggledsam_e4_four_target_keyframe_correction_policy.json \
  --timestamps 0,2.5,5,8 \
  --output-dir "runs/muggledsam-sam3-e4-four-target-keyframes-$(date -u +%Y%m%dt%H%M%Sz)"
```

At frame 0, label, review, human-select, accept, and mark the required initialization
mask for each exact target: `left_hand`, `right_hand`, `yellow_toy_top`, and
`black_toy_top_base`. At each later selected frame, label only targets that are visible;
after review, explicitly select, accept, and check **include as a later correction
keyframe**. The right panel lists both roles. Click **Finalize tracking plan** only after
the frame-0 set and any intended later corrections are checked. It writes
`multi_keyframe_correction_schedule.json`, binding the calibration manifest, policy,
target config, selected masks, stable object IDs/slots, exact frame indices, and SHA-256
fingerprints. Duplicate target/frame slots, duplicate masks, missing frame-0 slots, and
out-of-policy correction counts are rejected.

Finalizing seals the plan: an amber banner appears above the workspace, and candidate
review (mask options, eligibility, reject, unaccept, restore, and Finalize itself) is
disabled with that reason, so a finalized artifact can never be silently rewritten. To
keep iterating, click **Reopen for editing** in that banner. It changes nothing on disk
except the calibration manifest, which records the finalized files as a superseded plan
revision; their bytes and their calibration-manifest SHA-256 stay exactly as reviewed.
Finalizing again writes the next revision beside them
(`proposed_tracking_prompt.r2.json`, `multi_keyframe_correction_schedule.r2.json`, and so
on), each fingerprinting the manifest it was actually written with, and the manifest's
`final_proposal_uri` and `final_correction_schedule_uri` point at the newest revision.
Point downstream runs at those URIs rather than assuming the unsuffixed names.

Run only the bounded test with the new schedule:

```bash
uv run battle-muggled-smoke \
  --config configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json \
  --view ego-hmc21179183 --start-frame 0 --max-frames 300 \
  --multi-keyframe-correction-schedule \
  runs/<new-calibration-directory>/multi_keyframe_correction_schedule.json
```

MuggledSAM exposes batch mask-to-prompt-memory encoding and multiplex stepping, but no
documented per-slot prompt-memory patch API. The policy therefore has one conservative,
explicit meaning: at a correction frame the runner first obtains all four current masks,
replaces the selected human-corrected slots in that source-sized batch, writes one
replacement `encode_prompt_memory_from_mask` entry, and clears automatic frame-memory
history. Stable slots/object IDs (`sam3-00` through `sam3-03`) remain unchanged; this is
a temporal-memory reset at the correction frame, not an additive prompt-memory policy.
Choose whether that conservative replacement/reset meaning is acceptable before using
the schedule. An additive-memory experiment would need a distinct, separately reviewed
policy because the API does not document correction precedence.

## Historical e4 three-target manual-seed multiplexed smoke

The preserved dedicated three-target manual path validates its historical proposal
against its calibration manifest, loads only the three selected frame-0 mask-0 artifacts
(`left_hand`/`t000000-b01`,
`yellow_toy_body`/`t000000-b03`, and `toy_wheel`/`t000000-b04`), and excludes
`right_hand`/`t000000-b18`. It creates one SAM3.1 multiplexed prompt memory with
`encode_prompt_memory_from_mask`, then maintains one continuous three-target stream:

```bash
uv run battle-muggled-smoke \
  --config configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json \
  --view ego-hmc21179183 --start-frame 0 --max-frames 300 \
  --manual-seed-proposal \
  runs/muggledsam-sam3-e4-web-calibration-e55b5d0abe02/proposed_tracking_prompt.json
```

This is manual-seed multiplexed tracking, not out-of-box/text zero-shot. It has a
hard 300-frame/10.0-second limit, uses no chunks or intentional ID resets, records
fixed target-to-normalized-ID associations, and writes an external mask every analysis
frame. The worker refuses to begin while another GPU model process is active (an open
Rerun viewer is tolerated and recorded); close the local calibration workspace before
executing it. A completed run remains review-only and makes no segmentation,
association, or accuracy claim.

## Analysis frame-rate comparison (30 vs 60 fps)

The evidence for keeping the 30 fps analysis clock is three frame-0-seeded arms over the
same ten seconds of e4 footage; the result and decision are in the ledger's
[frame-rate comparison record](docs/method-ledger.md#sep-13-evening-muggledsam-sam3-e4-analysis-frame-rate-comparison)
and the report is [`docs/frame-rate-comparison-2026-09-13.json`](docs/frame-rate-comparison-2026-09-13.json).
The report measures output continuity and model self-estimated diagnostics; it is not a
ground-truth accuracy evaluation.
To reproduce:

```bash
bash scripts/create_assembly101_e4_60fps_proxy.sh   # 180 s, 954x720, 60 fps e4 proxy
P=runs/muggledsam-sam3-e4-four-target-keyframes-20260913t213159z/proposed_tracking_prompt.json
uv run battle-muggled-smoke --config configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json \
  --view ego-hmc21179183 --max-frames 300 --max-side-length 720 --manual-seed-proposal $P
uv run battle-muggled-smoke --config configs/clips/assembly101_nusar_9033_e4_60fps.json \
  --view ego-hmc21179183 --max-frames 600 --max-frame-memory 8 --max-side-length 720 --manual-seed-proposal $P
uv run battle-muggled-smoke --config configs/clips/assembly101_nusar_9033_e4_60fps.json \
  --view ego-hmc21179183 --max-frames 600 --max-frame-memory 4 --max-side-length 720 --manual-seed-proposal $P
uv run python scripts/compare_frame_rate_arms.py runs/<arm-30fps> runs/<arm-60fps-mem8> runs/<arm-60fps-mem4>
```

The `analysis` clock in a clip config may be 30 or 60 while the source, annotation, and
pose clocks stay fixed. Frame-zero-only seeds calibrated on the 30 fps proxy are accepted
on another proxy only after checking the same raw-source checksum, source start instant,
dimensions, and scaling policy. Proxy identity and fps may differ; source end time is not
part of this check. The run manifest records the transfer as a
`calibration_transfer_note`. Correction schedules are authored against the 30 fps clock
and are refused at any other rate.

To rebuild a run's profile-specific RRD after an exporter change without re-running
inference:

```bash
uv run python scripts/reexport_run_rrd.py runs/<run-id> [runs/<run-id> ...]
```

### Build the unified exploratory comparison

The final bounded review surface composes existing normalized outputs only; it does not import
or invoke model code. It validates the declared source/proxy/config fingerprints, retained
frame/source timestamps, native mask dimensions, and the shared 600-frame asset before writing
one RRD and its generated index under the ignored run directory:

```bash
uv run battle-build-exploratory-comparison
rerun rrd print runs/exploratory-first-20s-comparison/exploratory_first_20s_comparison.rrd
```

The default view shows the static RGB video with MediaPipe hands and BoxMOT boxes. Other method
roots are independently toggleable; WiLoR camera-relative non-metric 3D is in a separate 3D view.
ATHENA remains metadata-only and its synthetic fixture is never overlaid on Assembly101 frames.

Recordings are keyed by clip (`battle-<clip_id>`) and run id, so runs of different clips
or of the same clip open side by side without merging.

### Build the focused interaction review

The review-focused package combines the already verified exploratory artifacts with the
completed four-part comparison without running inference or opening a viewer:

```bash
uv run battle-build-interaction-review
rerun runs/interaction-review-first-20s/interaction_review.rrd
```

It validates all input fingerprints, source timestamps, the approved 30 FPS / `[0,600)`
contract, reference-mask dimensions, and the single 1280×720 embedded RGB asset before
writing `interaction_review.rrd`, `interaction_review_index.json`, `review_guide.md`, and a
labeled contact sheet under `runs/interaction-review-first-20s/`.

The default primary panel is deliberately focused: corrected focused SAM3 four-part masks with
the stabilized WiLoR 2D layer. Raw WiLoR and blue MediaPipe remain separately toggleable
comparison/fallback evidence; matching is same-frame nearest-wrist spatial assignment only,
never a cross-method or persistent-ID assertion. WiLoR's camera-relative non-metric 3D has a
separate 3D panel. BoxMOT is hidden from the default composition as optional
person/occlusion context, not part tracking or segmentation. Kineo exposes only partial 2D NLF
body context.

The contact time series is a review navigation aid: for each MediaPipe spatial proximity lane and
each named reference part, it retains palm/wrist, nearest-fingertip, and minimum source-pixel
distance. A raw candidate is inside the mask or at most 12 pixels away; it starts after two
observed candidate frames and ends after three observed non-candidate frames. Missing hand/mask
values are explicitly cleared and reset the debounce state rather than being treated as distant.
`contact_candidate_start`/`contact_candidate_end` are geometry heuristics, not touch or grasp
ground truth. The guide records deterministic required frames 0/300/599 plus high-disagreement,
missing-hand, transition, Kineo-gap, and late-occlusion bookmarks; every human disposition stays
pending.

### Overnight v2 rebuild

The v2 review defaults to corrected focused static SAM3 masks and a deterministic WiLoR-primary
stabilized layer. Raw WiLoR and MediaPipe remain separately toggleable evidence; no raw gap is
silently interpolated. Build the layer and package without opening a viewer:

```bash
uv run battle-stabilize-wilor --output-root runs/wilor-hands-stabilized-20s-overnight-v2-r3  # pre-gate layer; v5 below
uv run battle-kineo-nlf --seconds 20 --rtmlib-bbox-detection-frame-step 1 \
  --run-id kineo-nlf-headless-20s-frame-step-1-overnight-v2 \
  --sequence-name assembly101_focused_static_20s_step1_overnight_v2
uv run battle-build-interaction-review --output-root runs/interaction-review-overnight-v2
uv run rerun rrd print runs/interaction-review-overnight-v2/interaction_review.rrd
```

`docs/qa/overnight-interaction-review-v2.agent-review.json` is the structured mixed-provenance
record. It preserves supplied human feedback verbatim, keeps all human decisions pending, and
labels agent findings/correction proposals as `agent_authored_visual_review`. The 11-step
agent-authored substep timeline is primary navigation; crop-CLIP model arms remain secondary
exploratory evidence unless they materially improve both checkpoint and boundary diagnostics.

### Overnight v3 continuity rebuild

v3 preserves v2 and replaces only Kineo context with a deterministic fusion stream:

```bash
CUDA_VISIBLE_DEVICES=0 uv run battle-kineo-fusion \
  --output-root runs/kineo-nlf-fused-20s-overnight-v3
uv run battle-build-interaction-review --output-root runs/interaction-review-overnight-v3
uv run python scripts/render_overnight_v3_audits.py
uv run rerun rrd print runs/interaction-review-overnight-v3/interaction_review.rrd
```

It validates identical source/clock mapping before using independent BoxMOT person boxes, keeps
native Kineo/YOLOX boxes primary, accepts only nearby native-consistent BoxMOT fallbacks, and
runs NLF on every actual fused crop. Residual boxes may be interpolated only through a gap of at
most five frames (target three); longer outages remain explicit `missing` rows. The generated
`box_fusion.json` records one of `detected_native`, `boxmot_fallback`, `interpolated`, `held`, or
`missing` per frame. The review index retains all raw segmentation triggers while clustering
same-part neighboring triggers into compact episode bookmarks. The agent-authored v3 QA record
and dense raw/stabilized/fallback plus corrected-SAM3/contact sheets remain evidence only; human
pass/fail is still pending.

### First-minute v4 review

v4 extends the review to the retained first 60 seconds / frames `[0,1800)` while keeping the
same source-aligned static proxy and exactly one embedded RGB asset. Build the already-generated
inference-free package with:

```bash
uv run battle-build-interaction-review-v4
uv run rerun rrd print \
  runs/interaction-review-first-minute-v4/interaction_review_first_minute_v4.rrd
```

Its default layer is corrected SAM3 plus the deterministic WiLoR-primary stabilized 2D hands;
raw WiLoR, MediaPipe, BoxMOT person context, and partial Kineo NLF body context remain separately
toggleable. `interaction_review_index.json` retains source fingerprints, all 1,800 normalized
rows, hand/box provenance, geometry trigger episodes, and explicit segmentation validity
intervals. The late SAM3 display remains visible for comparison, but frames `[1200,1800)` are
`not_contact_eligible`: late contact diagnostics use `invalid_mask`, are cleared, and cannot
generate candidate events.

The first-20-second 11-step agent-authored navigation timeline remains the only fine action
timeline. Assembly101 coarse GT is logged separately as weak navigation context over the full
minute (`attach interior`, `screw chassis`, `attach body`, `screw chassis`); it is never a model
prediction or a replacement for visual review. The v4 guide and index make the retained coverage
and every human/agent claim boundary explicit.

#### Sep 17 human review follow-up (v4 rebuild)

`docs/qa/interaction-review-first-minute-v4.human-feedback.json` records the human's Sep 17
review verbatim (author `human`, no pass/fail added) and the agent's responses separately;
`docs/qa/interaction-review-first-minute-v4r2.agent-review.json` holds the agent findings and
the agent-proposed correction rows. What changed:

- **Missing panels.** The blueprint's Review guide, Drop-DTW and substep panels pointed at
  entities the v4 builder never logged (the 20 s builder also lacked `metadata/review_notes`).
  Both builders now log the guide, a per-frame navigation document (current agent substep,
  coarse GT segment, contact eligibility), step-index time series, and static coarse-GT /
  contract / Drop-DTW-status documents. No Drop-DTW alignment exists for the first minute; the
  panel says so. A test checks the exported RRD rows and blueprint references.
- **Chassis/interior swap.** Measured onset ~1020-1032 (interior mask leaking onto the chassis),
  chassis label lost 1110-1166, pure label swap 1167-1234 until the human-selected frame-1235
  correction. The grey interior block is not separately visible before ~1167, so the earliest
  credible correction is frame 1172. `battle-muggled-agent-correction` derives a draft
  calibration from the finalized one, decodes box prompts headlessly, and records acceptances
  with `selected_by: agent` (schema field; existing rows default to `human`, agents can never
  seed frame 0). Policy v3 permits a sixth keyframe. The rerun
  `runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t001210z`
  is bit-identical before 1172, fixes 1172-1234 (new chassis vs old interior IoU 0.926), and
  matches the old run after 1235. v4 references it and restricts `contact_eligible` to
  `[0,1020)` and `[1172,1200)`. Frames 1020-1171 stay wrong; frame 370 is a bookmark only.
- **Finger-only hands.** The stabilizer accepts real WiLoR detections with confidence in
  `[0.35,0.55)` only while they continue a lane accepted within 5 frames, for at most 5
  consecutive frames, tagged `low_confidence_continuation` (nothing is held or extrapolated).
  The v5 layer `runs/wilor-hands-stabilized-60s-v5` reduces missing frames 110 -> 79, short
  per-lane gaps 92 -> 69 and MediaPipe fallback frames 66 -> 15; its state counts are plotted in
  the disagreement time series. Kineo is body-only NLF context and is not a hand method.
- `battle-build-interaction-review-v4` accepts an empty existing output root and needs
  `--overwrite` to replace an existing package; `runs/interaction-review-first-minute-v4-local`
  is an older, superseded build kept only because a viewer may still have it open.

##### Sep 17–18 follow-up: leak onset, 20 s hand layer, DAM4SAM first minute

- **Leak-onset correction (not adopted).** Dense 1000-1180 evidence sheets show the black
  chassis clearly visible through 1020-1171 and the grey interior block partially visible only
  until ~1022 (the block protruding under the chassis plate, as the human accepted at 900). Agent
  corrections at frame 1020 (chassis 6,876 px, interior 1,152 px on that block) under policy v4
  (`..._correction_policy_v4.json`, eight later keyframes; the schema caps v3 at six) were rerun
  as `runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t004350z`.
  The interior slot re-leaks onto the chassis from ~1036-1046 along the same trajectory, so the
  memory reset alone does not hold; the 001210z run stays the reference and `contact_eligible`
  stays `[0,1020)` and `[1172,1200)`. Evidence and metrics live under
  `runs/muggledsam-sam3-four-part-focused-corrections-agent-leak-20260918t004115z/agent_review/`.
- **20 s hand layer.** `runs/wilor-hands-stabilized-20s-v5` applies the same
  `low_confidence_continuation` gate as the 60 s layer (missing frames 45 -> 29, MediaPipe
  fallback 32 -> 10, short per-lane gaps 46 -> 35); `battle-build-interaction-review` defaults
  to it and `runs/interaction-review-overnight-v3` was rebuilt in place.
- **DAM4SAM first minute.** `battle-four-part-segmentation dam4sam --frame-count 1800` runs the
  arm over `[0,1800)` from the same reviewed frame-0 seeds. `battle-segmentation-disagreement`
  computes per-frame per-target IoU between two runs and lists episodes with IoU < 0.5 for at
  least five consecutive frames (a mask missing on one side counts as 0; missing on both is
  neutral). This is **cross-method disagreement, not accuracy**: neither run is ground truth, and
  the corrected SAM3 reference carries human/agent corrections that the DAM4SAM arm does not.
  Against `runs/dam4sam-four-part-reviewed-seed-60s-20260918t005416z`: mean IoU cabin 0.968,
  rear_body 0.786, chassis 0.678, interior 0.553, 16 episodes (largest: interior `[707,1072)`,
  rear_body `[1522,1661)`, chassis `[570,706)`, interior `[1117,1252)`, chassis `[1055,1172)`).

```bash
uv run battle-stabilize-wilor --output-root runs/wilor-hands-stabilized-20s-v5
uv run battle-build-interaction-review --output-root runs/interaction-review-overnight-v3
uv run battle-four-part-segmentation dam4sam --frame-count 1800 --run-id dam4sam-four-part-reviewed-seed-60s-<utc>
uv run battle-segmentation-disagreement \
  --run-a runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t001210z --label-a corrected_sam3 \
  --run-b runs/dam4sam-four-part-reviewed-seed-60s-<utc> --label-b dam4sam_60s \
  --end-frame-exclusive 1800 --output runs/dam4sam-four-part-reviewed-seed-60s-<utc>/disagreement_vs_corrected_sam3.json
```

```bash
uv run battle-muggled-agent-correction derive \
  --source-calibration runs/muggledsam-sam3-four-part-focused-corrections-327-1235-20260916t022433z \
  --output-dir runs/<agent-calibration>
uv run battle-muggled-agent-correction decode --calibration runs/<agent-calibration> \
  --frame 1172 --prompt "chassis=870,463,995,583;bg=855,490" --prompt "interior=835,463,900,515"
# Several candidate frames at once, loading the checkpoint a single time:
uv run battle-muggled-agent-correction decode-batch --calibration runs/<agent-calibration> \
  --plan runs/<agent-calibration>/decode_plan.json
uv run battle-muggled-agent-correction accept --calibration runs/<agent-calibration> \
  --candidate-id t001172-b02 --index 1 --rationale "..."
uv run battle-muggled-agent-correction schedule --calibration runs/<agent-calibration> \
  --correction-policy configs/muggledsam_static_four_part_reassembly_focused_correction_policy_v3.json \
  --manual-seed-target-config configs/muggledsam_static_four_part_reassembly_focused_manual_seed.json
uv run battle-muggled-smoke --config configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json \
  --view static-c10379 --four-part-static-focused --max-side-length 720 \
  --multi-keyframe-correction-schedule runs/<agent-calibration>/multi_keyframe_correction_schedule.json
uv run battle-stabilize-wilor --wilor-run runs/wilor-hands-static-60s-overnight-v2 \
  --mediapipe-run runs/mediapipe-hands-static-60s-fused-dedup-th035-20260916t0430z \
  --parts-run runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t001210z \
  --output-root runs/wilor-hands-stabilized-60s-v5
uv run battle-build-interaction-review-v4 --overwrite
```

#### Assembly101 dataset hands and fine-grained labels (Sep 18)

`battle-build-assembly101-reference` resamples the recording's own dataset assets onto the
first-minute analysis clock and writes `runs/assembly101-reference-first-minute-v1/`
(`manifest.json` + `hands.jsonl`); `battle-build-interaction-review-v4` reads it by default
(`--no-assembly101-reference` builds without it). It is **external review context, not ground
truth for any method here**, and CC BY-NC 4.0 attribution applies.

- **Inputs** (ignored `data/raw/assembly101/<recording>/`, selectively acquired Sep 17):
  `landmarks3D`, `hand_confidences`, `timestamp`, `camera_extrinsics_fixed` from
  `AssemblyPoses.zip`, and this recording's fine-grained CSV rows. The C10379 intrinsics are
  the checked-in estimate `configs/assembly101/c10379_camera_estimate.json` (Brown model fitted
  to the dataset's own 2D/3D landmark pairs through the shipped camera-to-world pose; the
  archive ships no intrinsics). The builder refuses a camera estimate whose extrinsics differ
  from the dataset file.
- **Clock rule.** Proxy frame `p` -> pose frame `17649 + 2p` for the static view: the C10379
  video lags the 60 fps pose clock by 9 pose frames (~150 ms, +-1). A method-independent check
  agrees: the median distance from each dataset wrist to the nearest stabilized WiLoR wrist is
  minimal at +7..+10 pose frames (30.4 px at +9 vs 33.7 px at 0). Every earlier static/ego
  comparison assumed zero relative offset.
- **Projection check.** Our projection of the 3D joints reproduces the dataset's shipped 2D
  landmarks to 0.0002 px RMS over 73,500 points, so `landmarks2D` (1.1 GB) is never read.
- **What the viewer gets.** A third comparison view `comparison/assembly101_hands_2d` (dataset
  left hand yellow, right mint, drawn at confidence >= 0.5, 21 joints in the dataset's own
  MS-G3D joint order, never remapped onto the MediaPipe order); a `Spatial3DView` at
  `contexts/assembly101_world_mm_3d` with the world-mm hands and the estimated C10379 camera
  frustum; a `diagnostics/assembly101` panel (per-side dataset confidence and wrist distance to
  the nearest stabilized WiLoR wrist, a disagreement between two imperfect sources, not an
  error of either); and `metadata/fine_grained_gt` plus a `fine_gt_index` navigation series.
  The 27 fine-grained segments in the window (`position interior` 96-323 and 518-697, `screw
  chassis with screwdriver` 424-483, 737-1079 and 1668-1800, `position rear body` 1321-1518,
  pick-up/put-down/inspect steps; overlapping two-hand labels kept) replace the agent-authored
  substep track on the navigation panel; the substep contract stays as a tab.
- **Coverage.** Dataset hands in 1800/1800 frames (left 1800, right 1700); 72 of 3,500 hands
  fall below the draw threshold. Where a dataset wrist lands within 40 px of a WiLoR wrist
  (1,951 pairs) the wrist-to-middle-tip span ratio dataset/WiLoR has median 1.08 with a wide
  spread (p10 0.64, p90 1.57): the dataset's fixed-scale hand model and both trackers'
  articulation noise are visible, especially with fingers hidden behind the held part.

```bash
uv run battle-build-assembly101-reference            # writes runs/assembly101-reference-first-minute-v1
uv run battle-build-interaction-review-v4 --overwrite
uv run pytest -m real_data tests/test_assembly101_reference.py
```

#### All eight static views: focused window, per-view clock offsets, per-view cameras (Sep 18)

Track 0 of the overnight multicam pass puts every static camera of the pinned recording on
the same footing as C10379, for the focused window only (source 294.000-386.700 s). Nothing
here is an accuracy claim: dataset poses/extrinsics are external context, the fitted
intrinsics are estimates of the dataset's internal projection, CC BY-NC 4.0 applies.

- **Acquisition.** `battle-fetch-assembly101-view --view C10095` resolves the signed CDN URL
  of the pinned revision and points ffmpeg at a local counting proxy, so only the moov atom and
  the window's byte range are transferred (~10.5 % of each 1.6-4.0 GB file, 1.85 GB in total
  for the seven new views). One decode writes both `<view>_rgb_294.000-386.700_raw60.mp4`
  (sensor resolution, 60 fps, 5,562 frames, for clock scans) and the standard
  `<view>_rgb_294.000-386.700_1280x720_30fps.mp4` proxy (same filter chain and encoder as the
  C10379 proxy). `--local` runs the same recipe on recordings already on disk (used for the
  C10379 and HMC trims). `scripts/create_assembly101_all_static_focused_proxies.sh` runs all
  of it idempotently; `--write-report` writes the ignored acquisition report/manifest and the
  tracked eight-view clip config
  `configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_all_static_g2.json`
  (the single-view focused config is unchanged because several review builders assert
  `clip.views == ("static-c10379",)`).
- **Clock offsets.** `battle-assembly101-clock-offset --view C10095` ports the Sep 17 offset
  scan: three metrics (skin-mask hit rate for RGB views, fingertip gradient magnitude, frame
  difference) sampled at the dataset's projected fingertips of moving hands, three 30 s chunks,
  candidate offsets -6..+15 pose frames, sub-frame peak by parabolic interpolation. Each metric
  votes with the median of its informative chunks; metrics disagreeing by more than two frames
  mark the view ambiguous (no rule is forced). Per-view results go to
  `runs/assembly101-clock-offsets/<view>.json`; `--write-config` collects them into the
  tracked `configs/assembly101/clock_rules.json`. The static cameras do **not** share one
  offset: C10095 +5, C10115 +6, C10118 +6, C10119 +7, C10379 +9, C10390 +7, C10395 +6,
  C10404 +6 pose frames (+-1, +-2 for C10118/C10395); all four ego cameras 0 (+-1).
- **Cameras.** `battle-fit-assembly101-camera --all` generalises the C10379 fit: Brown model
  for static views (`cv2.calibrateCamera` on dataset 3D/2D pairs, verified by re-projecting
  through the shipped extrinsics: 0.0002-0.002 px RMS), rational model with per-frame
  `camera_extrinsics_ego` for the HMC cameras (0.27/0.31 px for e3/e4; 4-5 px for e1/e2, which
  barely see the hands). Written to `configs/assembly101/<view>_camera_estimate.json`; the
  schema gained `distortion_model`, `extrinsics_kind` and the shipped-pose residual, and the
  existing C10379 file still loads unchanged.

```bash
scripts/create_assembly101_all_static_focused_proxies.sh
uv run battle-assembly101-clock-offset --view C10095 --view C10115 --view C10118 --view C10119 \
  --view C10379 --view C10390 --view C10395 --view C10404 \
  --view HMC_21110305 --view HMC_21176623 --view HMC_21176875 --view HMC_21179183 --write-config
uv run battle-fit-assembly101-camera --all
uv run pytest -q tests/test_assembly101_acquisition.py
```

### First-minute review metrics (label-free triggers)

`battle-review-metrics` turns the v4 first-minute inputs into per-frame proxy metrics and a
ranked list of review triggers without running any model or touching the GPU:

```bash
uv run battle-review-metrics
uv run rerun rrd print runs/review-metrics-first-minute-v2/review_metrics_first_minute.rrd
```

It locates every input through the v4 index (`runs/interaction-review-first-minute-v4/`; the
reference segmentation run is read from the index rather than hardcoded, so the metrics always
measure the masks the review package displays), re-verifies the declared SHA-256 of each
manifest/observation file plus the bounded video before use, and writes `metrics.json` (typed,
NaN-free records; absence is always an explicit state),
`triggers.json` (ranked episodes with frame ranges, type, score, and a one-line rationale),
per-episode contact sheets for the top 12 episodes (source frame + mask overlays + stabilized
hand boxes, with before/after context for short episodes), `metrics_report.md`, and an
inference-free RRD whose scalar time series share the v4 `analysis_frame` / `analysis_time` /
`source_time` clocks and clip root so it can be opened next to the v4 recording.

Metrics (all label-free proxies, never accuracy):

- **Segmentation identity swap** per part pair and frame: IoU, centroid distance, and a swap score
  = max(label-exchange IoU against the other part's previous mask, same-frame IoU weighted by both
  masks being large, absorption of a collapsing part's footprint by an enlarged neighbor). A
  **label-crossing** heuristic fires when the centroid difference vector reverses over 10 frames
  while both areas stay ≥30 % of their medians. **Area anomalies** flag sustained runs ≥1.75× or
  ≤0.35× a part's own median area.
- **Appearance consistency**: per-mask median HSV and mean Lab, Lab distance from the same part
  on frame 0, a robust per-part z of that distance, the fraction of mask pixels inside a yellow
  band derived from the frame-0 `rear_body`/`cabin` masks (recorded, not hardcoded), and the
  fraction covered by stabilized-hand boxes. Leakage = unusual color drift (robust z ≥3 and
  distance ≥18) or a dark part turning yellow; hand overlap is context only.
- **Mask growth vs hand proximity**: frame-to-frame and 15-frame windowed area ratios, centroid
  velocity, and hand-box overlap, classifying growth events as `hand_capture_suspect` or
  `unexplained`.
- **Hands**: for frames where the stabilized layer is `missing`, a skin-color proxy near the last
  known pose (band calibrated from confident raw WiLoR boxes, checked against the frame
  background) plus last-pose fingertips-in-frame; re-entry jump after each gap in hand scales;
  nearest-wrist jitter normalized by hand scale, summarized per agent substep over `[0,600)` and
  per coarse GT phase over the minute.
- **Contacts**: debounced interval durations from the v4 index, sub-5-frame flicker count, and
  consistency against `configs/review_metrics/first_minute_v1.json`, which encodes the expected
  touched parts per substep as `agent_authored_assumption` records.
- **Kineo**: mean joint confidence and joint jitter grouped by fused-box provenance
  (`detected_native` / `boxmot_fallback` / `interpolated` / `held` / `missing`).

Scores are in threshold units and are comparable only within a type, so the global rank
interleaves types (round *k* holds the *k*-th strongest episode of every type). Every threshold is
applied uniformly to the whole minute. The report ends with a clearly marked GPU follow-up (60 s
DAM4SAM disagreement) that this CPU-only pass does not run.

`runs/review-metrics-first-minute-v2/` is the rerun against the current v4 index (reference
`...20260918t001210z` with the frame-1172 agent correction); v1 measured the superseded
`-v4-local` index. The v2 top episodes move accordingly: the chassis collapse anomaly stays at
f1089–f1171 (25.2x), the frame-1235 label exchange disappears and the frame-1172 correction now
registers as the single-frame chassis/interior exchange it is, and a chassis growth event at
f1172–f1186 marks the correction restoring the label.

### Per-target ensemble review reference (provenance-tracked fallback)

`battle-build-ensemble-reference` assembles a new reference run (`runs/ensemble-reference-
first-minute-v1/`, ignored) in exactly the observation/mask schema the review builders consume,
from retained runs only (no model, no GPU):

```bash
uv run battle-build-ensemble-reference            # --policy, --output-root, --overwrite
uv run battle-build-interaction-review-v4 --overwrite   # default reference is the ensemble
uv run battle-build-interaction-review-v4 --overwrite \
  --reference runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t001210z
```

The checked-in typed policy `configs/ensemble_reference/first_minute_v1.json`
(`EnsembleReferencePolicy`, tagged `agent_authored_assumption`) decides every frame x target:

- **Default** is the corrected SAM3 reference. Masks are copied whole from exactly one source;
  nothing is blended, morphed or interpolated (`no_blend` is a schema literal).
- **Fallback** to the DAM4SAM 60 s arm happens for one target only inside explicit frame
  intervals (chassis `[1055,1172)`, the disagreement episode), only when a rule fires on the
  SAM3 mask (area below 0.5x the rolling median of accepted areas, IoU with another SAM3 target
  above 0.3, discontinuity with the last sane accepted mask, or a missing mask), and only when
  the DAM4SAM mask passes sanity (area within 0.4–2.0x the rolling median, and IoU >= 0.5 with
  the last sane accepted mask or a centroid jump within 30 px + 5 px per elapsed frame, capped
  at 90 px). Rules are evaluated everywhere for diagnostics but only act inside the intervals;
  outside them the human/agent-reviewed SAM3 masks stay authoritative.
- **Hidden** intervals are agent-authored visibility labels written as explicit empty masks
  (`hidden_agent_label`): interior `[1024,1172)`, re-verified on zoomed source crops before
  adoption and pending human confirmation. Hidden frames carry no object row, so consumers see
  `missing_mask`/`invalid_mask`, never a guessed interior.
- Every frame x target record (`ensemble_provenance.json`) carries `provenance`
  (`sam3_corrected | dam4sam_fallback | hidden_agent_label | missing`), the rules that fired,
  sanity/eligibility flags and the measured areas/IoU/centroid jump; every substitution and
  every **declined** attempt is listed with its rationale. `segmentation_episode_check.json`
  reruns the review-metrics swap/crossing/area-anomaly detectors on the SAM3 reference and on
  the ensemble; `sheets/before_after_1000_1250.png` shows both side by side.

First run: chassis 111 DAM4SAM frames over `[1055,1056)` and `[1062,1172)`, 6 declined at
1056–1061 (DAM4SAM's chassis is equally undersized there); interior 148 hidden frames; rear_body
and cabin untouched. The 1089–1173 swap/collapse episodes vanish on the ensemble with no new
episode elsewhere (34 -> 29 episodes). The v4 builder now defaults to the ensemble, records
`reference_segmentation_method: ensemble_reference`, computes `contact_eligible` **per target**
(chassis `[0,1020)`, `[1055,1056)`, `[1062,1200)`; interior `[0,1024)`, `[1172,1200)`;
rear_body/cabin `[0,1200)`; the late `[1200,1800)` boundary is unchanged), and logs a toggleable
provenance layer: `diagnostics/reference_provenance/<part>` scalar series (0 missing, 1 sam3,
2 dam4sam, 3 hidden) in their own time panel, `diagnostics/segmentation_contact_eligible/<part>`,
and `primary/reference_provenance_overlay/<part>` magenta boxes around every DAM4SAM-sourced
mask in the primary view. Claim boundary: the ensemble is a review reference; cross-method
fallback is not accuracy and neither source run is ground truth.

### Four-part segmentation comparison

`configs/four_part_segmentation_comparison.json` is the checked-in fair-comparison
contract: the approved focused static RGB proxy, source interval 294.0–314.0 s, 30 FPS
frames `[0,600)`, exact ordered targets (`chassis`, `interior`, `rear_body`, `cabin`),
and SHA-256 fingerprints for the four reviewed frame-zero masks and their mask-derived
boxes. It binds the accepted focused SAM3 run as the baseline. It never copies a mask
into Git; the reviewed source artifacts remain under ignored `runs/`.

Run the four GPU arms serially (the commands use one process at a time):

```bash
uv run battle-four-part-segmentation grounding_dino_sam2_open_vocabulary
uv run battle-four-part-segmentation reviewed_seed_sam2_control
uv run battle-four-part-segmentation samurai
uv run battle-four-part-segmentation dam4sam
uv run battle-build-four-part-segmentation-comparison \
  --grounding-dino-sam2-open-vocabulary runs/<grounding-arm> \
  --reviewed-seed-sam2-control runs/<sam2-control-arm> \
  --samurai runs/<samurai-arm> --dam4sam runs/<dam4sam-arm>
rerun rrd print runs/four-part-segmentation-comparison/four_part_segmentation_comparison.rrd
```

Grounding-DINO receives only the independently recorded target prompts/synonyms in the
contract, never a reviewed box or mask. Its first-frame detector outcome is persisted per
target before SAM2 propagation. `reviewed_seed_sam2_control` uses the exact four
fingerprinted reviewed masks to isolate detector failure from propagation. SAMURAI runs
with `samurai_mode: true`; because its mode failed with a four-object state, it releases
one real SAMURAI predictor per reviewed target and combines only source-aligned labelled
observations. DAM4SAM runs four independent `DAM4SAMTracker` DRM streams from the same
masks and records DRM additions by label.

The side-by-side recording embeds one 600-frame video asset and gives SAM3 baseline,
Grounding-DINO+SAM2, SAM2 control, SAMURAI, and DAM4SAM independent colored entity
roots. At every frame, the render subtree is cleared before any new mask/box is logged,
so a missing target cannot persist visually. It is a mask-only comparison: MediaPipe and
WiLoR are hand-pose layers, BoxMOT is detector-conditioned box tracking, Kineo NLF is
body-pose output, and Drop-DTW is weak temporal alignment; none are segmentation
comparators. The earlier unified exploratory recording remains indexed at
`runs/exploratory-first-20s-comparison/exploratory_first_20s_comparison.rrd`.

## Validate and export fixtures

```bash
uv run ruff check .
uv run pytest
uv run battle-export-fixture --output artifacts/synthetic_fixture.rrd
```

### Mask cache

A review package walks the same masks three times: contact geometry, segmentation
triggers, then the RGBA cut-outs the viewer draws. `src/battle/mask_cache.py` keeps one
bounded in-memory cache per run directory and can persist `native/mask_cache.npz`,
holding the bit-packed masks plus the cut-outs each mask is actually logged with. Each
entry records its source PNG's size and modification time, so a rewritten mask falls
back to the PNG instead of serving a stale cut-out.

The SAM3 smoke and the SAM2-family video adapters write the sidecar after inference.
Write one for any other run with:

```bash
uv run battle-cache-masks runs/<run-id>
```

For the 1,800-frame first-minute package this took the rebuild from about 85 s to 60 s
through the shared in-memory cache, and to about 44 s with the sidecar present; the
recording holds the same rows, in fewer and larger chunks.

### Phase timing and narrowed builds

Every builder prints its split when it finishes, so the next regression is visible
without a profiler. `--quiet` suppresses it.

```text
interaction review v4: total 18.8s (validate 1.2s geometry 12.5s export 5.1s other 0.0s)
```

That report is what showed the geometry pass, not the export, dominated a rebuild:
centroids were measured with `np.nonzero`, mask areas were summed four times per
comparison, and a frame with two hands transformed each part mask twice. Measuring the
area and centroid in two axis reductions and sharing one distance map per part per frame
halved the phase, with byte-identical triggers, contacts, events, and episodes.

`--layers` narrows a build to the layers under review, which is useful while iterating on
one overlay. Anything short of the full set is recorded as `logged_layers` in the index so
a narrowed package cannot be mistaken for a reviewable one:

```bash
uv run battle-build-interaction-review-v4 \
  --output-root runs/iteration --layers reference_masks,stabilized_wilor
```

### Resuming a SAM3 correction rerun

Adding one correction keyframe used to mean re-streaming the whole clip, because the
tracker's memory at frame *k* only exists if frames `[0,k)` were stepped. The worker now
saves that memory at every correction keyframe (and every `--checkpoint-every` frames) to
`native/checkpoints/f<frame>.pt`, so a rerun can restart at the keyframe it is changing:

```bash
uv run battle-muggled-smoke --config configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json \
  --view static-c10379 --four-part-static-focused --max-side-length 720 \
  --multi-keyframe-correction-schedule runs/<calibration>/multi_keyframe_correction_schedule.json \
  --resume-run runs/<prior-run> --resume-at 1172
```

The resumed run copies the prior run's rows and mask PNGs for `[0,1172)` byte for byte and
steps only `[1172,2781)`. A checkpoint is refused unless the video, model, encoder
settings, preprocessing, and initialization hash to the same stream, and the run manifest
records `mode: checkpoint_resumed` with the prior run, the resume frame, and the
checkpoint's fingerprint, so a resumed stream is never presented as a continuous one.

Measured on the focused four-part run: worker inference fell from 191.1 s to 112.2 s when
resuming at frame 1172 of 2,781, and all 2,781 frames were bit-identical to the
continuous run. `uv run pytest -m gpu` asserts that equivalence on the 300-frame smoke.
Each checkpoint is about 6.6 MB, and they live under the ignored run directory.

### One warm worker per decode session

`battle-muggled-agent-correction decode` starts an isolated worker, which loads the
multi-gigabyte checkpoint before it answers anything. A review loop usually asks the same
model about several candidate frames, so `decode-batch` takes a plan and reuses one
worker for all of it:

```json
[
  {"frame": 1172, "prompts": ["chassis=870,463,995,583;bg=855,490"]},
  {"frame": 1235, "prompts": ["interior=835,463,900,515"]}
]
```

The plan is validated before the model is loaded, so a malformed frame or an empty prompt
list fails immediately rather than after the checkpoint is resident.

### Cached digests and probes

Builders also stop re-reading unchanged inputs. `src/battle/digest_cache.py` remembers a
file's SHA-256 against its size and modification time under `.cache/battle/`, and
`src/battle/media_probe.py` stores the exact frame count beside each video rather than
decoding the stream again with `ffprobe -count_frames`. The trimmed Rerun input video is
reused when a stamp shows the same proxy and frame count.

Both caches are keyed by stat metadata, not content, so a provenance-critical rebuild
should re-read the bytes:

```bash
uv run battle-build-interaction-review-v4 --verify-fingerprints
uv run battle-build-interaction-review --verify-fingerprints
```

The exploratory comparison now fingerprints the mask URIs its observations reference
instead of every file beside them, so a rebuild is not charged for caches and sheets that
no frame draws. That changes the recorded mask-tree digest for packages built before this
revision.

### Test tiers

The suite is tiered so the default run needs nothing but this repository. Tests that
read the ignored `data/`, `runs/`, or `models/` trees carry `real_data` and are
deselected by default; they name the missing path when the artifact is absent instead of
failing. `gpu` is reserved for checks that need a CUDA device and local weights.

```bash
uv run pytest                                        # default tier, no ignored artifacts
uv run pytest -m "not real_data and not gpu and not slow"  # fast loop
uv run pytest -n auto                                # same tier across cores
uv run pytest -m real_data                           # artifact integration checks
uv run pytest -m gpu                                 # device-bound checks
```

While iterating, `uv run ruff check .` plus the fast tier is enough to catch a mistake in
a few seconds. Before a commit, run the default tier; before claiming a run's provenance
in the ledger, run the `real_data` tier and the relevant builder with
`--verify-fingerprints`. The `gpu` tier is for the device-bound equivalence checks and
needs the local SAM3 checkpoint.

### Pruning runs

`runs/` accumulates every experiment. This reports the directories that nothing committed
cites, largest first, and separates the ones that only another run's manifest names:

```bash
uv run python scripts/prune_runs.py
uv run python scripts/prune_runs.py --json
```

It deletes nothing. The last line is a `rm -rf` to review and run by hand, because a run
cited only from an uncommitted note is still evidence.

### Overnight GPU queue

`scripts/overnight_queue.py` (runner in `battle.overnight_queue`) runs a JSON job list
strictly serially so only one process touches the GPU at a time:

```bash
cat > runs/overnight-multicam-20260918/jobs.json <<'EOF'
{"jobs": [
  {"name": "mediapipe-c10095", "interpreter": ["uv", "run"],
   "argv": ["battle-mediapipe-hands", "--view", "static-c10095"],
   "cwd": "/home/nick/src/battle", "timeout_s": 1800, "env": {"CUDA_VISIBLE_DEVICES": "0"}}
]}
EOF
uv run scripts/overnight_queue.py runs/overnight-multicam-20260918/jobs.json --dry-run
uv run scripts/overnight_queue.py runs/overnight-multicam-20260918/jobs.json
uv run scripts/overnight_queue.py jobs.json --continue-on-failure   # keep going past failures
```

Each job has `name`, `argv`, `cwd`, `timeout_s`, and optional `env` (merged over the current
environment) and `interpreter` (prepended to `argv`, e.g. `["uv", "run"]` or a venv python).
The queue re-executes itself under `systemd-inhibit --what=sleep:idle --why="battle overnight"`
(`--no-inhibit` to skip). Before every job it checks that `nvidia-smi` answers and that
`journalctl -k --since <queue start>` has no `NVRM`/`Xid` line (`--no-gpu-check` for CPU-only
lists); a failed check blocks the job and stops the queue. A job past its timeout is killed
with its whole process group (SIGTERM, then SIGKILL after 10 s). Events go to
`runs/overnight-multicam-20260918/queue.log` as JSON lines (`queue_start`, `gpu_check`,
`job_start`, `job_end`, `job_blocked`, `job_skipped`, `queue_end`) and each job's combined
stdout/stderr to `logs/<index>_<name>.log` beside it. The queue stops at the first non-zero
exit, timeout, spawn failure or GPU error unless `--continue-on-failure` is given; jobs after
the stop are logged as `skipped` with the reason. Exit code 0 only when every job succeeded.

The exporter writes already-normalized observations only: it never performs inference.
It can log an input video once when an approved local proxy is supplied, while mask
locations remain external/native artifact references. When a mask artifact root is
supplied, the referenced PNGs are also embedded as colored translucent segmentation
overlays in the RRD.

## Rerun hierarchy

The fixture exporter records the following stable hierarchy:

```text
world/<clip_id>/
  source/asset_reference                     # always: encoded-asset contract
  source/asset_policy                        # always: embedded-versus-contract policy
  views/<view_id>/video_asset                # conditional: video payload supplied
  views/<view_id>/video                      # conditional: frame references to that payload
  views/<view_id>/objects                    # conditional: object observations
  views/<view_id>/hands/{landmarks,skeletons,boxes} # conditional: hand observations
  views/<view_id>/hand_metrics/{count,mean_handedness_confidence}
  views/<view_id>/mask_references            # conditional: external mask references
  views/<view_id>/masks/<object_id>          # conditional: loaded per-object RGBA cut-outs
  views/<view_id>/segmentation               # conditional: sparse class-labelled masks
  views/<view_id>/tracker_diagnostics/
    object_score/lost_threshold              # conditional: diagnostics present
    object_score/<object_id>                 # conditional: diagnostics present
    iou_prediction/<object_id>               # conditional: model estimate present
  timing/{source, analysis, annotation, pose}
  quality/coverage
  frame_counter
```

Open the output in Rerun with `rerun artifacts/synthetic_fixture.rrd`. The blueprint
pins the spatial and timing views used by this session.

## Provenance

The approved Assembly101 G2 configuration records CC BY-NC 4.0 and G1/G2 user approval
for these local inputs. That approval does not decide whether a job-seeking or demo use
is noncommercial under the license or dataset terms. Never add raw recordings,
annotations, derived mask payloads, or checkpoint files to this repository. See
`docs/SOURCES.md`, `docs/LICENSES.md`, and `docs/method-ledger.md`.
