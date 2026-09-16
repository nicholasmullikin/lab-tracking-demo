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

In one paragraph: the repository has typed Pydantic manifests for clips, runs, timing,
coverage, and observations; an inference-free Rerun exporter; and one method, SAM3 via
MuggledSAM, exercised on Assembly101 RGB static and monochrome ego views. Text prompting,
human-seeded masks, reviewed keyframe corrections, and a browser calibration workspace
were all tested. A 196.7-second static exploration failed through identity drift, then
aligned 92.7-second static and monochrome ego runs were completed from a cleaner
separated-parts frame. Human review retained only their first 60 seconds for the
synchronized two-view comparison; later outputs remain failure evidence. The second
method in the original core spine, MediaPipe Hands, has a selected 60-second static run
merged into the focused first-minute comparison. The Sep 16 exploratory queue has one
Battle-integrated BoxMOT and CLIP+Drop-DTW smokes. WiLoR, Grounded-SAM-2, SAMURAI, DAM4SAM,
and Kineo remain external partials after an adversarial evidence audit; ATHENA is blocked.

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

### Grounded-SAM-2 — external partial / single-frame smoke

The native JSON records one 1280×720 `hand` box (score 0.953125) and a COCO-RLE SAM2 mask for
the `hand.` prompt. It used Hugging Face `IDEA-Research/grounding-dino-tiny` after the local
CUDA extension build failed, but does not pin that HF model revision or record a measured
runtime. `battle-import-external-smoke grounded-sam2` now creates normalized box observations
and `grounded_sam2.rrd`; it intentionally does not claim video propagation or a normalized mask.

### SAMURAI — external partial

`tracking.mp4` is a genuine 1280×720, 30-FPS, 602-frame output matching the 602-frame /
20.0667-second input, seeded by the preserved `(881,446,152,129)` hand box. The script selects
`configs/samurai/sam2.1_hiera_t.yaml`, which sets `samurai_mode: true`; it is genuine SAMURAI
mode despite calling the shared predictor constructor. No native masks, normalized output, runtime
log, or durable command manifest was preserved, so it remains an external partial.

### DAM4SAM — external partial

The native output has 602 unique, nonempty 1280×720 PNG masks named `00001.png` through
`00602.png`; its log measures 53.6 s. The source log is consistent with DAM4SAM/SAM2 code, but
the headless wrapper and exact command were not preserved, and the SAM2 post-processing extension
was skipped for an ABI error. The importer writes 602 normalized frame records, a
content-addressed mask index, and `dam4sam.rrd`, but this is not proof of DAM4SAM distractor
semantics.

```bash
uv run battle-import-external-smoke dam4sam
```

### Kineo — external partial

The trimmed NLF-only job produced readable PKLs: 462 `subject_0` 2D boxes and 462
`nlf_smplx` records (1,079 points each) over input frames 0–601, plus one estimated intrinsics
record. Its stage durations sum to 28.851 s; no total wall runtime is claimed. Seven raw boxes
extend outside image bounds. The Kineo checkout is dirty and the headless YAML is ignored, so it
is not a reproducible clean-upstream run. The importer preserves PKL hashes, normalizes its person
boxes (not its incompatible 1,079-point format), and creates `kineo_nlf_boxes.rrd`.

```bash
uv run battle-import-external-smoke kineo
```

## Exploratory methods still blocked

- **ATHENA:** blocked. Official Assembly101 documentation says the ~72 GB `AssemblyPoses.zip`
  contains 2D/3D hand poses, camera extrinsics, and positions; it does not establish that
  intrinsics are absent. The archive was not downloaded or inspected while ATHENA requires
  calibration inputs. The blocker is lack of approved, bounded-access calibration files—not a
  proven upstream absence of intrinsics.
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

Recordings are keyed by clip (`battle-<clip_id>`) and run id, so runs of different clips
or of the same clip open side by side without merging.

## Validate and export fixtures

```bash
uv run ruff check .
uv run pytest
uv run battle-export-fixture --output artifacts/synthetic_fixture.rrd
```

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
