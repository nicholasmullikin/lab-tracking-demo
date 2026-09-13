# Battle: video-understanding method lab

This is a no-training method lab for building a reviewable, machine-readable video
execution record. It includes versioned schemas, synthetic normalized fixtures, an
inference-free Rerun export, and a strictly bounded headless SAM3 smoke adapter—not
model accuracy claims.

## Current scope

The initial session establishes the artifact contracts needed before any dataset or
model work:

- typed Pydantic manifests for clips, runs, timing, coverage, and observations;
- YAML templates for method settings and an Assembly101 comparison clip;
- fixture-only validation and success-measure calculations;
- a deterministic Rerun `.rrd` exporter for video, points, boxes, hands, timing, and
  external mask references.

No recordings, annotations, or model weights are included. The SAM3 adapter lives in
the repository but executes only against the user-approved G2 proxies. `data/`, `runs/`,
caches, and Rerun outputs are ignored by Git.

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

## G3 decision and candidate outcome

The human G3 review approved only the static RGB view and rejected the ego monochrome
view as a likely grayscale-domain stress case. The completed static candidate is
`runs/muggledsam-sam3-g3-full-static-c10379-20260909t030710z/`; it covers all 5,400
analysis frames / 180.0 seconds with unchanged concepts and continuous-memory settings.
The completed ego 10-second smoke remains a documented negative stress-test result;
there is no full ego run and its prompts were not changed. This is not an aggregate
claim that SAM3 works across views.

Re-run the approved candidate only with:

```bash
uv run battle-muggled-smoke --view static-c10379 --g3-full-static
```

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
detections are logged at every analysis frame, and masks are external PNG artifacts at
at most 5 FPS. The RRD preserves those references and embeds a composed segmentation
image at each mask frame, rendered as a translucent, per-object-colored overlay.

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

1. Select a filmstrip timestamp or enter an approved timestamp, then load its source
   frame. The canvas preserves the complete 954×720 aspect ratio. Use scroll to zoom,
   Shift/Alt/middle-button drag to pan, `B` to draw/edit a box, `F` to add a green
   foreground point, and `N` to add a pink background-exclusion point. Drag a marker to
   move it, right-click it to remove it, or use **Clear selected points**. Prompt edits
   autosave atomically.
2. Set the target label, draw and select its box, then add target-specific points. The
   stored source-pixel clicks and normalized MuggledSAM `boxes`, `fg_points`, and
   `bg_points` payload are retained with the pending prompt and decoded candidate.
   The right panel shows the resulting compact manifest difference.
3. Use **Decode all pending on this frame** to process every target drawn at the active
   timestamp, even when the most recently drawn box is selected. **Decode selected prompt**
   remains available for an intentional one-prompt retry. The UI remains available while
   the dedicated decode queue runs. Every decoded target group remains visible in configured
   target order; each compact card exposes every returned mask with a thumbnail of its
   full-context/padded-ROI review and a link to the full-size review. Explicitly select a
   mask. Model IoU, mask area, and prompt overlap are diagnostics, not segmentation-accuracy
   measures.
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

For a no-model static route check, use `--no-worker`; frame decoding and masks will
intentionally report that the decoder is offline. The fixture tests exercise the persistent
JSONL protocol, queue/persistence, and API/static contract without CUDA, OpenCV, or a
checkpoint. Actual image decoding and browser interaction still require the local
MuggledSAM checkpoint, CUDA runtime, and a desktop browser; they are not model-accuracy
or human-GUI test results.

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
fixed target-to-normalized-ID associations, and writes external masks at most 5 FPS.
The worker refuses to begin while another GPU model process is active; close the local
calibration workspace before executing it. A completed run remains review-only and
makes no segmentation, association, or accuracy claim.

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
  source/asset_reference
  views/<view_id>/{objects, hands, segmentation, mask_references}
  timing/{source, analysis, annotation, pose}
  quality/coverage
```

Open the output in Rerun with `rerun artifacts/synthetic_fixture.rrd`. The blueprint
pins the spatial and timing views used by this session.

## Provenance

The approved Assembly101 G2 configuration records CC BY-NC 4.0 and G1/G2 user approval
for these local inputs. That approval does not decide whether a job-seeking or demo use
is noncommercial under the license or dataset terms. Never add raw recordings,
annotations, derived mask payloads, or checkpoint files to this repository. See
`docs/SOURCES.md`, `docs/LICENSES.md`, and `docs/method-ledger.md`.
