# Method ledger

Use one entry per method/configuration. Status values map to the versioned
`MethodState` schema: `pending`, `ready`, `running`, `succeeded`, `blocked`, `failed`,
or `not_run`.

## Required fields

- Method name and stage
- Schema/config version and Git revision
- State and owner
- Input clip/asset identifiers and provenance approval
- Environment and dependency lock reference
- Model/code version and license review status
- Clock mapping and sampling policy
- Chunk/continuity policy
- Output artifact URIs and checksum
- Success measure, data split, and uncertainty (when measured)
- Failure modes, blocker, and next decision

## Session-one entries

### fixture-rerun-export

- Stage: `export`
- State: `succeeded` on synthetic fixtures only
- Inputs: in-code normalized coordinates; no recording, annotation, or model call
- Clock policy: source/analysis/annotation/pose = 60/30/30/60 fps
- Output: local ignored `.rrd` created by `battle-export-fixture`
- Success measure: full fixture coverage and completed exporter contract
- Claim boundary: validates artifact shape and export behavior only; it is not an
  accuracy, performance, or dataset result.

### muggledsam-sam3-core-method-smoke (G2)

- Stage: `objects`
- State: `succeeded`; fixed smoke only, not an accuracy or full-duration result.
- Inputs: user-approved `assembly101-nusar-9033-g2` configuration
  (`configs/clips/assembly101_nusar_9033_g2.json`), its two G2 proxies, and no
  annotations or poses. The manifest carries approved raw/proxy checksums and a measured
  configuration SHA-256.
- Provenance/license: Assembly101 is recorded as CC BY-NC 4.0; G1/G2 approve the local
  procedure but do not resolve whether job-seeking, private demo, or public display is
  noncommercial. MuggledSAM source is Apache-2.0. Its SAM3.1 checkpoint was already
  locally available and was not downloaded or changed by this run; its separate
  agreement/license and any sharing rights remain unreviewed.
- Environment: Battle driver `uv` / CPython 3.12.13; isolated model worker
  `/home/nick/.pyenv/versions/muggled_sam/bin/python` (CPython 3.14.7) with source
  `/home/nick/src/muggled_sam`. The worker was intentionally dependency-light because
  that external environment does not import Battle's Pydantic dependency.
- Adapter/version: `battle.muggled_smoke` 0.1.0, based on the non-interactive
  `simple_examples/video_segmentation_multiplexed.py`; no external source was copied or
  modified.
- Fixed concepts: `hand`, `yellow toy body`, `toy wheel`.
- Clock/range: 30 FPS proxy analysis frames `[0, 300)` only, mapping to source
  215.000–225.000 s. Coverage per view is 10.0/180.0 s = 5.56%; no 180-second run was
  started.
- Continuity/memory: one sequential stream per view; no chunks or intentional ID
  resets; one prompt-memory entry, four frame-memory entries, maximum three initialized
  concepts. CUDA was restricted to device 0 and no concurrent GPU model process was
  detected.
- Static run: `runs/muggledsam-sam3-smoke-static-c10379-20260909t025854z/`;
  300/300 frames, elapsed 14.757 s, time-to-first usable output 4.651 s, peak allocated
  VRAM 1,965,193,216 bytes (1,874 MiB). Initial concepts produced persistent
  `sam3-00`/`hand` and `sam3-01`/`yellow toy body` IDs across all 300 frames; `toy wheel`
  had no initial detection.
- Ego run: `runs/muggledsam-sam3-smoke-ego-hmc21110305-20260909t025910z/`; 300/300
  frames, elapsed 14.056 s, time-to-first usable output 4.588 s, peak allocated VRAM
  1,965,065,728 bytes (1,874 MiB). Initial concepts produced persistent
  `sam3-00`/`hand` across all 300 frames; `yellow toy body` and `toy wheel` had no
  initial detection.
- Outputs: schema-validated `observations.jsonl`, `manifest.json`, runtime settings,
  and inference-free central-exporter `smoke.rrd` in each run directory. The RRD embeds
  one exactly 300-frame/10.0-second input clip, records boxes every analysis frame, and
  references rather than embeds mask PNGs. Static has 100 external masks (two IDs × 50
  samples); ego has 50 (one ID × 50); masks are at 5 FPS. Worker and video-export error
  logs are present and empty.
- Known limitations: absence of a concept in the first-frame detector is not recovery or
  an ID-continuity result; no re-detection/re-prompting occurred, by design. The smoke
  measures runtime/contract coverage only, not detection, segmentation, association, or
  scientific performance.
- G3 review artifacts: model-free overlays rendered from the completed normalized smoke
  outputs only:
  `runs/muggledsam-sam3-smoke-static-c10379-20260909t025854z/g3_review/static-c10379_contact_sheet.png`
  and
  `runs/muggledsam-sam3-smoke-ego-hmc21110305-20260909t025910z/g3_review/ego-hmc21110305_contact_sheet.png`.
  They show proxy analysis frames 0, 150, and 299 (0.000, 5.000, and 9.967 s), the
  masks recorded at the 5 FPS cadence, and boxes at all three frames. Frame 299 has
  boxes only because no mask was emitted at that non-5-FPS frame.
- G3 review checklist: (1) verify each returned ID follows the same physical target
  across all three timestamps; (2) flag any identity swap, mask/box drift, or loss of
  the target; (3) flag false-positive target assignments or implausible masks/boxes;
  (4) explicitly confirm the known initialization misses—static `toy wheel`; ego
  `yellow toy body` and `toy wheel`—rather than treating them as continuity failures.
- Reproduce the sheets without inference:
  `/home/nick/.pyenv/versions/muggled_sam/bin/python src/battle/g3_contact_sheet.py
  --run-directory <run-directory> --repository-root .`
- G3 decision: completed. The user approved static-only continuation and rejected the
  full ego continuation; the resulting view-separated candidate record follows.

### muggledsam-sam3 ego monochrome diagnostic

- Stage: `objects`; state: `succeeded` as two explicitly bounded diagnostic conditions,
  not as a new zero-shot or accuracy result.
- Baseline preserved: `runs/muggledsam-sam3-smoke-ego-hmc21110305-20260909t025910z/`
  remains the out-of-the-box three-text-concept (`hand`, `yellow toy body`, `toy wheel`)
  result. It initialized only `hand`; its visually poor continuous output is the
  comparison baseline, not a target-accuracy claim.
- Image evidence: `runs/muggledsam-sam3-ego-diagnostic-20260909t032450z/
  image_characterization.json` examined 300 decoded 954×720 BGR frames. All channel
  pairs had equality fraction 1.0, MAE 0.0, and correlation 1.0; the decoded proxy has
  no chroma. Aggregate OpenCV-gray p1/p50/p99 was 9/78/255, p1 ranged 8–10 and p99
  243–255 per frame, and 1.609% of pixels were exactly 255. The contrast condition is
  therefore a luminance-domain experiment; no pseudo-colorization/colorization was used.
- Prompt/preprocessing evidence: MuggledSAM's `prepare_image` accepts OpenCV BGR, then
  converts BGR→RGB, resizes, and normalizes. Its SAM3.1
  `tracking.encode_prompt_memory(encoded_image, box_xy1xy2_norm_list,
  fg_xy_norm_list, bg_xy_norm_list, ...)` cleanly supports direct normalized
  first-frame box/point prompts. The manual condition uses that API; it is not converted
  into a text-detection result.
- Condition configuration: `configs/muggledsam_ego_conditions.json` is versioned and
  schema-validated. `CONTRAST-NORMALIZED` uses only the `hand` text prompt and, per
  frame, gray p1–p99 scaling to 0–255 plus CLAHE clip limit 2.0 / 8×8 grid before
  three-channel replication. `MANUAL-SEED` keeps original decoded BGR and prompts
  frame 0 with pixel box `(380,300)–(555,435)`, normalized
  `(0.3987408,0.4172462)–(0.5823715,0.6050070)`, around the central hand holding the
  small part. Its initialized-observation confidence is a documented 1.0 sentinel,
  because the direct tracking-prompt API does not expose a detector confidence.
- Runs: `runs/muggledsam-sam3-smoke-contrast-normalized-hand-text-ego-hmc21110305-
  20260909t032357z/` and `runs/muggledsam-sam3-smoke-manual-seed-hand-box-ego-
  hmc21110305-20260909t032421z/`. Each contains a schema-valid separate manifest,
  worker result, observations, input-video Rerun export, and 50 external masks. Each is
  300/300 frames with continuous memory and no intentional ID resets.
- Factual proxy measures: baseline / contrast-normalized / manual-seed respectively:
  initial output count 1/1/1 (manual detector count not applicable); output coverage
  300/300 each; emission gaps 0/0/0; IDs introduced after initialization 0/0/0; TTFU
  4.588/4.834/4.183 s; elapsed 14.056/16.967/15.461 s; peak allocated VRAM
  1,965,065,728/1,965,065,216/1,943,994,368 bytes. Full structured measures are
  `runs/muggledsam-sam3-ego-diagnostic-20260909t032450z/proxy_metrics.json`.
- Visual QA: `runs/muggledsam-sam3-ego-diagnostic-20260909t032450z/
  ego_comparison_contact_sheet.png` compares all conditions at 0.000, 5.000, and
  9.967 seconds using the image representation supplied to each condition. It does not
  provide labels or prove accuracy. Neither condition provides a visually compelling,
  factual-proxy improvement over the known poor zero-shot track, so neither is proposed
  for a 60-second candidate. No 60- or 180-second ego inference was run.
- Next human decision: either approve a specifically defined, labelled evaluation
  protocol before any longer ego test, or stop ego continuation and retain this
  monochrome diagnostic as the view-scoped negative result.

### G3-approved static full candidate

- User decision: **approved** only `static-c10379` for the unchanged 180.0-second,
  5,400-frame run. **Rejected** `ego-hmc21110305` as a monochrome-domain stress-test
  negative result; its completed 10-second smoke remains the record, no full ego run
  was started, and its prompts were not changed.
- Run: `runs/muggledsam-sam3-g3-full-static-c10379-20260909t030710z/`; source,
  proxy, and config SHA-256 values are respectively
  `450731ebbb50f46cf8279383e4737db6d76e967f23580d3555de1888b78a9db9`,
  `ea9243591f6e8716ad7fe83f404e01ac45414071e9f06bc5ee68de29fcc543dc`, and
  `e6bb42376e72c8bdd58151afb748d4a4d0fbd3f6e097ae7190e70fe1a162fe36`.
- Result: `succeeded`; 5,400/5,400 frames and 180.0/180.0 seconds (100% of this
  static proxy), elapsed 206.621 s, TTFU 4.653 s, peak allocated VRAM
  1,965,193,216 bytes (1,874 MiB). CUDA device 0, bfloat16, 504-pixel max side,
  one prompt-memory entry, and four frame-memory entries were unchanged.
- IDs/continuity: created `sam3-00`/`hand` and `sam3-01`/`yellow toy body`; no
  re-detection or intentional ID restart occurred. The hand was emitted on 5,399/5,400
  frames and yellow toy body on 5,387/5,400, so their brief missing outputs must be
  reviewed as potential transient loss/drift rather than hidden restarts. `toy wheel`
  remained absent at initialization and was never started.
- Artifacts: normalized `observations.jsonl`, schema-valid `manifest.json`,
  `g3_full_static.rrd`, `input_5400f.mp4`, and 1,799 external masks at max 5 FPS.
  The RRD logs the 180-second input once and boxes at every analysis frame; it records
  external mask references only. Full-run QA:
  `runs/muggledsam-sam3-g3-full-static-c10379-20260909t030710z/g3_review/static-c10379_full_run_qa.png`
  at 0.000, 90.000, and 179.967 seconds; the last frame is boxes-only because of the
  5-FPS mask cadence.
- G4 decision required: choose exactly one: **accept** this static-only candidate as a
  view-scoped baseline (with the documented misses/losses and no cross-view claim), or
  **reject** it and specify whether the next approved work is manual initial-prompt
  review, a detection/re-prompt policy, or a separate monochrome-domain experiment.

### muggledsam-sam3 ego viewpoint screen

- Stage: `objects`; state: `succeeded` as three sequential, fixed 300-frame/10.0-second
  smoke screens. This is an additional e1/e2/e4 screen, not an alteration of the
  original static/e3 pair or its manifest.
- Inputs/provenance: `configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json`
  records user-approved G1/G2 scope, pinned `cvml-nus/assembly101` revision
  `bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967`, and only the three new HMC recordings.
  Raw/proxy paths, byte sizes, SHA-256 values, and ffprobe validation are in the ignored
  `data/raw/.../ego_viewpoint_screen_acquisition_report.md`. No annotations, poses, or
  checkpoint downloads were used.
- Fixed condition: original decoded BGR images; exact text concepts `hand`, `yellow toy
  body`, and `toy wheel`; CUDA 0, bfloat16, 504-pixel max side; one prompt-memory and
  four frame-memory entries; one continuous stream; no re-detection or intentional ID
  reset. No contrast normalization or manual seed was used.
- Runs: e1 `runs/muggledsam-sam3-smoke-ego-hmc21176875-20260909t033053z/`; e2
  `runs/muggledsam-sam3-smoke-ego-hmc21176623-20260909t033109z/`; e4
  `runs/muggledsam-sam3-smoke-ego-hmc21179183-20260909t033125z/`. Every run completed
  300/300 frames with schema-valid observations, manifest, worker result, and `.rrd`.
- View-separated factual outcome: e1 initialized `hand` only (270/300 hand-emission
  frames; gaps `[164,182)` and `[228,240)`; 45 masks); e2 initialized `toy wheel` only
  (no hand; wheel 294/300 with gap `[36,42)`; 49 masks); e4 initialized all three
  concepts (hand and wheel 300/300; yellow toy body 294/300 with gap `[36,42)`; 149
  masks). No newly introduced ID occurred after initialization in any screen run.
- Timings / peak allocated VRAM: e1 14.574 s / 4.327 s TTFU / 1,965,729,280 bytes; e2
  14.641 s / 4.327 s / 1,965,729,280 bytes; e4 15.261 s / 4.319 s / 1,965,320,704
  bytes. The retained e3 zero-shot baseline was read-only comparison evidence:
  14.056 s / 4.588 s / 1,965,065,728 bytes; hand 300/300; 50 masks.
- Review artifact: `runs/muggledsam-sam3-ego-viewpoint-screen-20260909t033053z/
  g3_review/ego_viewpoint_screen_4view_contact_sheet.png` shows e1/e2/e3/e4 at
  0.000/5.000/9.967 seconds. It renders only normalized smoke outputs and makes no
  accuracy claim. The corresponding machine-readable report is
  `runs/muggledsam-sam3-ego-viewpoint-screen-20260909t033053z/viewpoint_screen_report.json`.
- Human-review boundary: visual screening tentatively ranks e4 first (hand output is
  visibly co-located with the working hand across the three samples), e1 second
  (initial overlap but loss/drift away from a clearly hand-associated region), e3 third
  (preserved known poor track), and e2 last (no hand initialized). With no labels, this
  supports requesting—not approving—a 60-second e4-only candidate; it does not validate
  segmentation, association, or cross-view performance.
- Next decision: approve or reject a single 60-second e4 candidate under the unchanged
  condition, with a defined human review protocol; if rejected, choose a labelled
  evaluation protocol or stop ego continuation. Do not infer an aggregate ego result.

### muggledsam-sam3 G4 e4-only 60-second candidate

- Stage: `objects`; state: `succeeded` for only e4 / `ego-hmc21179183`, proxy frames
  `[0,1800)` / 60.0 seconds. This preserves every static/e1/e2/e3 result and does not
  begin the remaining 120 seconds or any 180-second ego inference.
- Inputs/config: existing approved viewpoint-screen G2 manifest
  `configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json`, SHA-256
  `df59392d786d2dffa3842a198098de6277ee09b11382a0577845f9ce8bbb63b4`; e4 raw SHA-256
  `ac5520a907acb4b31588e613d61c09e56d08e625770d4914242f66062c8b98d2`; e4 proxy
  SHA-256 `ce9ae52af7184c44faabb0a0dd5157eb1d4d69af6401674d157a5f2a3dde4007`.
- Fixed condition: original decoded BGR; exact text concepts `hand`, `yellow toy body`,
  `toy wheel`; one sequential continuous stream; no chunks/re-detection/intentional ID
  resets; one prompt-memory and four frame-memory entries; CUDA 0, bfloat16, max side
  504. No external repository/environment, model, or data was changed/downloaded.
- Run/artifacts: `runs/muggledsam-sam3-g4-e4-candidate-ego-hmc21179183-20260909t033519z/`
  contains schema-valid normalized observations and manifest, `input_1800f.mp4`,
  `g4_e4_60_second_candidate.rrd`, 887 external masks, worker logs/settings, factual
  `e4_candidate_report.json`, and
  `g3_review/ego-hmc21179183_g4_e4_candidate_qa.png`.
- Measurements: 1,800/1,800 frames, 60.0 seconds / 33.33% of the 180-second e4 proxy;
  elapsed 69.792 s, TTFU 4.297 s, and peak allocated VRAM 1,965,320,704 bytes.
- Initialization and continuity: all concepts initialized (`hand`/`sam3-00`, `yellow toy
  body`/`sam3-01`, `toy wheel`/`sam3-02`). Hand emitted 1,791/1,800 frames (99.50%) with
  gaps `[1345,1353)` and `[1365,1366)`; body 1,768/1,800 (98.22%) with five gaps; wheel
  1,742/1,800 (96.78%) with ten gaps. No track introduced a new ID after initialization;
  intentional ID resets were false.
- QA: fixed frames 0/900/1799 (0.000/30.000/59.967 seconds) render three, two, and
  zero external masks respectively because masks are capped at 5 FPS; boxes are present
  at 30 FPS. This is review-only evidence with no ground truth and no accuracy,
  segmentation, association, or general ego-performance claim.
- G4 decision required: choose exactly one: **accept** this e4-only 60-second candidate
  as a view-scoped baseline; **reject** it and stop/define labelled evaluation; or
  **run remaining 120 seconds** under this unchanged condition after explicit approval.

### muggledsam-sam3 e4 interactive box-prompt calibration

- Stage: `objects`; state: `ready` as a user-operated calibration utility, not a run.
- Scope: uses only the approved e4 `HMC_21179183_mono10bit` proxy from
  `configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json`. Default
  proxy-relative samples are 0.0, 10.0, 30.0, and 50.0 seconds / 30 FPS. It neither
  changes the G2 manifest nor starts a 60-/180-second tracking job.
- Manual operation: the user draws full-resolution pixel rectangles in OpenCV, provides
  `hand`, `yellow toy body`, `toy wheel`, or a custom label, reviews all returned
  MuggledSAM interactive-image candidates, and explicitly marks any candidate eligible
  for finalization. The schema-validated, gitignored calibration manifest retains
  source/analysis/proxy time, pixel and normalized coordinates, G2/proxy identity and
  hashes, decoder output metadata, and a compact external result directory.
- Decoder evidence: MuggledSAM SAM3.1 `get_interactive_context().generate_masks(...)`
  returns four mask candidates and model IoU estimates for the box prompt. The utility
  retains every candidate, labels maximum IoU (lowest index on tie) as deterministic
  best, saves per-candidate masks and a review overlay, and explicitly records that
  this API does not expose a stability score. The estimates are not ground-truth
  segmentation accuracy.
- Finalization: an explicit `--finalize --candidate-id ...` produces only a
  `proposed_non_authoritative` prompt configuration from user-selected candidates. A
  later tracking integration must address MuggledSAM's non-multiplexed direct box-memory
  API and reference frames after frame 0. No comparison to the preserved zero-shot
  baseline is claimed before a later human G gate defines and reviews it.
- e4 seed-set resolution: the user retained exactly these human-accepted frame-0 mask-0
  selections: `left_hand` / `t000000-b01`, `yellow_toy_body` / `t000000-b03`, and
  `toy_wheel` / `t000000-b04`. The extra human-accepted `right_hand` /
  `t000000-b18` mask-2 evidence remains in the calibration manifest but is explicitly
  excluded from finalization eligibility. The resulting schema-validated, non-authoritative
  proposal is `runs/muggledsam-sam3-e4-web-calibration-e55b5d0abe02/
  proposed_tracking_prompt.json`; its contact-sheet review is in the same run's `results/`.

### muggledsam-sam3 e4 rapid calibration workspace

- Stage: `objects`; state: `ready` as a localhost-only, selected-frame manual calibration
  workspace. It is not a tracking run and starts no video tracking inference.
- Inputs: the existing approved e4 proxy and G2 configuration. The default prompt frame
  requests are proxy seconds `0, 10, 30, 50`; previews are decoded only on request and
  stored in the ignored calibration run.
- Environment: Battle CPython 3.12/`uv` hosts the threaded standard-library HTTP server;
  the separately managed MuggledSAM Python process owns one warm interactive image
  decoder, OpenCV capture, frame cache, mask writes, and newline JSON protocol. CUDA is
  restricted to device 0 under the same source/PYTHONPATH isolation as the prior adapter.
- Review/audit: browser-drawn pending boxes, labels, exact frame references, all decoder
  candidates, explicit human mask selections, and compact persisted-state differences are
  schema-validated and atomically autosaved. Worker stderr stays in the ignored run
  directory; source previews, masks, candidate panels, and overlays remain external
  artifacts.
- Finalization gate: only a human-accepted selected candidate from proxy frame 0 can be
  marked eligible or enter a proposed non-authoritative tracking seed. Later frames are
  decoder checks. Legacy manifests stay readable but any former model-only eligibility is
  marked unverified and cannot be finalized until explicitly reviewed in the workspace.
- Known limitations: fixture/API tests do not initialize CUDA, OpenCV, a model, or a
  browser GUI. Decoder IoU, mask area, and prompt-overlap values are diagnostics, not
  accuracy measures. MuggledSAM model compatibility and detailed user interaction must
  be verified in a local desktop session before any human decision.

### muggledsam-sam3 e4 manual-seed multiplexed smoke

- Stage: `objects`; state: `blocked` before inference. The requested scope remains only
  e4 / `ego-hmc21179183` proxy frames `[0,300)` / 10.0 seconds; no 60- or 180-second
  run was started.
- Seed validation passed: proposal
  `runs/muggledsam-sam3-e4-web-calibration-e55b5d0abe02/proposed_tracking_prompt.json`
  matched calibration-manifest SHA-256
  `79cf1f6f29d3ea311f964f6c7d011ba3112848d8b4788c333d9a528716f34086`, and its exact
  mask-0 seeds are `left_hand`/`t000000-b01`, `yellow_toy_body`/`t000000-b03`, and
  `toy_wheel`/`t000000-b04`. Human-selected `right_hand`/`t000000-b18` is excluded.
- Implementation: the new manual path loads the three saved, source-sized selected masks,
  verifies each SHA-256, batches them into one
  `tracking.encode_prompt_memory_from_mask(...)` call, and advances one
  `step_video_masking_multiplex(...)` stream. Target-to-normalized-ID mapping is fixed
  as left hand→`sam3-00`, body→`sam3-01`, wheel→`sam3-02`; it makes no detector/text
  call and uses no chunks or intentional ID resets. Memory remains bounded to one
  prompt-memory and four frame-memory entries, CUDA device 0, bfloat16, and 504 pixels.
- Blocker: the local calibration workspace remains active and owns the same model on
  CUDA 0 (PID 19647, about 1.6 GiB). The safety preflight therefore refused a competing
  worker and recorded a schema-valid blocked attempt at
  `runs/muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260909t223016z/`.
  It contains no inference observations, RRD, or QA comparison and its placeholder
  zero-output metrics must not be treated as measurements.
- G3 decision: **do not run any longer duration.** This is not promising or unpromising
  evidence because inference did not start. Close the calibration workspace to free CUDA
  0, then rerun exactly this 300-frame manual-seed multiplexed smoke and review its
  measured 0/5/9.967-second comparison before requesting another duration.

### hand-pose adapter

- Stage: `pose`
- State: `not_run`
- Blocker: deferred until a provenance-approved source and session-two adapter plan
- Environment candidate: `~/.pyenv/versions/wilor`
- Weights: not downloaded
