# Method ledger and timeline

This is the record of what this repository set out to do, what actually happened, in
what order, and why it diverged from the plan. Part 1 is the short version: the original
ask, a goals scorecard, a dated timeline, and the plan-versus-actual list. Part 2 holds the
detailed per-method records with their claim boundaries; the timeline points into them.

Conventions: dates are local (UTC-4). Run directory names carry UTC timestamps, so a run
tagged `20260910t024052z` happened on the evening of Sep 9 local time. `runs/` is
gitignored; run IDs are cited so the on-disk evidence can be found, not because it is
tracked. States map to the versioned `MethodState` schema: `pending`, `ready`,
`running`, `succeeded`, `blocked`, `failed`, or `not_run`.

## Part 1: goals, timeline, and divergences

### The original ask (Sep 8, 2026)

The verbatim request is preserved at the top of
[`battle_plan.agent.final.md`](../battle_plan.agent.final.md). In short: get several
perception repositories working out of the box on Assembly101, visualize all of their
outputs in one Rerun recording, skip building a data pipeline, respect a 16 GB card and
limited time, and do it in a way that demonstrates a properly run ML project. Candidate
methods named at the outset: SAM3 via MuggledSAM (tested, "worked decently well"), WiLoR
("wasn't terrible"), Kineo ("interesting, but I don't think it'll end up helping a ton"),
FineBio access pending, audio "probably overkill for now".

That request was refined the same evening into the operating plan preserved as
[`docs/plan-2026-09-08-assembly-rerun-lab.md`](plan-2026-09-08-assembly-rerun-lab.md):
one pinned three-minute Assembly101 segment as the comparison unit, a core spine of
MuggledSAM/SAM3 plus a static-view MediaPipe hand baseline, a tiered exploratory queue
(WiLoR, BoxMOT, CLIP + Drop-DTW, Grounded-SAM-2, SAMURAI/DAM4SAM, ATHENA, Kineo), five
pre-accuracy success measures, five hard human gates (G1 data, G2 experiment contract,
G3 semantic QA, G4 stop/re-scope, G5 claims), and an explicit no-training,
no-annotation, no-accuracy-claims rule.

### Goals scorecard

| Goal (from the Sep 8 ask and plan) | Status | Evidence |
| --- | --- | --- |
| Typed manifests, fixture tests, inference-free Rerun exporter | Done | `src/battle/schemas.py`, `src/battle/exporter.py`; 335 tests in the 8 s default tier, 13 more behind `real_data`/`gpu` markers |
| Pin one Assembly101 segment with source/analysis/annotation/pose clocks | Done | `configs/clips/*.json`; nusar-9033, 215.000–395.000 s |
| MuggledSAM/SAM3 running over the full 180 s static view | Done (aligned hybrid: three text, one reviewed mask) | [Sep 14 aligned hybrid](#sep-14-aligned-static-hybrid-candidate) |
| MuggledSAM/SAM3 running over the full 180 s ego view | Done, but only with human-seeded masks | [Four-target 180 s baseline](#sep-9-evening-four-target-180-second-ego-baseline) |
| Both views on one synchronized Rerun timeline | Done, with a caveat found Sep 17: the static video lags the dataset pose clock by 9 pose frames (~150 ms); the existing comparisons assumed zero relative offset and are not yet corrected | `battle-build-ego-static-comparison`; [Sep 15 four-part experiment](#sep-15-four-part-static-reassembly-experiment); [Sep 18 dataset reference](#sep-18-assembly101-dataset-hands-and-fine-grained-labels-in-the-v4-review) |
| Five pre-accuracy measures recorded per run | Done | Every `worker_result.json` and `manifest.json` |
| MediaPipe Hands static-view baseline (core spine) | Selected 60 s run complete | [hand-pose adapter](#hand-pose-adapter) |
| Second method in the viewer (MediaPipe) | Done; merged into focused first-minute comparison | [hand-pose adapter](#hand-pose-adapter) |
| Fixed two-timestamp human QA per completed method | Records prepared; human dispositions pending | Human-selected source frames 14,868/21,732; aligned static, ego, and preserved historical records under `docs/qa/` |
| Exploratory queue (WiLoR, BoxMOT, CLIP + Drop-DTW, Grounded-SAM-2, SAMURAI, DAM4SAM, ATHENA, Kineo) | All eight attempted in one autonomous pass at smoke tier; four-part segmentation arms and a unified review surface built; ATHENA blocked on intrinsics until Sep 17; Kineo body-only partial | [Sep 16 queue](#sep-16-exploratory-queue-autonomous-pass); [review surface](#sep-16-final-unified-exploratory-review-surface) |
| No training, no annotation project, no accuracy claims | Held, with one gate crossed on request: dataset poses and fine-grained labels were acquired Sep 17 as review context only | Reviewed masks are calibration seeds, not labels; no metric vs. ground truth anywhere; [Sep 17 acquisition](#sep-17-assembly101-poses-extrinsics-and-fine-grained-annotations-selective-acquisition) |
| Four physical components through reassembly | Focused 92.7 s run completed; the first minute was human-reviewed Sep 17 and is the retained comparison window; identity failures at 279/573/1043 are the documented SAM3 limit | [Sep 15 four-part experiment](#sep-15-four-part-static-reassembly-experiment); [Sep 17 human review](#sep-17-first-minute-v4-human-review-and-follow-up-rebuild) |
| Git history from the start | Missed, then repaired | First commit Sep 13 after five days of uncommitted work |
| FineBio | Still pending | Not part of any run |
| Audio | Deferred by plan | Not revisited |

The honest summary, as of Sep 18: nine methods attempted, one clip, one minute reviewed
closely. SAM3 with human seeds is the only method that completed both target views;
MediaPipe and WiLoR completed the retained first minute of the static view; the rest are
smoke-tier evidence. The SAM3 spine is at its ceiling for this part taxonomy (label
migration between similar dark parts under rotation), the hand story has no trustworthy
3D from any monocular source, and the review tooling built to iterate on that one clip is
where most of the code and roughly half the time went. The multi-view line is now
unblocked but untried.

### Timeline

#### Sep 8, evening: plan, then foundation and first real runs

- 21:34–21:56. Plan written and revised through two external reviews. Key revisions:
  Assembly101 video is 60 fps, not 30, so a 30 fps analysis proxy with explicit clock
  mappings is required; the RGB static view is primary and the monochrome ego view is a
  stress test; tracker continuity is per adapter, no forced chunk resets; the "3-minute
  video" is the input, not an output deliverable; Kineo stays a final two-hour trial.
  The five success measures (coverage, time to first usable output, peak VRAM, runtime,
  ID resets) and a Rerun fidelity policy (video once, boxes at full cadence, masks at
  5 fps, native masks kept outside the `.rrd`) were added at the user's request.
- 21:56. "Ok begin." Scaffold, schemas, fixture exporter, clip configs, and proxy
  generation. G1/G2 approved the nusar-9033 recording, static `C10379` and ego
  `HMC_21110305`, interval 215–395 s.
- ~22:58. First SAM3 smokes (300 frames, text prompts `hand`, `yellow toy body`, `toy
  wheel`) on static and ego. Static initialized two of three concepts, ego only `hand`
  with a visibly poor track. Record: [core smoke (G2)](#sep-8-late-evening-muggledsam-sam3-core-method-smoke-g2).
- ~23:07. G3 decision: static approved for the full 180 s, ego rejected as a
  monochrome negative result. Full static run completed (5,400 frames, 207 s).
  Record: [G3 static full candidate](#sep-9-early-morning-g3-approved-static-full-candidate).
- ~23:24. Ego monochrome diagnostic: the proxy has no chroma; contrast normalization and
  a single manual hand box were tried as bounded conditions; neither was compelling.
  Record: [ego monochrome diagnostic](#sep-8-late-evening-muggledsam-sam3-ego-monochrome-diagnostic).
- ~23:30–23:35. Three other ego cameras screened; `HMC_21179183` (e4) initialized all
  three concepts and was promoted to a 60 s candidate. Records:
  [viewpoint screen](#sep-9-early-morning-muggledsam-sam3-ego-viewpoint-screen) and
  [G4 e4 60 s candidate](#sep-9-early-morning-muggledsam-sam3-g4-e4-only-60-second-candidate).
- 23:39. "I don't see anything": the first Rerun blueprint opened blank. Repaired
  without re-running inference, which is what the inference-free exporter was for.
- 23:46. Direction change. The user diagnosed the ego failure as a prompting problem
  ("SAM is likely getting confused" by black-and-white footage) and asked to draw
  rectangles and inspect what the image decoder returns for them. This is the moment the
  project pivoted from zero-shot text prompts to human-seeded tracking.

#### Sep 9, after midnight: calibration tooling

- 23:51–00:19. An OpenCV box-prompt CLI was built, hit interactive-terminal and ROI-mode
  problems, and was replaced within the hour by a browser workspace ("we need to make a
  better tool so I can rapidly do all the frames and all of the boxes"). Headless
  validation passed at 00:19. Records:
  [box-prompt calibration](#sep-9-after-midnight-muggledsam-sam3-e4-interactive-box-prompt-calibration)
  and [rapid calibration workspace](#sep-9-after-midnight-muggledsam-sam3-e4-rapid-calibration-workspace).

#### Sep 9, evening: manual-seed tracking, static comparison, and the target pivot

- 18:01. Frame-0 masks accepted for `left_hand`, `yellow_toy_body`, `toy_wheel`. The
  smoke was first refused because the calibration workspace held the GPU
  (`…20260909t223016z`), then ran at 18:39 (`…223928z`, 300 frames, 15.5 s).
  Record: [manual-seed multiplexed smoke](#sep-9-evening-muggledsam-sam3-e4-manual-seed-multiplexed-smoke).
- 18:42–18:46. Segmentation overlays added to the exporter; "Ok not bad. Good to
  continue." Full 180 s ego run with three seeded targets (`…224835z`, 5,400 frames,
  205 s, 1.85 GiB).
- 19:03. Red full-frame segmentation background removed from the viewer.
- 19:07. Synchronized ego-versus-static comparison recording added (the two views the
  plan called for, on one timeline).
- 19:25. Watching the comparison, the user identified the tracked "toy wheel" as the
  wrong part and pivoted the vocabulary to `yellow_toy_top` and `black_toy_top_base`.
  Assembly101 was checked for anything to cross-check against: it ships 30 fps action
  and mistake intervals and optional hand poses, but no object boxes or masks, so the
  new targets cannot be validated spatially from the dataset.
- 21:39–21:52. Proposal creation bugs fixed; the concept of a "proposal" (the frozen,
  hashed set of frame-0 masks that seeds the tracker) explained. First look at the
  oversized-box artifact: at 0.600 s the left-hand mask is a normal 6,077-pixel hand
  plus one isolated pixel near the bottom of the frame, and the min/max box spans both.
  Connected-component filtering was named as the fix; it was not implemented until Sep 13.
- 22:12. Right hand added as a distinct fourth multiplexed target rather than splitting
  a merged both-hands mask. Four-target 180 s baseline completed (`20260910t024052z`,
  5,400 frames, 190 s, 1.90 GiB; coverage 100 / 98.2 / 99.7 / 99.6 %). Record:
  [four-target baseline](#sep-9-evening-four-target-180-second-ego-baseline).
- 22:53. "The first frame is not super representative of the object": multi-keyframe
  human corrections designed. The user asked whether this was rebuilding something that
  already existed; the answer was that upstream SAM 3.1's interactive session takes
  points and boxes rather than reviewed masks, and CVAT would need heavy integration, so
  a thin correction layer was justified. Memory semantics decided: at a correction
  keyframe, replace that slot's prompt memory with the reviewed mask and clear frame
  memory, keeping object IDs. Reviewed masks are retained as future training data but no
  retraining is planned.
- 23:17–23:19. Labeling UX: masks rendered in the primary viewport, keys `1`–`4` to pick
  a decoder candidate.

#### Sep 10, morning: workflow simplification

- 10:05–10:13. Viewport tint fixed to positive mask pixels only. The two-button
  "create frame-0 proposal / create correction schedule" flow collapsed into one
  **Finalize tracking plan** action with plain labels; both artifacts are still written
  for integrity checks.

#### Sep 11–12: no work

#### Sep 13, afternoon: labeling finished, repository committed

- 16:02–16:11. Reject/remove for unwanted decoder candidates. Foreground/background
  point prompts added to the box prompt, specifically for the right hand against its own
  shadow. Backwards compatibility dropped; a fresh workspace started.
- 16:18–16:42. Display-only image view aids (adaptive threshold, brightness, contrast,
  Canny, Shen-Castan, zero crossings, boundary tensor, four corner operators) built on a
  from-source VIGRA 1.12.4, which required building Boost.Python against the venv's
  CPython 3.12 because Fedora ships it only for 3.14. Build notes:
  [`docs/vigra-build.md`](vigra-build.md). Panel made collapsible and 4–15x faster
  (per-operator caching, concurrent operators, browser-side brightness/contrast).
  These affect only what the human sees, never the pixels sent to SAM3.
- 17:04–17:29. Two workspace defects: a cosmetic "masks required at frame 0" control on
  later-keyframe cards, and a finalized-plan lock that silently swallowed mask clicks.
  A **Reopen for editing** path was added; the lock itself was kept because the plan's
  embedded manifest hash is only meaningful if the manifest stops changing.
- 17:43. Labeling finished: four targets at keyframes 0, 2.5, 5, 8 s. "How are we
  currently tracking progress?" surfaced that the repository had zero commits after five
  days. Eight layered commits were made on the spot; the ledger's "Git revision" field
  finally had something to point at.

#### Sep 13, evening: the tracker gets instrumented

- 17:52. First corrected run (`…215412z`, 300 frames, 15.0 s, 1.93 GiB): all four
  targets seeded, corrections applied at frames 75/150/240, three of four targets present
  on every frame. Best ego result so far. Record:
  [multi-keyframe corrections](#sep-13-evening-muggledsam-sam3-e4-multi-keyframe-human-corrections).
- 17:57. Frame counter panel added to every recording (commit `127e761`).
- 18:04–18:19. The user listed six failure timestamps (1.2, 5.04, 5.98, 6.9, 7.81,
  9.5 s) and asked what else SAM3 exposes. Finding: the worker discarded the model's
  predicted IoU and clamped its unbounded presence logit to `[0, 1]`, so all 1,193
  observations read `confidence: 1.0`. Both signals were recorded and plotted in Rerun
  (`…221457z`, commit `0d9b0bb`). The six timestamps split into two modes: 1.2 s is a
  tracker loss the model knew about (score decays over four frames to −3.6, IoU to 0),
  and later shown to be `yellow_toy_top` leaving the top of the frame under head motion,
  so the negative score was correct; the other five sit at healthy scores of 10–12 and
  are the model being confidently wrong, which no threshold will catch. Record:
  [tracker diagnostics](#sep-13-evening-tracker-score-and-iou-diagnostics).
- 18:19–18:32. Resolution and memory depth made run conditions (commit `93fdc22`).
  Sweep at 504/720/1008 px: stray-pixel box inflation fixed at ≥720, weakest-target IoU
  up 44 %, VRAM flat at ~2 GiB, runtime 16 → 25 → 37 s. Record:
  [resolution sweep](#sep-13-evening-encoder-resolution-sweep).
- 18:32–19:05. 30 vs 60 fps comparison: new 60 fps proxy and clip config, analysis fps
  plumbed through worker and schemas, and frame-zero-only seeds transferred after checking
  the same raw-source checksum, source start instant, dimensions, and scaling policy. Proxy
  identity and fps may differ; source end time is not part of this transfer check. The
  output-continuity and model self-estimate diagnostics showed no consistent/measurable
  quality benefit supporting 1.85x compute, so the technical recommendation was to keep
  30 fps. These are not ground-truth accuracy measurements. Along the way the raw e4
  recording turned out to be 636x480, so the 954x720 proxies are upscales and 720 is the
  sensible ceiling. Record:
  [frame-rate comparison](#sep-13-evening-muggledsam-sam3-e4-analysis-frame-rate-comparison).
- 19:05–19:30. Two exporter defects found while viewing the arms together (fixed
  application id shared by every recording; identical recording ids from one process).
  External mask PNG artifacts moved from `mask_period_frames=6` (5 fps at the 30 fps
  analysis clock) to every analysis frame. The viewer now logs those worker PNGs as
  per-object RGBA `EncodedImage` cut-outs every frame; the separate class-labelled
  `SegmentationImage` was and remains a sparse 1 Hz record. Same record as above.
- 19:56–20:11. "The bbox is less reliable than the actual segmented pixels": the box is
  the min/max of every positive pixel, so mask speckle balloons it. Replaced with the
  union of 8-connected components at ≥20 % of the largest component's area. Rerun
  viewer whitelisted in the worker's GPU guard. Arms re-run. Record:
  [box derivation](#sep-13-late-evening-box-derivation-and-gpu-guard).
- ~20:28. The user explicitly approved the earlier technical recommendation: "30fps is
  fine." This approval was separate from the 18:32–19:05 experimental work.
- 20:28–20:33. Documentation pass: this file restructured as a timeline.

#### Sep 14–15: static alignment, taxonomy correction, and four-part reassembly

- Sep 14 evening. A static-view hybrid aligned the existing four semantic outputs:
  text prompts for both hands and the yellow top, plus one reviewed mask for the black
  base. The user approved the smoke and the full 5,400-frame candidate completed. Record:
  [aligned static hybrid](#sep-14-aligned-static-hybrid-candidate).
- Reviewing the toy at part level invalidated the composite `yellow_toy_top` /
  `black_toy_top_base` vocabulary for the new task. A 25-piece visual catalog was made,
  then the experiment was deliberately narrowed to four physical body components:
  `chassis`, `interior`, `rear_body`, and `cabin`.
- The source interval was moved to 190.0 seconds (proxy frame 0) for a clearer initial
  view. The static calibration workspace gained live decoding and substantial state,
  queue, finalization, and candidate-table repairs while the user reviewed frame-0 and
  later masks.
- A 600-frame / 20-second pilot exposed semantic drift despite stable IDs, including
  near-identical chassis and cabin masks at sampled later frames. The pilot therefore
  failed as evidence of four stable physical identities.
- At the user's explicit request, the failed pilot did not stop an exploratory full run.
  The 5,901-frame run completed in 474.639 seconds with 2.06 GiB peak allocated VRAM and
  23,418 masks. Record:
  [four-part static reassembly](#sep-15-four-part-static-reassembly-experiment).
- Sep 15 evening. Visual review identified a better experimental boundary: proxy frame
  3120, where all four components begin separated and are subsequently merged during
  reassembly. This maps to proxy time 104.0 seconds and source time 294.0 seconds. The
  focused contract re-trims that source interval to a new proxy `[0,2781)`, or 92.7
  seconds, so old frame 3120 becomes its frame 0. It requires new calibration rather
  than transferred exploratory masks.
- The focused calibration used frame 0 plus corrections at frames 900/1800/2700. The
  2,781-frame run completed in 194.855 seconds. Only one object-pair frame exceeded 0.5
  mask IoU, versus 111 such pair-frames in the exploratory run; visual identity review
  remains pending.

#### Sep 16: MediaPipe, then the whole exploratory queue in one unattended pass

- MediaPipe Hand Landmarker became the second core method: full-frame plus a fixed
  workspace ROI, fused and capped at two hands per frame, over the retained first minute.
  Record: [hand-pose adapter](#hand-pose-adapter).
- The user left for eight hours with "do all non-gated work in the exploratory queue".
  WiLoR, BoxMOT, CLIP + Drop-DTW, Grounded-SAM-2, SAMURAI, DAM4SAM, ATHENA and Kineo were
  each attempted at smoke tier, the segmentation arms were re-run on the same four-part
  seeds, and one unified review recording was built. ATHENA stayed blocked on missing
  intrinsics; Kineo produced body-only NLF output. Records:
  [exploratory queue](#sep-16-exploratory-queue-autonomous-pass),
  [review surface](#sep-16-final-unified-exploratory-review-surface),
  [four-part comparison](#sep-16-focused-static-four-part-segmentation-comparison),
  [interaction review](#sep-16-focused-non-segmentation-interaction-review-package).
- Overnight rebuilds v2 and v3 stabilized the WiLoR-primary hand layer and made the
  first-minute package continuous. Records:
  [v2](#sep-1617-overnight-interaction-review-v2-rebuild),
  [v3](#sep-1617-overnight-interaction-review-v3-continuity-rebuild).

#### Sep 17: human review of the first minute, and what it set in motion

- The user's review of v4 named the real failures: chassis/interior identity swap around
  1100–1200 as a hand sweeps past, interior leaking into the chassis after 279/573/1043
  as the part rotates, three WiLoR hands, a frame-185 fusion drop, hopping WiLoR 3D, and
  coarse GT too coarse inside `screw chassis`. Fixes: two-hand cap, fusion regression test,
  wrist-relative 3D, an agent-selected correction at 1172, and missing panels logged.
  Record: [v4 review](#sep-17-first-minute-v4-human-review-and-follow-up-rebuild).
- A leak-onset correction at 1020 did not hold; DAM4SAM was run over the minute; a
  per-target ensemble review reference took DAM4SAM's chassis inside the one failing
  interval and labelled the interior hidden over [1024,1172). Label-free review metrics
  ranked 198 episodes. Records:
  [leak onset](#sep-1718-leak-onset-correction-attempt-20-s-v5-hand-layer-dam4sam-first-minute),
  [metrics](#sep-17-label-free-first-minute-review-metrics-and-ranked-triggers),
  [ensemble](#sep-1718-metrics-branch-merged-3d-view-defect-per-target-ensemble-reference).
- On request, the recording's dataset poses, extrinsics and fine-grained annotations were
  selectively acquired. Two findings: the static video lags the pose clock by 9 frames, and
  C10379 intrinsics can be recovered from the dataset's own projection. Record:
  [acquisition](#sep-17-assembly101-poses-extrinsics-and-fine-grained-annotations-selective-acquisition).
- An iteration-speed pass (test tiers, mask/digest/probe caches, SAM3 checkpoint/resume,
  warm decode worker) cut the review rebuild from 85 s to 19 s and a correction rerun from
  191 s to 112 s. Record: [speed pass](#sep-17-faster-test-and-iteration-cycle-plan-executed-four-tiers).

#### Sep 18: dataset reference in the review

- The dataset hand poses, projected through the estimated camera with the +9 offset, and
  the 27 fine-grained segments joined the v4 package as external context; the fine-grained
  labels replaced the agent-authored substeps for navigation. Record:
  [dataset reference](#sep-18-assembly101-dataset-hands-and-fine-grained-labels-in-the-v4-review).
- Clean-up: `ruff format` applied tree-wide, stale front matter here refreshed, 72
  unreferenced run directories (1.86 GB) listed for the user's deletion decision.

### Plan versus actual

What the plan said, what happened instead, and why, in one line each.

- One dataset, one clip, and a 60 s window for the close comparison, versus "as many
  repos as we can". Nine methods were eventually attempted, but only after the ego
  footage had consumed the original budget on one of them; the 180 s runs exist, the
  iteration loop lived at 10–60 s.
- SAM3 with human-seeded masks, versus zero-shot text prompts. Text prompts worked on
  the RGB static view and failed on the monochrome ego view (Sep 8, 23:46). Every ego
  result after that is human-in-the-loop initialization, and is labelled that way; it is
  not an out-of-the-box result and not a tuned model.
- Two calibration tools built (CLI, then browser workspace with view aids), versus a
  thin adapter wrapper. This is where most of the code is. It was the price of the seed
  pivot, and the user asked at the time whether it was reinventing something; it was
  not, for reviewed-mask provenance, but it is the largest scope expansion.
- Multi-keyframe corrections, versus frame-0 initialization only. Frame 0 was not
  representative (Sep 9, 22:53). Corrections reset a slot's memory at a human-verified
  frame; the run manifest records every correction.
- MediaPipe was delayed until Sep 16 by the calibration work, then completed over the
  retained first minute. The exploratory queue was reached only by running it unattended
  in one night at smoke tier, with delegated workers, rather than as the plan's serial
  time-boxed trials; the results are labelled accordingly.
- Review tooling (four review-package generations, label-free metrics, an ensemble
  reference, an agent-correction CLI, a speed pass), versus "the recording, the normalized
  artifacts and the ledger are the deliverables". This is the second large scope
  expansion after the calibration workspace; it made iterating on one clip fast and made
  the plan's fixed human QA gate slower to reach.
- Dataset poses and fine-grained annotations acquired Sep 17, versus "ground truth only as
  an optional visual reference". They are used exactly that way, but the acquisition
  itself crossed a gate the plan had closed, at the user's request, and it surfaced the
  static/pose clock offset that every earlier two-view comparison had missed.
- Masks per frame at full rate as RGBA cut-outs, versus the plan's 5 fps segmentation
  cadence in the `.rrd`; the plan's cadence rule is documented as superseded below.
- Masks on every frame in Rerun, versus the historical external-PNG cadence of 5 fps
  (`mask_period_frames=6` at 30 fps). The worker now writes compressed PNGs every analysis
  frame and Rerun logs each object's PNG as an RGBA `EncodedImage` cut-out. The distinct
  class-labelled `SegmentationImage` was and remains a sparse 1 Hz record.
- Evaluation harness, labeling, metrics with confidence intervals (the Sep 8 brief's
  "never-cut" items). Deliberately not done; the plan's no-annotation rule held, and the
  scorecard is five pre-accuracy measures plus human QA.
- Git from day one. Missed until Sep 13. The working tree was intact throughout, but
  nothing before that date is recoverable from history, and the ledger could not cite
  revisions.
- 24 GB assumptions in the brief. Retired on day one; peak VRAM never exceeded 2.2 GiB
  in any run, so the 16 GB card was never the constraint.

### Open items

Human gates (nothing below can be claimed until these are recorded):

- Fixed two-timestamp QA dispositions for every completed method under `docs/qa/`
  (aligned static hybrid, canonical ego, MediaPipe, WiLoR). Records are prepared; no
  pass/flag/fail has been assigned. The old static zero-shot record is historical.
- Hidden-interior semantics for focused frames [1024,1172): accept the agent's
  `hidden_agent_label` (explicit empty masks) or keep the gap `not_contact_eligible`.
- The four-part run beyond the first minute: frames 1946, 2381, 2578 and 2684 are the
  first target-loss frames; the late reassembly has never been dispositioned and is
  currently treated as failure evidence only.
- G5 claims gate, including the still-unanswered CC BY-NC question for any job-seeking or
  demo use; it now covers the poses and annotations as well.

Technical:

- Static/ego clock offset: only the Sep 18 dataset hand layer applies the +9 pose-frame
  static lag; `battle-build-ego-static-comparison` and the focused two-view build still
  assume zero relative offset.
- Multi-view: ATHENA triangulation and a static/ego correspondence audit are unblocked by
  the fitted intrinsics and untried.
- Human-review nits not yet addressed: SAM mask palette (orange on yellow), skeleton lines
  for Kineo body joints.
- 72 unreferenced run directories (1.86 GB) plus the superseded
  `runs/interaction-review-first-minute-v4-local` await a deletion decision;
  `scripts/prune_runs.py` lists them and never deletes.
- `yellow_toy_top` leaves the frame at ~216.2 s in every 180 s arm; its coverage numbers
  describe the scene, not the tracker.
- FineBio access and audio: never entered; close out explicitly or drop.

### Sep 13: fixed two-timestamp QA infrastructure

- Contract: `FixedTimestampHumanQARecord` uses the source clock and requires exactly one
  `easy_manipulation` checkpoint followed by one `hard_or_occluded_manipulation`
  checkpoint. Every checkpoint has content-addressed visual evidence. Any non-pending
  decision requires a human identity and timezone-aware review time; aggregate status is
  severity-conservative and `ground_truth_accuracy_claim` is fixed to false.
- Preparation: `battle-prepare-human-qa` accepts only explicitly named run directories or
  manifests, validates the completed run/config provenance, creates an inference-free
  two-column contact sheet, and writes pending JSON. It refuses to overwrite any record
  containing a human decision.
- Timestamp selection aid: `battle-human-qa-candidates` validates both canonical proxy
  fingerprints and their identical source mapping, then renders 12 evenly spaced raw
  static/ego frame pairs under gitignored `artifacts/qa/`. It adds no model overlays and
  assigns no semantic category.
- Canonical pending set: full static zero-shot
  `muggledsam-sam3-g3-full-static-c10379-20260909t030710z` and selected full ego
  four-target human-seeded
  `muggledsam-sam3-full-ego-manual-seed-multiplexed-ego-hmc21179183-20260910t024052z`.
  The earlier three-target full ego run is historical and was superseded by the target
  pivot, so it is not a third canonical QA baseline.
- Human selection and preparation: the user shifted candidate 03 and candidate 10 to the
  nearest historical mask-bearing frames and selected them as easy/clear and hard/occluded.
  Their canonical grid values are analysis frames 984/4,416, source frames 14,868/21,732,
  and source times 247.8/362.2 seconds. Both canonical records and their fingerprinted
  local evidence now exist; every disposition remains pending.
- Claim boundary: the schema and artifact fingerprints make human review auditable; they do
  not perform that review. Tracker object scores and IoU predictions remain model
  diagnostics and cannot substitute for this gate.

### Sep 14: aligned static hybrid candidate

- The common output contract is now `left_hand`, `right_hand`, `yellow_toy_top`,
  `black_toy_top_base` in that order. The first three identities are initialized from
  human-readable text prompts; the black base uses the user's reviewed frame-zero mask.
- `configs/muggledsam_static_aligned_hybrid.json` explicitly binds the three-target text
  config, one-target proposal and target policy, fingerprints, provenance, and canonical
  slots. This candidate is hybrid and must not be described as pure zero-shot or all-manual.
- Five bounded black-base text-prompt variants were visually rejected. One broad prompt
  emitted continuously but selected an unrelated black fixture; none reliably identified
  the intended base. This prompted the reviewed-mask hybrid pivot rather than broader
  prompt hacking.
- Approved smoke: the user reviewed
  `muggledsam-sam3-smoke-hybrid-static-static-c10379-20260915t005256z/g3_review/static-c10379_contact_sheet.png`
  and said “Looks good!” at approximately Sep 14 20:53 EDT / Sep 15 00:53 UTC. The earlier
  `...005001z` smoke was technically valid and rendered identical evidence, but it is not
  the evidence cited by the approval.
- Full run: `muggledsam-sam3-g3-full-hybrid-static-static-c10379-20260915t005919z`
  completed all 5,400 frames in one continuous stream. Its manifest binds the exact approved
  smoke manifest and QA fingerprints. Worker time was 192.724 s, TTFU 4.203 s, and peak
  allocated VRAM 2,125,744,128 bytes. Emissions were 5,215/5,400 left hand,
  5,388/5,400 right hand, 5,396/5,400 yellow top, and 5,397/5,400 black base, with one
  stable canonical ID per target and no restarts. These are output facts, not accuracy.
  The full contact sheet samples proxy 0/90/179.967 s.
- The pending fixed human-QA record samples source 247.8/362.2 s. The reviewed frame-zero
  mask remains a calibration seed, not annotation ground truth, and tracker diagnostics are
  not accuracy measurements.
- The Sep 9 full zero-shot static run is preserved as historical evidence for its original
  three-target/mismatched contract; it is no longer the aligned canonical static baseline.
  A new synchronized RRD pairs this hybrid run with the canonical four-target full ego run.

### Sep 15: four-part static reassembly experiment

- Stage: `objects`; state: `succeeded` as an exploratory execution, but failed as
  evidence of stable four-part segmentation.
- Taxonomy: `chassis` means the entire visible black lower chassis; `interior` means only
  the dark interior surface visible through the cabin; `rear_body` is the yellow rounded
  body shell; `cabin` is the yellow windowed cabin. All four use
  `visible_surface_only`. This vocabulary supersedes the composite top/base labels only
  for this experiment.
- Input contract:
  `configs/clips/assembly101_nusar_9033_four_part_reassembly_g2.json`, static
  `C10379`, source `[190.0,386.7)` seconds, 30 FPS, 5,901 proxy frames at 1280×720.
  The run used max side 720, four frame-memory entries, one prompt-memory entry, and no
  chunks or intentional ID resets.
- Calibration: four human-selected frame-0 masks and 11 later correction masks at proxy
  frames 36, 65, and 162. Corrections replace multiplexed prompt memory and clear frame
  memory while retaining slots `sam3-00` through `sam3-03`.
- Pilot: the first 600 frames showed visually incorrect identity transfer, including
  chassis/cabin masks with approximately 0.907 IoU at sampled frames 65 and 135. Stable
  slot IDs did not imply stable physical identities. The full run proceeded only because
  the user explicitly requested an exploratory execution despite that known failure.
- Full run:
  `muggledsam-sam3-four-part-static-full-exploratory-static-c10379-20260916t012945z`;
  5,901/5,901 frames, 474.6386 seconds elapsed, 4.9093 seconds to first usable output,
  2,209,013,760 bytes peak allocated VRAM, and 23,418 masks written. The 113 MB Rerun
  recording is `four_part_static_full_exploratory.rrd`; the complete ignored run
  directory is approximately 309 MB.
- Full-frame overlap audit: IoU above 0.5 occurred on 77 chassis/cabin frames, 18
  interior/cabin frames, 9 chassis/rear-body frames, and 7 interior/rear-body frames.
  This is a geometric conflict screen, not ground-truth accuracy, and projected
  occlusion can create some overlap; combined with the pilot's visual drift, it is enough
  to reject the run as evidence of four reliable identities.
- Next decision: old proxy frame 3120 (104.0 proxy seconds / 294.0 source seconds) is
  where the four components are separated before being merged. The focused G2 contract
  re-trims source `[294.0,386.7)` to a new proxy `[0,2781)`, 2,781 frames / 92.7
  seconds. Create new masks at focused frame 0; do not reuse the exploratory run's
  frame-0 masks.
- Focused implementation:
  `configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json`,
  `configs/muggledsam_static_four_part_reassembly_focused_manual_seed.json`, and its
  correction policy. The existing proxy script now builds the static
  `C10379_rgb_294.000-386.700_1280x720_30fps.mp4` asset. The focused run profile is
  `--four-part-static-focused`; long-run correction validation now follows the active
  profile budget rather than the historical 300-frame smoke limit.
- Focused run:
  `muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260916t020716z`;
  2,781/2,781 frames, 194.8546 seconds elapsed, 4.3494 seconds to first usable output,
  2,209,960,448 bytes peak allocated VRAM, and 11,042 masks. Output coverage was
  2,775/2,781 chassis, 2,774/2,781 interior, 2,751/2,781 rear body, and 2,742/2,781
  cabin frames. Corrections were applied at frames 900, 1800, and 2700.
- Focused overlap audit: only chassis/interior exceeded 0.5 IoU, on one frame (0.6891
  at frame 2381). All other pairs had zero frames above 0.5; their maxima were 0.2745
  or lower. This removes the exploratory run's dominant chassis/cabin geometric conflict,
  but it does not establish physical identity correctness without visual review.
- Visual follow-up identified chassis/interior confusion around frames 324–330 and
  screwdriver/rear-body confusion beginning around frame 1235. Correction policy v2
  raises the audited per-target ceiling from three to five later keyframes while leaving
  the v1 policy file unchanged so the completed focused run remains reproducible.
- The v2 rerun `muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-
  20260916t023700z` applied 14 later masks at frames 327/900/1235/1800/2700 and
  completed 2,781 frames in 197.5891 seconds with 11,044 masks. Chassis and rear-body
  output gaps fell, but cabin gaps rose to 64 frames. The requested frame-327 and
  frame-1235 samples look locally separated; a new rear-body/cabin identity merge appears
  around frame 2101, where their mask IoU reaches 0.9376. The rerun therefore remains
  review evidence, not an accepted result.
- Monochrome ego counterpart:
  `muggledsam-sam3-four-part-ego-focused-reassembly-ego-hmc21110305-20260916t031515z`
  uses the same source interval with fresh ego masks at frame 0 and corrections at
  97/1235/1800/2700. It completed 2,781 frames in 193.7752 seconds with 10,509 masks
  and 2.0 GiB peak allocated VRAM. No pair exceeded 0.5 mask IoU, while output gaps
  reached 2 chassis, 69 interior, 136 rear-body, and 408 cabin frames. Those gaps and
  the assembled-object contact sheet keep the result in visual review rather than
  establishing cross-view identity accuracy.
- Human scope decision: retain only focused frames `[0,1800)` / 60.0 seconds from each
  view and stop correcting the later decline. The focused-first-minute comparison command
  creates exact 1,800-frame CFR inputs and the inference-free
  synchronized recording under `runs/four-part-focused-first-minute-comparison/`.
  The selected static v2 run's frame-2101 merge lies outside this bounded comparison.
- Claim boundary: reviewed masks are calibration inputs, not evaluation labels. Runtime,
  mask counts, ID continuity, tracker diagnostics, and overlap counts are factual output
  properties, not segmentation or association accuracy.

## Part 2: detailed method records

Each record keeps its original claim boundary. Required fields per record: method and
stage; schema/config version and Git revision; state and owner; input clip/asset
identifiers and provenance approval; environment and dependency lock reference;
model/code version and license review status; clock mapping and sampling policy;
chunk/continuity policy; output artifact URIs and checksums; success measure, data
split, and uncertainty when measured; failure modes, blocker, and next decision.

### Sep 8: fixture-rerun-export

- Stage: `export`
- State: `succeeded` on synthetic fixtures only
- Inputs: in-code normalized coordinates; no recording, annotation, or model call
- Clock policy: source/analysis/annotation/pose = 60/30/30/60 fps
- Output: local ignored `.rrd` created by `battle-export-fixture`
- Success measure: full fixture coverage and completed exporter contract
- Claim boundary: validates artifact shape and export behavior only; it is not an
  accuracy, performance, or dataset result.

### Sep 8, late evening: muggledsam-sam3 core-method smoke (G2)

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

### Sep 8, late evening: muggledsam-sam3 ego monochrome diagnostic

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
- Outcome: the user chose neither option offered here (labelled evaluation protocol, or
  stop ego work). Instead, on Sep 8 at 23:46, the direction became human-drawn box
  prompts inspected through the image decoder, which led to the calibration tooling and
  every later ego result.

### Sep 9, early morning: G3-approved static full candidate

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
- Status: this remains the only zero-shot full-duration result and the static half of
  the synchronized ego-versus-static comparison built on Sep 9. The G4 accept/reject
  decision it asked for was never taken explicitly; it has served as the static baseline
  since. Its formal two-timestamp QA record is prepared with both human-selected source
  times; both dispositions remain pending.

### Sep 9, early morning: muggledsam-sam3 ego viewpoint screen

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
- Outcome: e4 (`HMC_21179183`) became the ego view for everything that followed. Note
  the `[36,42)` gap on `yellow toy body`: the same object at the same frames was
  identified on Sep 13 as leaving the top of the frame under head motion.

### Sep 9, early morning: muggledsam-sam3 G4 e4-only 60-second candidate

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
- Outcome: this is the last zero-shot ego run. Viewing it in Rerun (Sep 8, 23:39–23:46)
  is what prompted the switch to human-drawn box prompts; its "run remaining 120
  seconds" option was superseded rather than taken.

### Sep 9, after midnight: muggledsam-sam3 e4 interactive box-prompt calibration

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
- Superseded the same night by the browser workspace below; the CLI's interactive
  terminal prompts and OpenCV ROI mode were too slow for labelling several targets on
  several frames.

### Sep 9, after midnight: muggledsam-sam3 e4 rapid calibration workspace

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
- Grew over Sep 9–13 into the labelling tool used for every later ego run: named target
  sets (`configs/muggledsam_e4_*_manual_seed.json`), masks rendered in the viewport with
  `1`–`4` candidate selection, a fourth target slot, later-keyframe corrections under a
  correction policy (`configs/muggledsam_e4_four_target_keyframe_correction_policy.json`),
  one **Finalize tracking plan** action writing hashed proposal and schedule artifacts,
  a finalized-plan lock with **Reopen for editing** and revisioned artifacts,
  reject/remove for candidates, foreground/background point prompts, and display-only
  VIGRA view aids (`docs/vigra-build.md`). See the timeline for dates.

### Sep 9, evening: muggledsam-sam3 e4 manual-seed multiplexed smoke

- Stage: `objects`; state: `succeeded` after one `blocked` attempt. Scope was e4 /
  `ego-hmc21179183` proxy frames `[0,300)` / 10.0 seconds.
- Seed validation passed: proposal
  `runs/muggledsam-sam3-e4-web-calibration-e55b5d0abe02/proposed_tracking_prompt.json`
  matched calibration-manifest SHA-256
  `79cf1f6f29d3ea311f964f6c7d011ba3112848d8b4788c333d9a528716f34086`, and its exact
  mask-0 seeds are `left_hand`/`t000000-b01`, `yellow_toy_body`/`t000000-b03`, and
  `toy_wheel`/`t000000-b04`. Human-selected `right_hand`/`t000000-b18` is excluded.
- Implementation: the manual path loads the three saved, source-sized selected masks,
  verifies each SHA-256, batches them into one
  `tracking.encode_prompt_memory_from_mask(...)` call, and advances one
  `step_video_masking_multiplex(...)` stream. Target-to-normalized-ID mapping is fixed
  as left hand→`sam3-00`, body→`sam3-01`, wheel→`sam3-02`; it makes no detector/text
  call and uses no chunks or intentional ID resets. Memory remains bounded to one
  prompt-memory and four frame-memory entries, CUDA device 0, bfloat16, and 504 pixels.
- Blocked attempt: the local calibration workspace still owned the same model on CUDA 0
  (PID 19647, about 1.6 GiB). The safety preflight refused a competing worker and
  recorded a schema-valid blocked attempt at
  `runs/muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260909t223016z/`,
  with no inference observations; its placeholder zero-output metrics are not
  measurements.
- Completed run (18:39, after the workspace was closed):
  `runs/muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260909t223928z/`;
  300/300 frames, 15.5 s, 1.85 GiB peak allocated VRAM. QA sheet
  `qa/manual_seed_multiplexed_vs_e4_zero_shot.png` compares it with the retained e4
  zero-shot smoke. The user's verdict was "not bad, good to continue", which approved
  the 180-second run recorded below.
- Full 180-second three-target run:
  `runs/muggledsam-sam3-full-ego-manual-seed-multiplexed-ego-hmc21179183-20260909t224835z/`;
  5,400/5,400 frames, 204.9 s, 1.85 GiB. Its recording
  `full_ego_manual_seed_multiplexed_baseline.rrd` was the first with translucent mask
  overlays, and `ego_manual_seed_vs_static_g3_synchronized_comparison.rrd` puts it
  beside the static zero-shot run on one timeline (`battle-build-ego-static-comparison`).
- Claim boundary: this is manual-seed multiplexed tracking, not out-of-box/text
  zero-shot, and the 1.0 confidence on seeded observations is a sentinel. Watching the
  synchronized comparison is what revealed that the `toy_wheel` seed was on the wrong
  part, leading to the target pivot below.

### Sep 9, evening: four-target 180-second ego baseline

- Stage: `objects`; state: `succeeded`. Targets pivoted to `left_hand`, `right_hand`,
  `yellow_toy_top`, `black_toy_top_base` (`configs/muggledsam_e4_left_hand_right_hand_
  yellow_toy_top_black_toy_top_base_manual_seed.json`), each with a human-accepted
  frame-0 mask from the web workspace
  (`runs/muggledsam-sam3-e4-web-calibration-left-hand-right-hand-yellow-toy-top-black-toy-top-base-20260910t021538z/`).
- Why four: the user asked about tracking both hands as one mask and splitting; that was
  rejected because merged hands lose left/right identity and splitting would not fix the
  oversized-box artifact, which was traced to a single stray pixel far from the hand.
  Adding the right hand as its own slot reduced the two worst left-hand box areas by
  71% and 64% at 0.600 / 0.767 s; the underlying stray-pixel cause was fixed on Sep 13.
- Run: `runs/muggledsam-sam3-full-ego-manual-seed-multiplexed-ego-hmc21179183-20260910t024052z/`;
  5,400/5,400 frames, 190.1 s, 1.90 GiB, max side 504, memory 4. Output coverage 100%
  left hand, 98.2% right hand, 99.7% yellow top, 99.6% black base; no ID resets.
- Dataset cross-check: Assembly101 ships no object boxes or masks for these parts, only
  30 fps action/mistake intervals and optional hand poses, so the target identities rest
  on human review alone.
- Claim boundary: same as above; coverage is emission coverage, not correctness.
- Formal QA: this is the selected full ego baseline for the fixed two-timestamp gate. Its
  record is prepared with both human-selected source times; both dispositions remain
  pending.

### Sep 13, evening: muggledsam-sam3 e4 multi-keyframe human corrections

- Stage: `objects`; state: `succeeded` on the 300-frame smoke budget only, which is the
  only budget the schedule path permits.
- Motivation (Sep 9, 22:53): frame 0 is not representative of the objects over the
  clip. Human-reviewed masks at later keyframes are applied as scheduled corrections.
- Memory semantics: the tracker runs to the correction frame; the corrected slot's
  predicted mask is replaced with the reviewed mask; the multiplexed prompt memory is
  rebuilt; automatic frame memory is cleared; IDs are unchanged. This trades a brief
  loss of temporal context for a known-correct anchor and is recorded per correction in
  the manifest (`multi_keyframe_corrections`).
- Labelling: keyframes 0, 2.5, 5, 8 s, all four targets at all four keyframes, with
  foreground/background point clicks on the right hand to exclude its shadow. Plan
  `runs/muggledsam-sam3-e4-four-target-keyframes-20260913t213159z/` (proposal plus
  `multi_keyframe_correction_schedule.json`, both fingerprinting the calibration
  manifest). Schedules are authored against the 30 fps clock and are refused at any
  other analysis rate.
- Run: `runs/muggledsam-sam3-smoke-multi-keyframe-corrections-ego-hmc21179183-20260913t215412z/`;
  300/300 frames, 15.0 s, 1.93 GiB, TTFU 4.8 s, corrections applied at frames 75, 150,
  240 (12 later masks plus 4 seeds). Three of four targets present on every frame;
  `yellow_toy_top` 293/300.
- Claim boundary: human-in-the-loop initialization and correction; the reviewed masks
  are seeds, not evaluation labels, and nothing here is measured against ground truth.

### Sep 13, evening: tracker score and IoU diagnostics

- Finding: `step_video_masking_multiplex` returns per-slot predicted IoU and an
  unbounded presence logit. The worker discarded the IoU and clamped the logit to
  `[0, 1]`, so every one of the 1,193 recorded observations carried `confidence: 1.0`.
- Change (commit `0d9b0bb`): `object_score` (raw logit) and `iou_prediction` are
  recorded per object, and a gapless `tracker_diagnostics` list records every slot on
  every frame including frames where the slot was dropped at or below zero. Both are
  plotted per target in the Rerun blueprint beside the video, with a zero reference
  line on the score.
- Run: `runs/muggledsam-sam3-smoke-multi-keyframe-corrections-ego-hmc21179183-20260913t221457z/`
  (same plan as above; 300 frames, 15.9 s).
- Reading of the user's six failure timestamps (1.2, 5.04, 5.98, 6.9, 7.81, 9.5 s):
  at 1.2 s `yellow_toy_top` decays 9.6 → 8.7 → 7.3 → 2.8 → −3.6 over four frames with
  IoU falling to 0.00, then sits near −5 for seven frames and recovers; its box marches
  to the top edge and back, and the contact sheet shows head motion carrying the part
  out of view. The tracker was right. At the other five timestamps every score is in the
  healthy 10–12 band, so those are the model being confidently wrong and no threshold on
  its own outputs will catch them. `black_toy_top_base` runs chronically lower (6–8)
  than the other targets.
- Claim boundary: the model's own estimates, not ground truth.

### Sep 13, evening: encoder resolution sweep

- Condition change (commit `93fdc22`): `max_side_length` and `max_frame_memory` became
  explicit run conditions recorded in the manifest, replacing constants.
- Arms at 30 fps, memory 4, same plan: 504 (`…221457z`), 720 (`…222850z`), 1008
  (`…222916z`).

| max side | runtime | peak VRAM | inflated `right_hand` frames | worst box area |
| --- | --- | --- | --- | --- |
| 504 | 15.9 s | 1.93 GiB | 5 | 0.407 |
| 720 | 24.9 s | 1.99 GiB | 0 | 0.149 |
| 1008 | 37.0 s | 2.16 GiB | 0 | 0.171 |

- Mask quality rose monotonically on every target, most on the weakest:
  `black_toy_top_base` worst-case IoU 0.527 → 0.684 → 0.758. The `yellow_toy_top`
  dropout at 1.2 s was unchanged (7/6/7 frames), consistent with the object leaving
  the frame.
- Corrected the same evening: the raw e4 recording is 636x480 and the proxies are
  954x720 upscales, so above ~636 px the encoder sees no new information. 504 → 720
  crossed from below-native to native sampling and was real; 720 → 1008 resampled a
  1.6x upscale. Prefer 720, not 1008.

### Sep 13, evening: muggledsam-sam3 e4 analysis frame-rate comparison

- Stage: `objects`; state: `succeeded`. Three arms over the same ten seconds of e4
  source video, seeded from frame zero only so initialization is identical and
  clock-independent: 30 fps / 300 frames / memory 4, 60 fps / 600 frames / memory 8
  (matched 0.133 s memory span), and 60 fps / 600 frames / memory 4 (halved span).
  All at 720 pixels. Report: `runs/frame_rate_comparison.json`, copied to
  [`docs/frame-rate-comparison-2026-09-13.json`](frame-rate-comparison-2026-09-13.json).
- Plumbing: a 60 fps proxy (`scripts/create_assembly101_e4_60fps_proxy.sh`) and clip
  config (`configs/clips/assembly101_nusar_9033_e4_60fps.json`); analysis fps carried
  from the clip config through the worker instead of a hardcoded 30; the `analysis`
  clock may be 30 or 60 while the other clocks stay fixed; and the ten-second smoke
  budget expressed in seconds. Transfer is limited to frame-zero seeds and checks the
  same raw-source checksum, source-interval start instant, proxy dimensions, and scaling
  policy. Proxy identity and fps may differ; source end time is not checked. The run
  records this as a `calibration_transfer_note`.
- Result: median model IoU predictions for the 30 fps / 60 fps memory-8 / 60 fps
  memory-4 arms were `left_hand` 0.9414 / 0.9336 / 0.9336, `right_hand` 0.9023 /
  0.9023 / 0.9023, `black_toy_top_base` 0.9062 / 0.9062 / 0.9062, and
  `yellow_toy_top` 0.9102 / 0.9062 / 0.9258. `yellow_toy_top` first dropped out at
  source 216.2 s in every arm; coverage for the other three targets was 100% throughout.
  These are output-continuity and model self-estimate diagnostics, not ground-truth
  accuracy. They show variation between arms but no consistent/measurable quality
  benefit supporting the extra compute cost.
- Cost in the published report comes specifically from `…20260913t223959z` (30 fps,
  24.7069 s), `…20260913t224508z` (60 fps, memory 8, 45.8453 s), and
  `…20260913t224556z` (60 fps, memory 4, 44.4363 s). The matched-span comparison is
  therefore about 1.85x compute. Peak VRAM was about 1.96 GiB in all three report arms,
  so frame rate was compute-bound rather than memory-bound on this 16 GiB card.
- The matched-span 60 fps arm (memory 8) and halved-span arm (memory 4) did not establish
  a consistent benefit for either memory setting on this clip; that does not imply
  identical outputs.
- Decision: **keep the 30 fps analysis clock.** The 60 fps proxy, config, and runs are
  retained as the evidence for that choice, not as a new baseline.
- Exporter defect found while reviewing these arms: every recording was written with the
  fixed application id `battle-session-1`, but each ships a default blueprint whose views
  are anchored under a clip-specific entity root. Rerun keys blueprints by application id,
  so opening recordings of two clips together let one clip's blueprint activate for both
  and point every view at entity paths the other recording never logged, rendering it
  empty. The application id is now derived from the clip id, and a regression test asserts
  that two clips cannot share one. Recordings written before this fix still carry the old
  id; re-export them with `scripts/reexport_run_rrd.py` before viewing them together.
- Second exporter defect found here: exporting several runs from one process gave them the
  same random recording id, so the viewer merged them into one recording with duplicated
  rows and a red segmentation background. The recording id is now the run id.
- Masks now appear on every analysis frame. Each object's binary mask PNG is logged as an
  RGBA `EncodedImage` cut-out at `views/<view>/masks/<object_id>`, so the store holds only
  compressed bytes and PNG alpha gives a true overlay with no background veil. The
  class-labelled `SegmentationImage` is kept at 1 Hz as a sparse record and is excluded
  from the default view (toggle it on from the blueprint panel for class labels). The
  worker writes mask PNGs every frame (its internal `--mask-period-frames` setting
  defaults to 1). Arms with
  full-rate masks: `…-20260913t233645z` (30 fps), `…-20260913t233715z` and
  `…-20260913t233812z` (60 fps). Rendering was verified in the web viewer at frame 599,
  which the sparse segmentation does not cover.
- Timing provenance: those later full-rate-mask validation reruns measured 24.9664 /
  47.2285 / 45.9781 s in their manifests. They are separate from, and were not inputs
  to, `docs/frame-rate-comparison-2026-09-13.json`; the report's 24.7/45.8-second and
  ~1.85x figures come from the earlier `223959z`/`224508z`/`224556z` arms above. Still
  later connected-component reruns (`…20260914t000644z`, `…000716z`, `…000819z`) had
  the viewer competing for the GPU and should not be used for uncontended timing.

### Sep 13, late evening: box derivation and GPU guard

- Defect: the reported box was the min/max extent of every positive mask pixel, so a
  few stray speckles far from the object ballooned it while the mask itself read
  correctly. First observed Sep 9 at 0.600 s (one isolated pixel), named as a
  connected-component problem then, fixed now.
- Change: `_box_from_mask` runs `cv2.connectedComponentsWithStats` (8-connected) on
  the thresholded mask and boxes the union of components with area at least 20% of the
  largest (`BOX_COMPONENT_KEEP_FRACTION`), so an object split by occlusion keeps a box
  over both parts. The saved mask PNG is unchanged. Each run's `runtime_settings`
  records `box_derivation` and `box_component_keep_fraction`, so old and new runs are
  distinguishable.
- Effect, re-running the three frame-rate arms against their predecessors frame by
  frame: 8–10% of object-frames changed box; 28/58/80 boxes shrank by more than 25% in
  area; worst case shrank 99%. `black_toy_top_base` lost ~50% of its box area on average
  when it changed, `yellow_toy_top` 25–40%, hands 7–20%.
- Guard: the worker's concurrent-GPU-process check matched the Rerun viewer because its
  venv path contains `python3.12`; an executable named `rerun` or under
  `/rerun_sdk/rerun_cli/` is now tolerated and still recorded in
  `gpu_processes_before_initialization`. Per-process VRAM peaks are unaffected;
  wall-clock timing may see contention.
- Re-run arms with the new boxes:
  `…-20260914t000644z` (30 fps), `…-20260914t000716z` (60 fps, memory 8),
  `…-20260914t000819z` (60 fps, memory 4). Viewer was open; timings 26.9 / 52.7 /
  50.9 s are contended.

### hand-pose adapter

- Stage: `pose`
- State: `succeeded` for the bounded 20-second smoke and selected 60-second run.
- Adapter: MediaPipe 1.0.1 Hand Landmarker, official float16 v1 task bundle, VIDEO mode,
  CPU/XNNPACK, two-hand limit, default 0.5 detection/presence/tracking thresholds.
- Input: approved focused RGB static proxy, analysis frames `[0, 600)` / 20.0 seconds
  at 30 FPS. The model asset and input/config checksums are verified before inference.
- Output contract: one observation per decoded frame; each detection carries exactly 21
  normalized 2D landmarks, a landmark-derived box, raw model handedness/confidence, and
  handedness corrected for the unmirrored camera then temporally voted. Detection IDs are
  explicitly frame-local; no persistent hand-identity claim is made.
- Run: `runs/mediapipe-hands-static-20s-20260916t035031z/`. Inference processed 600
  frames in 5.374 seconds on CPU; first normalized output appeared after 0.059 seconds.
  At least one hand was detected in 543/600 frames: 57 zero-hand, 393 one-hand, and 150
  two-hand frames. These are output-presence counts, not recall or pose accuracy.
- Improvement ablation: a crop-only pass over normalized ROI
  `(x=0.45, y=0.35, width=0.55, height=0.65)` recovered some misses but regressed total
  detections (628 versus 693), so it did not replace the baseline. Fusing full-frame
  and 2× crop inference was complementary. The selected smoke candidate
  `runs/mediapipe-hands-static-20s-fused-dedup-th035-20260916t0428z/` uses 0.35
  detection/presence/tracking thresholds and strict spatial duplicate removal. It
  produced one hand on 292 frames and two on 308 (908 detections total), with no
  near-coincident wrist pairs under the 0.06 normalized-distance audit. Runtime was
  11.103 seconds on CPU. A 12-frame stratified review of gains showed plausible hand
  geometry, including recovery at frame 300; this remains visual review, not recall.
- Artifacts: normalized JSONL, manifest, bounded input, three-frame contact sheet, and
  a 9.62 MB inference-free Rerun recording with landmarks, skeletons, boxes, hand count,
  and mean handedness confidence. The contact sheet shows plausible geometry at frames
  0 and 599 and no detection at frame 300; human semantic disposition remains pending.
- Provenance: package source declares Apache-2.0; model bundle redistribution remains
  unapproved because its model-specific license was not independently confirmed.
- Selected minute run:
  `runs/mediapipe-hands-static-60s-fused-dedup-th035-20260916t0430z/`. It processed
  1,800 frames in 30.668 seconds on CPU and emitted 2,600 detections: 152 zero-hand,
  696 one-hand, and 952 two-hand frames. Coverage weakened in the final 20 seconds
  (120/600 empty frames) while hands grasped the black assembly; a stratified raw-frame
  sheet confirms visible hands in many misses, so this is model failure rather than
  absence. The tighter-crop ablation reduced final-section misses to 88 but regressed
  overall detections and was rejected.
- Combined viewer: the selected minute hand observations are frame/source-time checked
  against the static SAM3 run and merged into
  `runs/four-part-focused-first-minute-comparison/four_part_focused_first_minute_ego_static_comparison.rrd`.
  The 81.0 MB inference-free RRD contains synchronized ego/static SAM3 masks plus static
  MediaPipe landmarks, skeletons, boxes, hand count, and mean handedness confidence.
  Structure was verified without opening a viewer.
- Next: human visual review of the combined recording. WiLoR remains a separately
  licensed upgrade if MediaPipe's occlusion failures are unacceptable.

### Sep 16: exploratory queue (autonomous pass)

This section was rewritten after an adversarial artifact audit. The earlier 75 + 45 + 60 +
50 + 90 minute effort narrative is impossible: the relevant run artifacts were created from
00:37 through 01:01 EDT and commit `c07a7d8` was created at 01:02 EDT. Only durations written
by a worker, manifest, or native log appear below. Human gates G1–G5 remain deferred.

#### WiLoR hand-pose adapter

- Classification: `external partial`. The Battle adapter and RRD work, but the WiLoR checkout
  used by its worker is dirty (`demo.py`, `requirements.txt`, and a backbone loader). Its base
  revision `fcb9113…` is therefore insufficient to reproduce the observed run.
- Input/output: focused static proxy frames `[0,600)`; 600 normalized observations at indices
  0–599. 582 frames contain hands (not 598), with 1,171 total hands and 582 native JSON files.
  Each hand has exactly 21 normalized 2D points and 21 declared non-metric camera-relative 3D
  points. IDs are frame-local. `hands.rrd` contains boxes, 2D/3D joints, and skeletons.
- Measured worker runtime: 47.2474 s; TTFU 5.8854 s; peak allocated VRAM 2,819,373,056 bytes.
- License boundary: WiLoR checkpoints are CC-BY-NC-ND; MANO and Ultralytics terms remain
  separate. No result is usable as metric reconstruction, multi-view ground truth, or a
  noncommercial-use determination.

#### BoxMOT over independent YOLO detections

- Classification: `integrated smoke`. `boxmot_worker.py` runs Ultralytics `YOLO(...)(frame)`
  independently at every decoded frame, builds `[x1,y1,x2,y2,confidence,class]` arrays, and
  calls `BotSort.update` directly. No MuggledSAM ID appears in the worker.
- Evidence: 600 normalized rows at indices 0–599; 557 frame/box observations, IDs
  `boxmot-0` and `boxmot-1`; a 600-frame bounded video and `tracks.rrd`, whose structure
  includes 557 object-box rows. Measured worker runtime is 7.4826 s, TTFU 2.2372 s, and
  allocated VRAM 74,796,544 bytes.
- License/provenance: detector SHA-256 is recorded; package metadata for BoxMOT 25.0.0 and
  Ultralytics 8.1.34 says AGPL-3.0. No upstream detector acquisition URL or model-license
  decision was recorded. The result remains conditional COCO-`person` association only.

#### CLIP + Drop-DTW weak supervision

- Classification: `integrated smoke`. The worker directly imports `dp.exact_dp.drop_dtw` from
  `Drop-DTW` at `32ce9c8…`, creates OpenCLIP ViT-B-32 image/text embeddings, builds cosine
  costs, and calls the algorithm twice (cost and labels); no synthetic substitute was used.
- Evidence: exact coarse transcript source/hash is in the manifest; two GT weak-supervision
  steps overlap frames 8820–9420. Twenty 1-FPS samples produce cost 15.2950248 and matched
  counts 16 (`attach interior`) and 1 (`screw chassis`). `alignment.json` and scalar-only
  `alignment.rrd` are inference-free.
- Measured worker runtime: 2.6622 s in the pinned rerun
  `runs/drop-dtw-static-20s-pinned-openclip-rerun/`. It requires the verified
  `timm/vit_base_patch32_clip_224.openai` snapshot
  `a6f597a30f7b82c51704746581f9a4e41421e878`, with checkpoint SHA-256
  `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`; the runner is
  offline and fails rather than silently downloading a different weight.

#### Fine substep crop CLIP (`fine_substep_crop_clip`)

- Classification: `integrated smoke` / agent-review experiment. Eleven substeps are checked in
  at `configs/fine_substeps/assembly101_focused_static_first_20s_agent_labels.json` with
  provenance tag `agent_authored_visual_review` (not Assembly101 GT). WiLoR 2D boxes/landmarks
  are primary hand cues; MediaPipe is fallback only on WiLoR gaps. Part boxes come from baseline
  SAM3 masks (`--parts-reference baseline_sam3` default; reviewed-seed SAM2 control is
  CLI-selectable for later audit).
- Pipeline: 3 FPS crop sampling (60 samples / 20 s), multi-prompt OpenCLIP ViT-B-32 scoring,
  weak motion/contact fusion, monotonic DP arm plus Drop-DTW over eleven prototypes; baseline
  coarse 2-step Drop-DTW retained for comparison.
- Preserved run `runs/fine-substep-static-20s-20260916t2255z/`: pass1 weights
  motion=0.12/contact=0.08 collapsed monotonic DP to 3 substeps (checkpoint match 1/9); pass2
  weights motion=0.18/contact=0.14 selected with same monotonic collapse but Drop-DTW(11)
  recovered all 11 labels (checkpoint match 3/9). Only f405 matched both arms; screwdriver
  prompts dominate fused argmax (S07–S09). Artifacts: `scores.json`, `evaluation.json`,
  `boundary_contact_sheet.png`, `review_guide.md`, inference-free `fine_substep.rrd`.
- Measured worker runtime: 14.6 s (pass2). Command: `uv run battle-fine-substep-align --seconds 20`.
  Claim boundary: agent-review diagnostic only, not benchmark accuracy.

#### Grounded-SAM-2 (`transformers_grounding_dino_plus_sam2_video_smoke`)

- Classification: `smoke_only` bounded video propagation. Frame 0 uses pinned HF
  `IDEA-Research/grounding-dino-tiny` revision `a2bb814…` with prompt `hand.` (score 0.5595,
  box ≈881×446→1034×577); SAM2.1 tiny (`configs/sam2.1/sam2.1_hiera_t.yaml`, checkpoint SHA-256
  `7402e0…be69`) propagates one tracked object across the approved proxy prefix.
- Provenance: Grounded-SAM-2 checkout `b7a9c29…`; pyenv `grounded_sam2` (torch 2.11.0+cu128).
  Input proxy fingerprint verified from
  `configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json`
  (`bd57acd…b62e`, `static-c10379`, analysis frames 0–299 → source seconds 294.0–303.967).
- Measured runtime (300 frames / 10.0 s): 15.88 s wall; TTFU 15.39 s (includes model load +
  frame-0 detect + full propagation before normalized write); peak VRAM 6.37 GB. All 300 native
  masks nonempty; 300 distinct mask content hashes.
- Artifacts:
  `runs/transformers_grounding_dino_plus_sam2_video_smoke-10s-20260916t0518z/` with
  `observations.jsonl`, `native/masks/*.png`, `manifest.json`, and inference-free
  `propagation.rrd` (embedded bounded video, boxes, mask overlays at exporter cadence).
- Command: `uv run battle-grounding-dino-sam2-video --seconds 10`. This is the HF detector +
  SAM2 video predictor path, not the vendor CUDA Grounded-SAM-2 extension.
- Legacy one-frame external JSON (`score 0.953125`, un-pinned HF revision) remains importable via
  `battle-import-external-smoke grounded-sam2`; it is not relabelled as video propagation.

#### SAMURAI (`samurai_sam2_video_smoke`)

- Classification: `integrated smoke`. Checkout `/home/nick/src/samurai` @ `76ba195…`; worker uses
  `configs/samurai/sam2.1_hiera_t.yaml` with `samurai_mode: true` and SAM2.1 tiny checkpoint SHA-256
  `7402e0…be69`.
- Seed: deterministic frame-0 hand box `(881,446,152,129)` (xywh), shared with Grounded-SAM-2 and
  DAM4SAM on the approved focused static proxy.
- Frame contract: `round(seconds × 30)` analysis frames decoded from the proxy prefix — 300 for 10 s,
  600 for 20 s. This resolves the earlier 602-vs-600 mismatch from unbounded external decodes.
- Measured 10 s run (`runs/samurai_sam2_video_smoke-10s-20260916t052328z/`): 14.8 s wall, 7.0 s
  TTFU, 0.82 GB peak VRAM, 300/300 nonempty masks with 300 distinct content hashes.
- Measured 20 s run (`runs/samurai_sam2_video_smoke-20s-20260916t052348z/`): 26.7 s wall, 11.3 s
  TTFU, 1.06 GB peak VRAM, 600/600 masks.
- Command: `uv run battle-samurai-video --seconds 10`. Normalized observations, native masks,
  manifest, and inference-free `samurai.rrd` are written under `runs/<run_id>/`.

#### DAM4SAM (`dam4sam_video_smoke`)

- Classification: `integrated smoke`. Checkout `/home/nick/src/DAM4SAM` @ `9c95450…`; worker uses
  `DAM4SAMTracker('sam21pp-T')`, `sam21pp_hiera_t.yaml`, and `dam4sam_config.yaml`.
- Initialization: headless bbox seed `(881,446,152,129)` on frame 0 via
  `estimate_mask_from_box`→`add_new_mask`. Official VOT integration initializes from mask prompts
  instead; no extra initialization frames are consumed here.
- Compatibility: pyenv `samurai` (torch 2.11+cu128) because upstream torch 2.1+cu121 fails on
  sm_120. DAM4SAM-specific code remains `dam4sam_tracker.py`, `return_all_masks`, and `add_to_drm`.
- Frame contract: same source-aligned `round(seconds × 30)` decode as SAMURAI (300/600, not 602).
- Measured 10 s run (`runs/dam4sam_video_smoke-10s-20260916t052430z/`): 12.0 s wall, 2.6 s TTFU,
  0.81 GB peak VRAM, 22 DRM additions, 300/300 masks with distinct hashes.
- Measured 20 s run (`runs/dam4sam_video_smoke-20s-20260916t052447z/`): 21.9 s wall, 3.2 s TTFU,
  1.05 GB peak VRAM, 31 DRM additions, 600/600 masks.
- Command: `uv run battle-dam4sam-video --seconds 10`. Does not prove distractor-scene semantics on
  identical parts. Legacy 602-frame import remains via `battle-import-external-smoke dam4sam`.

#### ATHENA multi-view hand triangulation

- Classification: `blocked` for Assembly101 real data; `fixture_smoke` only via
  `uv run battle-athena-fixture-smoke`.
- Checkout: `/home/nick/src/athena` @ `e85bd494…`. ATHENA requires per-camera intrinsics plus
  extrinsics in JARVIS YAML or Anipose TOML form.
- HTTP Range probe (`uv run battle-athena-calibration-probe`, evidence
  `docs/athena_hf_calibration_probe.json`): `AssemblyPoses.zip` at
  `cvml-nus/assembly101` @ `bfc15ea5…` supports Range requests; central directory has 3381
  members. For recording `nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532` the
  archive exposes six calibration-like members (`camera_extrinsics_fixed`, `camera_position_fixed`,
  `camera_extrinsics_ego`, `camera_position_ego`, `timestamp`, `xf_transf`) and no path containing
  `intrinsic`. Selective extract (no whole-archive download) retrieved fixed extrinsics (2347 B),
  fixed positions (596 B), timestamps (1.1 MB), and `xf_transf` (31.4 MB).
- Fixture smoke (`runs/athena-fixture-triangulation-smoke/`): synthetic two-view normalized 2D
  inputs triangulated with ATHENA DLT logic; separate 2D view logs and 3D world points in
  `athena_fixture.rrd`. No accuracy claim and no Assembly101 landmarks consumed.

#### Kineo offline pipeline

- Classification: `kineo_nlf_only_partial`. Checked-in config
  `configs/kineo_nlf_headless_only.yaml`; wrapper `uv run battle-kineo-nlf --seconds 20`.
- Kineo checkout `/home/nick/src/kineo` @ `03b36e31…` remains dirty (tracked edits in
  `rerun_export.py`, `sfm_camera_extrinsics_initialization.py`, `pyproject.toml`, `pixi.lock`).
- Person selection: `best_bbox_only=True` on RTMLib detections (highest score per frame; not largest
  area), `frame_step=5`, `bbox_thr=0.3`, `nms_iou_thr=0.65`.
- Latest bounded rerun `runs/kineo-nlf-headless-20s-20260916t0540z/`: wall inference 30.4 s;
  PKLs hold 456/600 frames with bbox + 55-body-joint NLF outputs; MoGe intrinsics remain native
  only. Normalized export retains all 600 rows, with 456 containing body joints; RRD
  `kineo_nlf_partial.rrd`. Not SfM, metric world pose, BVH, or multi-view Kineo.

#### Explicitly deferred (not integrated)

- LM-EEC, ObjectRelator, Qwen video VLMs, supervised temporal-action models, and
  long-video VLMs remain out of scope per the Sep 8 plan line 59.

### Sep 16: final unified exploratory review surface

`uv run battle-build-exploratory-comparison` composes the approved focused static RGB prefix
only. It performs no inference and writes
`runs/exploratory-first-20s-comparison/exploratory_first_20s_comparison.rrd` plus an ignored,
typed `exploratory_comparison_index.json`. The index fingerprints every consumed manifest,
normalized observation/alignment artifact, native mask tree when used, shared raw/proxy/config
input, bounded video, final RRD, and each source manifest's declared inference input fingerprint.

| Method | Included range | Review layer | Queue status |
| --- | --- | --- | --- |
| MediaPipe Hands | frames 0–599 / 20 s | 2D landmarks, skeletons, boxes, confidence/count | selected 2D baseline |
| WiLoR | frames 0–599; output in 582 frames | projected 2D hands; separate camera-relative non-metric 3D | external partial |
| BoxMOT + YOLO | frames 0–599; 557 detector-conditioned box frames | boxes and local track IDs | integrated smoke |
| Grounding-DINO + SAM2 | frames 0–299 / 10 s only | propagated masks and boxes | smoke-only, no extension |
| SAMURAI | frames 0–599 / 20 s | seeded-hand masks and boxes | integrated smoke |
| DAM4SAM | frames 0–599 / 20 s | seeded-hand masks and boxes | integrated smoke |
| Kineo NLF-only | frames 0–599; 456 NLF-output frames | person boxes and 2D body joints | external partial |
| CLIP + Drop-DTW | 20 s temporal samples | GT-transcript weak-supervision intervals/cost | contextual only |
| ATHENA | no real-data overlay | fixture URI/status metadata only | blocked: no intrinsics |

Every spatial method retains its own entity root, so IDs are not merged or equated. Missing
observations clear their render subtree and coverage/output-presence traces expose gaps. The
default blueprint shows MediaPipe and BoxMOT over the one shared embedded static video; other
layers remain available from the entity tree. G3/G4/G5 human review gates remain deferred, and
this comparison makes no accuracy or cross-method identity claim.

### Sep 16: focused static four-part segmentation comparison

- Contract: `configs/four_part_segmentation_comparison.json` pins the reviewed focused static
  RGB proxy, source 294.0–314.0 s / frames `[0,600)` at 30 FPS, and the exact ordered
  physical targets `chassis`, `interior`, `rear_body`, `cabin`. It validates the focused
  correction schedule and all four user-reviewed frame-zero mask SHA-256 values before each
  run. Its mask-derived boxes are 793,468–955,549; 845,254–909,313; 773,572–844,628; and
  997,504–1179,632 respectively. The matching accepted SAM3 v2 run is a read-only baseline.
- Open-vocabulary arm:
  `runs/grounding-dino-sam2-open-vocabulary-four-part-20s-20260916t1004z/`, 600 observations,
  55.911 s, 3,011,449,856 B peak allocated VRAM. Grounding-DINO independently tried each
  recorded prompt/synonym, then SAM2 propagated every threshold-passing candidate. All four
  calls produced a candidate, but this is not four semantic successes: chassis, rear body, and
  cabin selected essentially the same cabin-region box, while `vehicle interior` selected almost
  the full frame. The contact sheet at frames 0/300/599 therefore flags the arm as visually
  non-one-to-one; no reviewed mask/box was substituted. The raw attempt list and scores remain
  in `worker_result.json`.
- Reviewed-seed SAM2 control:
  `runs/reviewed-seed-sam2-control-four-part-20s-20260916t1005z/`, 600 observations,
  52.594 s, 3,033,928,704 B. All four exact reviewed masks initialized a multi-object SAM2
  state; every target emitted 600 nonempty masks with 600 distinct mask byte hashes. This is a
  control for detector failure, not a Grounding-DINO result.
- SAMURAI:
  `runs/samurai-four-part-reviewed-seed-20s-20260916t1007z/`, 600 observations, 103.557 s,
  1,058,056,704 B. `samurai_mode: true` was active. Its four-object propagation raised the
  upstream ambiguous-tensor error, so the completed arm uses four independent real SAMURAI
  predictor streams, one shared reviewed mask per named target, combined only on the common
  source clock. Each target emitted 600 nonempty masks; rear body has 599 distinct byte hashes.
- DAM4SAM:
  `runs/dam4sam-four-part-reviewed-seed-20s-20260916t1009z/`, 600 observations, 78.429 s,
  3,534,988,800 B. Four independent `DAM4SAMTracker('sam21pp-T')` streams use
  `initialize(image, init_mask)` rather than the prior hand-box fallback. Each target emitted
  600 nonempty masks; rear body has 599 distinct byte hashes. Native DRM additions were
  chassis 27, interior 32, rear body 20, cabin 27.
- Skeptical visual screen (not human QA): sampled frame 0, 300, and 599 for the baseline and
  every arm. The reviewed-seed SAM2, SAMURAI, and DAM4SAM samples retain nonempty output for
  all labels, but the interior/chassis regions visibly migrate around the hand/tool area by the
  middle/final samples; their physical attachment is not confirmed. Cabin remains visually
  localized to the orange cab in those samples and rear-body remains localized near the lower
  yellow component, but this is only an agent visual observation, not acceptance or accuracy.
  The open-vocabulary arm fails this screen for the reasons above. There were no output gaps in
  `[0,600)` and hence no target-loss frames to add; masks clear on any later missing observation.
- Viewer: `runs/four-part-segmentation-comparison/four_part_segmentation_comparison.rrd` is
  inference-free, embeds the bounded video exactly once, and contains separate colored roots for
  `baseline_sam3`, `grounding_dino_sam2_open_vocabulary`, `reviewed_seed_sam2_control`,
  `samurai`, and `dam4sam`. Its typed fingerprint index links the earlier mixed-modality viewer
  at `runs/exploratory-first-20s-comparison/exploratory_first_20s_comparison.rrd`; MediaPipe,
  WiLoR, Kineo NLF, BoxMOT, and Drop-DTW are deliberately excluded because they do not produce
  comparable part masks. ATHENA remains metadata-blocked.

### Sep 16: focused non-segmentation interaction review package

- `uv run battle-build-interaction-review` is inference-free and composes the verified static
  RGB source outputs with the reviewed-seed SAM2 four-part control. It validates every available
  source/proxy/config/bounded-input artifact fingerprint, common source clock, 600 retained
  timeline rows where applicable, exact source timestamps, reference-mask dimensions, and the
  one 600-frame 1280×720 video asset before writing the ignored
  `runs/interaction-review-first-20s/interaction_review.rrd`, typed
  `interaction_review_index.json`, `review_guide.md`, and contact sheet.
- Reference choice: `reviewed_seed_sam2_control` is the default because every one of the four
  exact reviewed-mask-seeded targets has a nonempty 600-frame mask sequence. It is a
  detector-failure control/reference layer—not ground truth, an accuracy result, or confirmed
  physical part attachment. `--reference-segmentation baseline_sam3` is an explicit review-time
  override; it does not conflate either source with the other comparison arms.
- The primary panel contains only the reference part masks and MediaPipe's blue skeletons. WiLoR
  orange 2D is side-by-side and its non-metric camera-relative 3D is separate; same-frame
  assignment is greedy nearest wrist and never claims hand ID equivalence across methods or
  persistent identity. The compact time series records 21-point mean/max pixel disagreement,
  one-method-only detection counts, and handedness disagreement counts. It is a diagnostic, not
  hand-pose accuracy.
- Hand-to-part candidates take the minimum source-pixel distance among wrist/palm and five
  fingertips for each MediaPipe spatial proximity lane and `chassis`, `interior`, `rear_body`,
  and `cabin`. Inside-mask or ≤12 px is raw positive; starts require 2 observed positive frames
  and ends 3 observed negative frames. Missing hand/mask values are explicit nulls, clear their
  Rerun paths, and reset debounce state. The generated `contact_candidate_start/end` records are
  geometry heuristics only, never touch/grasp labels or ground truth.
- BoxMOT is retained under a disabled-by-default person/occlusion context root, not part tracking
  or segmentation. Kineo remains its own NLF-only 2D body/box panel with 456/600-frame coverage:
  no hand articulation, SfM, metric 3D, multi-view, or BVH claim. Drop-DTW is only a navigation
  bookmark based on Assembly101 coarse GT transcript weak supervision. ATHENA is metadata-only,
  blocked on missing real intrinsics, and its fixture never appears on the real timeline.
- Generated deterministic review pins retain required frames 0, 300, and 599 and add the first
  contact transition (frame 1), first Kineo gap (91), late one-method-only detection/possible
  occlusion (597), and highest same-frame landmark disagreement (599; ties resolve by frame).
  Human dispositions are all pending. Fixture coverage includes contact distance/inside behavior,
  debounce/missing propagation, nearest-wrist pairing, deterministic pins, timestamp rejection,
  output paths, and Rerun clear behavior. Full suite count after this addition: 222 passing tests.

### Sep 16–17: overnight interaction-review v2 rebuild

- Human feedback is preserved verbatim in
  `docs/qa/overnight-interaction-review-v2.agent-review.json`; it is explicitly separate from
  agent-authored review and from ground truth. No feedback is represented as an approval.
- WiLoR stabilization: `runs/wilor-hands-stabilized-20s-overnight-v2-r3/` preserves raw WiLoR
  inputs and adds a deterministic review layer: WiLoR confidence ≥0.55, duplicate suppression,
  evidence-defined part-workspace filtering for multi-detection phantoms, and One-Euro
  wrist/palm rigid smoothing (min cutoff 1.5, beta .007, derivative cutoff 1.0). MediaPipe is
  only a confidence ≥0.85 shape/workspace-gated fallback across future-confirmed WiLoR gaps of
  five frames or fewer. No long gap is interpolated. Output-stability diagnostics changed median
  normalized wrist jitter from 0.003228 to 0.002775 (14.0% reduction); 523 frames are WiLoR
  primary, 32 use the guarded MediaPipe fallback, and 45 remain missing. These are not pose
  accuracy values.
- WiLoR full-minute rerun: `runs/wilor-hands-static-60s-overnight-v2/` completed serially in
  128.438 seconds wall time. Its raw/native evidence is retained for the apples-to-apples
  MediaPipe comparison; the v2 interaction package stays bounded to the validated first 20 s.
- Kineo: the Battle wrapper now writes a generated, hashed config override rather than modifying
  the dirty external checkout. Per-frame RTMLib detection (`frame_step=1`) yielded 478/600
  bbox/NLF frames, versus the prior 456/600 baseline. Measured wall inference was 40.577 s:
  bbox detection 4.921 s, MoGe 0.768 s, NLF 28.726 s. The remaining 122 gaps show that eyes/HMD
  visibility is not the control point; person detector boxes are. No bbox/keypoint hold or
  interpolation was introduced because the residual gaps remain material.
- Segmentation/reference: corrected focused static SAM3 is now the interaction default; the
  reviewed-seed SAM2 arm remains a selectable control. The structured agent record retains the
  verified human frame-327 chassis/interior correction separately and offers DAM4SAM
  chassis/interior experiment candidates at frames 480/520. They are proposals, not accepted
  human masks or semantic accuracy claims; no tracker correction API was used to avoid
  relabelling that method.
- Fine substeps: a 6 FPS phase-conditioned crop-CLIP experiment at
  `runs/fine-substep-static-20s-phase-conditioned-overnight-v2/` soft-penalized screwdriver
  language before independently documented tool onset frame 345 and used a measured yellow-pixel
  crop cue after it. It does not encode agent substep boundaries. Drop-DTW reached 4/9 checkpoint
  matches but mean absolute boundary error worsened to 168.6 frames; monotonic DP was 3/9 and
  80.0 frames. It is not promoted. The original 3/9, 158-frame run remains preserved failure
  evidence, while the 11-step `agent_authored_visual_review` timeline is primary navigation.
- Integrated package:
  `runs/interaction-review-overnight-v2/interaction_review.rrd`, generated guide, contact
  sheet, and typed index. It embeds one 600-frame video; its default is corrected SAM3 plus
  stabilized WiLoR, with raw WiLoR/MediaPipe evidence, per-frame Kineo context, agent substep
  timeline, coarse GT context, and deterministic correction/action/occlusion pins. `uv run rerun
  rrd print` completed successfully; no Rerun viewer was opened.

### Sep 16–17: overnight interaction-review v3 continuity rebuild

- Kineo fusion: `battle-kineo-fusion` verifies the Kineo and BoxMOT source asset, approved static
  view, 30-FPS clock mappings, and every retained source timestamp before it can use BoxMOT. It
  keeps all 478 native RTMLib/YOLOX boxes primary. A missing native box may use an independent
  YOLO/BotSort person box only when a nearby native box passes a deterministic IoU/center/size
  consistency gate; 94 did. Nine residual boxes were linearly interpolated through gaps of at
  most five frames (target three); no held boxes were needed. The 13-frame 246–258 and 6-frame
  266–271 dual-detector outages remain explicit missing state.
- NLF was run on each of the 581 actual fused crops, not by interpolating output keypoints. The
  resulting `runs/kineo-nlf-fused-20s-overnight-v3/` has 581/600 valid bbox/keypoint rows:
  478 `detected_native`, 94 `boxmot_fallback`, 9 `interpolated`, 0 `held`, and 19 `missing`.
  Native Pkl structural validation reports 55 finite body joints per output row. NLF wall time
  was 21.064 s in the dedicated fused-crop worker. This remains `kineo_nlf_only_partial`, without
  face, eye, HMD, SfM, metric-3D, BVH, or multi-view claims.
- Hand and mask audit: deterministic sheets under
  `runs/interaction-review-overnight-v3/audits/` cover every requested WiLoR and corrected-SAM3
  review frame. The visible evidence did not establish that more One-Euro smoothing would improve
  hand contact/re-entry behavior, so v2's 14.0% wrist-jitter reduction and explicit missing
  state were retained. Corrected SAM3 remains the default mask source; contact diagnostics remain
  geometry-only candidates.
- Review package: `runs/interaction-review-overnight-v3/interaction_review.rrd` retains the
  single video asset and uses fused Kineo context. It preserves all 359 raw segmentation triggers
  in the typed index while reducing navigation to 15 same-part trigger episodes and 29 total
  bookmarks. Raw WiLoR, MediaPipe fallback, BoxMOT, and the exploratory substep arm remain
  separately labeled; the 11 agent-authored substeps are primary. `rerun rrd print` completed
  without a viewer popup. The v3 QA record preserves user feedback as human-authored text and
  leaves human pass/fail pending. Final agent-authored visual review
  (`runs/interaction-review-overnight-v3/final_agent_review.md`, gitignored) fixed a
  `mediapipe_fallback` audit-sheet labeling bug in `scripts/render_overnight_v3_audits.py`.
  Full suite after that fix: 245 passing tests.

### Sep 17: first-minute v4 human review and follow-up rebuild

- Human feedback (verbatim, `docs/qa/interaction-review-first-minute-v4.human-feedback.json`,
  author `human`, no pass/fail added): chassis/interior identities swap around frames
  1100-1200 as a hand sweeps across the part; at 370 the interior briefly grows into the
  chassis and recedes; hands are good but drop briefly when only a couple of fingers are
  visible; Kineo is decent on the body and less good on hands than WiLoR; the Review guide,
  Drop-DTW weak supervision and agent-authored substep panels showed nothing.
- Missing panels (definite defect): the shared blueprint referenced `metadata/review_notes`,
  `metadata/drop_dtw` and `metadata/agent_substeps`; the v4 builder logged none of them (its
  substeps were `TextLog` rows under `metadata/agent_substeps_first_20s/timeline`, which a
  `TextDocumentView` cannot render) and the 20 s builder also never logged `review_notes`. Both
  builders now log the guide, a per-frame navigation document, agent-substep and coarse-GT
  step-index series, and static coarse-GT / contract / Drop-DTW-status documents; the 20 s
  package was rebuilt in place and `rerun rrd print` confirmed the rows. The v4 builder accepts
  an empty existing output root and requires `--overwrite` for a non-empty one.
- Swap analysis on the earlier reference run (dense 1040-1260 metrics and outline sheets under
  `runs/muggledsam-sam3-four-part-focused-corrections-agent-swap-20260918t000947z/agent_review/`):
  interior leak onset ~1020-1032, chassis label jumps at 1070-1074/1089, chassis <600 px over
  1110-1166, swapped labels 1167-1234, and the human-selected frame-1235 correction undoing the
  swap with 0.89 IoU both ways. The grey interior block is not separately visible before ~1167;
  a frame-1100 decode found only the yellow body being attached where the tracker's small
  chassis blob sat. The 20 s DAM4SAM/SAMURAI/SAM2-control arms are irrelevant here (600 frames).
- Agent-attributed correction: `MuggledSAMCalibrationCandidate` and
  `MuggledSAMMultiKeyframeCorrection` gained `selected_by: human|agent` (default `human` for
  every existing record; agent rows are later-frame corrections only and can never seed frame
  0); run metadata records `agent_selected_correction_frame_indices`. `battle-muggled-agent-
  correction` derives a draft calibration from the finalized one (hard-linked results, origin
  fingerprinted), decodes box prompts through the isolated worker, accepts with a written
  rationale, and finalizes an augmented schedule. Policy v3 allows six later keyframes because
  chassis already had five. Accepted at 1172: chassis `t001172-b02` candidate 1 (6954 px, box =
  tracked interior-slot bbox plus a background point on the grey block) and interior
  `t001172-b03` candidate 0 (2375 px, box = tracked chassis-slot bbox); mutual overlap 18 px.
- Rerun `muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t001210z`:
  2,781 frames in 192.65 s, 2,209,637,376 bytes peak VRAM, 11,091 masks, corrections at
  327/900/1172/1235/1800/2700 with 1172 tagged agent-selected. Compared with the old run:
  identical before 1172 (IoU 1.000), new chassis vs old interior 0.926 over 1172-1234 with
  0.002 chassis/interior overlap, and >=0.99 agreement after 1235. Visual sheets confirmed blue
  on the black body and green on the grey block; v4 now references this run with
  `contact_eligible` = `[0,1020)` and `[1172,1200)`. Frames 1020-1171 remain uncorrected; 370
  was left alone and added as a bookmark. GPU checks before/after each command: no NVRM/Xid.
- Finger-only hands: 92 short (1-4 frame) per-lane gaps covered 154 frames in the v4 layer;
  102 of those frames had a real WiLoR detection in lane range with confidence 0.35-0.55, 34
  were two-hand dedup merges, 10 were below 0.35 and 7 had none. The stabilizer now accepts
  sub-gate detections only while continuing a lane accepted within 5 frames, at most 5
  consecutive frames, tagged `low_confidence_continuation`; no hold/extrapolation. v5 layer:
  missing frames 110 -> 79, short gaps 92 -> 69, <=2-frame flicker lanes 10 -> 1, MediaPipe
  fallback 66 -> 15, 405 continuation instances, median wrist jitter 0.00265 -> 0.00274 (raw
  0.00317). Random (24) and MediaPipe-disagreeing (18) samples showed the continuation landmarks
  on real partially occluded hands; 192/405 lack a MediaPipe hand within 0.08 and are the
  conservative phantom bound. Kineo remains NLF body-only context, not a hand method.
- Rebuilt `runs/interaction-review-first-minute-v4/interaction_review_first_minute_v4.rrd`
  (1,800 navigation rows, 13 blueprint views, all contact rows over 1020-1171 `invalid_mask`).
  `runs/interaction-review-first-minute-v4-local` is the superseded build a viewer may still
  hold open. `docs/qa/interaction-review-first-minute-v4r2.agent-review.json` lists the agent
  findings and the two agent-proposed correction rows; all human decisions remain pending.

### Sep 17–18: leak-onset correction attempt, 20 s v5 hand layer, DAM4SAM first minute

- Visibility review of the reference run over 1000-1180 (every 4 frames plus every frame
  1015-1040, zoomed raw and gamma-lifted crops, coordinate grids at 1020/1024; sheets under
  `runs/muggledsam-sam3-four-part-focused-corrections-agent-leak-20260918t004115z/agent_review/`,
  summary in `visibility_findings.json`): the black chassis is clearly visible through the whole
  1020-1171 window (rotated to its underside from ~1024, partly under the fingers 1064-1171,
  never hidden). The grey interior block is partially visible only until ~1022 as the matte
  block protruding below the chassis plate between the two underside posts, the same surface the
  human accepted at frame 900 (1,084 px); from ~1024 the region under the tracked interior mask
  is the chassis's own lower body, and a lighter grey inner surface reappears at the left of the
  object from ~1128 (first boxable at 1172, as recorded earlier).
- Leak-onset correction attempt (not adopted): agent corrections at frame 1020, chassis
  `t001020-b02` candidate 0 (6,876 px, box 808,418,978,486 with background points on the
  protruding block and the finger) and interior `t001020-b04` candidate 0 (1,152 px on the
  protruding block), 12 px mutual overlap, rationales in `agent_acceptances.jsonl`. Chassis
  reached seven later keyframes, so
  `configs/muggledsam_static_four_part_reassembly_focused_correction_policy_v4.json`
  (`policy_version` 4, eight later keyframes per target; the schema now caps v3 at six and v4
  at eight) was added and the schedule finalized under it. Rerun
  `muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t004350z`: 2,781
  frames in 181.6 s, 2,210,630,656 bytes peak VRAM, 11,061 masks, corrections at
  327/900/1020/1172/1235/1800/2700 with 1020 and 1172 agent-selected. Against the reference
  (`compare_1000_1300.txt`): identical before 1020; interior IoU to the old run 0.901 over
  1020-1070 and 0.980 over 1070-1110 (the interior slot re-leaks onto the chassis from
  ~1036-1046 along the same trajectory); chassis mean area 1,338 vs 609 px over 1110-1172 but
  still <600 px on 29 frames; the reviewed 1172-1235 interior changed (IoU 0.770, mean 2,325 vs
  3,089 px). Conclusion: the memory reset with a credible chassis mask does not stop the leak,
  because the interior's refreshed memory is a block that disappears four frames later and the
  slot latches onto the black body. The 001210z run remains the v4 reference; `contact_eligible`
  stays `[0,1020)` and `[1172,1200)`; no interior mask was invented for 1024-1171. A
  hidden-object (empty mask) interior correction would need new schedule/worker semantics and
  human sign-off.
- 20 s v3 package on the v5 gate: `runs/wilor-hands-stabilized-20s-v5` applies the bounded
  `low_confidence_continuation` gate to the 20 s WiLoR/MediaPipe inputs (same corrected
  023700z parts run). Against `overnight-v2-r3`: missing frames 45 -> 29, MediaPipe fallback
  frames 32 -> 10, WiLoR-primary frames 523 -> 561, stabilized hand frames 555 -> 571,
  stabilized instances 772 -> 939 (190 continuation), short 1-4 frame per-lane gaps 46 -> 35,
  <=2-frame lanes 4 -> 0, wrist jitter 0.002775 -> 0.002697 (raw 0.003228).
  `battle-build-interaction-review` now defaults to the v5 layer;
  `runs/interaction-review-overnight-v3` was rebuilt in place (audits and the final agent
  review preserved; `rerun rrd print` confirmed the stabilized rows).
- DAM4SAM first minute: `battle-four-part-segmentation dam4sam --frame-count 1800` extends the
  arm to frames `[0,1800)` from the same four reviewed frame-0 seeds (the contract itself is
  unchanged; the run metadata records 60 s).
  `runs/dam4sam-four-part-reviewed-seed-60s-20260918t005416z`: 1,800 frames in 238.8 s,
  7,316,468,224 bytes peak VRAM, coverage
  chassis/interior/cabin 1,800 and rear_body 1,780, DRM additions 49/77/72/54.
  `battle-segmentation-disagreement` (per-frame per-target IoU, episodes = IoU<0.5 for >=5
  consecutive frames, a mask missing on one side counts as 0, missing on both is neutral)
  versus the corrected SAM3 reference, written to
  `disagreement_vs_corrected_sam3.json` in that run: mean IoU cabin 0.968, rear_body 0.786,
  chassis 0.678, interior 0.553; 16 episodes. Largest: interior `[707,1072)` (DAM4SAM's
  independent interior stream merged onto the black chassis body, ~6.6k px vs SAM3's
  human-corrected ~1.4k), rear_body `[1522,1661)` (DAM4SAM mostly lost it, 20 missing frames),
  chassis `[570,706)`, interior `[1117,1252)`, chassis `[1055,1172)` (DAM4SAM keeps a ~8.3k px
  chassis mask where the reference has lost the chassis), chassis `[327,364)` (starts at the
  human correction the reference applies and DAM4SAM does not). This is cross-method
  disagreement between two uncorrected/corrected trackers, not accuracy; neither run is ground
  truth.
- `docs/qa/interaction-review-first-minute-v4r3.agent-review.json` records the findings and the
  two not-selected agent candidates; all human decisions remain pending. GPU checked before and
  after every GPU command (nvidia-smi, current-boot kernel journal): no NVRM/Xid faults.

### Sep 17: label-free first-minute review metrics and ranked triggers

- Claim boundary first: every number in `runs/review-metrics-first-minute-v1/` is a geometry,
  appearance, or provenance proxy computed on CPU from already retained v4 inputs. None is an
  accuracy metric, a ground-truth label, or a statement about what the worker touched; the
  ranked episodes are prompts for a human to look at a frame range. No model ran and no viewer
  was opened. Expected touched parts per substep are `agent_authored_assumption` records layered
  on the agent-authored substeps (`configs/review_metrics/first_minute_v1.json`).
- Inputs were located through the v4 index and every manifest/observation file plus the bounded
  video was re-verified against its declared SHA-256 before use. `metrics.json` records 7,200
  part stats, 10,800 pair metrics, 7,200 appearance rows, 7,200 growth rows, 114 gap proxies,
  20 gaps, 2,577 jitter rows, 145 contact intervals, and 1,800 Kineo rows; absence is an explicit
  state and the file contains no NaN.
- Segmentation: part area medians were chassis 7,164, interior 2,627, rear_body 2,136, cabin
  16,445 px. The chassis/interior swap score reached 0.89 at f1235 (a clean label exchange, also
  the only chassis/interior label-crossing) and stayed 0.36–0.70 over f1091–f1164 through the
  absorption term (interior 3.4–4.3× its median while chassis fell below 0.1×). Uniform thresholds
  also fired at f326 (matching the previously verified human frame-327 correction), f616–f623, and
  f1062–f1073. The user-reported 1100–1200 swap is therefore caught by the swap detector, the
  chassis collapse anomaly (f1089–f1173, rank 1), and the chassis appearance drift (f1089–f1180).
  The reported transient interior growth near f370 is a slow ramp (444 px at f279 to 5,255 px at
  f365) that no frame-to-frame step of ≥1.3× marks; it is caught by the sustained interior area
  anomaly f346–f393 (only 1.17× threshold, rank 156) and the ramp-onset growth event f291–f298.
- Appearance: the yellow band derived from frame-0 rear_body/cabin masks is HSV hue 17–24,
  S ≥147, V ≥179. The first version flagged leakage whenever a stabilized hand box covered ≥35 %
  of a mask, which fired on 77 % of chassis and 97 % of interior frames because hands legitimately
  cover the handled parts; hand overlap is now recorded as context and the leakage flag uses each
  part's own robust Lab-distance z (≥3, distance ≥18) or a dark part turning yellow. This is a
  design correction, not a tuning to the reported frames.
- Hands: 20 frame-level stabilized-WiLoR gaps cover 110 frames. The skin band calibrated from
  confident raw WiLoR boxes (YCrCb Cr 131–152, Cb 111–130) also admits 20 % of non-hand scene
  pixels, so a gap frame is a visible-but-undetected suspect only when its last-pose window beats
  the frame background by 2× as well as half the reference; 67/114 gap frames qualify. Four gaps
  re-enter ≥1 hand-scale away (f377–388, f394–414, f1119–1120, f1132–1156). Normalized wrist
  jitter medians per substep are 0.00–0.044 with p90 up to 1.6 in S04/S05; these include real
  motion.
- Contacts: 145 debounced intervals on `[0,1200)`, 51 shorter than 5 frames, median 9 frames,
  p90 42, max 317; 53 expected, 39 unexpected, and 53 outside the labeled `[0,600)` against the
  agent-assumed parts. Kineo: 1,619 `detected_native` frames with mean joint confidence 0.412 and
  median joint jitter 4.01 px, 150 `boxmot_fallback` at 0.384 / 5.46 px, 14 `interpolated` at
  0.348 / 3.50 px (p90 126 px), 17 `missing`.
- Ranking: scores are threshold units and comparable only within a type, so the rank interleaves
  types (round k holds the k-th strongest episode of every type). 220 episodes were written;
  the top 12 spans all ten detectors and each has a contact sheet with before/after context.
- Tests: 13 synthetic-fixture tests cover swap/exchange/absorption, label crossing, missing-mask
  explicitness, step and slow-ramp growth classification, area anomalies, gap/re-entry/proxy,
  jitter by phase, contact durations/flicker/substep consistency, provenance grouping, ranking,
  and the checked-in config. They need no real data or models.
- **TODO (GPU, not run):** DAM4SAM over the 60 s window and its cross-method disagreement against
  corrected SAM3 as an additional swap/leakage trigger.

### Sep 17–18: metrics branch merged, 3D-view defect, per-target ensemble reference

- **Claim boundary first.** The ensemble reference is a *review* reference assembled from
  retained runs; cross-method fallback is not accuracy, neither the corrected SAM3 run nor the
  DAM4SAM arm is ground truth, and the hidden interior interval is an agent visibility label
  pending human confirmation (`docs/qa/interaction-review-first-minute-v4r4.agent-review.json`).
  No model ran and no viewer was opened; every step was CPU-only.
- **Metrics branch merged** (`review-metrics-first-minute`, d0fb900) with a non-fast-forward
  merge commit; the three expected conflicts (`pyproject.toml`, README, this ledger) were
  resolved keeping both sides. Follow-up fixes: the module hardcoded the 20 s SAM3 reference and
  rejected the v5 layer's `low_confidence_continuation` state; it now reads the reference run
  from the v4 index (accepting `baseline_sam3` or `ensemble_reference`), describes contact
  eligibility from the index's validity intervals, and defaults to
  `runs/review-metrics-first-minute-v2/`. v2 against the current index (reference 001210z):
  198 episodes (v1: 220 on the superseded `-v4-local` index). Top of the interleaved rank: chassis
  growth f1172–1186 (40.2x, the correction restoring the label), chassis collapse anomaly
  f1089–1171 (25.2x), chassis appearance drift f1089–1171 (4.6x), interior growth f1220–1234,
  chassis/interior identity swap at f1172 (2.1x, the correction's single-frame exchange), then
  the same contact/hand detectors as v1; the v1 frame-1235 exchange no longer exists because the
  1172 correction removed the swap it undid. Part-area medians moved by <1.2 %.
- **3D-view defect (user screenshot).** The "WiLoR camera-relative non-metric 3D" view reported
  "2D visualizers require a pinhole ancestor" for `.../hands/{boxes,landmarks,skeletons}`:
  `_log_hands` logged the 2D archetypes under the Spatial3DView root together with the 3D pose.
  Hand logging gained `include_2d`; both builders pass `include_2d=False` for the 3D root, the
  3D view sweeps only `camera_relative_3d/**`, and two tests (a fixture RRD plus `rerun rrd
  print` of the built v4 and 20 s v3 recordings) assert no 2D archetype under the 3D root and no
  3D archetype under any 2D root. Both packages were rebuilt (v3 in place, v4 with
  `--overwrite`).
- **Hidden interior verification.** Before adopting the label, zoomed source crops were
  re-rendered (1016–1034 every 2 frames at 3x; 1020–1028 every frame at 5x, gamma 2.2, with
  SAM3 chassis/interior and DAM4SAM chassis outlines). The matte block protrudes below the
  chassis plate through 1022, is faint at 1023, and from 1024 the region under the tracked
  interior mask is plate/finger with no separate block while the mask grows onto the chassis
  body (1,119 px at 1020, 5,245 at 1059, ~11k over 1128–1164). Onset ambiguity about two frames.
  DAM4SAM's interior stream merged onto the chassis over [707,1072) and is not a substitute.
  Adopted: interior `hidden_agent_label` over `[1024,1172)` as explicit empty masks.
- **Ensemble builder.** `battle-build-ensemble-reference` + typed policy
  `configs/ensemble_reference/first_minute_v1.json` (`EnsembleReferencePolicy`,
  `agent_authored_assumption`): default SAM3; per-target fallback only inside explicit intervals
  (chassis `[1055,1172)` = the disagreement episode) when a rule fires (area < 0.5x rolling
  median of sane accepted areas over 300 frames, IoU with another SAM3 target > 0.3,
  discontinuity with the last sane accepted mask, or primary missing) and the DAM4SAM mask passes
  sanity (area within 0.4–2.0x the rolling median; IoU >= 0.5 with the last sane accepted mask or
  centroid jump within 30 px + 5 px/frame, capped at 90 px). Rules run everywhere for
  diagnostics (out-of-interval firings: chassis 174, interior 317, rear_body 174, cabin 0 –
  mostly legitimate occlusion shrinkage, which is why they do not act outside the intervals).
  Masks are copied whole; hidden frames are zero PNGs referenced from the sidecar and carry no
  object row. Outputs: `manifest.json` (`RunManifest.ensemble_reference` metadata),
  `observations.jsonl`, `masks/`, `ensemble_provenance.json` (7,200 frame x target records,
  substitutions, declined attempts, per-target summaries), `segmentation_episode_check.json`,
  `sheets/before_after_1000_1250.png`.
- **Result.** Chassis: 111 DAM4SAM frames over `[1055,1056)` and `[1062,1172)`; declined
  1056–1061 (DAM4SAM chassis 1,807–2,595 px < 0.4 x ~6.7k; SAM3 kept, flagged unsane). One
  policy iteration: the first pass returned to the SAM3 chassis at 1086–1088 although it sat on
  the yellow body piece (IoU 0.31 with the accepted mask just missed a separate 0.3
  discontinuity threshold), so the discontinuity rule now reuses the sanity continuity test
  (IoU < 0.5 and jump beyond the allowance). Interior: 148 hidden frames; rear_body and cabin
  1,800 SAM3 frames each. Review-metrics swap/crossing/area detectors on the ensemble: the
  check window 1089–1173 goes from four episodes (chassis collapse 25.2x, swaps f1091–1164 and
  f1172, interior enlargement f1049–1171) to none, with no new episode anywhere (34 -> 29). The
  remaining chassis collapse anomaly f1050–1061 (1.95x) is exactly the declined/undersized run.
  The DAM4SAM chassis grows to ~10–11k px from 1085 as the assembly rotates; whether the attached
  body counts as chassis is a semantic question left to the human.
- **v4 on the ensemble.** `battle-build-interaction-review-v4` defaults to the ensemble
  (`--reference <run>` keeps a SAM3-only build with the old frame-level eligibility), verifies the
  policy/primary/fallback/sidecar fingerprints, records `reference_segmentation_method:
  ensemble_reference` plus provenance counts and the sidecar fingerprint, computes
  `contact_eligible` **per target** from the sidecar (chassis `[0,1020)`, `[1055,1056)`,
  `[1062,1200)`; interior `[0,1024)`, `[1172,1200)`; rear_body/cabin `[0,1200)`; late boundary
  1200 unchanged; chassis `[1020,1055)` stays ineligible because both trackers are undersized
  there), writes per-target `segmentation_validity_intervals` with the policy rationales, and
  logs the provenance layer (`diagnostics/reference_provenance/<part>` series in a new time
  panel, `diagnostics/segmentation_contact_eligible/<part>`, and magenta
  `primary/reference_provenance_overlay/<part>` boxes on DAM4SAM frames; per-frame navigation
  lists each part's provenance). Contact rows: 7,598 observed, 5,178 `invalid_mask`, 1,624
  `missing_hand`; 168 events. `rerun rrd print` completed on the rebuilt RRD (14 blueprint views).
- **Tests.** 297 passing: policy typing and interval-overlap rejection, primary standing when no
  rule fires, area rule inside vs. diagnostic outside, sanity declines (area and continuity),
  discontinuity catching a label jump, other-target overlap and missing-primary rules, hidden
  explicit empty masks, whole-mask copy (no blend) with a schema guard against out-of-interval
  fallback, ineligible/late intervals, summary + validity rationales, provenance RRD entities,
  per-part contact eligibility, the spatial-root archetype checks, and an integration check of
  the built ensemble run.

### Sep 17: Assembly101 poses, extrinsics and fine-grained annotations (selective acquisition)

- **Gate crossed on request.** This is the first pose or annotation data pulled into the project;
  it was fetched for this recording only, at the user's explicit ask, and lives under ignored
  `data/raw/assembly101/<recording>/` with a full report
  (`selective_poses_annotations_acquisition_report.md`) and `acquisition_manifest.json`. Nothing
  here has been used by any run, builder, or review package yet. Licence remains CC BY-NC 4.0.
- **What was fetched.** The ten `AssemblyPoses.zip` members for the recording via HTTP Range
  (~420 MB compressed; the 72 GB archive was never downloaded): `landmarks2D`/`landmarks3D`,
  `hand_bboxes`, `hand_confidences`, fixed/ego extrinsics and positions, `timestamp`, `xf_transf`.
  Fine-grained annotations were streamed from the split CSVs and only this recording's rows kept
  (432 segments x 12 views; the segment set is view-independent). TSM/DINOv2 features were
  inspected and skipped: single-LMDB archives of 20-51 GB per view, and DINOv2 exists only for
  C10119.
- **Conventions verified on the data.** Pose files are keyed by 60 fps pose frame; hand `"0"` is
  left, `"1"` right; 21 joints in the MS-G3D order (tips 0-4, wrist 5, palm 20); `landmarks3D` is
  world-frame millimetres with a fixed-scale hand model; `landmarks2D` is raw sensor pixels
  (1920x1080 static, 636x480 ego) and is an exact projection of the 3D (reprojection RMS
  0.0001 px), so it carries no independent detection signal; `camera_extrinsics_*` are
  camera-to-world.
- **Clock finding.** Proxy frame p is raw frame 17640 + 2p. The ego view HMC_21110305 has no
  offset to the pose clock. The static C10379 video **lags the pose clock by 9 pose frames
  (~150 ms, +-1)**, established by two independent image metrics and a zoomed overlay; use
  `pose = 17649 + 2p` for the static proxy. Every static-vs-ego comparison built so far assumed
  zero relative offset; this is a correction to apply, not yet applied.
- **Intrinsics.** The archive ships none. Fitting the dataset's own 2D/3D landmark pairs with the
  provided extrinsics recovers a 5-parameter Brown model for C10379 to numerical precision
  (fx ~ fy ~ 1250.7 px on 1920x1080, principal point (954.0, 528.6), k1 -0.130); scale K by 2/3
  for the 720p proxy. Ego HMC_21110305 fits a rational model (fx ~ 189 on 636x480). These are
  estimates derived from the dataset's internal projection, not an official calibration file,
  but they remove ATHENA's stated blocker in practice.
- **Fine-grained segments in the 60 s window.** 27 segments (7 overlapping two-hand pairs)
  including `position interior` 96-323 and 518-697, `screw chassis with screwdriver` 424-483,
  737-1079 and 1668-1800, `position rear body` 1321-1518, and pick-up/put-down/inspect steps.
  This is the substructure the human review found missing inside coarse `screw chassis`
  (318-1031) and is the intended replacement for the agent-authored substep track.

### Sep 17: faster test and iteration cycle (plan executed, four tiers)

- **Claim boundary.** No perception result changed. Every commit in this pass is infrastructure;
  the one GPU run (the resume equivalence test) reproduced an existing run bit-for-bit.
- **Tier 1, tests** (`8239888`). `real_data`, `slow` and `gpu` markers; `addopts` deselects
  `real_data` and `gpu` by default; shared `tests/conftest.py` (`require_artifact`,
  `require_executable`, a 10 ms-poll `serve` helper, session-scoped synthetic video);
  `pytest-xdist`. Default tier 18.5 s -> 8.2 s (324 passed, 11 deselected); `-m "not slow"`
  3.4 s; `-m real_data` 1.75 s; `-n auto` 5.2 s.
- **Tier 2, builders** (`51cc7ef`, `d87961a`, `b99c2da`). One bounded LRU `MaskCache` per run
  directory shared by the exporter and review passes, persisted as `native/mask_cache.npz`
  (bit-packed masks + RGBA PNG bytes) and written by every mask-producing worker
  (`battle-cache-masks` for old runs); a stat-keyed SHA-256 `digest_cache` and an `ffprobe`
  `media_probe` cache; fingerprinting only referenced masks; a stamp-cached ffmpeg trim;
  `PhaseTimer` with `--quiet`, `--overwrite`, `--layers`, `--verify-fingerprints` on the
  builders; `mask_summary` plus one distance map per part per frame in the contact pass.
  `battle-build-interaction-review-v4` 85 s -> 18.8 s with identical row counts (RRD shrank
  63.6 -> 51.4 MB from chunk packing, not data).
- **Tier 3, SAM3 resume** (`a6dfe1c`, `ab5bb86`). The worker writes tracker state
  (`prompt_memories`, `frame_memories`) to `native/checkpoints/f{idx:06d}.pt` at every
  correction keyframe and every N frames, keyed by a stream identity; `--resume-from-checkpoint`
  restores it. `battle-muggled-smoke --resume-run <prior> --resume-at k` copies
  `observations.jsonl` rows and masks for `[0,k)` verbatim and steps only the tail;
  `StreamContinuityPolicy` gains `mode: checkpoint_resumed` with `resumed_from_run`,
  `resumed_at_frame` and `checkpoint_fingerprint`. Continuity claims for resumed runs must say:
  bit-identical to the prior run before `k`, a resumed stream after. The `gpu` test resumed
  the focused static run at 1172 and matched a continuous run byte-for-byte; worker inference
  191.1 s -> 112.2 s for the 2,781-frame clip. `battle-muggled-agent-correction decode-batch`
  reuses one warm worker across a JSON plan of decode requests instead of reloading the model
  per call.
- **Tier 4, hygiene** (`da800fe`). `scripts/prune_runs.py` lists run directories that no
  tracked file or other run manifest cites (dry-run only, never deletes); README documents the
  test tiers and when to run each.

### Sep 18: Assembly101 dataset hands and fine-grained labels in the v4 review

- **Claim boundary first.** The dataset poses come from Assembly101's own multi-view tracker
  with a fixed-scale hand model; the intrinsics are an estimate recovered from the dataset's
  own 2D/3D projection; the fine-grained segments are the dataset's human annotations. All of
  it is external review context. It is not ground truth for any method compared here, and no
  accuracy number rests on it. CPU-only; no model ran; the one viewer opened was the web
  viewer in a background browser tab for a layout check.
- **Reference window.** `battle-build-assembly101-reference` (new module
  `assembly101_reference.py`, typed contracts in `assembly101_pose_schemas.py`) reads
  `landmarks3D`, `hand_confidences`, `timestamp`, `camera_extrinsics_fixed` and this recording's
  fine-grained CSV rows, maps every analysis frame `p` of the focused static proxy onto pose
  frame `17649 + 2p` (the measured +9 static offset, recorded with its evidence and +-1 frame
  uncertainty in `Assembly101ClockRule`), projects the world-mm joints through the checked-in
  camera estimate `configs/assembly101/c10379_camera_estimate.json` (the builder refuses a
  camera whose extrinsics differ from the dataset file), and writes
  `runs/assembly101-reference-first-minute-v1/{manifest.json,hands.jsonl}` with input
  fingerprints. 2.6 s. The dataset's 21-joint MS-G3D order and edge list are kept verbatim
  rather than remapped onto the MediaPipe order (the dataset thumb has three joints plus a
  palm centre), so the layer has its own schema and drawing code.
- **Checks.** Our projection reproduces the dataset's shipped `landmarks2D` to 0.0002 px RMS
  (max 0.0004) over 73,500 points, so the 1.1 GB 2D file is never read. A third,
  method-independent confirmation of the clock offset: the median distance from each dataset
  wrist to the nearest stabilized WiLoR wrist across the minute is minimal for +7..+10 pose
  frames (30.4 px at +9 vs 33.7 px at 0; the mean is minimal at exactly +9). The residual
  ~30 px is a definition difference (dataset joint 5 vs WiLoR joint 0) plus both trackers'
  noise, not a claim about either. Coverage: dataset hands in 1800/1800 frames (left 1800,
  right 1700; 72 of 3,500 hands below the 0.5 draw threshold). Where a dataset wrist lands
  within 40 px of a WiLoR wrist (1,951 pairs) the wrist-to-middle-tip span ratio dataset/WiLoR
  has median 1.08, p10 0.64, p90 1.57: scale agrees on average, articulation does not,
  especially with fingers hidden behind the held part; zoomed overlays at 300/800/1172/1400
  show the dataset skeleton on the right hand but stretched wide when the hand grips the part.
- **v4 package.** New layer `assembly101_hands` (on by default; `--no-assembly101-reference`
  builds without it): `comparison/assembly101_hands_2d` (left yellow, right mint) as a third
  comparison view, `contexts/assembly101_world_mm_3d` as a `Spatial3DView` with the hands in
  world mm and the estimated C10379 camera as a `Transform3D` + `Pinhole` frustum,
  `diagnostics/assembly101/{confidence,wrist_distance_to_stabilized_wilor_pixels}/{left,right}`
  on a new time panel, and `metadata/fine_grained_gt` (27-row table) plus a
  `metadata/navigation/fine_gt_index` series. Per the Sep 17 human nit ("agent substep is
  confusing"), the fine-grained dataset segments replace the agent-authored substep on the
  navigation panel and lead the per-frame navigation document; the substep series stays
  logged and the contract stays a tab. `InteractionReviewIndexManifest.assembly101_reference`
  fingerprints the window manifest; its input fingerprints join `input_artifacts`; the guide
  gains a dataset section. Rebuilt in place (20.4 s; 18 blueprint views, was 15).
- **Tests.** 335 passing default tier (+13: clock rule, joint graph, pinhole projection,
  synthetic hand-frame build with zero-confidence drop and inside-image counts, missing pose
  frame is an error, fine-grained CSV parsing with view filter/window clipping/overlaps,
  segment and hand-side validators, `load_reference` fingerprint refusal, the dataset
  blueprint referencing only logged entities, fine-GT series clearing); `real_data` tier
  checks the built window (27 segments, +9 offset, projection RMS < 0.01 px) and that the
  built v4 index and RRD carry the dataset entities; the 2D/3D root dimensionality guard now
  covers the new 3D root.
- **Open.** The +9 static offset is applied only to this dataset layer. The canonical
  ego/static comparisons (`battle-build-ego-static-comparison`, focused two-view) still
  assume zero relative offset; correcting them is the next multicam step, together with the
  static/ego correspondence audit and an ATHENA triangulation trial now that the intrinsics
  blocker is gone in practice. Hidden-interior semantics for `[1024,1172)` and the pending
  human QA dispositions are unchanged.

### Sep 18: Track 0 of the overnight multicam pass (all static views, per-view clocks, per-view cameras)

- **Claim boundary first.** Everything in this track is external review context: the dataset's
  poses and extrinsics are the dataset's, the fitted intrinsics are estimates of the dataset's
  own internal projection (the archive ships none), the measured clock offsets describe when
  each camera's video started relative to the pose clock. None of it is ground truth for any
  method compared here and no accuracy number rests on it; CC BY-NC 4.0 attribution applies.
  CPU only; no model ran; no viewer opened.
- **Acquisition (new module `assembly101_fetch_view.py`, CLI `battle-fetch-assembly101-view`).**
  For each of C10095, C10115, C10118, C10119, C10390, C10395, C10404 the pinned Hugging Face
  file resolves to a signed CDN URL that answers Range requests; ffmpeg reads it through a local
  counting proxy, seeks to source 294.000 s and writes, from one decode, the 60 fps trim at
  sensor resolution (`<view>_rgb_294.000-386.700_raw60.mp4`, 5,562 frames) and the standard
  1280x720 CFR-30 proxy (2,781 frames, identical filter chain and encoder settings to the C10379
  proxy). Two requests per view (3,145,728 B for the moov atom at the head; one open-ended range
  for the mdat window). Coverage: 7/7 remote views, 12/12 views with trims (the local C10379 and
  four HMC recordings ran through the same recipe; their pinned proxies were kept and three new
  954x720 HMC proxies at 294.000 s were added). Bytes: 1,852,833,792 B total, 10.4-11.0 % of each
  file, never the whole file. Time to first output: 53 s for the first view (C10095, network +
  encode); runtime 45-90 s per remote view, ~19 s per local trim; the whole set finished in
  under 5 min with three fetches in parallel. VRAM n/a. Frame alignment verified on C10379
  (local full file): trim frame t = raw frame 17640 + t and proxy frame p = trim frame 2p by
  downscaled MAE minimum. Report + manifest under the ignored raw tree
  (`static_views_focused_acquisition_report.md`, `..._manifest.json`, per-view JSON records);
  G1 entry in `docs/SOURCES.md` with byte ranges, sizes and SHA-256s. The tracked eight-view clip
  config is `configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_all_static_g2.json`
  (view ids `static-c10095` ...; remote raw sources cite the `hf://` path and the LFS etag,
  which is the file's SHA-256). The single-view focused config was deliberately not extended:
  `interaction_review`, `exploratory_comparison` and `kineo_fusion` assert
  `clip.views == ("static-c10379",)` on run manifests that embed it.
- **Per-view clock offsets (new module `assembly101_clock_offset.py`, CLI
  `battle-assembly101-clock-offset`).** Port of the Sep 17 ad-hoc scans onto the 60 fps trims and
  the dataset's own 2D landmarks: three metrics (YCrCb skin-mask hit rate for RGB views,
  Gaussian-smoothed Sobel gradient magnitude, frame-to-frame absolute difference) sampled at
  the five fingertips of hands with confidence >= 0.7 moving faster than 5 px/frame (3 for ego),
  velocity-weighted, three 30 s chunks, offsets -6..+15, parabolic sub-frame peak. A chunk
  whose maximum sits on the edge of the range or whose z-prominence is below 1.2 is reported but
  not counted; each metric votes with the median of its informative chunks; the rule is the
  rounded mean of the metric medians, uncertainty the largest metric deviation (never below
  one frame); metrics disagreeing by more than two frames make the view ambiguous with no rule.
  Coverage: 12/12 views scanned, 5,400 frames each, none ambiguous. Runtime 47-51 s per static
  view, 8-9 s per ego view (429 s total, first output = end of the first view). Result: the
  static cameras do not share one offset. C10095 +5 (+-1, sub-frame +4.96), C10115 +6 (+-1,
  +5.65), C10118 +6 (+-2, +5.85), C10119 +7 (+-1, +6.83), **C10379 +9 (+-1, +8.51)**, C10390 +7
  (+-1, +6.67), C10395 +6 (+-2, +6.48), C10404 +6 (+-1, +6.26); all four HMC cameras 0 (+-1;
  sub-frame +0.38, -0.02, -0.11, -0.44). The C10379 regression check therefore agrees with the
  committed +9 rule, with the honest caveat that its sub-frame estimate is 8.5: the true lag sits
  between eight and nine 60 fps frames and either integer is within the stated +-1. Tracked
  summary `configs/assembly101/clock_rules.json`; per-view curves under
  `runs/assembly101-clock-offsets/`. Two curve-level findings worth keeping: a gradient chunk
  can produce a ramp with its maximum on the range edge (C10395 chunk 0 at -6), which is why
  edge maxima are excluded; and the +9 offset previously measured for C10379 is the largest of
  the eight, so applying it to another static camera would have been wrong by up to four frames
  (67 ms).
- **Per-view cameras (new module `assembly101_camera_fit.py`, CLI
  `battle-fit-assembly101-camera`).** Generalisation of `intrinsics2.py`: 112 frame groups every
  50 pose frames over [17640, 23202), hands with confidence >= 0.8, joints inside the image;
  `cv2.calibrateCamera` full Brown for static views, rational model (`CALIB_RATIONAL_MODEL`,
  zero tangential) with per-frame `camera_extrinsics_ego` for the HMC cameras; each result is
  verified by re-projecting the dataset 3D through the *shipped* extrinsics with the fitted
  intrinsics. Static RMS through the shipped pose: C10095 0.00039 px, C10115 0.00096, C10118
  0.00062, C10119 0.00027, C10379 0.00028 (identical parameters to the checked-in file), C10390
  0.00062, C10395 0.00021, C10404 0.00228 (max 0.0029); 4,388-4,389 points each. Ego: e3
  HMC_21110305 0.275 px (max 0.93, 4,229 points), e4 HMC_21179183 0.305 px (max 1.32, 4,192
  points), e2 HMC_21176623 5.06 px (max 19.0, only 1,120 points / 54 frames), e1 HMC_21176875
  4.06 px (max 5.3, 1,532 points / 61 frames). OpenCV 5's fisheye model was tried for the ego
  cameras and is far worse (27-33 px through the shipped pose), so the rational model stays; the
  e1/e2 residuals reflect how rarely those cameras see the hands, and their estimates are
  labelled accordingly in `fit_notes`. Runtime 4.5 s for all twelve fits (2.3 s of it loading the
  pose members). `Assembly101CameraModel` gained `distortion_model` (`brown` | `rational`),
  `extrinsics_kind` (`fixed` | `per_frame_ego`, ego models carry no constant pose),
  `fit_frame_count`, the shipped-pose residual and `fit_notes`; the existing C10379 file loads
  unchanged and was not rewritten (built runs fingerprint it). Twelve
  `configs/assembly101/<view>_camera_estimate.json` files are checked in.
- **Tests.** `tests/test_assembly101_acquisition.py`: ffmpeg command shape (single seek, split
  filter graph, trim-only variant), the counting proxy against an in-process Range server,
  sub-frame peak / edge / flat curve handling, the metric-median decision with an outlier chunk
  and the ambiguity rule, a synthetic 60 fps trim whose known offset (3 and 11) is recovered end
  to end, `rescore` round trip, camera-model validators and legacy-file compatibility, a
  synthetic Brown recovery through a given pose, and default-tier checks that the twelve tracked
  camera files and the tracked clock rules are complete; `real_data` checks the acquired windows
  and that the on-disk scans match the tracked rules.
- **Open.** The offsets are per-camera constants for this recording; nothing was checked about
  drift within the 92.7 s window beyond the three-chunk agreement. Track 1 consumes
  `clock_rules.json` and the camera files through `multiview_geometry.CameraRig`.

### Sep 18: overnight GPU queue runner

- **What.** `scripts/overnight_queue.py` (logic in `battle.overnight_queue`, so it is unit
  tested) runs a JSON job list strictly serially under
  `systemd-inhibit --what=sleep:idle --why="battle overnight"`; before each job it requires
  `nvidia-smi` to answer and `journalctl -k --since <queue start>` to carry no `NVRM`/`Xid`
  line, kills a job that exceeds its timeout together with its process group, appends one JSON
  line per event to `runs/overnight-multicam-20260918/queue.log`, keeps each job's combined
  output under `logs/`, and stops at the first GPU error, non-zero exit, timeout or spawn
  failure unless `--continue-on-failure` is given (later jobs are logged as `skipped` with the
  reason). Job states: `succeeded`, `failed`, `timed_out`, `blocked_gpu`, `skipped`,
  `spawn_failed`.
- **Checks.** Ten default-tier tests drive the state machine with fake commands and a fake GPU
  check (serial order and full event log, stop-on-first-failure with skips, continue-on-failure,
  timeout kill in under 10 s with a shortened grace, GPU-gate block before the job starts, spawn
  failure as a recorded state, the `NVRM|Xid` pattern, spec round trip with interpreter prefix,
  `--dry-run`, and the CLI without inhibit/GPU check). One real smoke (`echo queue-ok`) ran under
  the inhibitor with the live GPU check (`NVIDIA GeForce RTX 5070 Ti, 1197 MiB, 50 C`, no
  kernel errors) and is the first record in the overnight queue log. CPU only; no model ran.
- **Boundary.** The runner knows nothing about what the jobs do or produce; it only sequences
  them and records their exit. It does not resume a partially run list; re-run with a trimmed
  job file.

### Sep 18: Track 1, one calibrated rig for all twelve views and the two-view clock fix

- **Claim boundary first.** The rig is geometry on dataset context (shipped extrinsics, per-frame
  ego poses) and on fitted estimates (intrinsics). Its residuals against the dataset's own 2D
  and 3D landmarks show that it reproduces the dataset's internal projection; they are not
  accuracy claims about the dataset, the cameras or any method. Triangulated points are
  cross-view agreement. CPU only; no model ran; no viewer opened. CC BY-NC 4.0 applies.
- **Module.** `multiview_geometry.CameraRig` (`CameraRig.load(repository_root)` reads the twelve
  `configs/assembly101/<view>_camera_estimate.json`, `configs/assembly101/clock_rules.json` and
  the window's `camera_extrinsics_ego`). Time is the 60 fps pose clock; `pose_frame(view,
  analysis_frame)` applies the view's measured rule. API: `project(view, X_world_mm,
  pose_frame=None)` (raw px, distortion applied, ego views need the pose frame),
  `depth`, `undistort` (raw px to normalised coordinates, Brown or rational), `ray(view,
  pixel, pose_frame=None)` (world `Ray(origin, direction)`), `triangulate(points_by_view,
  pose_frame=None, reproj_filter_px=30, min_views=2)` (equal-weight DLT on normalised
  coordinates as in ATHENA's `triangulaterefine`, then, while a point's worst reprojection
  error exceeds the threshold and more than `min_views` views remain, drop the view whose
  removal leaves the smallest worst-case error; the fixture showed that dropping the largest
  residual can remove a good view when the compromise spreads a gross error), returning
  points, the used-view mask and the per-view reprojection error so a disagreeing two-view
  point stays visible; `epipolar_distance(view_a, px_a, view_b, px_b, pose_frame=None)`;
  `fit_table_plane(landmarks3d, confidences, pose_frames, lowest_fraction=0.05)` where
  "down" is the mean image-down (y) axis of the static cameras, cross-checked against the
  camera-to-scene direction, never an assumed world axis; `TablePlane.intersect_ray` for seed
  transfer. `battle-multiview-rig-check` writes `runs/assembly101-multiview-rig-check/report.json`.
- **Measures (rig check, every 25th pose frame of the window, hands with confidence >= 0.8).**
  Coverage 12/12 views loaded; runtime 3.3 s (time to first output the same). Projection of
  dataset 3D vs shipped 2D: C10095 0.00039 px RMS, C10115 0.00096, C10118 0.00062, C10119
  0.00027, C10379 0.00028, C10390 0.00062, C10395 0.00021, C10404 0.00228 (8,733-8,736 points
  each); ego HMC_21110305 0.274 px (max 2.2, 8,406 points), HMC_21179183 0.306 (max 1.3),
  HMC_21176875 4.04 (max 5.3, 3,232 points), HMC_21176623 5.14 (max 30.1, 2,313 points).
  Triangulating the dataset's own 2D back to 3D: all eight static views median 0.0004 mm, p95
  0.0004, max 0.001 (8 views used everywhere); C10379+C10395 median 0.0007 mm, p95 0.0029,
  max 0.098; C10115+C10404 median 0.0020 mm, p95 0.0024; eight static + HMC_21110305 median
  0.062 mm, p95 1.94, max 16.0 (mean 8.96 views used: the ego's 0.3 px residual at a 190 px
  focal length is millimetres at arm's length and the 30 px filter rarely drops it, so the
  ego view should be weighted or filtered tighter when it joins a static triangulation).
  Table plane: normal (-0.331, 0.942, 0.048), 2,604 of 52,075 fingertips (lowest 5 %), residual
  RMS 9.98 mm, p95 19.7 mm, 0.2 % of all fingertips more than 20 mm below the plane, normal 3.5
  degrees from the mean static image-down axis; the camera-position-to-scene direction is 56
  degrees from that axis, which says the cameras look at the table obliquely, not that either
  estimate is wrong.
- **Two-view clock fix.** `export_synchronized_comparison` gained `static_clock_shift_seconds`
  and `rerun_comparison.measured_static_clock_shift` derives it from the tracked rules: C10379
  +9 pose frames minus HMC 0 = 9 pose frames = 4.5 analysis frames = 0.150 s; the static video
  started later, so static analysis frame p shows the scene 0.150 s after ego frame p.
  Decision on the half frame: nothing is resampled. Every static entry (video frame reference,
  masks, hands, tracker diagnostics) is logged at `analysis_time = p/30 + 0.150` and
  `source_time + 0.150`, the static `VideoFrameReference` still addresses frame p of the static
  file, and on the integer `analysis_frame` timeline the static entries move by
  round(4.5) = 4 frames (documented in the recording's `comparison_metadata`). Scrubbing
  `analysis_time` gives exact alignment; the static view is empty for the first 0.150 s and
  runs 0.150 s past the ego's end. Rebuilt: focused first minute to
  `runs/four-part-focused-first-minute-comparison-offset-v2/` (84.5 MB, 58 s) and the canonical
  180 s pair (C10379 vs HMC_21179183, same 0.150 s) to
  `runs/ego-static-synchronized-comparison-offset-v2/` (122.0 MB, 20 s). Verified in the RRD:
  static frame 0 sits at analysis_frame 4 / analysis_time 0.150 s / source_time 294.150 s with
  video timestamp 0, ego frame 0 at 0 / 0 / 294.000. The earlier zero-offset recordings
  (`runs/four-part-focused-first-minute-comparison/`, 81.0 MB, and the 131.5 MB one inside the
  ego run directory) are superseded and kept. `--static-clock-shift-seconds 0` reproduces
  them. The offset is assumed constant over the recording; it was measured on the focused
  window only.
- **Tests.** `tests/test_multiview_geometry.py`: a three-static-plus-one-ego fixture rig with
  Brown and rational distortion (bookkeeping and clock rules, projection equals OpenCV and rays
  pass through the points, ego pose changes per frame, triangulation recovers points to 0.01 mm
  and removes a corrupted third view while the unfiltered solve does not, NaN gaps and lone
  views, `dlt_triangulate` patterns, epipolar distance zero for a consistent pair, table plane
  on a tilted synthetic table with camera-derived down and ray intersection); `real_data`:
  the loaded rig reproduces the shipped 2D for a static and an ego view and triangulates a
  dataset hand from three static views to 0.01 mm, and the rig-check report meets the
  thresholds above. Exporter/comparison tests cover the shift metadata and the shifted static
  rows, view-id-to-camera mapping and the measured 0.150 s.

### Sep 18: Track 3a, MediaPipe on every static view plus the ego mono stress test

- **Claim boundary first.** Detection presence, handedness and landmark placement are model
  outputs with no hand-pose ground truth. The per-view "wrist vs dataset 2D" numbers compare
  MediaPipe's wrist against the dataset's own tracker projection (raw sensor pixels,
  handedness-agnostic nearest match): cross-source distance, not accuracy. CPU only, no GPU
  touched; no viewer opened. CC BY-NC 4.0 applies to the dataset assets.
- **Adapter change.** `battle-mediapipe-hands` accepts any view of the given clip config (it
  previously refused everything but `static-c10379`), and gained `--roi-from-dataset-2d
  [--roi-margin 0.05]`: the inference crop is the union box of the dataset's shipped 2D hand
  joints for that camera over the run (hands with confidence >= 0.5, joints inside the sensor)
  padded by 5 % of the image per side, recorded in the manifest as `roi` plus
  `roi_source`. The Sep 16 hand-tuned C10379 crop `0.45,0.35,0.55,0.65` is reproduced by the
  derived rule to within 0.07 (`0.520,0.355,0.480,0.619`), which is why the derived ROI was
  trusted on the other seven views instead of a copied crop; per-view crops span 12-31 % of
  the image (C10390 smallest, C10118 largest). All seven static runs used the selected Sep 16
  condition otherwise (full frame + 2x upscaled crop fusion, two-hand cap, 0.35 thresholds).
  The ego mono proxy ran full-frame only (its joints fill the frame) with the same thresholds;
  OpenCV decodes the yuv420p mono file as three identical channels, so MediaPipe saw a
  grayscale RGB image, which is the stress: this model was trained on colour.
- **Runs** (`runs/mediapipe-hands-<view>-60s-20260918/`, 1,800 frames each, four in parallel,
  ~35 s wall each; `dataset_2d_check.json` in each run from `battle-hands-2d-check`). Five
  measures per view: coverage (frames with >= 1 hand / with 2), time to first output, runtime,
  VRAM n/a (CPU), ID resets n/a (hand ids are frame-local by contract). Wrist column: median
  (p90) distance in raw px from each dataset wrist (confidence >= 0.5, inside the image) to the
  nearest detected wrist, and how many of those dataset wrists had a detection within 150 px.

  | view | frames with hand (two) | first output s | runtime s | wrist median (p90) px | within 150 px |
  | --- | --- | --- | --- | --- | --- |
  | C10095 | 1795 (304) | 0.126 | 35.3 | 32.2 (134.5) | 3226 / 3426 |
  | C10115 | 1797 (1301) | 0.129 | 34.8 | 20.8 (207.3) | 2965 / 3427 |
  | C10118 | 1800 (951) | 0.122 | 36.4 | 35.5 (280.6) | 2641 / 3427 |
  | C10119 | 1800 (1122) | 0.130 | 34.1 | 19.8 (259.7) | 2773 / 3427 |
  | C10379 (Sep 16 run, fixed crop) | 1648 (952) | 0.081 | 30.7 | 91.9 (340.1) | 2026 / 3428 |
  | C10390 | 1759 (681) | 0.119 | 34.3 | 24.4 (118.3) | 3281 / 3427 |
  | C10395 | 1350 (69) | 0.081 | 36.4 | 50.7 (249.9) | 1708 / 3427 |
  | C10404 | 1800 (1492) | 0.095 | 34.1 | 17.3 (68.1) | 3119 / 3427 |
  | HMC_21110305 mono (stress) | 446 (14) | 0.057 | 14.1 | 40.7 (265.9) | 483 / 2714 |

  Reading: C10404, C10115 and C10119 are the strongest views (both hands in most frames,
  wrists within ~20 px of the dataset's). C10395 finds a second hand in only 69 frames and
  C10095 in 304, so they contribute mostly one hand to any triangulation. C10379, the close-up
  that every earlier MediaPipe number was measured on, has the largest wrist disagreement
  (91.9 px raw = 61 proxy px); a clock sweep over +5..+11 pose frames moves that median by
  under 1 px, so it is not a synchronisation error but the view itself: hands are large and
  frequently occluded by the held part, and the nearest-detection rule pays the full distance
  to the other hand whenever one is missed (only 59 % of dataset wrists have a detection within
  150 px, against 91 % on C10404). The ego mono stress arm fails as expected: hands in 446 of
  1,800 frames, two hands in 14, on a camera the dataset says sees both hands almost always.
- **Tests.** `tests/test_mediapipe_hands.py` +3: view-id to camera-name mapping, `real_data`
  derived-ROI agreement with the Sep 16 crop, and the eight Sep 18 run directories with their
  dataset checks (coverage and wrist-median floors).

### Sep 18: Track 3b, ATHENA multi-view hand triangulation against the dataset 3D

- **Claim boundary first.** The triangulated hands are a DLT on MediaPipe detections through
  fitted intrinsics and the dataset's shipped extrinsics; the dataset `landmarks3D` are the
  output of the dataset's own multi-view tracker with a fixed-scale hand model. Their
  millimetre difference is cross-source disagreement between two estimates, never accuracy of
  either. Hand-side correspondence borrows the dataset wrist projected into each view (method
  handedness is unreliable); positions do not. Intrinsics are estimates. CPU only; the GPU was
  not touched; no viewer opened. Assembly101 CC BY-NC 4.0; ATHENA MIT.
- **ATHENA itself, not a reimplementation.** `/home/nick/src/athena` @ `e85bd494` was installed
  into its own py3.12 virtualenv (`/home/nick/src/athena/.venv`, `uv pip install -e .`:
  mediapipe 0.10.21, numpy 1.26.4, opencv 4.11.0, scipy 1.17.1). Its `labels2d` module imports
  tkinter and MediaPipe 0.10 at import time, so pulling `triangulaterefine` into the battle env
  (mediapipe 1.0.1, no scipy) was not an option; `battle-athena-hands` instead writes the
  undistorted normalised points, matching pixel coordinates, world-to-camera extrinsics and
  intrinsics to an `.npz`, runs `scripts/athena_triangulate_worker.py` under the ATHENA
  interpreter (`_triangulate_with_filtering` per batch, then `_smooth3d`, 20 Hz Savitzky-Golay,
  order 3, gaps over five frames restored) and reads the points back; only numpy crosses the
  boundary. Two ATHENA conventions were handled rather than copied: the dataset's
  camera-to-world poses are inverted to world-to-camera (`CameraRig.projection_matrix`), and
  because ATHENA's filter reprojects with `K @ E` and no distortion, the "pixel" coordinates it
  compares against are the *undistorted-image* pixels (`K` applied to the Brown/rational
  undistorted normalised coordinates), so the 30 px filter acts in a consistent space. The
  zero-distortion shortcut was not used: every observation goes through
  `cv2.undistortPoints` with the view's fitted coefficients. ATHENA returns points only, so a
  view counts as contributing when its final reprojection error is within the filter threshold
  (the loop's own acceptance rule). `--triangulator rig` runs the Track 1 leave-one-out DLT
  instead; on the same inputs it lands within 0.4 mm of ATHENA on every headline number
  (wrist median 24.3 / 19.8 mm vs 23.9 / 19.8) in 31 s against ATHENA's 6.7 s, which says the
  two filters agree here and that ATHENA's batched SVD is the one to keep.
- **Inputs and alignment.** Per view: the Sep 18 MediaPipe runs (C10379: the selected Sep 16
  run), landmarks normalised -> raw px (x1.5). Analysis timeline = C10379 clock
  (`pose_frame = 17649 + 2p`, the dataset reference window); every other view contributes the
  analysis frame nearest that pose frame under its own measured rule: C10095 +2.0 frames
  (residual 0), C10119/C10390 +1.0 (0), C10115/C10118/C10395/C10404 +1.5 (rounds to +1 pose
  frame = 16.7 ms), HMC_21110305 +4.5 (+1). Nothing is resampled; the residual is in the
  manifest. Side matching: minimum-total-distance assignment of <= 2 detections to <= 2 dataset
  wrists (confidence >= 0.5) within 150 raw px; match-distance medians 16-30 px on the seven
  other static views, 51.6 px on C10379 (671 of 2,600 C10379 detections unmatched, the view's
  known weakness from Track 3a). Twenty common joints (table in the module docstring; dataset
  palm and MediaPipe thumb_cmc dropped; the thumb pair is the least certain correspondence).
- **Measures, static arm** (`runs/athena-hands-first-minute-mediapipe/`, 8 static views, 1,800
  frames; runtime 6.7 s, first output 6.6 s; VRAM n/a; coverage: left hand solved in 1,757
  frames, right in 1,634, dataset present 1,800 / 1,700; ID resets n/a, sides come from the
  dataset match). Disagreement vs `landmarks3D`, raw DLT (smoothed in brackets):

  | hand | mean contributing views | wrist median / p90 mm | fingertips median / p90 mm | all 20 joints median / p90 mm |
  | --- | --- | --- | --- | --- |
  | left | 3.64 | 23.9 / 37.1 (23.9 / 36.3) | 38.8 / 84.3 (38.8 / 82.4) | 25.4 / 68.7 (25.6 / 66.6) |
  | right | 6.59 | 19.8 / 31.2 (19.7 / 31.1) | 31.3 / 59.6 (31.3 / 59.4) | 21.9 / 48.2 (21.9 / 48.1) |

  Per joint (raw median mm, left / right): wrist 23.9 / 19.8; index_mcp 17.7 / 13.9, middle_mcp
  18.6 / 12.5, ring_mcp 18.9 / 14.3, pinky_mcp 23.2 / 18.8 (the palm agrees best); thumb_cmc
  16.5 / 26.2, thumb_ip 20.3 / 26.9 (the uncertain correspondence); tips thumb 26.0 / 28.7,
  index 37.5 / 26.0, middle 46.6 / 32.8, ring 44.8 / 35.9, pinky 42.9 / 35.3. Fingertips
  disagree roughly twice as much as the palm, on both hands, which matches the Sep 18 2D
  finding that dataset and method agree on hand scale but not on articulation when fingers are
  hidden behind the held part. Per-view reprojection RMS of the points a view contributed to
  (left / right): C10095 15.3 / 7.8 px, C10115 12.8 / 9.0, C10118 15.8 / 12.3, C10119 10.7 / 9.8,
  C10379 15.8 / 13.4, C10390 12.8 / 8.7, C10395 11.9 / 12.5, C10404 11.5 / 10.4; over all
  observed points (including the ones the filter dropped) C10379 is worst at 59.1 / 41.1 px and
  C10118 second at 51.4 / 17.1. The left hand is seen by half as many views as the right
  (3.6 vs 6.6): C10095 and C10395 rarely detect a second hand (Track 3a) and C10379 misses it.
- **Ego arm** (`runs/athena-hands-first-minute-mediapipe-ego/`, + HMC_21110305 with its
  per-frame pose, 1,800 single-frame ATHENA batches, 9.8 s). MediaPipe found hands in only 446
  ego frames (mono stress test), 445 matched at 20.2 px median; the ego view was used for 4,820
  of 5,020 observed left-hand points (RMS 10.7 px) and 2,931 of 3,880 right (14.7 px). Headline
  numbers move by under 0.4 mm (left wrist 23.9 -> 23.9, right 19.8 -> 19.6; mean views 3.64 ->
  3.75 and 6.59 -> 6.67). So with this few ego detections the ego camera neither helps nor
  hurts measurably; the Track 1 warning about weighting it stands for a future arm with a
  colour ego stream.
- **First honest answer on "is WiLoR 3D hopping fixable".** Median per-frame wrist displacement
  on the C10379 clock: multi-view DLT 2.8 mm (left) / 3.3 mm (right) raw, 2.5 / 2.9 mm after
  ATHENA smoothing; dataset tracker 1.9 / 3.5 mm; p90 10.4 / 9.8 mm vs dataset 4.6 / 9.8. In
  C10379 pixels the multi-view wrist reprojected moves 4.3 / 4.0 px per frame (p90 19.5 / 12.5)
  against WiLoR's own 2D wrist 3.1 / 6.3 px (p90 11.2 / 17.4): in the image the two are equally
  steady. WiLoR's *camera-frame* wrist (joint 0 plus `pred_cam_t_full` from the native evidence,
  in WiLoR's own units under its 25,000 px scaled focal length, not millimetres) jumps a median
  145 (left) / 333 (right) units x 1e-3 per frame, p90 535 / 1,169, almost entirely in depth,
  while its wrist-rooted `joints_3d_camera_relative` is constant by construction (the wrist sits
  at a fixed MANO offset, ~2e-5 per frame). Reading: the hopping lives in WiLoR's per-frame
  depth/translation guess, not in its 2D or its hand articulation; a calibrated multi-view
  estimate of the same wrist is two orders of magnitude steadier frame to frame and within
  ~20 mm (median) of the dataset's tracker. "Fixable" therefore means replacing WiLoR's
  translation with a triangulated wrist and keeping WiLoR's articulation, which is what the
  queued three-view WiLoR arm will test; it is not a claim that either estimate is right.
- **WiLoR arm, queued not run.** `runs/overnight-multicam-20260918/jobs_wilor_3views.json`
  (validated with `--dry-run`): `battle-wilor-hands` on C10379, C10395, C10115 (60 s, native
  evidence on, 900 s timeouts) followed by `battle-athena-hands --hand-source wilor` on the
  three and its review recording. `--hand-source wilor` reads the same `observations.jsonl`
  contract (WiLoR's 21 joints share the MediaPipe order) and attaches `pred_cam_t_full` from
  `native_evidence/` for the steadiness series; verified as a loading smoke against the
  existing `runs/wilor-hands-static-60s-overnight-v2` (1,800 frames, 1,751 with hands,
  camera-frame depth ~31.8 units), and a single view is refused with "needs at least two views".
- **Viewer.** `battle-build-athena-hands-review` writes `hands.rrd` beside each run (dataset
  hands yellow/mint, triangulated orange/blue with thin smoothed skeletons, eight camera frusta,
  time series of contributing views, wrist disagreement mm and per-view reprojection RMS; 2.8 s).
  `interaction_review_v4` gained an `athena_hands` layer (on by default, skipped when
  `runs/athena-hands-first-minute-mediapipe/` is absent) that logs the same entities under the
  existing `contexts/assembly101_world_mm_3d` and `diagnostics/assembly101` roots through the
  new module `athena_hands_review.py`, so they land in the dataset 3D view and time panel
  without blueprint changes; a scratch rebuild confirmed the entities. The tracked v4 package
  was not rebuilt in place tonight (another worker is rebuilding it for the multiview layer).
- **Tests.** `tests/test_athena_hands.py` (9): mapping table invariants, nearest-frame residual
  rule, handedness-agnostic matching with threshold, numpy Savitzky-Golay preserving cubics and
  the undistorted-pixel helper, a three-camera synthetic scene (Brown distortion, moving hands,
  one view with swapped detection order) recovered to < 2 mm median through the rig path, the
  same scene through the ATHENA worker with one corrupted wrist dropped by the filter (skipped
  when the ATHENA venv is missing); `real_data`: WiLoR loading smoke with camera translation and
  single-view refusal, and both built runs (>= 1,500 frames solved, mean views >= 3, wrist
  median in (5, 40) mm, per-view used RMS <= 30 px, hands.jsonl length, claim boundary, RRD).
- **Open.** Per-view weights (ego especially) and a tighter ego filter; WiLoR arm pending GPU;
  the thumb correspondence; a per-frame time interpolation of views to the reference pose frame
  would remove the 16.7 ms residual for fast motion.
