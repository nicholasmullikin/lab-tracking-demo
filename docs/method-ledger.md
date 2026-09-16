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
| Typed manifests, fixture tests, inference-free Rerun exporter | Done | `src/battle/schemas.py`, `src/battle/exporter.py`, 125 passed, 1 skipped |
| Pin one Assembly101 segment with source/analysis/annotation/pose clocks | Done | `configs/clips/*.json`; nusar-9033, 215.000–395.000 s |
| MuggledSAM/SAM3 running over the full 180 s static view | Done (aligned hybrid: three text, one reviewed mask) | [Sep 14 aligned hybrid](#sep-14-aligned-static-hybrid-candidate) |
| MuggledSAM/SAM3 running over the full 180 s ego view | Done, but only with human-seeded masks | [Four-target 180 s baseline](#sep-9-evening-four-target-180-second-ego-baseline) |
| Both views on one synchronized Rerun timeline | Done, rebuilt with aligned hybrid static | `battle-build-ego-static-comparison`; [Sep 14 aligned hybrid](#sep-14-aligned-static-hybrid-candidate) |
| Five pre-accuracy measures recorded per run | Done | Every `worker_result.json` and `manifest.json` |
| MediaPipe Hands static-view baseline (core spine) | Not started | Displaced by the ego calibration work |
| Second method in the viewer (WiLoR or any exploratory item) | Not started | [hand-pose adapter](#hand-pose-adapter) is `not_run` |
| Fixed two-timestamp human QA per completed method | Records prepared; human dispositions pending | Human-selected source frames 14,868/21,732; aligned static, ego, and preserved historical records under `docs/qa/` |
| No training, no annotation project, no accuracy claims | Held | Reviewed masks are calibration seeds, not labels; no metric vs. ground truth anywhere |
| Four physical components through reassembly | Full exploratory run completed; segmentation not accepted | [Sep 15 four-part experiment](#sep-15-four-part-static-reassembly-experiment) |
| Git history from the start | Missed, then repaired | First commit Sep 13 after five days of uncommitted work |
| FineBio | Still pending | Not part of any run |
| Audio | Deferred by plan | Not revisited |

The honest summary: one method, one dataset, one clip. The comparison lab the plan
described has a working spine for its first method and a well-instrumented viewer, and
essentially all of the time went into making SAM3 usable on the monochrome ego view.

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
  next contract is `[3120,5901)`, or 2,781 frames / 92.7 seconds, with a new
  frame-3120 calibration rather than transferred frame-0 seeds.

### Plan versus actual

What the plan said, what happened instead, and why, in one line each.

- One dataset, one clip, one ten-second interval for most experiments, versus "as many
  repos as we can". The ego footage was hard enough that making one method usable on it
  consumed the budget; the 180 s runs exist, but the iteration loop lived at 10 s.
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
- No MediaPipe, no WiLoR, no exploratory queue. Never reached. These are the cheapest
  next steps precisely because the exporter, schemas, and viewer are done.
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

- Create a new four-part calibration and bounded run for proxy frames `[3120,5901)`.
  The components begin separated at frame 3120 and are assembled afterward; this is a
  cleaner test of identity preservation than carrying tracks through the earlier
  disassembly/warm-up interval.
- Inspect the aligned hybrid static and canonical ego model-overlay sheets and complete
  their pending records under `docs/qa/`. The old static zero-shot record remains historical;
  no new hybrid pass/flag/fail has been assigned.
- Next method: MediaPipe Hands on the static view is the plan's core-spine item and
  needs no GPU; WiLoR is the named second wave. Either can reuse the SAM3 hand masks and
  boxes now that they are trusted. Source and license review comes first
  (`docs/SOURCES.md`, `docs/LICENSES.md`).
- `yellow_toy_top` leaves the frame at ~216.2 s in every arm; its coverage numbers
  describe the scene, not the tracker.

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
- Next decision: start at proxy frame 3120 (104.0 proxy seconds / 294.0 source seconds),
  where the four components are separated before being merged. The bounded range is
  `[3120,5901)`, 2,781 frames / 92.7 seconds. Create new masks at frame 3120 and treat it
  as the run's local initialization frame; do not reuse the exploratory run's frame-0
  masks.
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
- State: `not_run`
- Blocker: deferred until a provenance-approved source and adapter plan. The plan's
  core-spine item is MediaPipe Hand Landmarker on the RGB static view (CPU-only); WiLoR
  is the named second-wave upgrade.
- Environment candidate: `~/.pyenv/versions/wilor`
- Weights: not downloaded
