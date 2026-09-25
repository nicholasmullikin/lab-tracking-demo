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

The verbatim request is preserved at the top of `battle_plan.agent.final.md` at the repository
root (kept locally, untracked and gitignored since Sep 24; tracked until then, so it is in the
history). In short: get several
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

Final statuses as of the Sep 24 close (the Sep 18 wording each row replaced is in the git
history of this file; the closing record is
[Sep 24: Assembly101 phase closed](#sep-24-assembly101-phase-closed)).

| Goal (from the Sep 8 ask and plan) | Status at close (Sep 24) | Evidence |
| --- | --- | --- |
| Typed manifests, fixture tests, inference-free Rerun exporter | Done. 762 tests in the default tier, 62 behind `real_data`, 2 behind `gpu`; ruff clean; three dedup passes (Sep 22) removed 1,675 lines inside the pre-existing modules with equivalence proven on three CPU artifacts | `src/battle/schemas.py`, `src/battle/exporter.py`; [dedup summary](#sep-22-dedup-passes-1-3-summary); [close](#sep-24-assembly101-phase-closed) |
| Pin one Assembly101 segment with source/analysis/annotation/pose clocks | Done, twice: nusar-9033 215.000–395.000 s (focused four-part window 294.000–386.700 s, first minute retained), and recording 2 nusar-9061 374–454 s fetched and calibrated Sep 20 for the generalization test | `configs/clips/*.json`; [Track C1](#sep-20-track-c1-of-the-multicam-plan-a-second-recording-fetched-and-prepared-nusar_9061) |
| MuggledSAM/SAM3 running over the full 180 s static view | Done (aligned hybrid: three text, one reviewed mask). Superseded as the comparison unit by the first minute of the four-part window at 1280 px `pm-append`, **0.743** mean IoU on the 52 human anchor cells (13 frames of C10379) | [Sep 14 aligned hybrid](#sep-14-aligned-static-hybrid-candidate); [memory arms](#sep-19-sam3-correction-memory-arms-on-c10379-at-1280-plan-step-1b); `runs/anchor-scoreboard-20260919/anchor_iou.md` |
| MuggledSAM/SAM3 running over the full 180 s ego view | Done, only with human-seeded masks. On the retained first minute e4 scores 0.558 (`r1280`, 3 parts) -> 0.589 (human-accepted 4-part seeds) on its 26-frame anchors; the interior stays behind the head camera at frame 0 and a per-slot start frame was never built | [Four-target 180 s baseline](#sep-9-evening-four-target-180-second-ego-baseline); [Sep 21, night](#sep-21-night-the-three-labelling-sessions-acted-on-human-accepted-seeds-an-interior-seed-from-two-human-masks-the-four-part-rerun-the-c10119-rear_body-finding-and-a-distractor-guard); `runs/multiview-reprompt-20260921/anchor_iou_e4.md` |
| Both views on one synchronized Rerun timeline | Done. The 9-pose-frame static lag (Sep 17) is applied in the `-offset-v2` two-view recordings and per view in every multi-camera build; the final surface is the **v6** recording (`runs/interaction-review-first-minute-v6/interaction_review_combined.rrd`) with three blueprint presets (`segmentation.rbl`, `hands.rbl`, `multiview.rbl`) | [Sep 18 Track 1](#sep-18-track-1-one-calibrated-rig-for-all-twelve-views-and-the-two-view-clock-fix); [v6](#sep-20-v6-review-surface-one-recording-with-three-blueprint-presets-track-d-of-the-multicam-plan); [`docs/review-guide-2026-09-20-multiview-presets.md`](review-guide-2026-09-20-multiview-presets.md) |
| Five pre-accuracy measures recorded per run | Done | Every `worker_result.json` and `manifest.json` |
| MediaPipe Hands static-view baseline (core spine) | Done: selected 60 s run, then every static view (Track 3a) and the ATHENA triangulation below | [hand-pose adapter](#hand-pose-adapter); [Track 3a](#sep-18-track-3a-mediapipe-on-every-static-view-plus-the-ego-mono-stress-test) |
| Second method in the viewer (MediaPipe) | Done; in the v6 recording under the `hands.rbl` preset with WiLoR, the dataset hand poses and ATHENA | [hand-pose adapter](#hand-pose-adapter); [v6](#sep-20-v6-review-surface-one-recording-with-three-blueprint-presets-track-d-of-the-multicam-plan) |
| Fixed two-timestamp human QA per completed method | **Closed without dispositions.** Records prepared Sep 13 (`docs/qa/*.human-qa.json`, all `pending`); never marked pass/flag/fail. Superseded Sep 19 by the human review anchors and `battle-anchor-iou` as the yardstick (52 + 64 + 75 cells on three cameras) | `docs/qa/`; [anchors](#sep-18-human-review-anchors-labeling-run-prepared-not-labelled-sep-18-labelled-sep-19); [close](#sep-24-assembly101-phase-closed) |
| Exploratory queue (WiLoR, BoxMOT, CLIP + Drop-DTW, Grounded-SAM-2, SAMURAI, DAM4SAM, ATHENA, Kineo) | All eight attempted at smoke tier (Sep 16). DAM4SAM then given a fair run (shared predictor, large checkpoint, the SAM3 schedule): **0.715**, tied with SAM3-1280 (0.724) within the 0.01 rule and the ensemble's fallback arm; SAMURAI+schedule unsupported upstream; the rest stay smoke-tier evidence. Cutie / XMem (E2) never tried | [Sep 16 queue](#sep-16-exploratory-queue-autonomous-pass); [DAM4SAM arms](#sep-19-dam4sam-and-samurai-arms-under-the-sam3-run-conditions-plan-step-2) |
| Multi-view geometry (8 static + 4 ego cams, dataset extrinsics, fitted intrinsics) | Done, as external context: 7 static views fetched for the 92.7 s window, per-view clock offsets measured, 12 camera estimates checked in; the rig reproduces the dataset's shipped 2D to <= 0.0023 px RMS (static) and 0.27-5.1 px (ego); the same tooling calibrated recording 2 | `configs/assembly101/`, `runs/assembly101-multiview-rig-check/`; [Track 0](#sep-18-track-0-of-the-overnight-multicam-pass-all-static-views-per-view-clocks-per-view-cameras); [Track 1](#sep-18-track-1-one-calibrated-rig-for-all-twelve-views-and-the-two-view-clock-fix) |
| Cross-view SAM3 consensus + visual hull | Done and answered: **multicam is a detector, not yet a corrector.** Consensus-only corrections with decoder-score ranking reach 0.659 (chassis 0.609, within 0.03 of the human's 0.637) against 0.743 human-corrected and a 0.591 seed-only floor; the interior gap (0.288 vs 0.585) is what no other camera tracks. Seeds for rear_body / cabin transfer automatically (0.80 / 0.95 held-out), chassis / interior do not (0.53 / 0.47). The 13 consensus and 37 hull `not_contact_eligible` proposals of Sep 18 were never dispositioned | [multicam closing](#closing-the-multicam-plan-the-two-goals-answered-with-numbers); [decoder-score run](#sep-20-evening-the-named-next-experiment-run-decoder-score-ranking-in-the-consensus-re-prompt-loop-commit-625321a-tooling-this-entrys-commit); `runs/multiview-reprompt-2026092{0,1}/anchor_iou_arms.md` |
| Multi-view hand triangulation (ATHENA on MediaPipe/WiLoR) | Done: ATHENA's own filter on an 8-view MediaPipe arm and a 3-view WiLoR arm; wrist disagreement vs the dataset tracker 20-29 mm median, fingertips roughly twice that; WiLoR's 3D hopping shown to be its per-frame depth and a triangulated wrist is 3.5-6x steadier (measured, not built). Unchanged since Sep 18 | `runs/athena-hands-first-minute-mediapipe/`, `runs/athena-hands-first-minute-wilor/`; [Track 3b](#sep-18-track-3b-athena-multi-view-hand-triangulation-against-the-dataset-3d); [GPU results](#sep-18-gpu-queue-results-wilor-arm-kineo-ego-exo) |
| Kineo multi-camera | Done, both arms in 8.4 min of the 2 h box: known-camera wrists 17 mm median from the dataset's; self-calibrated cameras 7.3 deg / 115 mm median from the dataset's after similarity alignment. Unchanged since Sep 18 | `runs/kineo-multiview-{known,selfcal}-first-minute-20260918/`; [GPU results](#sep-18-gpu-queue-results-wilor-arm-kineo-ego-exo) |
| Ego-exo correspondence (LM-EEC) | Done at 12 keyframes: exo->ego IoU vs the ego SAM3 mask 0.41 chassis / 0.27 interior median, 0.00 rear_body and cabin; checkpoint direction inferred from the file name, never checked. Unchanged since Sep 18 | `runs/egoexo-correspondence-first-minute-20260918/`; [GPU results](#sep-18-gpu-queue-results-wilor-arm-kineo-ego-exo) |
| No training, no annotation project, no accuracy claims | Held. Two gates crossed on request and labelled: dataset poses and fine-grained labels acquired Sep 17 as review context; human review anchors (52 cells C10379, 64 C10119, 75 e4) labelled Sep 19-21 as review evidence for ranking arms, one person's choice of decoder masks, not ground truth. No metric against ground truth anywhere; every table says so | [Sep 17 acquisition](#sep-17-assembly101-poses-extrinsics-and-fine-grained-annotations-selective-acquisition); [anchors](#sep-18-human-review-anchors-labeling-run-prepared-not-labelled-sep-18-labelled-sep-19) |
| Four physical components through reassembly | First minute closed at **0.743** (`pm-append`, human seeds + four corrections, C10379). On C10119 the chassis reaches 0.740 with five exemplar corrections at agent onsets (4-part seeds alone 0.339; 3-part `r1280` 0.758); the interior is the part that stays human on every camera (0.585 human, 0.288-0.363 agent). The late reassembly beyond the first minute (target loss at 1946, 2381, 2578, 2684) was never dispositioned | [Sep 17 human review](#sep-17-first-minute-v4-human-review-and-follow-up-rebuild); [exemplar Track B](#sep-22-exemplar-detections-as-correction-candidates-track-b-of-the-exemplar-plan-the-c10119-chassis-arm); [close](#sep-24-assembly101-phase-closed) |
| Git history from the start | Missed, then repaired; first commit Sep 13 after five days of uncommitted work. The phase closes at the annotated tag `assembly101-lab-close` | `git tag -n9 assembly101-lab-close` |
| FineBio | Started Sep 21 (SAM3 smoke, shipped detector run); next phase planned, see [`docs/plan-2026-09-24-finebio-detector-seeded-lab.md`](plan-2026-09-24-finebio-detector-seeded-lab.md) | [FineBio first look](#sep-21-finebio-first-look-sam3-zero-shot-text-prompts-on-one-first-person-clip); [shipped detector](#sep-21-finebio-shipped-detector-first-run-mmdetection-dino-and-deformable-detr-on-the-cpu-beside-the-sam3-smoke) |
| Audio | Deferred by plan; never revisited; closed with the phase | [close](#sep-24-assembly101-phase-closed) |

The honest summary, as of the morning of Sep 18: nine-plus methods attempted, one clip,
one minute reviewed closely. SAM3 with human seeds is still the only method that completed
both target views; MediaPipe and WiLoR completed the retained first minute of the static
view; the rest are smoke-tier evidence. The SAM3 spine is at its ceiling for this part
taxonomy (label migration between similar dark parts under rotation), and the review
tooling built to iterate on that one clip is where most of the code and roughly half the
time went. The multi-view line was tried overnight, on the dataset's own extrinsics and
fitted intrinsics, with agent-authored seeds no human has looked at. What it found:
cross-view consensus and a carved visual hull independently flag the human-reported C10379
identity failures (chassis 585-702 and 1089-1172) and do not flag 279-408; WiLoR's 3D
hopping is its per-frame depth guess, and a triangulated wrist removes it (depth step cut
3.5-6x) while its 2D and articulation stay untouched; Kineo's body agrees with the dataset
up to a similarity but its self-calibrated cameras sit 7 deg / 115 mm (median) from the
dataset's under any alignment; LM-EEC finds the chassis and interior in the monochrome ego
view in about half the keyframes and never the rear body or cabin; and the dataset's own
hand poses, which every one of those numbers is measured against, are themselves a
tracker's estimate that stretches when fingers hide behind the held part. Every number on
that line is cross-source disagreement between estimates, not accuracy.

The closing summary, Sep 24: what changed after Sep 18 is the yardstick and what it showed.
The human labelled review anchors on three cameras (Sep 19-21) and `battle-anchor-iou`
replaced self-consistency; on those anchors 1280 px and an appended prompt bank are the SAM3
run condition (`pm-append` 0.743 against the old reference's 0.668), DAM4SAM-large ties it,
and ensemble v2 is a named candidate with no adoption decision recorded. The multicam block
answered its two questions with numbers: SAM3's own object score detects its failures as a
ranker (AUROC 0.91-0.96) but no threshold transfers; other cameras seed the rear body and
cabin automatically and not the chassis or interior, and their corrections reach the human's
level on the chassis (0.609 vs 0.637) and nowhere near it on the interior. SAM3's own
exemplar detector then cleared the correction bar on one camera for one part (C10119 chassis
0.740). Recording 2 ran with zero human input and was never scored. The phase closes there;
the numbers, dispositions and open items are in
[Sep 24: Assembly101 phase closed](#sep-24-assembly101-phase-closed).

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

#### Sep 18, overnight: multi-camera pass

Run unattended from the user's plan (`overnight_multicam_pass`), one commit and one ledger
section per track; GPU work only through a serial queue with a kernel-error watchdog (25
jobs, 25 succeeded, no `NVRM`/`Xid` line). The morning review guide is
[`docs/review-guide-2026-09-18-multicam.md`](review-guide-2026-09-18-multicam.md).

- Track 0 (`0fdfeb7`). The seven remaining static views fetched for the 92.7 s window only
  (1.85 GB, 10-11 % of each file), 720p proxies, per-view clock offsets (+5..+9 pose frames
  static, 0 ego; C10379's +9 is the largest) and twelve camera estimates checked in. Record:
  [Track 0](#sep-18-track-0-of-the-overnight-multicam-pass-all-static-views-per-view-clocks-per-view-cameras).
- Queue runner (`994fbd5`). Record: [queue runner](#sep-18-overnight-gpu-queue-runner).
- Track 1 (`a2e4157`). `CameraRig` over all twelve views; the two-view recordings rebuilt
  with the measured 0.150 s static lag, closing the Sep 17 debt. Record:
  [Track 1](#sep-18-track-1-one-calibrated-rig-for-all-twelve-views-and-the-two-view-clock-fix).
- Track 3 (`a7d9807`, `5c0d87e`). MediaPipe on every static view with a dataset-derived
  crop (C10379, the close-up, is the weakest view); ATHENA triangulation of the eight views
  against the dataset 3D, and the first honest reading of WiLoR's hopping. Records:
  [Track 3a](#sep-18-track-3a-mediapipe-on-every-static-view-plus-the-ego-mono-stress-test),
  [Track 3b](#sep-18-track-3b-athena-multi-view-hand-triangulation-against-the-dataset-3d).
- Track 4 preparation (`3a81384`). Record:
  [Track 4 prep](#sep-18-track-4-preparation-kineo-on-all-eight-static-views-cpu-only-gpu-jobs-queued).
- Tracks 2, 5, 6 (`577df77`, `cd6d857`, `c7fe790`). Table-plane seed transfer failed from
  the grazing C10379 camera, so seeds were triangulated from C10379 and the e3 human masks;
  seven static SAM3 runs plus e4; cross-view consensus, visual hull, e1/e2 visibility audit.
  Records:
  [Track 2](#sep-18-track-2-of-the-overnight-multicam-pass-cross-view-sam3-with-a-geometric-combiner),
  [Track 5](#sep-18-track-5-of-the-overnight-multicam-pass-per-part-visual-hulls-from-eight-silhouettes),
  [Track 6](#sep-18-track-6-of-the-overnight-multicam-pass-the-other-ego-cameras).
- Track 7 preparation (`4e6f0e2`). LM-EEC installed inside its 90 min box. Record:
  [Track 7 prep](#sep-18-track-7-preparation-lm-eec-ego-exo-correspondence-cpu-only-gpu-job-queued).
- GPU results (`5e11133`). WiLoR three-view arm, both Kineo arms, LM-EEC; 14.1 min of GPU
  wall for the four job files. Record:
  [GPU queue results](#sep-18-gpu-queue-results-wilor-arm-kineo-ego-exo).

#### Sep 19-20: human anchors, the resolution and DAM4SAM follow-up, ensemble v2

The human labelled 52 review anchors on 13 C10379 frames (Sep 19), and `battle-anchor-iou`
replaced self-consistency as the yardstick. The follow-up plan then ran through the queue
(19 arms scored) and closed with a named candidate. The review guide is
[`docs/review-guide-2026-09-20-ensemble-v2.md`](review-guide-2026-09-20-ensemble-v2.md).

- Anchors (`30be958`): 51 masks, one hidden mark (rear_body 1700, a screwdriver every arm
  latches onto, `distractor_confusion`). Record: [anchors](#sep-18-human-review-anchors-labeling-run-prepared-not-labelled-sep-18-labelled-sep-19).
- Resolution (`072fe0b`, `024dbf9`, `c0cc32b`): the label-free proxies had rejected 720 px
  wrongly; 1280 is best on the anchors (0.724-0.728), 1920 saturates (0.708); all eight views
  rerun at 1280. Record: [1280 on all views](#sep-19-sam3-at-1280-px-on-all-eight-views-and-1920-px-on-c10379-steps-0-1-and-1c-of-the-follow-up-plan).
- Queue gap (`7380a9e`): a dead worker now exits 3 and stops the queue. Record:
  [queue gap](#sep-19-queue-gap-closed-a-dead-worker-no-longer-counts-as-a-succeeded-job).
- Memory arms (`68d0b49`, `49b6159`): appending corrections to the prompt bank is the one
  change that helps (`pm-append` 0.743, +0.019); drop-900 refuted; frame memory 6/8 no gain.
  Record: [memory arms](#sep-19-sam3-correction-memory-arms-on-c10379-at-1280-plan-step-1b).
- DAM4SAM, fairly (`340f1a5`, `6eed418`, `8ee1b9f`): shared predictor, large checkpoint, the
  SAM3 schedule; large+sched ties SAM3-1280 (0.715); 1536 VRAM-gated (14.79 GiB projected);
  SAMURAI+schedule unsupported upstream. Record: [DAM4SAM arms](#sep-19-dam4sam-and-samurai-arms-under-the-sam3-run-conditions-plan-step-2).
- Scoreboard (`d4bb226`): 19 arms x 52 cells in one table. Record:
  [scoreboard](#sep-19-anchor-scoreboard-over-every-first-minute-arm-plan-step-3-scoring-half).
- Ensemble v2, the candidate (`c02a2d8`, `a8362a8` and the docs commit after them): arms by anchors,
  fallback intervals label-free from the consensus rebuilt on `pm-append`, hidden interior
  withdrawn, 1700 distractor interval; v5 review package and review metrics. Adoption is the
  human's disposition. Record:
  [ensemble v2](#sep-20-ensemble-reference-v2-the-named-candidate-plan-step-3-deciding-half-closes-the-resolution-and-dam4sam-follow-up-plan).

#### Sep 20: the multicam block (detect failure, remove the human, prove it generalizes)

One autonomous day against two goals; every number is cross-view disagreement or IoU against
the 52 anchors on one view. Closing record, with the goals answered in plain terms:
[closing](#closing-the-multicam-plan-the-two-goals-answered-with-numbers).

- Detector scorecard (Track A): SAM3's object score is the best failure detector (AUROC
  0.91-0.96 in-sample); leave-one-frame-out the combined detector's P/R is 0.36-0.62 /
  0.44-0.64; consensus and hull undefined on a third of the cells; nothing scored on the
  279-408 leak. Record: [scorecard](#sep-20-detector-scorecard-against-the-52-review-anchors-track-a-of-the-multicam-plan).
- Seeds (B0-B3): rear_body / cabin transfer automatically (0.80 / 0.95 held-out), chassis /
  interior do not (0.53 / 0.47); eight views rerun with the accepted seeds. Records:
  [B0](#sep-20-b0-tooling-of-the-multicam-plan-anchors-mapped-to-every-view-commit-6c6cd66),
  [B2](#sep-20-b2-of-the-multicam-plan-seeding-strategy-search-on-the-c10379-human-masks-and-transfer-commits-2861d2f-df1edff-afb8475),
  [B3](#sep-20-b3-of-the-multicam-plan-eight-views-rerun-at-1280--append-with-the-search-seeds).
- Corrections (B4): consensus-only 0.590 / 0.663 vs 0.743 human vs 0.591 seed-only floor; the
  accepted corrections were hand+chassis blobs; an acceptance-rule search on the frames with
  human truth found no rule that meets the bar (chassis held-out 0.470, harm 0.20; oracle
  0.596), and one fixed improvement (rank by decoder score). Verdict: multicam is a detector,
  not yet a corrector. Records: [tool](#sep-20-battle-multiview-reprompt-the-consensus-re-prompt-loop-b4-of-the-multicam-plan-tool-built-and-unit-tested-gpu-arms-not-run),
  [arms](#sep-20-b4-arms-on-c10379-the-consensus-re-prompt-loop-run-multicam-plan-headline-commits-78f2eca-21da378-cf8007e),
  [C10119](#sep-20-the-multiview-profile-takes-agent-corrections-c10119-as-re-prompt-target-plan-b4-second-target-commits-78f2eca-cf8007e),
  [acceptance search](#sep-20-acceptance-rule-search-for-the-consensus-corrections-and-the-multicam-plan-closed-commits-12f48e9-833343d).
- Recording 2 (Track C): fetched, calibrated, seeded and run with zero human input for
  rear_body and cabin; seven views agree on 3.3-3.6 of 7 per frame; unscored until labelled.
  Records: [C1](#sep-20-track-c1-of-the-multicam-plan-a-second-recording-fetched-and-prepared-nusar_9061),
  [C2/C3](#sep-20-track-c2c3-recording-2-nusar_9061-seeded-and-run-with-zero-human-input-commits-78f2eca-21da378-cf8007e).
- Viewer (Track D): one v6 recording with three presets. Record:
  [v6](#sep-20-v6-review-surface-one-recording-with-three-blueprint-presets-track-d-of-the-multicam-plan).

#### Sep 21: FineBio first look

FineBio access granted and the archives extracted (non-commercial research licence). One
20-second first-person trim, five text prompts, SAM3 zero-shot detect-then-track, 2 min 20 s
of GPU: three prompts detected at frame 0, the transparent 6-well plate and the tube racks
found nothing. A toolchain smoke with no accuracy claim. Record:
[FineBio first look](#sep-21-finebio-first-look-sam3-zero-shot-text-prompts-on-one-first-person-clip).

#### Sep 21, night: the labelling sessions acted on

The human labelled C10119 (63 cells), e4 (52 + 23 hidden) and decided the 24 seed proposals
(chassis 7/8 and rear_body 8/8 accepted, interior 1/8). Acted on in one pass: the accepted
candidates became seeds; the interior was seeded at frame 0 on five statics from two human
masks (C10379 f0 x C10119 f41; hand-luminance rule needed; e4 blocked, the part is behind its
camera); eight views rerun with four parts; the C10119 rear_body correction that raised
agreement to 0.997 scores **0.00 on six labelled frames** (screwdriver; the decoder rated it
0.95), so a `--distractor-guard` reading the human `hidden` marks now blocks such onsets; the
four-part consensus proposes interior corrections on C10379 but chassis-sized ones
(consensus-only-ds interior 0.288 -> 0.363, all 0.659 unchanged vs 0.743 human); the fourth slot
destabilises the chassis on C10119 (0.758 -> 0.339) and the guarded re-prompt repairs it (0.667)
while tracking the interior on that camera for the first time (0.74-0.87 over 1651-1771).
Record: [Sep 21, night](#sep-21-night-the-three-labelling-sessions-acted-on-human-accepted-seeds-an-interior-seed-from-two-human-masks-the-four-part-rerun-the-c10119-rear_body-finding-and-a-distractor-guard).

#### Sep 21: FineBio shipped detector, first run

FineBio's own MMDetection detector (DINO and Deformable DETR, 35 classes, the authors'
checkpoints) run on the CPU over the same 600 frames as the SAM3 smoke and put in one Rerun
recording beside it. The transparent plate is `cell_culture_plate` on all 600 frames and the
racks get boxes under six rack classes; 3-4 pipettes per frame are boxed separately, the
right-hand box swallows the upright pipette, nothing fires on the fiducials. Qualitative, no
annotations here, no accuracy claim. Record:
[FineBio shipped detector](#sep-21-finebio-shipped-detector-first-run-mmdetection-dino-and-deformable-detr-on-the-cpu-beside-the-sam3-smoke).

#### Sep 22: exemplar detectors and three dedup passes

SAM3's own memory-free signals (mask-pooled backbone embeddings, the visual-exemplar
detector) scored as failure detectors on three cameras and used as a correction candidate
pool: the one pool to clear the bar (C10119 chassis, cross-view exemplars, 0.740 on the
anchors). Three deduplication passes over `src/battle` and `scripts` (-1675 lines inside the
pre-existing modules, tests 646 -> 762), each proven equivalent on three CPU artifacts. Records:
[Track A](#sep-22-sam3-appearance-detectors-on-three-cameras-track-a-of-the-exemplar-plan-commits-0ef1868-fb426c2-d914bef-d5298a8-7f608e5-329eea1-bc794a1),
[Track B](#sep-22-exemplar-detections-as-correction-candidates-track-b-of-the-exemplar-plan-the-c10119-chassis-arm),
[dedup summary](#sep-22-dedup-passes-1-3-summary).

#### Sep 24: the Assembly101 phase closed

Decision: close now; nothing else runs on Assembly101 unless a FineBio result sends us
back. No labeling, no GPU, no run re-scored. Dispositions recorded as they stand (ensemble v2
a candidate with no adoption decision recorded; recording 2 unscored; interior seeds
unconfirmed; the resolution plan's E2-E9 and the exemplar plan's remaining items not run), the
goals table above finalized, one table of final numbers, the `runs/` archive list written
(nothing deleted), the tag `assembly101-lab-close` placed, and the next phase written down as
[`docs/plan-2026-09-24-finebio-detector-seeded-lab.md`](plan-2026-09-24-finebio-detector-seeded-lab.md)
(planned, not started). Record: [Sep 24: Assembly101 phase closed](#sep-24-assembly101-phase-closed).

#### Sep 24, evening: cleanup pass

The user approved the deletion and a small tidy-up. Deleted from `runs/`: 72 of the 76
unreferenced runs (four kept because a surviving run's provenance names them), the whole
`runs/dedup-equivalence/` and `runs/dedup-pass2-smokes-20260922/` roots, every `code-snapshot-*`
directory (22; all `git archive`s of commits still in history), the `blocked-by-gpu-guard/` and
`failed-worker-edit-*` sub-directories: 3.67 GB, `runs/` 60G -> 56G. In the repository: fourteen
dead constants, two example YAML configs, five one-off scripts moved to `scripts/archive/`,
`configs/methods.yaml` moved to `docs/methods-inventory-2026-09.md`, the Tailscale serve script
pointed at the v6 package with its segmentation preset, superseded banners on the Sep 18 and
ensemble-v2 review guides, `battle_plan.agent.final.md` untracked. No GPU, no run re-scored.
Record: [Sep 24: cleanup pass](#sep-24-cleanup-pass).

#### Sep 24, night: FineBio preflight, the plan's assumptions checked before it starts

The user asked for every assumption behind the FineBio 3D-tracking plan to be challenged
first. Checked on the data, CPU plus about five GPU minutes: `T1..T5` are cameras `1,2,3,4,6`
in order and P03 is day 221013 (ArUco markers vs the shipped extrinsics, 2-7 px on cameras 1-4);
the top-down camera 6 is 6.4 cm / 94 px off its shipped pose and is re-solved from the markers;
the shipped fpv pose is a marker PnP good to 0.9 px on 97-99% of frames; the six videos are
synchronised to +/-1 frame; five-view triangulation of static objects returns their half
heights in centimetres with a leave-one-out residual of 10 px median / 25 px p90; the plate
triangulated from the fixed cameras lands inside the fpv detector box on 45/45 frames. SAM3
given the FineBio detector's box as its only prompt segments the transparent plate in all six
views (bbox IoU 0.77-0.94) and holds it, the pipette, the centrifuge and a tube for 300 frames
in four views with one hand occlusion re-acquired; encoder side 1920 adds nothing; objects under
~30 px (single micro tubes) are marginal. The in-hand object is the weakest detection in every
fixed view (0.44-0.57), so seeding by score persistence picks the bench, not the hand; protocol
01 barely moves anything and protocol 03 (`P03_03_01`, same rig, six centrifuge cycles,
six-view annotated frame 916) is the trial to use. Eleven plan modifications proposed, none
applied. Record: [`docs/preflight-2026-09-24-finebio.md`](preflight-2026-09-24-finebio.md);
outputs `runs/preflight-finebio-20260924/` (gitignored) including `preflight.rrd`.

#### Sep 24, night: FineBio contracts and fixtures (p0-contracts)

The data contracts the FineBio lanes split on: observation, camera-config, track and event
schemas in `multiview_schemas.py`; the preflight's camera library moved into
`battle.finebio_cameras` with the `mapping` and `rig` outputs reproduced byte for byte; the
P03_01_01 camera config committed (`configs/finebio/cameras/P03_01_01.json`); the preflight's
detections and SAM3 series converted into 20,209 observation rows under
`tests/fixtures/finebio_preflight/` (4.5 MB, numbers only) that re-triangulate the eleven
static objects and the plate hand-off to the preflight's numbers; the frame-index contract as a
`real_data` test (proxy frame k == raw frame start+k at 0.7-0.9 grey levels against 1.0-5.2 one
frame off, pose length == raw frame count on two trials, markers on the proxy at 0.7-1.0 px). CPU
only, no GPU, no viewer. Record: [Sep 24, night: FineBio contracts and fixtures
(p0-contracts)](#sep-24-night-finebio-contracts-and-fixtures-p0-contracts).

#### Sep 24, night: FineBio detector, CUDA build and `battle-finebio-detect` (p2-detect tooling)

The plan's first GPU-lane item, time-boxed to an hour: the detector venv got a CUDA build in
nine minutes of wall clock (`/home/nick/src/finebio-detector/.venv-cuda`: torch 2.13.0+cu130,
mmcv 2.1.0 compiled from the PyPI sdist for sm_120, mmdet 3.3.0; no mmcv source patch, the one
fix is `CC=gcc-15 CXX=g++-15` because CUDA 13.2's nvcc refuses GCC 16), so every frame of the
trial window can be detected: FineBio DINO at **51-59 ms/frame** on the RTX 5070 Ti, peak
reserved **925 MB**, Deformable DETR 42 ms. GPU boxes agree with the CPU preflight to 0.2 px /
0.009 in score once cuDNN's TF32 is off (the driver's default; with it, scores move by up to
0.03). `battle-finebio-detect` (`src/battle/finebio_detect.py`) replaces the two Sep 21/24
scripts: raw frames by raw index, six views, CPU or CUDA venv chosen by `--device`, strides
with IoU-associated linear interpolation as the flagged CPU fallback, one manifest with the
GPU-guard decision. It reproduces the preflight's T5 rows exactly on the CPU venv and ran the
78-frame preflight set on six views in 32 s on the GPU. A 3600-frame window on six views is
about 25 minutes of GPU. Record: [Sep 24, night: FineBio detector, CUDA build and
battle-finebio-detect (p2-detect tooling)](#sep-24-night-finebio-detector-cuda-build-and-battle-finebio-detect-p2-detect-tooling).

#### Sep 24, night: plan approved, trial windows chosen (p0-trials, p-docs)

The rewritten plan was approved and copied verbatim into
[`docs/plan-2026-09-25-finebio-3d-tracking.md`](plan-2026-09-25-finebio-3d-tracking.md) (its 19
todos as a table, the Sep 24 draft marked superseded). The trials were then chosen on top-down
evidence, CPU only: every 30th frame of the T5 videos through FineBio DINO (centrifuge box,
hand boxes), frame-difference energy at every frame, and the centrifuge lid read at every frame
as the teal fraction of a fixed rotor region of the top-down view (bimodal, threshold 0.24 at the
widest gap, the same state runs at every threshold 0.15-0.35, checked by eye on 36 + 16 1:1
crops). `P03_03_01` has the lid closed for its first 37 s and then **four** spins of 1.7-3.0 s
with a hand on the lid (the preflight's "six" was a loose count); trial 1 is raw frames
**[600, 4200)** (20.0-140.1 s): frame 916 at 10.5 s in, two full cycles ([1176, 1228) and
[3224, 3311)) with loading and unloading, three shipped annotated frames inside, the busiest
120 s that contains 916, fpv pose valid on 3587/3600. `P20_03_01` passes every check (six views
x 6045 frames, pose file of that length, 94.6% valid) and has four spins of 1.1-1.4 s; the
proposed window is the same **[600, 4200)** (frame 1442 at 28.1 s, three cycles, six of seven
annotated frames), with one finding: its fpv pose drops out for 2-3 s around every spin (the
head camera loses the markers over the centrifuge), so in room 2 the fixed cameras carry
containment alone. `P03_01_01` stays the smoke. SOURCES and LICENSES gained the camera-pose
archive and the seven shipped checkpoints with SHA-256s. Record: [Sep 24, night: plan approved,
trial windows chosen (p0-trials, p-docs)](#sep-24-night-plan-approved-trial-windows-chosen-p0-trials-p-docs);
evidence [`docs/qa/finebio-trials-2026-09-25.md`](qa/finebio-trials-2026-09-25.md), config
`configs/finebio/trials.json`.

#### Sep 24, night: SAM3 worker modes for the FineBio arms (p3-worker)

The two worker modes the tracking arms need, built on the preflight's two SAM3 paths. (i) A
memory-free per-frame box decode (`--prompt-mode box_stream`): a JSONL box stream in, the
standard `observations.jsonl` + mask PNGs out, one image encode per frame and all boxes in one
batched decoder call, no video memory; on the preflight's fpv window it reproduces the
preflight's masks (mask-bbox IoU vs the prompt box 0.94 / 0.95 / 0.99 / 0.97 for plate /
pipette / centrifuge / tube) at **152 ms/frame** (149 ms of it the encoder; four decodes 2 ms),
against 220 ms/step for the four-slot tracker, 2.15 GiB. (ii) Video-memory hooks: a tau
memory-write gate (`--memory-write-min-score`, the score test alone, added beside the Sep 18
`gate` rather than reusing it, since that gate bundles IoU, contest and area tests that starved
slots); corrections and seeds that are boxes (`prompt_box`, provenance `detector_reseed` or
`track_reproject`) decoded on the tracker's own image tokens and installed through the existing
correction path; per-slot start frames (a slot allocated empty at frame 0, forced absent until
its seed frame; the multiplex object count stays fixed). A sibling driver `battle-muggled-arms`
(`box-decode`, `video-memory`) writes SHA-256-bound manifests for videos that have no G2
config. 61 default-tier tests, 2 `gpu` smokes run. Record: [Sep 24, night: SAM3 worker modes
for the FineBio arms (p3-worker)](#sep-24-night-sam3-worker-modes-for-the-finebio-arms-p3-worker).

#### Sep 25: thin slice on the preflight window (p0-slice)

The whole chain on the 60-frame preflight window of `P03_01_01` before anything scales:
fixtures -> per-frame same-class triangulation (>= 3 fixed views, or 2 fixed plus a valid fpv
pose) -> 3D points, per-view and leave-one-view-out residuals, the fixed -> fpv hand-off, and
the seven preflight cross-checks as standing entities in one Rerun recording
(`battle-finebio-slice`, 3 s, no viewer). The eleven static classes land 0.00-0.01 cm from the
preflight's points; the plate triangulates on 60/60 frames at 16.9 px with the preflight's LOO
per view (34.9 / 17.1 / 44.5 / 17.2 / 33.7) and projects into the fpv box on 60/60 frames
(36.8 px median). Four multi-instance classes (50 ml tubes, micro tubes, pens, the 50 ml rack)
fail a one-object flag under the top-box-per-class rule, and the preflight's per-view SAM3
tube slots turn out to track different tubes: the identity problem the tracker is for, seen
on real rows. Record: [Sep 25: thin slice on the preflight window
(p0-slice)](#sep-25-thin-slice-on-the-preflight-window-p0-slice).

#### Sep 25: battle-multiview-tracks core (p3-tracker)

The 3D tracker's core, built against the fixtures only: observations (mask centroid or box
centre, class, scores, per-view slot, fpv only with a valid pose), a stationary prior,
predict-project-gate update with weighted re-triangulation, states observed / single_view /
coasting / lost, birth by pairwise same-class triangulation + Hungarian + cliques with >= 3
fixed views or 2 fixed + fpv, re-acquisition only when detector-confirmed in >= 2 views and
unambiguous (two candidates -> nobody resumes, `possibly_same_as`), hand-off re-seed events
with the reprojected box for the worker's `track_reproject` correction, `tracks.jsonl` /
`events.jsonl` / `residuals.jsonl` / `identity_metrics.json`. Gates from lane B's rig
output or the CLI, P03 defaults 30 / 55 px. 12 synthetic-rig tests plus the preflight
fixtures: static classes within 1 cm of the slice, one plate id over 60/60 and 300/300
frames, the two 50 ml tubes the slice collapsed separated at 1.5-4 px; the in-hand pipette
fragments and the identical micro tubes are ambiguous, as the plan expects. Record: [Sep 25:
battle-multiview-tracks core (p3-tracker)](#sep-25-battle-multiview-tracks-core-p3-tracker).

#### Sep 25: battle-finebio-cameras, the per-trial camera solve (p0-cameras)

The preflight's camera mapping as a standing step with a committed config per trial: ArUco on
N frames per fixed view, every (day, camera) shipped pose ranked, the day by summed residual
and a second witness (the shipped fpv pose fits only the marker layout of its own day), one
PnP per camera against the chosen day's markers, shipped pose kept only where it fits within
10 px, else marker PnP, else the camera is dropped. P03_01_01 reproduces the preflight (one
recorded residual corrected: T4 4.78 px on the chosen day, not 2.23 px on another day's best
fit); P03_03_01 is day 221013 with the same rig (camera 6 within 0.14 cm of the smoke's
solve); **P20_03_01 is day 221124 with all four side cameras 15-32 px off their shipped poses
and re-solved to 0.7-2.1 px**, the top-down camera kept shipped at 7.5 px; no camera dropped.
Record: [Sep 25: battle-finebio-cameras, the per-trial camera solve
(p0-cameras)](#sep-25-battle-finebio-cameras-the-per-trial-camera-solve-p0-cameras).

#### Sep 25: battle-finebio-rig and the gate formulas (p1-rig)

The preflight's rig check generalised to any camera config and detector pass, with the
tracker's gates as formulas written into `rig.json`: association = 3x the static
leave-one-out median, hand-off = the widest view's p90 of the tracked objects' held-out
residuals (fixed LOO and the fixed -> fpv hand-off), floor 15 / cap 80 px at 1920, clock offsets
with +/-1 uncertainty, birth from 3 fixed views or 2 + fpv. P03_01_01 gives **31.1 / 51.9 px**
(the preflight's 30 / 50-60) and reproduces every preflight number; the negative control puts
camera 6's shipped pose beside the marker PnP (markers 93.7 vs 0.7 px, static LOO 94 vs 7 px,
left hand 48 vs 8 px). Record: [Sep 25: battle-finebio-rig and the gate formulas
(p1-rig)](#sep-25-battle-finebio-rig-and-the-gate-formulas-p1-rig).

#### Sep 25: finebio_preprocessing and the trial proxies (p1-configs)

Six native-resolution, native-rate proxies per trial window with the p0-contracts recipe
unchanged, a manifest that records the command verbatim, the frame-index contract on three
frames per view and the fpv markers, a per-window camera config with `frame_index_offset =
start`, and a clip config (views, targets, window, proxies, the GPU-phase rig command). Cut:
the smoke window `P03_01_01` 1798-2398, trial 1 `P03_03_01` 600-4200 and trial 2 `P20_03_01`
600-4200 (3600 frames each, about 2 min per trial), every sample's minimum at offset 0, fpv
markers 0.8-1.0 px. The Assembly101 `static-c10379` / four-part asserts in the interaction
review, the exploratory comparison and the Kineo fusion now read a clip config with the
Assembly101 values as defaults. Record: [Sep 25: finebio_preprocessing and the trial proxies
(p1-configs)](#sep-25-finebio_preprocessing-and-the-trial-proxies-p1-configs).

#### Sep 25: battle-detector-seed, observation adapters, gate-1 sheets (p2-seeds, p2-gate1)

Which objects get a SAM3 slot in which view, from the detector alone: per-view instances by
IoU association joined across gaps, singleton classes and rack footprints (T1 1433 fragments
-> 468 instances), a slot for what **moves** (sustained median displacement > 20 px at 1920;
40 px against the head's own motion in the fpv, and only for classes a fixed camera saw move)
or sits **in a hand** (> 10% of frames), the named containers as static volumes, tube groups
per rack, a cap of 10 ordered landmarks -> movers -> groups -> held -> racks. Each seed is the
tight detector box through the SAM3 image decoder, accepted by mask-bbox IoU >= 0.6: trial 1
**60/60 accepted** (0.63-0.99, 24 image encodes, 13 s, 2.4 GiB), per view the three machines,
the four pipettes in use, the tubes that move, micro-tube groups of 29-54; the plate never
moves in protocol 03 and stays detector-only. Worker-ready box streams (arm b) and schedules
(arms c/d) in analysis frames, the two observation adapters, per-view contact sheets and the
15-minute gate-1 brief ([`labeling-sessions-2026-09-25-finebio.md`](labeling-sessions-2026-09-25-finebio.md);
soft: the pipeline runs on `provenance: auto`, decisions filter afterwards). Record: [Sep 25:
battle-detector-seed, observation adapters, gate-1 sheets (p2-seeds, p2-gate1)](#sep-25-battle-detector-seed-observation-adapters-gate-1-sheets-p2-seeds-p2-gate1).

#### Sep 25: detector runs on the trial windows (p2-detect, runs)

Every frame of both trial windows detected in all six views, DINO and Deformable DETR: four
GPU runs of 14-18 min each (86,400 frames in 65 min of wall clock; DINO 46-47 ms/frame, DDETR
38-43, peak 882 / 702 MiB, the card shared with nothing but the compositor at every start),
every manifest `succeeded` / `full`, 3600 rows per view with `frame_index` 600..4199 and no
interpolated row. The two detectors agree on **95.4%** of confident DINO boxes in room 1 and
**90.4%** in room 2 (same class, IoU >= 0.5; 0.99 in the fpv, 0.96-1.00 on the static bench
objects, 0.72-0.79 on the moving pipette, the hands and the centrifuge); the one target-level
disagreement is the trial-2 plate, which DDETR sees in T2 and T4 as a ~100 px far box and DINO
does not. Box-centre displacement over 30-frame steps in the fixed views gives the seeding rule
its input: in trial 1 only the **blue pipette moves** (median 23-37 px, relocates up to 146 px,
in a hand on 52% of frames), the micro and 50 ml tubes are handled without their top instance
moving (in a hand on 44% / 11% of frames), and **the plate is static to 0.6 px and never in a
hand**, so the plan's rule does not seed it, a named-object decision the seeds lane reached
independently. Trial 2 with nothing changed: the yellow pipette and its tip rack move, the
plate is static again, no 15 ml tube on that bench. Record: [Sep 25: detector runs on the trial
windows (p2-detect, runs)](#sep-25-detector-runs-on-the-trial-windows-p2-detect-runs).

#### Sep 25: tracking arms on trial 1 (p4-arms; p1-rig run)

The rig on trial 1's window (association **30.1 px**, hand-off **27.0 px**: the plate does not
move in protocol 03, so the moving-object p90 halves against the preflight's 51.9), the plate
added to the seeds as a plan-driven landmark (66/66 seeds accepted), and four arms through the
3D tracker on six views at 1280 in about 4 h of GPU: (a) boxes only, (b) per-frame box decode
(150 ms/frame, 2.2 GiB), (c) SAM3.1 video memory seeded once (195-217 ms/step for 11 slots,
3.6-4.2 GiB), (d) (c) + 986 tracker-emitted re-seed boxes (250-320 ms/step). Against the same
reference, the best same-class DINO box of the frame, **(b) reads 0.926 median with 99.1% of
masks >= 0.5 IoU, (c) 0.914 with 79.3%, (d) 0.911 with 75.7%**: video memory drifts on the
tubes that leave, the in-hand pipette and the centrifuge whose lid opens (0.45 in T1/T2), and
the re-seeds under `append` injected wrong prompts more often than they repaired (one box took a
97%-clean slot to 0.1%). (c) wins only on identity through appearance change (the centrifuge one
id across the lid cycles, 8-9 ids in (a)/(b)) and on the tracker's proxy metrics (264 vs 629 id
switches), which is why (d) ran under the pre-registered rule and came out negative. **The
memory-free per-frame decode is the mask source for the floor.** The occlusion inventory
(284 / 317 / 247 / 218 support-0 episodes per arm, hand tracks apart) says `contained` would
serve 125-144 episodes, four fifths of them micro tubes under a hand in the micro-tube rack;
`held` 89-166 by the loose test but 5-19 by the strict one, the in-hand pipette's real failure
being the stationary prior at a 30 px gate (22-80 ids before any occlusion); group tracks
145-171 episodes of identical instances, half ending ambiguous. One adapter fix on the way:
video-memory masks carry stray single pixels that set the bbox; the worker's own component rule
is now applied before a mask is measured. Record: [Sep 25: tracking arms on trial 1 (p4-arms;
p1-rig run)](#sep-25-tracking-arms-on-trial-1-p4-arms-p1-rig-run).

#### Sep 25: anchor frames and workspace for gate 2 (p6-anchors, prepared)

Gate 2 prepared, not labelled: `battle-finebio-anchors select` chose the frames from arm (b)
(the six-view annotated frame **916** in all six views; **12 disagreement frames** on the fpv and
**T4**, the fixed view with the most individually tracked slots and the plate visible, where arm
(b)'s mask disagrees most with its own detector box, at least 3 s apart, three of them around the
centrifuge cycles; **5 random frames** with a fixed seed), `workspace` decoded the candidates the
human chooses among rather than draws (c0 = arm (b)'s mask, a margin box, the box plus a point,
the decoder's second-ranked mask; 440 cells, 337 with candidates, 33 s of GPU beside the trial-2
worker), and `score` runs on the empty record today (0 / 440 labelled) and will re-run as labels
land: mask and box IoU per arm and cell, hidden false positives, and identity F1 between the
human's instance names and the tracker's ids on 916 and pooled. The Assembly101 web workspace was
not generalised; gate 2 uses static sheets and a decisions JSON like gate 1. Record: [Sep 25:
anchor frames and workspace for gate 2 (p6-anchors,
prepared)](#sep-25-anchor-frames-and-workspace-for-gate-2-p6-anchors-prepared).

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
- Multi-view was reached, versus the plan's "Kineo as a final two-hour trial" and an ATHENA
  trial time-boxed to format conversion on "shipped intrinsics" the archive turned out not
  to have. It came through a door the plan did not draw: eight static views acquired for
  the focused window, the dataset's own extrinsics, and intrinsics fitted to the dataset's
  own 2D/3D projection, so every multi-view number is measured against dataset context
  rather than a calibration of our own. Kineo used 8.4 min of its two hours; LM-EEC (the
  brief's deferred ego-exo correspondence line) was un-deferred at the user's request and
  ran inside its 90 min box. Seeds on the new views are agent-authored and labelled so,
  per the overnight plan's ground rule; no human has reviewed a mask on any of them.
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
  demo use; it now covers the poses, annotations and the seven newly fetched views as well.
- Dispositions on the 13 consensus-proposed `not_contact_eligible` intervals for C10379
  (`runs/multiview-part-consensus-first-minute/manifest.json`,
  `proposed_validity_intervals`, all `applied: false`): chassis 585-604, 643-658, 665-689,
  697-702, 1089-1163, 1167-1172, 1464-1471; rear_body 1525-1530, 1677-1751, 1759-1781,
  1786-1800. The 37 hull proposals in
  `runs/multiview-visual-hull-first-minute/manifest.json` are the noisier second signal.
- Dispositions on the agent-authored frame-0 seeds of the seven new static views and e4
  (`runs/multiview-seed-transfer-20260918/<view>/seed_manifest.json`, `selected_by: agent`):
  no human has looked at a mask on any of those eight views, so every consensus and hull
  number rests on them.
- Whether the consensus/hull disagreement triggers should enter `contact_eligible` at all
  (as `SegmentationValidityInterval` rows with provenance `human_feedback_report`), or stay
  review triggers beside the ensemble reference.
- LM-EEC checkpoint direction: `ExoEgo_checkpoint.pt` was used for exo->ego on its file
  name alone; the other checkpoint is one `--checkpoint` swap away in `pairs.json`.
- Whether to seed e1 (HMC_21176875) mid-minute: every part is outside its frame at frame 0,
  so a run needs a profile that starts at the first frame the cabin is in view.

Technical:

- Static/ego clock offset: fixed for the two rebuilt two-view recordings
  (`runs/*-offset-v2/`) and applied per view everywhere in the multi-camera pass; the
  offsets are per-camera constants measured on the focused window only, with the half
  analysis frame left unresampled (16.7 ms residual in the manifests).
- Interior was never seeded on any view other than C10379 (its two-clock triangulation is
  invalid because it is in the hand at frame 0), so it has no consensus, no hull and no
  ego-exo hull reference; a same-clock second human view, or seeding at the first frame it
  rests on the table, would fix it.
- Kineo self-calibration: cameras land ~37 % closer to the subject with 4-11 deg of
  compensating rotation (one subject in a ~1 m volume at 2-3 m range); the camera
  comparison is the honest result of that arm and its wrist numbers under the camera
  alignment are dominated by it.
- WiLoR wrist substitution (triangulated wrist + WiLoR articulation as a hand series) is
  measured, not built; per-view weights for the ego camera in triangulation; the thumb
  correspondence.
- Human-review nits not yet addressed: SAM mask palette (orange on yellow), skeleton lines
  for Kineo body joints.
- Disk: closed Sep 24 evening. Of the 76 unreferenced run directories listed in
  [`docs/qa/runs-archive-list-2026-09-24.md`](qa/runs-archive-list-2026-09-24.md), 72 were
  deleted with the dedup roots, the code snapshots and the two failed sub-directories (3.67 GB;
  the record is that file's "Deleted Sep 24" section); four stay because a surviving run's
  provenance names them ([cleanup pass](#sep-24-cleanup-pass)). `scripts/prune_runs.py` still
  lists and never deletes; note that the archive list is itself tracked and names every run, so
  the script now counts them as cited.
- `yellow_toy_top` leaves the frame at ~216.2 s in every 180 s arm; its coverage numbers
  describe the scene, not the tracker.
- FineBio: entered Sep 21 (SAM3 smoke, shipped detector); the next phase is planned in
  [`docs/plan-2026-09-24-finebio-detector-seeded-lab.md`](plan-2026-09-24-finebio-detector-seeded-lab.md).
  Audio: never entered, closed with the phase (Sep 24).

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
  `mediapipe_fallback` audit-sheet labeling bug in `scripts/render_overnight_v3_audits.py`
  (since Sep 24 `scripts/archive/render_overnight_v3_audits.py`).
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

### Sep 18: Track 4 preparation, Kineo on all eight static views (CPU only, GPU jobs queued)

- **Claim boundary first.** The self-calibration arm compares Kineo's estimated extrinsics with
  the dataset's shipped extrinsics after a similarity alignment of camera centres: a calibration
  comparison against dataset context, not pose accuracy, and the only place in the pass where the
  dataset extrinsics act as a reference for a method. Wrist numbers in both arms compare Kineo's
  SMPL-X body wrists with the dataset hand-tracker wrists: cross-source disagreement between two
  estimates. Intrinsics are fitted estimates. Ego cameras are excluded (Kineo assumes static
  cameras). Kineo research/evaluation-only, Assembly101 CC BY-NC 4.0. Tonight: CPU only, the
  GPU was never touched (every pixi check ran with `CUDA_VISIBLE_DEVICES=""`); no model ran; no
  viewer opened.
- **What exists now (`battle-kineo-multiview`, module `kineo_multiview.py`, review
  `kineo_multiview_review.py`, runner `scripts/kineo_multiview_runner.py`).** `prepare --arm
  selfcal|known` wrote `runs/kineo-multiview-{selfcal,known}-first-minute-20260918/` in 27-29 s
  each: eight 1,800-frame trims (`trims/<view>_first_minute_1800f.mp4`, libx264 crf 18, frame
  counts verified with `ffprobe -count_frames`), the generated Kineo YAML, the dataset cameras
  as Kineo annotation PKLs (+ JSON twins), `queue_job.json`, `prepare.json`, and the queue
  files `runs/overnight-multicam-20260918/jobs_t4_kineo_{known,selfcal}.json` (1800 s / 3600 s
  timeouts, 5400 s together inside the plan's 2 h box; `--dry-run` clean). `validate` then ran
  the runner's `--validate-only` path under pixi: both YAMLs load and resolve, all 21 (selfcal)
  / 7 (known) stage classes import, every runtime-config dataclass instantiates, constructor
  signatures and model paths check, 14 (selfcal) / 5 (known) model-free stages were fully
  instantiated, the GT PKLs load through `CameraIntrinsicsAnnotations.from_dict` /
  `CameraExtrinsicsAnnotations.from_dict` (`brown_conrady`, `resolution_hw [720, 1280]`, eight
  view ids), and cv2 reports 1,800 frames for every trim. Not instantiated in the dry run:
  rtmlib (its `YOLOX(..., device="cuda")` opens an ONNX Runtime CUDA session in `__init__`), MoGe,
  NLF detection/fitting, background subtraction and MoGe scene reconstruction, all of which load
  models in their constructors; signatures and file paths only.
- **Decisions.** (1) *Offsets by trimming, not `camera_temporal`.* The reference is C10379
  (latest-starting camera, +9 pose frames); every other view's trim starts at
  `floor((17649 - view.pose_frame(0)) / 2 + 0.5)` proxy frames: C10095 2 (residual 0), C10115 2
  (-1 pose frame), C10118 2 (-1), C10119 1 (0), C10390 1 (0), C10395 2 (-1), C10404 2 (-1), so
  trimmed frame q shows pose frame 17649 + 2q in every view to within 16.7 ms, the same rule and
  residual Track 3b used, and Kineo frame q == analysis frame q of the dataset reference window
  (`hands.jsonl`). The measured offsets say the earlier-started cameras must *skip* frames; a
  formula shifting the later-started ones would have doubled the error. Kineo's
  `global_time_resampling` supports a `camera_temporal` annotation (seconds per view, then a
  50 Hz resample with linear interpolation of 2D keypoints), but only through `annotations`, its
  resampled global timeline would no longer index the dataset frames, and the identity path
  (zero offsets, identical frame counts) is the one its benchmarks exercise; the trim leaves the
  residual explicit in `prepare.json` instead. (2) *Metres.* The known-arm extrinsics are the
  inverted dataset `camera_to_world` (nearest proper rotation first; the JSON rotations are
  orthonormal to ~1e-7 and Kineo inverts by transposition) with `t / 1000`, so Kineo's world is
  the dataset world in metres and its metre-scaled defaults (SMPL, Rerun radii, z clipping)
  apply; `evaluate` multiplies back by 1000 with identity rotation. (3) *GT in both arms.* The
  same PKLs go to `gt_annotations` in the self-calibration arm, where no estimation stage reads
  them (only `rerun_export`, which similarity-aligns its own recording and logs the dataset
  cameras); the exported `camera_extrinsics.pkl` stays Kineo's unaligned estimate and battle does
  its own Umeyama. (4) *Known arm stage list*: `transfer_gt_annotations` (order 0) then rtmlib,
  NLF, time resampling, MVS triangulation, Rerun (`log_pred_smpl`/world reconstruction off),
  annotations export; MoGe intrinsics, pairs sampling, SfM, BA sampling, BA 1-3, SMPL scale,
  scale application, reorientation, BA history, SMPL fitting, background subtraction, scene
  reconstruction and BVH removed. Feasible without editing Kineo: the stock CLI's only obstacle is
  `gt_annotations={}` (`kineo/demo/offline/demo.py:72`), which the battle runner replaces.
- **Evaluate / rerun (CPU, ready).** `evaluate <run_dir>` reads the exported PKLs through a
  restricted unpickler (builtins + a stand-in for `CameraDistortionModel`, everything else
  refused), Umeyama-aligns Kineo camera centres to the dataset's (selfcal) or applies the fixed
  1000 (known), reports scale, camera-centre RMS, per-camera rotation/translation error, the
  wrist disagreement per side (median/p90/mean, swapped-side-closer fraction) against
  `runs/assembly101-reference-first-minute-v1/hands.jsonl` (confidence >= 0.5), Kineo's own
  `global_scale`, its stage timings, and the five measures (coverage = frames with body 3D /
  both wrists; time to first 3D output = cumulative stage time through MVS triangulation;
  runtime = queue `job_end.duration_s`; VRAM = torch `max_memory_reserved` from the runner's
  `kineo_runtime.json`, which excludes ONNX Runtime; ID resets n/a with one `best_bbox_only`
  subject). Output `manifest.json` (typed) + `body_aligned_mm.npz`; `rerun <run_dir>` writes
  `comparison.rrd` (dataset frusta as in `_log_assembly101_static`, aligned Kineo frusta with
  magenta labels, Kineo 55-joint body in green, dataset hands, wrist series). Both were exercised
  end to end on a synthetic Kineo export built from the real rig and the real dataset wrists
  under a random similarity (scale 1000): recovered scale 1000.000, camera errors < 1e-3 deg /
  mm, wrist medians < 1e-6 mm, 1,800/1,800 frames, RRD written.
- **Tests (`tests/test_kineo_multiview.py`, 18 default + 3 `real_data`).** Intrinsics
  scaling, world-to-camera inversion against `inv(C2W)`, Umeyama recovering a known similarity
  (with and without scale, refusing < 3 points), geodesic angle, start frames on a four-camera
  fixture (+5/+6/+7/+9 -> 2/2/1/0 with the half-frame residual), GT annotation dicts (proxy
  scale, metres, brown_conrady, ego refusal) round-tripping through the unpickler, selfcal and
  known YAML generation from a stock-shaped fixture (SAM2 removed, rtmlib inserted, stage
  order, transfer at order 0, export paths, missing-stage refusal, YAML round trip, stock not
  mutated), synthetic PKLs matching Kineo's `to_dict` field names (enum pickled under Kineo's
  module path) parsed to arrays with the 1024 vertices dropped, unpickler refusal, alignment
  recovering scale/rotation with zero errors and flagging a perturbed camera, the known arm's
  fixed scale, wrist statistics with side swaps and absences, queue job validating against
  `QueueSpec`, runner view parsing and shared model-free list, queue-log record selection;
  `real_data`: both prepared run directories (start frames, 1,800 verified frames, YAML, git
  head, queue spec, validation report, GT centres round trip to the rig within 1e-6 mm) and the
  synthetic end-to-end evaluate + rerun above. Suite: 403 default tests pass; the two
  `real_data` failures in the tree belong to the concurrent multiview worker's in-flight v4
  rebuild and seed-transfer manifests, not to this track.
- **Not run.** Both GPU jobs (waiting for the queue owner). The evaluation numbers, the five
  measures and the native `.rrd`/`.bvh` fingerprints therefore do not exist yet; when the jobs
  finish: `uv run battle-kineo-multiview evaluate runs/kineo-multiview-<arm>-first-minute-20260918`
  then `rerun`. Known risks for the night: MoGe loads `Ruicheng/moge-2-vitl` from the HF cache
  (the Sep 17 NLF smoke used the same id); the 8-view SfM/BA on 1,800 frames is the part most
  likely to approach the 3600 s box; `background_subtraction`/`moge_scene_reconstruction` are
  kept for the stock stage list and could be dropped from the YAML if the box is tight.

### Sep 18: Track 2 of the overnight multicam pass, cross-view SAM3 with a geometric combiner

- **Claim boundary first.** Every seed on a view other than C10379 was chosen by an agent from
  geometry (`selected_by: agent`, provenance `geometric_seed_transfer`); no human reviewed a mask
  on the seven new static views or on e4. The consensus, agreement scores and disagreement
  episodes below measure how far runs of one tracker disagree with each other across calibrated
  views; they are not accuracy. The human-corrected C10379 masks stay the reference and were not
  modified; the `not_contact_eligible` intervals emitted here are *proposals* for human review.
  GPU work ran only through `scripts/overnight_queue.py` (five queue starts tonight, 16 jobs,
  16 succeeded, no timeout, no `NVRM`/`Xid` line). CC BY-NC 4.0 applies to the dataset assets.
- **Seed transfer: the table plane does not work from C10379, so the seeds are triangulated.**
  The plan lifted the human C10379 frame-0 mask centroids onto the fitted table plane. Rendering
  frame 0 of all eight views showed why that cannot work here: C10379 (and C10395) sit at table
  height (`y` = -1 mm and +1 mm in a world whose table plane is `y` ~ 0, normal
  (-0.17, 0.985, 0.01)) and see the parts at a grazing angle, so a mask centroid a few
  centimetres above the table sends its ray upward past the plane (the chassis ray hit the plane
  63 mm from the camera; the interior, held at the face, never does). `battle-multiview-seed-transfer`
  (new module `multiview_seed_transfer.py`, typed in `multiview_schemas.py`) therefore
  *triangulates* each part's frame-0 centroid from the two views that carry human frame-0
  masks, C10379 and the ego e3 run of Sep 16 (`HMC_21110305`, its own pose frame 17640; the two
  views are 9 pose frames = 0.15 s apart, irrelevant for parts resting on the table): chassis
  (-79, 66, -84) mm, 74 mm above the table, C10379/e3 reprojection 6.5/2.9 px; rear_body
  (-28, -8, 44), -6 mm, 9.2/3.3 px; cabin (-172, 6, -296), 30 mm, 0.1/0.0 px; interior
  (-121, 273, -20), 286 mm above the table, 28/39 px (hand-held and moving between the two
  clocks). Each part becomes a sphere of the masks' equivalent-circle radius at depth (57, 31,
  41, 29 mm), projected into every view as two box prompts (margins 0.25 and 0.60) plus
  background points at the other parts' projected centroids; hands get one box from the 21
  projected dataset joints with the wrist as a foreground point. Decoding reuses the calibration
  workspace's isolated image-decoder worker (`batch_decode`, 9 prompts x 4 masks per static
  view) rather than `decode-batch`, because that tool and the calibration schema deliberately
  forbid agent-selected frame-0 seeds and only know three view ids; both invariants were kept.
- **Acceptance.** A candidate passes when its area is within [0.3, 3.0] x the e3 human mask
  area carried by the squared focal/depth ratio, and either its warp through a table-parallel
  plane at the part's height into e3 reaches IoU >= 0.25 with the e3 human mask, or the ray
  through its own centroid passes within 1.5 radii of the triangulated point. The second rule
  was added after the first pass: for the two low cameras (C10118, C10395) the plane warp is as
  ill-conditioned as it is for C10379 and every candidate scored IoU 0.10-0.17 while its
  centroid ray passed 2-11 mm from the triangulated point and its area ratio was 0.8-1.3.
  Result, all agent decisions, in `runs/multiview-seed-transfer-20260918/<view>/seed_manifest.json`
  with every rejected alternative: chassis/rear_body/cabin accepted on all 7 new static views
  (plane-warp IoU 0.45-0.80 and centroid ray 1-11 mm on C10095, C10115, C10119, C10390,
  C10404; centroid-ray basis on C10118 and C10395); **interior blocked on every view**
  (`seed_transfer_failed`: its two-clock triangulation is invalid because it is in the hand and
  moving, so the expected area is off by 20-190x); left hand accepted everywhere (>= 60 % of
  projected joints inside), right hand below the 0.5 confidence floor at frame 0. Decode 5-6 s
  per view after a ~4 s model load (queue duration 9-10 s per view), CPU acceptance 3-4 s. Seven
  views run with three parts each; none skipped.
- **Runs.** `battle-muggled-smoke --four-part-multiview-first-minute --geometric-seed-manifest`
  is a new profile: any non-C10379 static view of the all-static config (or e4), analysis frames
  `[0, 1800)` only, the accepted seeds as `manual_seeds` with `selected_by: agent`, no later
  corrections, `FourPartMultiviewRunMetadata` in the manifest (seed fingerprints, IoU, blocked
  targets), the worker's confidence sentinel reworded to say the seed is agent-selected. Seven
  runs (`runs/muggledsam-sam3-four-part-multiview-first-minute-static-<view>-20260918t05*`),
  720 px, checkpoints every 300 frames, 3 concepts each. Measures per view: coverage 1800/1800
  frames with 5,394 masks (3 x 1,798); runtime 117-120 s worker (queue duration 158-163 s with
  export and QA); time to first usable output 4.0-4.2 s; peak VRAM 2.10 GB on every view;
  part presence 0.98-1.00, first frame where a part is missing: C10095 chassis 1537 / rear_body
  1526, C10115 chassis 234 / rear_body 1739, C10390 chassis 1642 / rear_body 1525, C10395
  chassis 594 / cabin 228, none on C10118, C10119, C10404 (a missing frame is a dropped slot,
  not a judged failure).
- **Combiner** (`battle-build-multiview-part-consensus`, `multiview_consensus.py`,
  `runs/multiview-part-consensus-first-minute/`, 79 s, static views only). Per frame and part
  the mask centroids (raw px) of every static view with the part are triangulated on the rig
  (DLT, drop-worst filter at 30 raw px, >= 2 views); a view's error is 0 inside its mask, else
  the pixel distance to the mask; agreement = share of frames with error <= 40 raw px; an
  episode is >= 5 consecutive frames over 40 px; it contradicts the majority when the view was
  dropped from a consensus formed by >= 2 other views. Consensus exists on 1800/1800 frames for
  chassis (mean 6.45 views), rear_body (7.07) and cabin (7.14), never for interior (blocked
  everywhere but C10379). Agreement: C10379 0.88 / 0.93 / 1.00 (chassis / rear_body / cabin);
  C10095 0.96 / 0.92 / 1.00; C10115 0.98 / 0.97 / 1.00; C10118 0.99 / 0.92 / 1.00; C10119
  0.99 / 0.94 / 1.00; C10390 0.87 / 0.91 / 0.98; C10395 0.73 / 0.81 / 0.73 (the other
  grazing camera; 30 of the 83 episodes are its); C10404 0.99 / 0.94 / 1.00. **C10379 is
  contradicted by the majority** on chassis `[585,604)`, `[643,658)`, `[665,689)`, `[697,702)`
  (inside the human-reported 573-722 failure), `[1089,1163)` and `[1167,1172)` (the human-reported
  1020-1172 chassis/interior swap, ending exactly at the agent correction frame 1172), and
  `[1464,1471)`; on rear_body `[1525,1530)`, `[1677,1751)`, `[1759,1781)`, `[1786,1800)` (the
  retained late-degradation note: the rear-body label stays on the table piece). The
  human-reported 279-408 window is *not* contradicted: the other views agree with C10379 there,
  so whatever the human saw in that window is not a centroid displacement the static cameras
  can resolve. Thirteen typed `multiview_disagreement` proposals (`not_contact_eligible`,
  `applied: false`) sit in the manifest; nothing was substituted. Dataset-wrist anchor: the
  shipped per-view 2D triangulated at one pose frame recovers the shipped 3D to median 0.0004
  mm (p95 0.0004, 656 points, rig check); with every view read on its own clock rule the median
  is 2.9 mm, p95 13 mm, max 350 mm, which is hand motion over the up-to-4-pose-frame (67 ms)
  inter-camera spread, not rig error, and is the size of the clock-skew effect the mask consensus
  lives with.
- **Review surfaces.** (a) `battle-build-multiview-static-comparison` (`multiview_review.py`)
  writes `runs/multiview-static-comparison-first-minute/multiview_static_comparison.rrd`: eight
  2D views (one per static proxy, part masks as RGBA cut-outs, the consensus point reprojected
  as a marker labelled with that view's pixel error), the world-mm 3D view with consensus
  centroids, dataset hands, all eight frusta and, when the Track 5 run exists, the hull voxels
  at 1 fps and hull projections as a toggleable overlay per view, a per-part time panel of
  every view's error, and the episode document. 282 MB with masks on every frame (43 s);
  rebuilt with `--mask-every 2`. (b) v4 layer `assembly101_multiview` (on by default, gated on
  the consensus run existing): consensus centroids in the existing `contexts/assembly101_world_mm_3d`
  view, `diagnostics/multiview/<part>/{views_used,c10379_error_px}` on a new time panel, the
  episode document as a static tab, per-frame consensus lines and active episodes in the
  navigation document, `InteractionReviewIndexManifest.multiview_consensus` and coverage keys;
  rebuilt with `--overwrite` (23.5 s, 67.4 MB). No viewer opened.
- **Tests.** `tests/test_multiview_pass.py`: plane lift/warp round trip on the synthetic rig,
  behind-camera projection, box clamping, decode-request normalisation, seed status validators,
  the new run range guard, ordered-seed metadata validators, episode/majority/merge logic on a
  fixture, mask distance and centroid; `real_data`: every seed manifest is agent-authored with
  verified mask fingerprints and in-band area ratios (or skipped with a reason), every multiview
  run declares agent seeds and the first minute, the consensus has >= 6 static sources and a
  sub-0.01 mm wrist anchor, and the v4 RRD carries the multiview entities. 403 default-tier
  tests pass (+9), `real_data` 34.
- **Open.** Interior is unseeded on every new view; a same-clock second human view, or seeding
  at the first frame the interior rests on the table, would fix it. The ego pose gate used by
  Track 6 is a data-consistency check (shipped 2D vs our projection), not a video check, and
  passes on all 1800 frames.

### Sep 18: Track 5 of the overnight multicam pass, per-part visual hulls from eight silhouettes

- **Claim boundary first.** A visual hull is the intersection of the views' silhouette cones. It
  is an upper bound on the object's volume only when every silhouette is a superset of the
  object; with SAM3 masks that miss a part's occluded half, the intersection is *over-carved*
  and the hull shrinks or collapses. Hull counts, centroids, hull-vs-mask IoU and the
  `hull_disagreement` episodes are therefore geometry-only comparison evidence and never a
  reconstruction or accuracy claim. Nothing is substituted into any reference. CPU only; no
  model ran; no viewer opened. CC BY-NC 4.0 applies.
- **Builder** (`battle-build-visual-hull`, `multiview_visual_hull.py`,
  `runs/multiview-visual-hull-first-minute/`). A 5 mm axis-aligned world grid over the dataset
  hand joints (confidence >= 0.5) of the minute plus 150 mm, cut at the fitted table plane:
  140 x 133 x 149 cells from (-505, -210, -380) mm, 2.02 M of 2.77 M cells above the plane. Every static
  camera is constant, so each voxel's proxy pixel in each of the eight views is computed once
  (distortion on, out-of-frame and behind-camera voxels marked). Per frame and part the grid is
  carved by the static views that have a mask for the part and are not inside an active
  `multiview_disagreement` episode for it (>= 2 views); the survivors give the voxel count,
  centroid, bounding box and height above the table (`per_frame.jsonl`), the occupied indices at
  1 fps (`hull_voxels_1fps.npz`, `<target>/<frame:06d>` -> (N, 3) int16; centre = origin +
  (index + 0.5) x 5 mm), and a `hull_projection` mask per view (voxel centres rasterised, then
  closed and dilated by the voxel footprint f x 5 mm / depth) compared with the view's own SAM3
  mask: IoU, mask/hull area ratio, "mask larger than hull" (> 1.25 x). Hull projection PNGs are
  kept at 1 fps under `hull_projection_masks/<view>/`; the per-frame series are in
  `hull_series.npz`. Runtime 184 s for 1,800 frames x 3 parts x 8 views, one frame in memory
  at a time.
- **Measures.** Hull exists (>= 2 views, > 0 voxels) on 1514 / 1496 / 1592 frames for chassis /
  rear_body / cabin (median 8 views); median voxel count 843 / 214 / 440 (p10 98 / 86 / 29, p90
  2771 / 250 / 1741), i.e. 105 / 27 / 55 cm3 at 0.125 cm3 per voxel; the frame-0 cabin hull
  centroid (-167, 12, -304) mm sits 8 mm from the triangulated seed centroid. Hull-vs-mask
  median IoU per view: chassis 0.30-0.56 (C10395 0.30, C10118 0.37, the rest 0.45-0.56),
  rear_body 0.37-0.80 (C10115 0.80, C10395 0.37), cabin 0.15-0.68 (C10395 0.15, C10390 0.68);
  the median mask/hull area ratio is 1.1-2.9 and the "mask larger than hull" fraction 0.26-0.99
  in every view, which is the over-carving bias above, not a per-view verdict. Around the
  human-reported C10379 failures: chassis `[573,722)` median C10379 hull IoU 0.11 (hull present
  on 149/149 frames, median 504 voxels) and `[1020,1172)` median IoU 0.00 (152/152 frames,
  1186 voxels) against 0.46 in `[279,408)`; rear_body 0.41-0.47 in all three windows; cabin
  `[279,408)` collapses to a median of 28 voxels (the hand covers it in several views) with
  C10379 IoU 0.06, against 0.72-0.74 in the other two windows.
- **Disagreement.** Because every mask exceeds an over-carved hull, a view is flagged only when
  it disagrees clearly more than the other views on the same frame: IoU < 0.3 *and* 0.15 below
  the frame's median IoU, or area ratio above max(2.0, 1.5 x the frame's median ratio); >= 5
  consecutive flagged frames form a `hull_disagreement` episode. 183 episodes; C10395 has 67 and
  C10118 43 (the two low cameras), C10379 37, the top-down and corner cameras 4-13 each. The 37
  C10379 episodes are typed proposals (`not_contact_eligible`, `applied: false`): chassis
  `[240,272)`, `[585,598)`, `[643,651)`, `[665,689)`, `[697,702)`, `[1046,1084)`, `[1089,1101)`,
  `[1103,1161)`, `[1167,1172)`, `[1487,1493)`, `[1537,1547)`; rear_body `[199,324)`,
  `[414,613)`, `[614,629)`, `[696,701)`, `[1179,1228)` (four adjacent runs), `[1552,1643)` (four
  runs), `[1671,1677)`; cabin `[94,114)`, `[156,171)`, `[247,254)`, `[276,298)`, `[342,413)`,
  `[513,518)`, `[529,572)`, `[700,706)`, `[722,764)`, `[769,875)`, `[921,969)`, `[1111,1116)`. The
  chassis intervals coincide with the Track 2 centroid contradictions (585-702, 1089-1172); the
  rear_body and cabin lists are far longer than Track 2's and include windows where the hull
  itself is tiny, so the hull proposals are the noisier of the two signals and should be read
  next to `voxel_count`.
- **Not run, by design.** MVDet-style learned BEV fusion (`not_run`: needs scene-specific
  ground-plane training data and would be training, which the plan forbids; the carved hull is
  the training-free equivalent) and homography-to-BEV identity handoff (`not_run`: a
  multi-person tracking device; one person and four rigid parts leave nothing to hand off).
  Both are recorded in the manifest's `not_run` map. Hand-occlusion handling (excluding voxels
  behind a dataset hand in a view) was not implemented; the collapses above are what it would
  have prevented.
- **Viewer.** Hull voxels per part at 1 fps in the world-mm 3D view and hull projections as a
  toggleable overlay in each of the eight 2D views of the static comparison recording; not
  added to the v4 layer (the hull is a separate run with its own manifest; the v4 3D view
  already carries the consensus centroids).

### Sep 18: Track 6 of the overnight multicam pass, the other ego cameras

- **Claim boundary first.** e4's seeds are agent-authored geometric transfers like the static
  ones; its per-frame pose is the dataset's; its intrinsics are the 0.31 px rational estimate;
  e1/e2 intrinsics are 4-5 px estimates fitted on few hand observations ("projection-only
  cameras"). Visibility is geometric (inside the sensor, in front of the camera) and ignores
  occlusion. Nothing here is accuracy. GPU jobs went through the queue (two jobs, both
  succeeded). CC BY-NC 4.0 applies.
- **e4 (HMC_21179183).** New clip config
  `configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_ego_e4_g2.json` (the
  954x720 proxy from Track 0 at 294.000 s, sha `40e8b4a8...`). Seeds from the same triangulated
  frame-0 centroids as Track 2, projected with the e4 pose at pose frame 17640 (proxy px = raw
  px x 1.5): chassis accepted (plane-warp IoU 0.77 into e3, centroid ray 7 mm, area x0.74),
  rear_body (0.75, 10 mm, x0.52), cabin (0.77, 4 mm, x1.45), left hand accepted; interior
  projects behind the camera (depth -17 mm: it is at the face) and is blocked. Decode 5.1 s
  (queue 9.0 s). SAM3 run
  `runs/muggledsam-sam3-four-part-multiview-first-minute-ego-hmc21179183-20260918t061608z`:
  1800/1800 frames, 5,273 masks, worker 132 s (queue 173 s), time to first output 4.6 s, peak
  VRAM 2.10 GB; presence chassis 1.00, cabin 1.00, rear_body 0.93 with the first missing frame
  at 216. As a moving camera in the consensus (`--ego-view HMC_21179183`,
  `runs/multiview-part-consensus-first-minute-with-e4/`, 80 s) it joins a frame only behind the
  wrist gate (both dataset wrists reproject within 10 raw px of the shipped 2D through the e4
  pose); the gate passes on 1800/1800 frames, which says the pose data are self-consistent, not
  that the video is (the gate has no video-side observation; the 30 px triangulation filter is
  what actually drops e4 when its mask disagrees). e4 agreement: chassis 0.97 (mean error 2.9
  px), cabin 1.00, **rear_body 0.58 (39 px)**: its rear_body mask left the part early and the
  filter drops it (27 of the 121 episodes are e4's, mostly rear_body). With e4 the C10379
  contradictions are unchanged for chassis (585-702, 1037-1042, 1089-1172, 1459-1470,
  1499-1504) but the late rear_body contradiction `[1677,1800)` shrinks to `[1525,1530)`,
  `[1678,1683)`, `[1698,1704)`: the consensus that contradicted C10379 late was fragile enough
  for one more (imperfect) view to move it, so the static-only run stays the canonical
  `runs/multiview-part-consensus-first-minute/` and the e4 variant sits beside it. Carving with
  e4 (`runs/multiview-visual-hull-first-minute-with-e4/`, 872 s because the ego pose projects
  the 2 M voxels per frame): chassis hull on 1476 frames (median 705 voxels), cabin 1585 (427),
  rear_body 1221 (196, down from 1496 static-only: e4's wrong rear_body mask empties the
  intersection); e4 hull-vs-mask median IoU chassis 0.37, cabin 0.41, rear_body 0.03; 266
  episodes, 43 C10379 proposals. Both e4 artefacts are kept as the moving-camera arm; the
  static-only hull is canonical.
- **e1/e2 audit** (`battle-multiview-ego-visibility`, `multiview_ego_audit.py`,
  `runs/multiview-ego-visibility-audit/report.json`, 3.6 s, hull voxels at 1 fps and dataset
  joints at 30 fps through each camera's per-frame pose). Share of sampled frames with > 50 %
  of the part's hull inside the sensor, and mean share of dataset hand joints inside:
  **e2 HMC_21176623**: chassis 6 %, rear_body 45 %, cabin 0 %; hands left 2 %, right 31 % ->
  `not_run` (best part 45 % <= 50 %). **e1 HMC_21176875**: chassis 61 %, rear_body 4 %, cabin
  88 %; hands 27 % / 23 % -> the rule says a run is warranted, so a clip config was written
  (`..._focused_ego_e1_g2.json`) and seeds planned, but at frame 0 every part projects outside
  e1's frame (cabin at proxy row 745 of 720; the rest farther out), so frame-0 seeding is
  impossible: recorded as `blocked: seed_transfer_failed` in
  `runs/multiview-seed-transfer-20260918/HMC_21176875/seed_manifest.json` without spending a
  decode. A later-frame seed (the first frame the cabin is in view) would need a run profile
  that starts mid-minute; not built tonight. For reference the same audit gives e3 100 / 100 /
  38 % (hands 95 / 97 %) and e4 100 / 91 / 100 % (hands 100 / 88 %), matching the Track 6 plan
  note that e4 sees the left hand throughout.
- **Tests.** The Track 2 fixture tests cover the range guard's e4 exception and the skip
  decision for a view with no in-frame part; `real_data` covers the e4 run's agent-seed
  declaration through the shared multiview-run check. No viewer opened.

### Sep 18: Track 7 preparation, LM-EEC ego-exo correspondence (CPU only, GPU job queued)

- **Claim boundary first.** Every number this track will produce is an IoU between two
  estimates of the same ego-view region: the LM-EEC prediction from a human C10379 mask, the ego
  SAM3 mask of the Sep 16 run (human frame-0 seed, one tracker), and Track 5's visual hull
  (agent-seeded static masks, fitted intrinsics, dataset extrinsics) projected into the ego
  camera. Cross-source disagreement, not accuracy; no ego ground truth exists for the parts.
  The ego video is monochrome and LM-EEC was trained on colour Aria frames; both views are
  squashed to 480x480 inside the model. LM-EEC's released checkpoints are research artefacts of a
  NeurIPS 2025 paper with no separate licence file (SAM 2 code Apache-2.0); Assembly101 is
  CC BY-NC 4.0. Tonight: CPU only, the GPU was never touched (every check ran with
  `CUDA_VISIBLE_DEVICES=""`); no viewer opened.
- **Install (LM-EEC, inside the 90 min box; the ObjectRelator fallback was not needed).**
  `juneyeeHu/LM-EEC` cloned to `/home/nick/src/LM-EEC` at `b37e50e50fd03ae8625e6100da37bad3dfeb6aa4`
  (the pre-flight HEAD), own venv (Python 3.10 as in its `environment.yml`), torch 2.7.1+cu128 /
  torchvision 0.22.1 (the upstream cu118 pins cannot drive the RTX 5070 Ti; 2.7.1+cu128 is the
  build already proven on this machine), `pip install -e .` with `SAM2_BUILD_CUDA=0` (the
  connected-components CUDA extension is optional post-processing), plus four imports the
  package needs at construction time that its `setup.py` never declares (`timm`, `matplotlib`,
  `scikit-learn`, `networkx`) and the inference tool's `natsort`/`pycocotools`/OpenCV. The
  released weights are on the authors' Google Drive, not HF: `ExoEgo_checkpoint.pt`
  (SHA-256 `b3130bcb…83dcd9c`, 1,003,932,430 B) and `EgoExo_checkpoint.pt` (`a79234ab…af259ad`,
  1,003,932,238 B), fetched with `gdown`; both carry identical training metadata (epoch 60,
  11,280 steps) and no direction label, so `ExoEgo` is used for exo->ego on its file name alone
  (recorded as a claim boundary). The SAM 2.1 base-plus weights are symlinked from
  Grounded-SAM-2 into `./checkpoint/` (the training config's path; inference does not read
  them, the fine-tuned checkpoint holds every tensor: 627 keys load with no missing/unexpected
  keys). Clock: clone at 06:32Z, model constructing on CPU at 06:43Z (11 min). All steps in
  `scripts/install_lm_eec.sh` (idempotent, re-run clean in 8 s). Recorded in `docs/SOURCES.md` /
  `docs/LICENSES.md`.
- **What the CPU dry run proved.** (1) Config parses, `build_sam2_video_predictor_ego` constructs
  `sam2.sam2_correspondence_predictor.SAM2VideoPredictor` on CPU (83.6 M parameters, image size
  480) in 3.2 s, checkpoint loads with `map_location="cpu"`, `forward_image` runs on a dummy pair
  (FPN 32x120x120 / 64x60x60 / 256x30x30). (2) The whole driver path ran end to end on CPU on a
  copy of the real run directory with the model's five hard-coded `.to("cuda")` calls shimmed
  (`--cpu-smoke`; `sam2/modeling/sam2_base.py:738,982,1237,1422,1648`): 47 pairs (4 parts x 12
  keyframes minus interior at 1050), 23.8 s total, first output 3.3 s after start, 0.42 s per
  frame, every prediction non-empty, model predicted-IoU 0.26-0.64 and object-score logits
  8-13 on the first pairs. Those CPU masks are smoke output and were deleted; the real run
  directory holds no `predictions.json` until the queue runs. (3) Reading the predictor: its
  `ego_*` tensors are the *query* view and `exo_*` the *predicted* view whatever the cameras are
  (the tool's `--swap False` therefore means exo->ego); `_get_orig_video_res_output1` returns
  480x480 logits, not frame-sized ones, so the driver resizes to the target frame itself; frames
  are read as `<key>.jpg` from one directory per view with a shared key list; the query mask
  must be present on every frame fed (`propagate_in_video` skips frames without one), so the
  twelve keyframes of one part are one twelve-frame clip in `sequence` mode (memory spans the
  150-frame gaps it was not trained on; `--mode independent` resets per keyframe).
- **Prepared (`battle-egoexo-correspondence prepare`, 3.3 s).**
  `runs/egoexo-correspondence-first-minute-20260918/`: frames at C10379 analysis frames 0, 150,
  ..., 1650 and ego HMC_21110305 frames `p + 4` (`floor((17649 + 2p - 17640) / 2)`; the half
  frame is rounded down, so every ego frame is one pose frame = 16.7 ms *before* its exo frame,
  the same integer-timeline convention as the Track 1 synchronized recordings; residual `-1`
  recorded per pair), JPEG quality 95, 1280x720 and 954x720. Query masks: 47 human/agent
  C10379 masks from `runs/ensemble-reference-first-minute-v1` (interior absent at 1050).
  Reference masks: 43 ego SAM3 masks from the Sep 16 e3 run (absent: cabin at 150, 600, 900 and
  rear_body at 900). Hull voxels available (Track 5, `hull_voxels_1fps.npz`, 1 fps covers every
  keyframe): chassis on 7, rear_body on 9, cabin on 10 keyframes, interior never. Ego->exo for
  the two hands is wired in the driver and `pairs.json` but `skipped`: neither run carries a hand
  mask (`PerFrameHand` has no mask field). Queue job
  `runs/overnight-multicam-20260918/jobs_t7_egoexo.json` (name `lm-eec-egoexo-correspondence`,
  interpreter `/home/nick/src/LM-EEC/.venv/bin/python`, cwd the LM-EEC checkout because Hydra
  resolves the config module from the package, `CUDA_VISIBLE_DEVICES=0`, 1800 s); `--dry-run`
  clean. `run` prints the same argv and refuses to execute with CUDA hidden.
- **Evaluate / rerun (CPU, ready; exercised on the smoke copy).** `evaluate` projects each hull
  voxel's eight corners through `CameraRig` into the ego camera at the ego frame's pose frame
  (raw 636x480 px scaled x1.5 to the proxy, pixel boxes filled, corners behind the camera
  dropped) and writes `hull_projection/ego-hmc21110305/<part>/<key>.png`; visually the
  projections sit on the parts in the ego frame (frame 0: chassis, cabin and rear_body coincide
  with the ego SAM3 masks). On the smoke copy the reference-vs-hull IoU (SAM3 vs hull, no model
  involved) had medians chassis 0.006 (n=7), rear_body 0.23 (n=9), cabin 0.0 (n=10): the ego
  SAM3 run and the hull agree at frame 0 and disagree later (at 450 the SAM3 chassis is not where
  the hull puts it), which is the same tracker drift the Sep 16 ego review reported and is the
  yardstick any model number has to be read against. Typed `manifest.json`
  (`egoexo_correspondence_evaluation`): per pair IoU vs SAM3, vs hull, SAM3 vs hull, the model's
  predicted IoU and object score, areas, notes; per-part summaries; the five measures (coverage
  = non-empty predictions / pairs, first output and driver time from `predictions.json`, runtime
  = queue `job_end.duration_s`, VRAM = torch `max_memory_reserved` from the driver, ID resets
  n/a); the queue record; claim boundaries. `rerun` writes `correspondence.rrd` (2.2 MB on the
  smoke): two 2D views per keyframe (exo frame + query cut-outs; ego frame + prediction in
  magenta, hull in white, SAM3 in part colours), an IoU time panel, the manifest.
- **Tests (`tests/test_egoexo_correspondence.py`, 15 default + 2 `real_data`).** Keyframe list;
  the clock mapping on the +9/0 rules (`p + 4`, residual -1), on equal rules, on +6/0, on the
  inverse ego->exo direction (frame 10 -> 5, residual -1) and its refusal before the target
  proxy (the inverse mapping initially forgot the target's own pose offset; the test caught it);
  IoU edge cases; mask PNG round trip; hand-mask lookup by label; voxel corners and the box
  rasteriser (fill, NaN drop, clipping, empty); hull projection on a fixture rig landing on the
  rig's own projection and dropping voxels behind the camera; `prepare` on a fixture repository
  (synthetic runs, fake frame reader, tiny checkpoints: mapping, JPEG sizes, missing masks ->
  None, hull availability, skipped direction, fingerprints, queue job against `QueueSpec`);
  `prepare` without a hull; `evaluate` + `rerun` on driver-shaped predictions (perfect / empty /
  disjoint masks, per-pair notes, summaries, measures, queue record, de-duplicated skips) and
  without a hull; `PredictionsFile`/`KeyframePair` typing; driver argv and queue job;
  `summarize`. `real_data`: the prepared run directory (12 pairs, `p + 4`, files present, >= 40
  query masks, sizes, queue job) and the LM-EEC checkout at the pinned commit with the 1 GB
  checkpoint present. Default suite 419 passed.
- **Not run.** The GPU job. After it: `uv run battle-egoexo-correspondence evaluate` then
  `uv run battle-egoexo-correspondence rerun`. Known risks: the model asserts a 2-D query mask
  and needs it on every frame fed (satisfied by construction); bf16 autocast on Blackwell with
  torch 2.7.1 is the same stack the other queue jobs use; the direction of the two checkpoints is
  a file-name inference, so if the exo->ego masks look like exo-view shapes the other checkpoint
  is one `--checkpoint` swap away in `pairs.json`.

### Sep 18: GPU queue results (WiLoR arm, Kineo, ego-exo)

- **Claim boundary first.** Everything below is cross-source disagreement between estimates.
  The WiLoR-arm millimetres compare a DLT on WiLoR detections (fitted intrinsics, dataset
  extrinsics) with the dataset's own hand tracker; neither is ground truth. The Kineo
  self-calibration numbers compare Kineo's estimated cameras with the dataset's shipped
  extrinsics after a similarity alignment: a calibration comparison against dataset context, not
  pose accuracy; Kineo wrist numbers are body-model wrists against hand-tracker wrists. The
  ego-exo IoUs compare a correspondence model with one tracker's ego masks and a carved hull; no
  ego ground truth exists for these parts. Licences: Assembly101 CC BY-NC 4.0; WiLoR checkpoints
  CC-BY-NC-ND (MANO and Ultralytics separate); Kineo research/evaluation only, checkout dirty
  (fingerprinted in `prepare.json`); LM-EEC weights are research artefacts of a NeurIPS 2025
  paper with no licence file, SAM 2 Apache-2.0; ATHENA MIT. No viewer opened.
- **Queue.** Four job files, one at a time, every job through `scripts/overnight_queue.py`
  (systemd-inhibit, `nvidia-smi` + `journalctl -k` between jobs): `jobs_wilor_3views.json`
  (dry-run clean, then 5/5 succeeded: WiLoR C10379 134.9 s, C10395 77.3 s, C10115 118.9 s,
  `battle-athena-hands --hand-source wilor` 6.9 s, its review 2.1 s), `jobs_t4_kineo_known.json`
  (1/1, 222.2 s of the 1800 s box), `jobs_t4_kineo_selfcal.json` (1/1, 280.2 s of 3600 s; the
  2 h Kineo box used 8.4 min), `jobs_t7_egoexo.json` (appeared while Kineo ran; dry-run clean,
  1/1, 6.3 s of 1800 s). GPU wall 14.1 min in total; every `gpu_check` ok, no NVRM/Xid line in
  the kernel journal, idle memory 1.19-1.31 GiB before each job. Nothing failed, timed out or was
  skipped; no job was rerun.
- **Five measures per run.** WiLoR C10379 / C10395 / C10115 (`runs/wilor-hands-<view>-60s-20260918`,
  native evidence on): coverage 1,751 / 1,712 / 1,800 of 1,800 frames with a detection (3,490 /
  2,140 / 3,622 detections); first usable output 6.3 / 5.1 / 5.1 s; runtime 126.8 / 70.0 /
  110.7 s (queue 134.9 / 77.3 / 118.9); peak VRAM 2,819,373,056 B (2.63 GiB) on each; ID resets
  n/a (frame-local ids). ATHENA WiLoR arm (`runs/athena-hands-first-minute-wilor`): left hand
  solved in 1,623 frames, right in 1,603; first output 4.8 s; runtime 6.3 s; VRAM n/a; ID resets
  n/a (sides from the dataset match). Kineo known: 1,800 / 1,800 frames with body 3D and both
  wrists; first 3D output 147.0 s (cumulative through MVS triangulation); runtime 222.2 s
  (pipeline 206.9); peak VRAM 5,737,807,872 B (5.34 GiB, torch reserved, ONNX Runtime excluded);
  ID resets n/a (`best_bbox_only`). Kineo selfcal: 1,800 / 1,800; first 3D 155.0 s; runtime
  280.2 s (pipeline 246.0); VRAM 5,880,414,208 B (5.48 GiB); n/a. LM-EEC
  (`runs/egoexo-correspondence-first-minute-20260918`): 47 / 47 pairs predicted non-empty (43
  with an ego SAM3 mask to compare, 26 with a hull projection); first output 4.06 s; runtime
  6.28 s (driver 5.54 s including model load); peak VRAM 788,529,152 B (752 MiB); ID resets n/a
  (one object per query).
- **WiLoR arm (Track 3), three views, 1,800 frames, C10379 clock.** Same alignment, matching,
  20-joint mapping and ATHENA filter as the MediaPipe arm (Track 3b). Disagreement vs
  `landmarks3D`, raw DLT (smoothed in brackets); the MediaPipe eight-view arm in the last column
  for reference:

  | hand | mean views | wrist median / p90 mm | fingertips median / p90 mm | all 20 joints median / p90 mm | MediaPipe 8-view wrist; tips |
  | --- | --- | --- | --- | --- | --- |
  | left | 2.19 | 28.9 / 43.1 (28.8 / 43.1) | 29.6 / 64.9 (29.3 / 65.1) | 20.9 / 48.6 (20.9 / 48.3) | 23.9 / 37.1; 38.8 / 84.3 |
  | right | 2.58 | 21.0 / 34.7 (21.0 / 34.9) | 32.3 / 63.8 (32.3 / 63.9) | 22.5 / 50.1 (22.7 / 50.4) | 19.8 / 31.2; 31.3 / 59.6 |

  Per-view reprojection RMS of used points (left / right): C10379 13.1 / 12.0 px, C10395 10.4 /
  9.7, C10115 8.5 / 5.9; over all observed points C10379 26.0 / 25.9, C10395 14.1 / 12.1, C10115
  8.8 / 8.0 (C10379 again the worst view; C10395 detects a second hand in only 11,400 of the
  left-hand point slots, as in Track 3a). Per joint (raw median, left / right): wrist 28.9 /
  21.0; palm mcp 14.4-16.4 / 12.6-16.2; tips thumb 23.3 / 28.9, index 27.8 / 27.7, middle 33.1 /
  33.8, ring 34.7 / 36.1, pinky 31.6 / 33.8. Reading: with three views WiLoR's wrist disagrees
  slightly more than MediaPipe's eight-view wrist (+5 mm left, +1 mm right) while its fingertips
  disagree less (-9 mm left, +1 mm right, p90 -20 / +4), so WiLoR's articulation is at least as
  consistent with the dataset as MediaPipe's, from fewer cameras.
- **The wrist-step test ("substitute a triangulated wrist for WiLoR's translation").** All on the
  C10379 clock, per-frame displacement median (p90), left / right. Triangulated WiLoR-arm wrist,
  raw: 2.4 (7.0) / 4.0 (11.2) mm; ATHENA-smoothed 1.9 (5.5) / 3.3 (10.7); dataset tracker 1.9
  (4.6) / 3.5 (9.8); the MediaPipe arm was 2.8 / 3.3. In C10379 pixels the triangulated wrist
  reprojected moves 3.2 (11.4) / 4.9 (14.1) px against WiLoR's own 2D wrist 3.1 (11.2) / 6.3
  (17.4): equally steady in the image. WiLoR's camera-frame wrist (joint 0 + `pred_cam_t_full`)
  jumps 145 (535) / 333 (1,169) units x 1e-3 per frame, identical to the Track 3b numbers from
  the Sep 16 C10379 run (WiLoR is deterministic on this proxy). To put that in millimetres the
  triangulated wrist was moved into the C10379 camera frame (`CameraRig.world_to_camera`) and
  WiLoR's camera-frame wrist fitted to it per axis on the 1,588 / 1,349 frames where both exist:
  lateral 754 / 751 (left) and 766 / 887 (right) mm per WiLoR unit by least squares, depth by the
  median depth ratio 30.2 mm per unit (the least-squares depth slope, 13 / 6 mm per unit, is
  diluted by WiLoR's depth noise: depth correlation 0.43 / 0.35). The lateral/depth ratio of ~25
  is what the pinhole predicts for WiLoR's 25,000 px scaled focal length over the fitted 834 px
  proxy focal (30), within the MANO-vs-real hand-scale factor. In those millimetres WiLoR's
  wrist moves 5.1 (17.8) / 12.0 (37.1) mm per frame, of which depth 4.4 (16.2) / 10.1 (35.3) and
  lateral 2.2 / 4.8; the triangulated wrist's depth step is 1.25 (4.3) / 1.59 (6.4) mm and lateral
  1.7 / 3.2. The two wrists sit 25.0 (69) / 36.8 (114) mm apart after the fit, |dz| median 21 /
  33 mm against |dx| 10 / 11 and |dy| 5 / 6. Reading: the hopping is WiLoR's per-frame depth;
  replacing its translation with the triangulated wrist cuts the depth step 3.5x (left) / 6x
  (right) and lands the wrist series on the dataset tracker's own steadiness, while WiLoR's 2D and
  articulation are untouched. That is a statement about steadiness and cross-source agreement,
  not about which wrist is right. The substitution itself is not built; the arm only measures it.
- **Viewer.** `interaction_review_v4` now logs both ATHENA arms when present: the MediaPipe arm
  under `contexts/assembly101_world_mm_3d/athena_hands` and `diagnostics/assembly101/athena_hands`
  as before, the WiLoR arm beside it under `.../athena_hands_wilor` (magenta / green, series names
  carry the source), through a `label` argument on `athena_hands_review.log_static` / `log_frame`.
  Rebuilt in place (`--overwrite`, 30.9 s); `rerun rrd print` shows both `athena_hands` and
  `athena_hands_wilor` joints / skeletons / skeletons_smoothed / manifest entities and their
  diagnostics series. The standalone `hands.rrd` of the WiLoR arm was written by the queue.
- **Kineo known-camera arm** (`runs/kineo-multiview-known-first-minute-20260918`, dataset cameras
  injected, Kineo only detects and triangulates). Fixed scale 1000 mm per unit; the exported
  cameras round-trip to the dataset's at < 1e-6 deg / < 1e-4 mm (identity check). Wrist
  disagreement vs the dataset hand tracker: left median 16.9 mm, p90 37.8, mean 19.3 over 1,800
  frames; right 17.4 / 31.9 / 20.5 over 1,628; the opposite dataset side is never closer.
  Stages: rtmlib 28.4 s, NLF 114.5 s, MVS triangulation 4.2 s, Rerun export 59.8 s.
- **Kineo self-calibration arm** (`runs/kineo-multiview-selfcal-first-minute-20260918`, full
  stock stage list minus SAM2, `shared_intrinsics: false`). Kineo's own SMPL global scale 15.14.
  Umeyama with scale on the eight camera centres: recovered scale 597.9 mm per Kineo unit,
  camera-centre RMS 148.7 mm. Per camera after alignment (rotation deg / translation mm): C10095
  10.77 / 247.0, C10115 7.94 / 151.9, C10118 3.64 / 226.1, C10119 5.09 / 61.3, C10379 10.22 /
  102.0, C10390 3.98 / 69.5, C10395 11.41 / 127.3, C10404 6.57 / 80.4; medians 7.26 deg / 114.7
  mm. Wrist disagreement through that alignment: left 83.1 / 111.1 (mean 84.8, n 1,800), right
  44.3 / 67.3 (45.7, n 1,628); swapped side closer 0.4 % / 0 %. Runtime 280 s; SfM 6.2 s and
  the three BA passes 0.4 s each on 1,800 frames x 8 views, so the 3600 s box was never near
  (Rerun export at 76.6 s and NLF at 112.9 s dominate). Native `.rrd`, `_ba_history.rrd` and
  `.bvh` fingerprinted in the manifest; `comparison.rrd` written by `rerun` (1.5 s).
  *Diagnostic, not in the manifest* (`/tmp` script on `body_aligned_mm.npz`): aligning instead on
  the 3,428 Kineo-vs-dataset wrist pairs asks for a further similarity of scale 1.579, 2.25 deg
  and 135 mm on top of the camera alignment (total 944 mm per unit), after which the wrists
  disagree by 17.1 / 17.6 mm median (p90 32.9 / 33.2), the known arm's numbers, while the camera
  centres then sit 264-842 mm from the dataset's. Reading: Kineo's body 3D is consistent with the
  dataset up to a similarity; its camera placement is not the dataset's under any similarity.
  Kineo puts the cameras ~37 % closer to the subject (598 / 944) with 4-11 deg of compensating
  rotation, which is the expected weak spot of self-calibration from one subject in a ~1 m volume
  at 2-3 m range (camera depth and rotation trade off in BA). The camera comparison is therefore
  the honest result of this arm and the wrist numbers under the camera alignment are dominated by
  it; ego cameras were excluded (Kineo assumes static cameras).
- **Ego-exo correspondence (Track 7), LM-EEC `ExoEgo_checkpoint.pt`, `sequence` mode, 12
  keyframes every 150 frames, C10379 -> HMC_21110305.** IoU of the exo->ego prediction against
  the ego SAM3 mask / against the hull projection, median (n, count >= 0.5), plus the model's own
  IoU prediction and the SAM3-vs-hull reference-to-reference disagreement:

  | part | model IoU pred. median | vs ego SAM3 | vs hull projection | SAM3 vs hull |
  | --- | --- | --- | --- | --- |
  | chassis | 0.70 | 0.41 (12, 5) | 0.37 (7, 1) | 0.006 (7, 1) |
  | interior | 0.71 | 0.27 (11, 3) | no hull projection on any keyframe | - |
  | rear_body | 0.03 | 0.00 (11, 2) | 0.00 (9, 0) | 0.23 (8, 3) |
  | cabin | 0.05 | 0.00 (9, 0) | 0.003 (10, 0) | 0.00 (8, 2) |

  Per keyframe, chassis vs SAM3: 0.35, 0.73, 0.41, 0.26, 0.01, 0.40, 0.70, 0.69, 0.96, 0.57,
  0.40, 0.33 (frames 0-1650); interior: 0.00, 0.12, 0.51, 0.69, 0.93, 0.33, 0.02, 0.00, 0.27,
  0.39, 0.23 (no pair at 1050); rear_body has two hits (0.73 at 150, 0.74 at 1350) and zeros
  elsewhere with 1,200-10,800 px predictions; cabin peaks at 0.34 (frame 0) and 0.10 and is
  otherwise zero, with predictions of 12,000-84,000 px that are not the cabin. Ego->exo for the
  hands was skipped (no ego hand masks in any run, recorded in the manifest). Reading: the model
  finds the two large parts in roughly half the keyframes and its own IoU prediction separates
  the cases (0.70 for chassis/interior, 0.03-0.05 for the two small dark parts), so its
  confidence is a usable trigger; the hull projection disagrees with the ego SAM3 masks as much
  as the model does (SAM3 vs hull median 0.006 for the chassis), so on this monochrome ego view
  there is no reference the other two agree with. `correspondence.rrd` (2.2 MB) written.
- **Tests.** `tests/test_athena_hands.py` +1 default (both v4 arms on distinct entity paths and
  colours) +1 `real_data` (the WiLoR arm: three views, 2 <= mean views <= 3, wrist median in
  (5, 40) mm, per-view used RMS <= 30 px, the `wilor_wrist_camera_frame_C10379` series present).
  Default suite 419 passed, `real_data` 37 passed (includes the rebuilt v4 package, both Kineo run
  directories, the ego-exo run directory and LM-EEC checkout). Ruff clean.
- **Not done / open.** The substitution (triangulated wrist + WiLoR articulation as a hand
  series) is measured, not built. Kineo's camera comparison could be repeated with the body-wrist
  alignment as the manifest's primary alignment if a future arm wants the body numbers; the
  `evaluate` code keeps the camera-centre Umeyama the plan asked for. LM-EEC `independent` mode
  and the other checkpoint direction were not run (one queue job as prepared). Plan todos
  `t4-kineo` and `t7-lmeec` marked completed.

### Sep 18: slot exclusivity and score-gated memory (tracker policy ablation)

- **Claim boundary first.** Every number in this section is either disagreement between two runs
  of one tracker (an arm against the human-corrected reference
  `runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t001210z`,
  frames `[0,1800)`, which is itself known to be wrong inside `[279,408)`, `[573,722)` and
  `[1020,1172)`), self-consistency of one run (its own slot overlaps, area traces and gate
  decisions), or cross-view disagreement on the calibrated rig. None is accuracy; a low IoU with
  the reference inside a flagged window is as consistent with a fix as with a different failure.
  The policy is a **run condition**, recorded as `runtime_settings.tracker_memory_policy`
  (typed `TrackerMemoryPolicy`) in every manifest and named in the run id. Nothing was swapped
  into any reference; the candidate below is named, not adopted. GPU work ran only through
  `scripts/overnight_queue.py` (three passes, 20 jobs, 20 succeeded, no timeout, no `NVRM`/`Xid`;
  `runs/sam3-policy-ablation-20260918/queue.log`). CC BY-NC 4.0 applies to the dataset assets.
- **What was built.** In `muggled_worker.py`, two pure functions on the multiplex step's own
  outputs: `resolve_slot_exclusivity(masks, scores, mode)` (`argmax`: a pixel positive in more
  than one present slot, raw score > 0, goes to the larger logit and the losers are clamped to
  at most `-8` before `encode_frame_memory` and before `_observation`; both modes report each
  slot's contested fraction) and `memory_gate(scores, ious, contested, areas, history, policy)`
  (a slot whose raw score `<= 0`, predicted IoU `< 0.5`, contested fraction `> 0.2` or area
  outside `[0.5, 2]` x the rolling median of its last 30 trusted frames is handed to the memory
  encoder with score `-1`, which adds MuggledSAM's `no_object_embed` for that multiplex entry
  only, so the frame is memorised as absent for that slot; the band is never applied during the
  first 30 trusted frames, reason `warmup`; corrected frames are never gated, reason
  `corrected`, and the corrected slot's area history restarts). Flags `--slot-exclusivity
  off|argmax`, `--memory-gate off|on`, `--gate-min-object-score`, `--gate-min-iou`,
  `--gate-max-contested-fraction`, `--gate-area-band`, `--gate-area-history-frames`; the
  area history rides in the checkpoint payload under a new key with a default on load and a
  non-default policy joins the stream identity. `TrackerSlotDiagnostic` gains optional
  `contested_fraction`, `memory_written`, `memory_gate_reason`; the raw `object_score` is
  unchanged. `battle-muggled-smoke` passes the flags through, suffixes the run id
  (`-xargmax-gon-r1008-seed0`), lets the focused C10379 profile stop at `--max-frames 1800`
  (corrections past the bound are dropped and listed in the metadata) and run
  `--frame-zero-seeds-only`; `discover_multiview_runs` skips policy runs so last night's
  combiner inputs cannot be replaced by accident. `battle-policy-ablation` (new
  `policy_ablation.py`) computes the table below, the sheets and the recording;
  `battle-review-metrics --reference-run` measures any first-minute C10379 run with the override
  recorded in the package.
- **Regression.** `off-r720-sched` (policy off, 720 px, the schedule, first minute) is
  byte-identical to the reference over `[0,1800)`: 1800 observation rows (same md5 as the
  reference's first 1800 lines) and 7197 mask PNGs (`tests/test_worker_policy_gpu.py`, run as
  queue job `regression-policy-off-byte-identical`). The policy is additive.
- **Resolution axis** (focused C10379, first minute, with the schedule; six runs):

  | arm | side | mask grid | runtime s | ttfu s | peak VRAM GiB | IoU chassis / interior vs ref | chassis IoU 573-722 / 1020-1172 | episodes vs ref (outside windows) | gated chassis / interior % |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | off-r720-sched | 720 | 180 | 125.6 | 4.40 | 2.06 | 1.000 / 1.000 | 1.00 / 1.00 | 0 (0) | - |
  | off-r1008-sched | 1008 | 252 | 199.4 | 4.35 | 2.16 | 0.699 / 0.646 | 0.35 / 0.24 | 14 (8) | - |
  | off-r1280-sched | 1280 | 320 | 348.5 | 4.18 | 2.43 | 0.693 / 0.608 | 0.39 / 0.22 | 17 (8) | - |
  | xg-r720-sched | 720 | 180 | 126.5 | 4.46 | 2.06 | 0.896 / 0.862 | 0.53 / 0.90 | 3 (1) | 15 / 18 |
  | xg-r1008-sched | 1008 | 252 | 201.8 | 4.32 | 2.16 | 0.676 / 0.639 | 0.22 / 0.29 | 12 (8) | 15 / 7 |
  | xg-r1280-sched | 1280 | 320 | 346.4 | 4.17 | 2.43 | 0.725 / 0.606 | 0.13 / 0.35 | 16 (10) | 19 / 8 |

  Runtime 1.6x and 2.8x the 720 run; VRAM 2.2-2.4 GiB (the 8 GiB stop was never near). A larger
  encoder side rewrites the whole trajectory: 1008 and 1280 open eight or more disagreement
  episodes outside the windows (the largest an interior episode over `[1464,1800)`, 336 frames,
  IoU 0.29-0.35 with the reference), and the in-window self-consistency proxies (pairwise slot
  overlap, area jumps, gate fraction) do not improve. By the plan's rule the **working
  resolution stays 720**; whether 1008 is better or worse late in the minute is a labelling
  question.
- **Ablation at 720 px** (four policies x {reference schedule, frame-0 seeds only}; `off/sched`
  is the reference itself). Five pre-accuracy measures for every arm: coverage 1800/1800 frames,
  7091-7200 masks, per-part mask presence 1742-1800 frames (lowest `xg-r720-seed0`: chassis 1749,
  interior 1742); runtime 125.4-127.0 s; time to first usable output 4.07-4.46 s; peak VRAM
  1.99-2.06 GiB; mask grid 180 x 180. The policy costs nothing measurable.

  | arm | policy | corrections | IoU ch / in / rb / cab vs ref | ch IoU 279-408 / 573-722 / 1020-1172 | in IoU 1020-1172 | IoU outside windows | episodes (outside) | slot pair IoU > 0.3 frames | area-jump frames ch / in | gated ch / in % | review-metric swaps (in windows / outside) |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | off-r720-sched | off | full | 1.000 / 1.000 / 1.000 / 1.000 | 1.00 / 1.00 / 1.00 | 1.00 | 1.000 | 0 (0) | 0 | 141 / 123 | - | 7 (4 / 3) |
  | x-r720-sched | x | full | 0.849 / 0.807 / 0.968 / 0.999 | 0.99 / 0.48 / 0.40 | 0.51 | 0.931 | 7 (2) | 0 | 109 / 180 | 0 / 0 | 4 (2 / 2) |
  | g-r720-sched | g | full | 0.881 / 0.870 / 0.986 / 0.999 | 1.00 / 0.42 / 0.93 | 0.98 | 0.941 | 4 (1) | 0 | 141 / 149 | 20 / 18 | 6 (3 / 3) |
  | **xg-r720-sched** | x+g | full | 0.896 / 0.862 / 0.978 / 0.999 | 0.99 / 0.53 / 0.90 | 0.96 | 0.938 | 3 (1) | 0 | 150 / 147 | 15 / 18 | 4 (2 / 2) |
  | off-r720-seed0 | off | frame 0 | 0.592 / 0.589 / 0.941 / 0.989 | 0.47 / 0.42 / 0.05 | 0.35 | 0.805 | 10 (3) | 0 | 164 / 125 | - | 13 (5 / 8) |
  | x-r720-seed0 | x | frame 0 | 0.560 / 0.341 / 0.941 / 0.989 | 0.48 / 0.59 / 0.05 | 0.34 | 0.708 | 11 (4) | 0 | 120 / 163 | 0 / 0 | 12 (4 / 8) |
  | g-r720-seed0 | g | frame 0 | 0.576 / 0.559 / 0.948 / 0.988 | 0.49 / 0.58 / 0.05 | 0.34 | 0.785 | 13 (4) | 0 | 134 / 102 | **53 / 34** | 6 (2 / 4) |
  | xg-r720-seed0 | x+g | frame 0 | 0.543 / 0.354 / 0.939 / 0.988 | 0.37 / 0.59 / 0.05 | 0.35 | 0.707 | 12 (5) | 0 | 140 / 152 | **59 / 67** | 13 (4 / 9) |

  Where the schedule arms differ from the reference: `xg-r720-sched` interior `[707,900)` (IoU
  0.18, up to the 900 correction), chassis `[617,697)` 0.26, interior `[1220,1235)` 0.01;
  `g-r720-sched` adds chassis `[702,827)`; `x-r720-sched` adds chassis `[1089,1172)` 0.03 and
  interior `[1096,1172)` 0.08, i.e. exclusivity alone changes the swap window and the gate
  undoes that. Every frame-0-only arm loses the chassis to the reference from frame ~695 until the
  1172 correction (IoU 0.02-0.04) and the interior over `[707,1050)`: no policy removes the need
  for the 900 and 1172 corrections. Review-metric leakage / growth / area-anomaly episode counts
  (arm masks as the tool's reference) move by at most a few episodes between the schedule arms
  (leakage 17-22, growth 54-58, area anomaly 23-27). Pairwise slot IoU above 0.3 never occurs in
  the first minute in any arm, so exclusivity has little to act on here (contested fraction >
  0.2 on 2-43 frames per slot).
- **Gate behaviour.** With the schedule, `area_jump` is 80-90 % of every gated frame (`xg`:
  chassis 212 of 264 gated frames, `contested` 25, `low_iou` 23, `low_object_score` 4; interior
  308 of 327), cabin is never gated and rear_body 19 %; no slot crosses the 30 % starvation
  line. Without corrections the gate **starves**: chassis 53 % / interior 34 % (`g`) and 59 % /
  67 % (`xg`). The cause is by construction: the rolling median only advances on trusted frames,
  so once a part's apparent size changes for good the band never catches up until a correction
  resets the history. The same happens on the grazing camera in the seven-view pass (below).
- **Seven other views and the combiner** (candidate `xg`, 720 px, last night's geometric seeds,
  `runs/sam3-policy-ablation-20260918/views/<view>/`): 1800/1800 frames each, 123.7-125.3 s,
  1.96 GiB; gated chassis / rear_body / cabin: C10095 35 / 10 / 0, C10115 4 / 9 / 0, C10118
  10 / 12 / 0, C10119 0 / 2 / 0, C10390 19 / 11 / 25, C10395 **42** / 6 / **64**, C10404 0 / 3 / 0;
  first missing frames unchanged where they existed (C10095 chassis 1537, C10115 chassis 234,
  C10395 cabin 228 / chassis 594). Consensus rebuilt with the eight `xg` runs into
  `runs/multiview-part-consensus-first-minute-policy/` (63 s) and the hull into
  `runs/multiview-visual-hull-first-minute-policy/` (148 s); last night's roots untouched.
  C10379 chassis contradicted by the majority: before `[585,604)` `[643,658)` `[665,689)`
  `[697,702)` `[1089,1163)` `[1167,1172)` `[1464,1471)` (147 frames), after `[476,494)`
  `[586,604)` `[618,703)` `[1088,1162)` `[1167,1172)` `[1358,1366)` `[1464,1471)` `[1502,1509)`
  `[1514,1520)` (228 frames); rear_body before `[1525,1530)` `[1677,1751)` `[1759,1781)`
  `[1786,1800)` (115), after `[1524,1530)` `[1705,1751)` `[1759,1765)` `[1767,1781)`
  `[1790,1800)` (82). C10379 agreement chassis / rear_body / cabin 0.88 / 0.93 / 1.00 before,
  0.83 / 0.95 / 1.00 after; consensus episodes over all views 83 (C10379 14) before, 99 (15)
  after. Hull-vs-mask median IoU for the C10379 chassis in `[279,408)` / `[573,722)` /
  `[1020,1172)`: 0.46 / 0.11 / 0.005 before, 0.47 / 0.00 / 0.003 after; overall 0.456 -> 0.396;
  hull episodes 183 (C10379 37) -> 146 (32), the chassis ones now including `[477,494)` and
  `[1571,1641)`.
- **Reading and the candidate.** The policy is cheap and additive, and with the reference
  schedule `xg-r720-sched` (`runs/sam3-policy-ablation-20260918/arms/xg-r720-sched/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t153507z-xargmax-gon`)
  agrees with the reference everywhere except the two windows the human flagged and 15 frames
  before the 1235 correction, with the fewest swap episodes and no starved slot; it is the
  **CANDIDATE reference** for the anchor-frame labelling and is **not adopted**. The label-free
  multi-view evidence does not endorse it: the same arm on all eight views contradicts C10379's
  chassis on more frames (147 -> 228), lowers its chassis agreement (0.88 -> 0.83) and takes the
  `[573,722)` hull IoU from 0.11 to 0.00, while rear_body improves (115 -> 82, 0.93 -> 0.95). The
  gate is the part to fix before wider use (an area band that follows lasting size changes, or a
  median over all recent frames rather than trusted ones); exclusivity is nearly inert on this
  minute. Deliverables: `runs/sam3-policy-ablation-20260918/{README.md,summary.json,summary.md}`,
  `sheets/<arm>.png` for the 12 arms and the reference at frames 300, 370, 400, 600, 650, 700,
  900, 1050, 1100, 1150, 1200, 1500, 1700 with the gated slots named per tile,
  `best_vs_reference.png` (reference | `xg-r720-sched`, cropped to the assembly region),
  `best_vs_reference.rrd` (`uv run rerun runs/sam3-policy-ablation-20260918/best_vs_reference.rrd`:
  the video once, both runs' masks as toggleable layers, object score and gated flag per slot),
  `analysis/<arm>/review_metrics/`. No viewer was opened.
- **Tests.** `tests/test_worker_policy.py` (12 tests: exclusivity on synthetic 3-slot logits
  incl. an absent slot and bfloat16, off-is-identity, gate reason order and thresholds, rolling
  median warm-up, history padding, stream identity, area-band parsing) passes under the battle
  interpreter (9 passed, 9 torch cases skipped) and under the MuggledSAM interpreter (12 passed;
  pytest installed into that env for this); `tests/test_worker_policy_gpu.py` (`gpu`) is the
  byte-identity check; `tests/test_policy_ablation.py` (5) covers the metrics, summary and sheet
  rendering on synthetic runs; schema round trips with and without the new fields, the focused
  first-minute bound and the seeds-only schedule (`real_data`) and the combiner's discovery guard
  are in `test_schemas.py`, `test_muggled_smoke.py`, `test_multiview_pass.py`. Default suite 433
  passed, `real_data` 38 passed. Ruff clean.
- **Not done, by instruction.** The anchor-frame labelling set-up (`x-anchor-setup`) and the
  labels themselves; no reference was swapped; the v4 package and the contact eligibility are
  unchanged. (The set-up followed in the next section; the labels were made Sep 19 and scored
  in the subsection below.)

#### Anchor IoU (human, Sep 19): the anchors do not support the candidate

- **Claim boundary first.** 13 human review anchors on one view (`static-c10379`, focused first
  minute), 51 accepted SAM3 image-decoder masks and one hidden mark, chosen by one person in one
  36-minute session. They rank arms against each other on those 52 cells; they are not a
  dataset, not ground truth and not accuracy. Every IoU below is between a tracker mask and a
  human-chosen decoder mask whose boundary is the decoder's. At the human's own correction frame
  900 the reference and the anchors agree 0.97 / 0.84 / 0.82 / 0.97 (chassis / interior /
  rear_body / cabin); that is about the noise between two SAM3 masks the same person picked on
  different days, so per-cell differences under ~0.15, and arm means within ~0.02, are not a
  ranking. CC BY-NC 4.0 covers the frames and everything derived from them. No GPU work: the
  scorer reads PNGs. The calibration server the human used was left running (Tailscale bind,
  pid noted in the session), untouched.
- **Completeness.** 52 / 52 cells answered (`n/52 done` in the workspace): 51 accepted masks
  and one hidden mark, rear_body at frame 1700. On the three `hidden_prompt` interior cells
  (1050, 1100, 1150) the human accepted a mask (698, 1546, 4161 px), i.e. judged the interior
  visible. Two decoder candidates were rejected before re-prompting (chassis 300, chassis 650);
  the 32 "pending boxes" in the workspace are the drawn prompts of already-decoded candidates,
  nothing outstanding. Exported with `battle-anchor-export export`: 51 labeled / 1 hidden / 0
  unlabeled into `runs/human-review-anchors-first-minute/anchors/` and the committed decision
  record `docs/qa/first-minute-review-anchors.human-record.json` (author = the git identity,
  `reviewed_at` 2026-09-20T00:45:30Z = the manifest's last autosave; the workspace stores no
  per-acceptance timestamps). `configs/qa/first_minute_review_anchors.json` carries the same
  provenance; the test that compared the committed config with the builder now compares
  everything but provenance and requires provenance to be filled.
- **Scoring.** `battle-anchor-iou` over the 12 arms, the reference, `ensemble-reference-first-minute-v1`
  and `dam4sam-four-part-reviewed-seed-60s-20260918t005416z` ->
  `runs/sam3-policy-ablation-20260918/anchor_iou.{json,md}`. The scorer gained an `outside`
  window mean (anchor frames in no window: 900, 1200, 1500, 1700) and `--sheet`, which renders
  `anchors_vs_reference_vs_best.png` (rows = 13 frames, columns = human anchors | reference |
  best arm, per-part IoU under each run tile) through `policy_ablation`'s crop / tile / grid
  helpers. Built-in check passed: `off-r720-sched` and `reference` are identical on all 52
  cells. No run mask needed resizing.

  | arm | IoU ch | IoU in | IoU rb | IoU cab | IoU all | 279-408 | 573-722 | 1020-1172 | outside | missing | hidden FP px |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | **xg-r1280-sched** | 0.602 | 0.525 | 0.827 | 0.964 | **0.728** | 0.888 | 0.620 | 0.590 | 0.796 | 0 | 1224 |
  | off-r1280-sched | 0.608 | 0.540 | 0.789 | 0.964 | 0.724 | 0.861 | 0.603 | 0.659 | 0.764 | 0 | 1228 |
  | dam4sam-60s | 0.649 | 0.436 | 0.833 | 0.959 | 0.717 | 0.811 | 0.648 | 0.713 | 0.700 | 0 | 1236 |
  | off-r1008-sched | 0.544 | 0.518 | 0.798 | 0.968 | 0.705 | 0.866 | 0.607 | 0.554 | 0.775 | 0 | 1235 |
  | xg-r1008-sched | 0.500 | 0.498 | 0.798 | 0.968 | 0.689 | 0.866 | 0.573 | 0.534 | 0.764 | 0 | 1232 |
  | ensemble-reference-v1 | 0.621 | 0.375 | 0.775 | 0.960 | 0.681 | 0.824 | 0.628 | 0.586 | 0.684 | 3 | 1306 |
  | x-r720-sched | 0.576 | 0.406 | 0.773 | 0.960 | 0.677 | 0.825 | 0.597 | 0.597 | 0.686 | 0 | 1307 |
  | g-r720-seed0 | 0.470 | 0.488 | 0.771 | 0.960 | 0.670 | 0.637 | 0.639 | 0.692 | 0.705 | 0 | 1283 |
  | off-r720-sched = reference | 0.520 | 0.427 | 0.775 | 0.960 | 0.668 | 0.824 | 0.628 | 0.533 | 0.684 | 0 | 1306 |
  | g-r720-sched | 0.495 | 0.427 | 0.775 | 0.960 | 0.662 | 0.824 | 0.600 | 0.533 | 0.685 | 0 | 1308 |
  | xg-r720-sched (CANDIDATE) | 0.494 | 0.421 | 0.774 | 0.960 | 0.660 | 0.825 | 0.589 | 0.532 | 0.687 | 0 | 1309 |
  | off-r720-seed0 | 0.427 | 0.384 | 0.766 | 0.960 | 0.632 | 0.624 | 0.618 | 0.695 | 0.598 | 0 | 1310 |
  | x-r720-seed0 | 0.404 | 0.333 | 0.765 | 0.960 | 0.613 | 0.624 | 0.631 | 0.692 | 0.526 | 0 | 1302 |
  | xg-r720-seed0 | 0.387 | 0.338 | 0.766 | 0.960 | 0.610 | 0.597 | 0.640 | 0.695 | 0.529 | 1 | 1303 |

- **The reference against the human, per frame** (chassis / interior / rear_body / cabin):
  300 0.75 / 0.79 / 0.80 / 0.96; 370 0.72 / 0.52 / 0.86 / 0.97; 400 0.95 / 0.74 / 0.87 / 0.97;
  600 **0.27** / 0.54 / 0.88 / 0.98; 650 0.51 / 0.58 / 0.81 / 0.96; 700 **0.00** / **0.20** /
  0.83 / 0.98; 900 0.97 / 0.84 / 0.82 / 0.97; 1050 **0.27** / **0.14** / 0.81 / 0.97; 1100
  **0.00** / **0.15** / 0.87 / 0.96; 1150 **0.00** / **0.40** / 0.87 / 0.96; 1200 0.86 / **0.27**
  / 0.88 / 0.96; 1500 0.81 / **0.10** / **0.00** / 0.87; 1700 0.65 / **0.30** / FP 1306 px /
  0.97. The area ratios say what happened: in `[573,722)` and `[1020,1172)` the reference's
  interior mask is 2.4-6.8x the human's and its chassis 0.09-0.55x, i.e. the interior slot is on
  the part the human called chassis (the reported swap, now measured; visible in the sheet). The
  1172 agent correction restored the chassis by 1200 (0.86) but not the interior (0.27); at 1150,
  22 frames before it, the chassis is still 0.00. Late in the minute the reference's interior is
  a third of the human's (1500, 1700) and its rear_body at 1500 is disjoint from the human's
  588 px. Cabin 0.87-0.98 and rear_body 0.80-0.88 outside 1500 / 1700 are stable in every arm.
  The one hidden mark (rear_body 1700) was confirmed by the human on a second look (Sep 19):
  the rear body is not visible there, and the yellow piece next to it that all 15 runs put
  1.2-1.3k px on is a screwdriver. It is recorded as the named failure case
  `distractor_confusion` in the human record (`failure_case` / `note` on that cell); the
  scorer's `hidden FP` area on that cell is the distractor-confusion signal, not label noise.
  No relabel.
- **What each arm changed on the anchor cells** (|IoU| >= 0.05 against the reference).
  `xg-r720-sched`: 650 chassis 0.51 -> 0.17 (0.25x the human's area), 650 interior 0.58 -> 0.45,
  1500 interior 0.10 -> 0.15, nothing else; the `[1020,1172)` swap is untouched (chassis 0.27 /
  0.00 / 0.00 in both). `g-r720-sched`: 600 interior +0.10, 650 chassis and interior as `xg`.
  `x-r720-sched`: 1100 / 1150 chassis 0.00 -> 0.61 / 0.48 and 1100 interior +0.12, against 1150
  interior 0.40 -> 0.00 and the same 650 chassis loss: exclusivity alone moved the swap window
  toward the human and the gate undid it. Frame-0-only arms lose chassis 370 / 400 / 900
  (0.00-0.21; the 327 and 900 corrections are needed) but have chassis 0.87 / 0.52 at
  1100 / 1150 where the reference has 0.00; `g-r720-seed0`, called starved label-free (53 % /
  34 % gated), has the best outside-window score of any 720 arm (interior 1200 / 1500 / 1700
  0.89 / 0.73 / 0.68 vs 0.27 / 0.10 / 0.30). 1008 and 1280 gain at 300-400 (interior and
  rear_body +0.06-0.13), at 1500 / 1700 (chassis 0.88-0.94, interior 0.62-0.91) and, at 1280,
  chassis 1050 / 1100 / 1150 0.53 / 0.31-0.44 / 0.51; they lose 1200 interior (0.27 -> 0.00, all
  four) and, for the two `xg`, 600 chassis (0.27 -> 0.00). `ensemble-reference-v1` has no
  interior at 1050 / 1100 / 1150 (3 missing) and chassis 0.79 / 0.52 at 1100 / 1150.
  `dam4sam-60s`: chassis 700 / 1100 / 1150 / 1700 0.64 / 0.79 / 0.52 / 0.84 and rear_body 1500
  0.48, but interior 900 / 1050 0.02 / 0.00.
- **Ranking, and agreement with the label-free ranking.** `xg-r1280-sched` 0.728 >
  `off-r1280-sched` 0.724 > `dam4sam-60s` 0.717 > `off-r1008-sched` 0.705 > `xg-r1008-sched`
  0.689 > `ensemble-reference-v1` 0.681 > `x-r720-sched` 0.677 > `g-r720-seed0` 0.670 > reference
  0.668 > `g-r720-sched` 0.662 > `xg-r720-sched` 0.660 > `off-r720-seed0` 0.632 > `x-r720-seed0`
  0.613 > `xg-r720-seed0` 0.610. The four 720 schedule arms are within 0.017 (no ranking among
  them), and the direction is against the candidate. The anchors **disagree with the
  self-consistency proxies** (fewest swap episodes, no starved slot and highest IoU-with-
  reference picked `xg`; the "starved" `g-seed0` scores above it), **agree with the consensus /
  hull direction** (the eight-view `xg` pass contradicted the C10379 chassis on more frames and
  took the `[573,722)` hull IoU to 0.00; the anchors find no gain there either), and **reverse
  the resolution rule**: 720 was kept because 1008 / 1280 opened disagreement episodes outside
  the windows, and the anchors at 1500 / 1700 say those episodes were the higher resolutions
  being right about the interior (0.62-0.91 vs 0.10 / 0.30) and the chassis (0.88-0.94 vs
  0.81 / 0.65).
- **Decision on the CANDIDATE: do not adopt `xg-r720-sched`.** 0.660 vs 0.668 overall; windows
  0.825 / 0.589 / 0.532 vs 0.824 / 0.628 / 0.533; it fixed neither swap window and made frame 650
  worse. The 13 anchors settle the Sep 18 ambiguity ("fix or a different failure") as "no change
  in `[1020,1172)`, a different failure at 650". What the anchors put forward instead is the
  1280 px encoder side (`off-` or `xg-r1280-sched`: +0.06 overall, +0.11 outside the windows,
  2.8x the runtime), still only 0.59-0.66 inside the two swap windows, with DAM4SAM within 0.01
  of it and best inside the windows. Neither is adopted; a reference swap needs the seven-view
  and full-clip evidence, not 13 frames of one view.
- **Deliverables.** `runs/sam3-policy-ablation-20260918/{anchor_iou.json,anchor_iou.md,anchors_vs_reference_vs_best.png}`,
  `runs/human-review-anchors-first-minute/anchors/` (51 PNGs + `anchor_masks.json`), the README
  section "Anchor IoU (human), Sep 19" in the ablation run, the signed
  `docs/qa/first-minute-review-anchors.human-record.json` (tracked; the masks are not).
- **Tests.** `tests/test_review_anchors.py`: the config test compares everything but provenance
  and requires it filled; the table test covers the `outside` column and the sheet renderer on
  synthetic frames; new default-tier test that the committed record is signed with 51 / 1 / 0
  and the hidden cell is (1700, rear_body). Counts in the commit message.

### Sep 18: human review anchors (labeling run prepared, not labelled Sep 18; labelled Sep 19)

- **Claim boundary first.** Nothing in this section is a label. The agent prepared a labelling
  session and the tooling around it; no mask was drawn, accepted or marked hidden, and the
  workspace manifest holds zero candidates and zero hidden marks. What the human will produce
  are `human_review_anchor` masks: review evidence for scoring tracker arms against each other
  on 13 frames, not a dataset, not ground truth, and no accuracy claim (the boundary is SAM3's
  image decoder's, the choice is the human's). CC BY-NC 4.0 covers the frames and everything
  derived from them. GPU use in this step was one MuggledSAM worker process spawned by the
  calibration workspace for its live decode (1.2 GiB, `cuda:0`), started and stopped during the
  verification below; no tracking ran.
- **Anchor list** (`configs/qa/first_minute_review_anchors.json`, typed `ReviewAnchorConfig`
  in the new `review_anchors.py`, `VersionedModel` style): the focused C10379 clip
  (`assembly101_nusar_9033_four_part_reassembly_focused_g2.json`, view `static-c10379`),
  analysis frames 300, 370, 400, 600, 650, 700, 900, 1050, 1100, 1150, 1200, 1500, 1700 with
  proxy seconds `f/30` and source seconds `294 + f/30`, targets chassis / interior / rear_body /
  cabin, `expected_visible` all `visible` except the interior on the three frames inside
  `[1024,1172)` (1050, 1100, 1150), which is `hidden_prompt`: the human must answer explicitly
  there, a mask of what is visible or a hidden mark. The config carries the three review windows,
  the claim boundary and license text, and `provenance.author` / `reviewed_at` left null. The
  validator pins the clocks and that `hidden_prompt` is used exactly on the frames inside the
  declared interval.
- **Hidden affordance (added; it did not exist).** The calibration workspace could accept,
  unaccept, reject and restore masks but had no way to say "reviewed, and the part is not
  visible", so an unlabelled cell and an absent part looked the same. Added the smallest thing:
  `MuggledSAMCalibrationHiddenTarget` (`intended_target`, `frame`, `state="hidden"`,
  `marked_by="human"`) and `hidden_targets` on `MuggledSAMBoxCalibrationManifest` (default
  empty, so every existing manifest still loads; validator: one mark per cell, on a configured
  frame, clock-consistent, never coexisting with an accepted mask on the same cell);
  `Workspace.mark_target_hidden` / `clear_hidden_target` behind `POST /api/hidden-targets` and
  `POST /api/hidden-targets/clear` (browse-only frames refused, plan lock respected; accepting a
  mask on a cell removes its hidden mark, marking hidden while an accepted mask exists is
  refused with "unaccept the accepted mask"); in `app.js` a small **hidden** button in every
  not-done cell of the *Decoder candidate review* status table, a **— Hidden** state with `×`
  to clear, and hidden cells counted in `n/52 done`. Hidden marks never enter a proposal or
  correction schedule. Also `--timestamps` on `battle-muggled-calibration-web` now defaults to
  the persisted frames under `--resume` (previously the default `0,10,30,50` made resuming any
  workspace without frame 0 impossible without retyping every timestamp).
- **Workspace** `runs/human-review-anchors-first-minute/` (`battle-anchor-export prepare`): a
  plain calibration workspace whose 13 configured frames are the anchors, targets from the
  focused four-part policy, plus `anchor_session.json` naming the config and target policy it
  serves. **Frame 0 was not needed and not pre-populated**: the workspace only requires frame-0
  masks to finalize a plan, which this session never does, so `add_or_update_prompt` and
  acceptance work on every anchor frame from the first load (`preview_1100.png` shows frame
  1100 loaded as a *Calibration frame*). Nothing was derived from the 20260916t022433z
  calibration.
- **Export and scorer.** `battle-anchor-export export` reads the workspace, and for each of the
  52 cells writes `labeled` (the human-selected candidate's PNG copied byte for byte to
  `anchors/masks/f<frame>_<part>.png`, SHA-256, area, source candidate id and index), `hidden`,
  or `unlabeled`, into `anchors/anchor_masks.json` (`ReviewAnchorMaskSet`, fingerprinting the
  config and the calibration manifest) and the committed skeleton
  `docs/qa/first-minute-review-anchors.human-record.json` (`ReviewAnchorHumanRecord`: author
  and reviewed_at null, the per-cell states and SHA-256s, counts, claim boundary). Run once now
  in the unlabelled state: 0 labeled / 0 hidden / 52 unlabeled. `battle-anchor-iou --anchors
  <workspace> --run <run or arms/<arm>> ... --output <json>` scores each run at each cell:
  labeled anchor with a run mask -> IoU and area ratio (run / anchor; masks of another size are
  resized nearest and flagged); labeled anchor without a run mask -> IoU 0, `run_mask_missing`;
  hidden anchor with a run mask -> `hidden_false_positive` with its area, without ->
  `hidden_correct`; unlabeled -> skipped and counted. Report `AnchorIoUReport` plus a markdown
  table (rows = arms; columns = per-part mean IoU, overall, per-window means over the anchor
  frames inside `[279,408)` = {300, 370, 400}, `[573,722)` = {600, 650, 700}, `[1020,1172)` =
  {1050, 1100, 1150}, missing count, hidden false positives with area, scored / unlabeled).
  Run masks are read through `mask_cache.cache_for(run_dir).mask(uri)` from
  `observations.jsonl`; the twelve `runs/sam3-policy-ablation-20260918/arms/*/` directories and
  the reference run are accepted as given (smoke-run on the unlabelled set: every cell skipped,
  as it should be).
- **Verification without labelling.** Server started headlessly with the worker (URL printed,
  no window), `GET /api/state` showed `worker_online`, the four targets and exactly the 13
  frames; `GET /api/frame` extracted all 13 frames (`results/frames/frame-000300.jpg` ...
  `frame-001700.jpg`); `POST /api/workspace` accepted each anchor frame and refused frame 0 as
  browse-only; the served page (Cursor browser tab, loopback) listed 13 filmstrip buttons, `0/52
  done`, and a **hidden** button in all 52 cells, including the interior on 1050 / 1100 / 1150;
  frame 1100 loaded as *Calibration frame* (`preview_1100.png`). Server and worker were stopped
  (port 8765 free, no GPU process left); the manifest afterwards: 0 candidates, 0 hidden marks,
  0 pending prompts (only `active_proxy_timestamp_seconds` moved to 36.667 s by the frame click).
- **Tests.** `tests/test_review_anchors.py` (12): the committed config equals the builder and
  has the 13 frames with `hidden_prompt` exactly on 1050/1100/1150; config validators;
  `score_cell` on synthetic masks (perfect, disjoint, larger, missing, empty, hidden-correct,
  hidden-false-positive, unlabeled skip, resize); `score_runs` on two synthetic arms with the
  markdown table; `prepare_workspace` configures exactly the anchor frames; export with one
  accepted, one hidden and fifty unlabeled cells then scores 1.0 against a run reproducing the
  mask; hidden-target schema round trip (duplicate, wrong clock, legacy manifests without the
  field); `real_data`: the reference run as its own anchor set scores 1.0 and the xg arm reads
  through the real layout. `tests/test_muggled_calibration_web.py` +2: the HTTP hidden-mark
  lifecycle (browse-only refusal, idempotent mark, superseded by acceptance, refused while
  accepted, clear, clear-again 404 semantics) and `--resume` without `--timestamps`.
- **Next (the human's).** Label per `runs/human-review-anchors-first-minute/README.md` (~1 h),
  then `battle-anchor-export export` and `battle-anchor-iou ... --output
  runs/sam3-policy-ablation-20260918/anchor_iou.json`; the agent appends the IoU columns to the
  ablation table. `off-r720-sched` and the reference must score identically (they are byte-equal
  over the first minute), which is the built-in check that the scorer read the right masks.

### Sep 19: calibration workspace reachable over the tailnet (still not labelled)

- **Exposure, and its stance.** `battle-muggled-calibration-web` was hard-wired to `127.0.0.1`
  (`--host` existed but errored on any other value). It now accepts `--host` (default still
  loopback) and `--tailscale`, which resolves this machine's Tailscale IPv4 with
  `tailscale ip -4` and binds exactly that address, failing with a plain message when
  tailscaled is down or the CLI is missing; wildcard binds (`0.0.0.0`, `::`) are refused
  outright. The reason for the narrow shape: the workspace has **no authentication**, and a
  prompt POST runs a MuggledSAM decode on `cuda:0`, so the only remote bind offered is one that
  Tailscale's WireGuard layer already restricts to the operator's own devices. Inside the tailnet
  it stays unauthenticated: any peer can edit the manifest or trigger decodes while it runs. The
  server prints every reachable URL (numeric plus MagicDNS, `.Self.DNSName` from
  `tailscale status --json`) and a one-line warning whenever it is bound beyond loopback, flushed
  so the URL appears immediately in a log. The frontend already used root-relative URLs
  (`fetch("/api/...")`, `/static/app.js`, `/static/style.css`), so no change was needed there;
  a test now pins that no served page contains a `127.0.0.1`, `localhost`, `http(s)://` or
  `ws(s)://` literal. Verified headlessly with the anchor session: bound
  `100.64.0.7:8765`, `GET /api/state` answered via the IP and via
  `host.example.ts.net`, assets loaded through the same host, loopback refused, then server
  and worker stopped (port free, no GPU process, manifest still 0 candidates / 0 hidden). The
  zero-code alternative is documented too: `tailscale serve --bg --https=8443
  http://127.0.0.1:8765` keeps the bind on loopback and adds HTTPS (443 already carries another
  serve rule on this machine; the operator user is set, so no sudo). No browser was opened and
  nothing was labelled. Tests `tests/test_muggled_calibration_web.py` +10: real bind on the
  requested host with the printed URL, `--tailscale` binding the monkeypatched address and
  advertising MagicDNS, the down-tailscale error path, `--tailscale` with `--host` refused,
  four wildcard refusals, IPv6 bracketing, and the relative-URL sweep over the served pages.

### Sep 19: SAM3 at 1280 px on all eight views, and 1920 px on C10379 (steps 0, 1 and 1c of the follow-up plan)

- **Claim boundary first.** Every run below is the same tracker (MuggledSAM SAM3, policy off)
  at a larger encoder side; the eight-view numbers are cross-view disagreement between runs of
  that tracker, and the C10379 numbers are IoU against the 13-frame human review anchors on one
  view (review evidence for ranking arms, not a dataset, not ground truth). Seeds on the seven
  static views and e4 are last night's agent-authored geometric transfers, unreviewed; the
  1920 arm's seeds and corrections are resampled copies of masks chosen on the 720p proxy,
  unreviewed at 1080p. Nothing was swapped into any reference. CC BY-NC 4.0 applies. GPU:
  16 queue jobs, one at a time, no `NVRM`/`Xid`; no viewer opened.
- **Step 0.** The anchor calibration workspace (`battle-muggled-calibration-web --tailscale`,
  pid 3361603/3361607, and its MuggledSAM worker pid 3361702, 2.5 GiB on `cuda:0`) was stopped
  with SIGTERM (all three exited within 1 s; port 8765 free; `nvidia-smi` showed no Python
  process). The human confirmed the (1700, rear_body) hidden mark on a second look and named
  the yellow piece a screwdriver; recorded as failure case `distractor_confusion` in
  `docs/qa/first-minute-review-anchors.human-record.json` (new optional per-cell
  `failure_case` / `note` fields on `ReviewAnchorRecordEntry`, default null so earlier records
  load; `battle-anchor-export` rewrites the file without them, which the record's `notes` says),
  and the Sep 19 ledger line that called the mark "contradicted by all 15 runs" was reworded:
  the 1.2-1.3k px every arm puts on that cell is the distractor signal. Commit `30be958`;
  `tests/test_review_anchors.py` +1 (13 pass).
- **Step 1, runs** (`runs/sam3-views-r1280-20260919/`, `scripts/overnight_queue.py`, same
  `--four-part-multiview-first-minute --geometric-seed-manifest ... --checkpoint-every 300`
  command lines as Sep 18 with `--max-side-length 1280`, policy off, all-static config for the
  seven static views and the e4 config for `ego-hmc21179183`). Eight runs, 1800/1800 frames
  each, 5314-5400 masks; worker elapsed 326-352 s per view (2.5-3.0x the 117-132 s at 720),
  peak VRAM 2.39 GiB on every view (1.96 at 720), time to first output 4.2-5.6 s. Part presence
  unchanged to two decimals except C10395 chassis 0.99 -> 1.00 and e4 rear_body 0.93 -> 0.95;
  first-missing frames moved (C10095 chassis 1537 -> none, C10390 chassis 1642 -> none but
  rear_body 1230 and cabin 1079 appear, C10395 chassis 594 -> none, cabin 228 -> 105). Run
  directories `views/<VIEW>/muggledsam-sam3-four-part-multiview-first-minute-<view>-20260920t0*-r1280`;
  table in `README.md` / `views_table.md`. **Incident:** the first queue pass finished only
  C10095: a concurrent worker's live edit left `src/battle/muggled_worker.py` without `main` for
  a few minutes, the SAM3 worker of jobs 2-8 died at import (`NameError`), the smoke recorded
  `failed` and exited 0, and the queue counted the seven as succeeded in 0.4 s each (kept under
  `failed-worker-edit-20260920t0156z/`). The retry pass and the 1920 arm ran with `PYTHONPATH`
  on `code-snapshot-072fe0b/src`, a `git archive` of `src/battle` at commit 072fe0b, so working-
  tree edits cannot reach a queued GPU job; the queue's "exit 0 = success" reading of a smoke
  that recorded a failed core method is a gap to close in `overnight_queue`.
- **Step 1, consensus and hull** (`runs/multiview-part-consensus-first-minute-r1280/`, 77 s,
  reference `runs/sam3-policy-ablation-20260918/arms/off-r1280-sched`, eight explicit
  `--view-run`, `--ego-view HMC_21179183`; `runs/multiview-visual-hull-first-minute-r1280/`,
  835 s). Compared with the Sep 18 `-with-e4` roots by the new `battle-compare-multiview-builds`
  (`summary.md` / `summary.json` in both new roots; commit `024dbf9`, whose `real_data` test
  reproduces the Sep 18 policy numbers 228 frames / 99 (15) / 146 (32) / 0.396). C10379 chassis
  contradicted by the majority: 720 `[585,604)` `[643,658)` `[665,693)` `[697,702)` `[1037,1042)`
  `[1089,1163)` `[1167,1172)` `[1459,1470)` `[1499,1504)` (166 frames) -> 1280 `[296,313)`
  `[475,494)` `[508,515)` `[594,629)` `[649,722)` `[1058,1063)` `[1065,1085)` (174 frames): the
  `[1020,1172)` swap window drops from 84 contradicted frames to 25 and ends at 1085, while
  `[573,722)` grows from 66 to 106 and three short new intervals appear (296, 475, 508).
  rear_body 16 -> 37 frames (`[1662,1667)` `[1762,1794)`); cabin none in both. C10379 agreement
  0.88 / 0.98 / 1.00 -> 0.89 / 0.97 / 1.00; consensus episodes over all views 121 (C10379 14) ->
  111 (12); C10395 chassis / rear_body agreement 0.74 / 0.81 -> 0.84 / 0.93 but cabin 0.73 ->
  0.58; e4 rear_body 0.58 -> 0.63. Hull: frames with hull chassis / rear_body / cabin 1476 /
  1221 / 1585 -> 1371 / 1209 / 1726, median voxels 704 / 196 / 427 -> 450 / 207 / 779 (the cabin
  hull no longer collapses under the hand at 279-408: C10379 cabin hull IoU there 0.061 ->
  0.860); C10379 chassis hull-vs-mask median IoU overall 0.418 -> 0.420, in `[279,408)` /
  `[573,722)` / `[1020,1172)` 0.451 / 0.036 / 0.005 -> 0.465 / 0.000 / 0.466; rear_body 0.407 ->
  0.447 (0.465 / 0.408 / 0.379 -> 0.583 / 0.446 / 0.401); hull episodes 266 (C10379 43) -> 251 (33),
  the C10379 chassis list losing `[1089,1161)` and gaining `[477,493)` `[510,515)` `[699,722)`
  and three short ones before frame 200.
  Reading, label-free and consistent with the anchors: at 1280 the eight views agree with
  C10379 through the `[1020,1172)` window (the anchors gave the 1280 chassis 0.53 / 0.31-0.44 /
  0.51 at 1050 / 1100 / 1150 where 720 had 0.27 / 0.00 / 0.00) and disagree with it through
  more of `[573,722)` (the anchors gave the 1280 chassis 0.29 / 0.17 / 0.00 at 600 / 650 / 700).
- **Step 1c, 1080p proxy and 1920 arm** (`runs/sam3-c10379-r1920-20260919/`). New proxy
  `data/derived/.../C10379_rgb_294.000-386.700_1920x1080_30fps.mp4` from the local raw60 trim
  with the 720p proxy's filter and encoder settings (`fps=30:round=near`, libx264 crf 18
  medium yuv420p cfr, 12.6 s), 2781 frames, sha256 `c98180f4acd978c9ae3d8886e320e028c1cf4157fd7bdb995461f6c59628dc62`,
  frame k the same instant as the 720p proxy's frame k (downscaled |diff| 1.7-1.9 vs 2.0-3.2 for
  the neighbouring frame at five probes). Registered in
  `configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_1080p_g2.json` with the new
  `scaling_policy` literal `preserve_aspect_ratio_height_1080` (`G2PreprocessingManifest`;
  the old literal still validates) and sibling manual-seed / correction-policy configs bound to
  it. Seed plumbing: the worker verifies every mask's SHA-256 and pixel size against the
  calibration and the schedule fingerprints the calibration, the policy and the G2 config, so
  instead of touching the worker (another worker owns it tonight) `battle-rescale-calibration`
  (`calibration_rescale.py`, commit `072fe0b`, 3 tests) writes a derived calibration: all 104
  candidate masks resampled x1.5 nearest (area ratio 2.21-2.26), boxes and clicks scaled and
  re-normalised, review URIs pointing back at the source workspace, `derived_from_calibration`
  set, plus a derived schedule and a `rescale_provenance.json` sidecar; the derived schedule
  loads through `_load_multi_keyframe_correction_schedule` unchanged (seeds in order, schedule
  327 / 900 / 1172 / 1235, 1800 and 2700 dropped as out of range). Run `off-r1920-sched`
  (`arms/off-r1920-sched/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260920t024532z-r1920`):
  1800/1800 frames, 7199 masks at 1920x1080, worker 911 s (queue 1036 s; 2.6x the 1280 arm's
  348 s, 7.3x 720), peak VRAM 3.36 GiB, first output 4.5 s; the manifest's
  `multi_keyframe_corrections.schedule_fingerprint` names the derived schedule and the
  provenance sidecar is copied into the run directory.
- **Anchor IoU, 1920** (`anchor_iou.{json,md}`, `anchors_vs_reference_vs_r1920.png`; run masks
  downsampled nearest to the 720p anchors on all 51 cells, `resized_run_mask` set). `off-r1920-sched`
  0.597 / 0.530 / 0.749 / 0.960 (chassis / interior / rear_body / cabin), **0.708** overall,
  windows 0.868 / 0.558 / 0.652, outside 0.746, 0 missing, hidden FP 2726 px at 1080p (about
  1212 px at 720p, the same screwdriver). Against `off-r1280-sched` (0.724; 0.861 / 0.603 /
  0.659 / 0.764): gains chassis 370 0.72 -> 0.82 and 700 0.00 -> 0.65, interior 370 0.64 -> 0.82,
  1100 0.24 -> 0.80, 1150 0.53 -> 0.64; losses chassis 600 0.29 -> 0.00 and 1050 0.54 -> 0.09,
  interior 650 0.46 -> 0.02 and 1500 0.70 -> 0.59, rear_body 0.03-0.06 lower on nine of twelve
  visible cells. The resolution trend does not continue past 1280 on these anchors: 1920 sits
  with 1008 (0.705), below both 1280 arms (0.724 / 0.728), inside the ~0.02 band where the anchors
  do not rank. Two confounds are named, not resolved: the seeds are 1.5 px-blocky upscales, and
  the encoder side 1920 on a 1920x1080 proxy is native pixels where 1280 was a downscale.
- **Deliverables.** Commits `30be958` (step 0), `072fe0b` (proxy, configs, rescaler),
  `024dbf9` (build comparison); run roots `runs/sam3-views-r1280-20260919/`,
  `runs/multiview-part-consensus-first-minute-r1280/`, `runs/multiview-visual-hull-first-minute-r1280/`,
  `runs/sam3-c10379-r1920-20260919/`, the derived calibration
  `runs/muggledsam-sam3-four-part-focused-corrections-agent-swap-20260918t000947z-rescaled-1920x1080/`,
  each with a README or summary. Not done here, by instruction: the memory arms (step 1b), DAM4SAM
  (step 2), scoring tables across all new runs and the ensemble v2 (step 3), README and review
  guide updates.

### Sep 19: queue gap closed (a dead worker no longer counts as a succeeded job)

- **The gap.** `battle-muggled-smoke` writes the run manifest whether or not the SAM3 worker
  survived; a dead worker left `method_statuses[objects].state = failed` behind an exit code of
  0, and `scripts/overnight_queue.py` reads exit 0 as `succeeded`. That is how seven 1280 view
  jobs were reported "succeeded" in 0.4 s each on Sep 19 (`runs/sam3-views-r1280-20260919/failed-worker-edit-20260920t0156z/`).
- **The fix (commit `7380a9e`).** The driver, not the queue: `battle-muggled-smoke` now reads
  its own manifest after the run (`core_method_failure`), prints `core method failed: <blocker>`
  to stderr and exits `3` (`CORE_METHOD_FAILED_EXIT_CODE`) when the `objects` method recorded
  `failed`; the `Wrote MuggledSAM/SAM3 run: <dir>` line is still printed first so the run
  directory stays findable. The queue is unchanged: a non-zero exit is already `failed` and
  stops the queue under `continue_on_failure: false`. Nothing depended on the exit-0 behaviour:
  the two GPU tests that invoke the CLI (`tests/test_worker_policy_gpu.py`,
  `tests/test_worker_resume_gpu.py`) use `check=True` on runs that succeed, and no script chains
  the command. `battle-four-part-segmentation` (SAM2 arms) already raised on a dead worker.
  Tests: `tests/test_muggled_smoke.py` +2 (`core_method_failure` reads only the `objects` stage
  and returns the blocker; the CLI exits 3 with the failed manifest and 0 with a succeeded one,
  `run_smoke` monkeypatched). 49 pass in that file and `test_overnight_queue.py`; ruff clean.
- **Frozen code for queued GPU jobs.** Every job in tonight's two queues
  (`runs/sam3-memory-arms-20260919/jobs_1_memory_arms.json`,
  `runs/dam4sam-arms-20260919/jobs_dam4sam_arms.json`) carries
  `env.PYTHONPATH = <root>/code-snapshot-7380a9e/src`, a `git archive 7380a9e src/battle`
  extracted into the run root (83 files, byte-equal to the committed tree; `battle.__file__`
  resolves into the snapshot and the SAM3 worker is spawned from the snapshot's
  `muggled_worker.py`, the SAM2 worker from its `four_part_video_worker.py`). Working-tree edits
  during the queue cannot reach a GPU job; each root's README names the snapshot commit.

### Sep 19: SAM3 correction-memory arms on C10379 at 1280 (plan step 1b)

- **Claim boundary first.** Six runs of the same tracker (MuggledSAM SAM3, `--max-side-length
  1280`, policy off, the four-frame schedule `[327, 900, 1172, 1235]`) that change only the
  memory-bank condition, scored against the 13-frame human review anchors on one view (review
  evidence for ranking arms, not a dataset, not ground truth; arm means within ~0.02 do not
  rank). No mask was reviewed. CC BY-NC 4.0. GPU: six queue jobs, one at a time, 03:26-04:07 UTC,
  no `NVRM`/`Xid`; no viewer opened.
- **Flags (commit `68d0b49`, the flags worker; unit tests only there, first GPU use here).**
  `battle-muggled-smoke` / `muggled_worker.py` gained `--prompt-memory-semantics replace|append`
  (`append` keeps the frame-0 seed and every correction in a prompt bank, default 32 entries,
  `--max-prompt-memory`), `--keep-frame-memory-at-correction`, `--recent-first`, and
  `--drop-correction-frame FRAME` (repeatable; the frame must be one the schedule corrects;
  recorded as `cli_dropped_correction_frame_indices`). Defaults reproduce the earlier runs
  exactly. Every manifest and `runtime_settings.json` now records `correction_memory_semantics`
  (one of `replace|append_prompt_memory_and_reset|keep_frame_memory`),
  `frame_memory_position_encoding` (`within_trained_range` for <= 6 frame memories,
  `clamped_beyond_6` above; MuggledSAM's SAM3 clamps frame deltas past position 5) and a
  `correction_memory_settings` record; checkpoints store the condition and a resume refuses a
  checkpoint whose memory condition differs, while legacy checkpoints still resume into default
  runs. The `recent-first` arm the plan listed was **dropped before running**
  (`runs/sam3-memory-arms-20260919/jobs_recent_first_dropped.json`): the worker appends frame
  memories oldest-first, so `is_recent_first=True` tells the fusion model the oldest entry is
  the most recent and reverses the temporal position encoding instead of testing a memory
  policy (`runtime_settings.is_recent_first_semantics` says so in every run).
- **Runs** (`runs/sam3-memory-arms-20260919/`, `jobs_1_memory_arms.json` through
  `scripts/overnight_queue.py`, `PYTHONPATH` on `code-snapshot-7380a9e/src`; provenance of the
  first arm checked before the rest ran, and every arm's recorded semantics matches its flags).
  All 1800/1800 frames, first output 4.2 s: `drop-900` (schedule 327 / 1172 / 1235) 327 s,
  2.43 GiB, 7198 masks; `fm6` 342 s, 2.45 GiB; `pm-append` (append, 32, reset) 341 s, 2.54 GiB;
  `pm-append-keepfm` (append, 32, keep) 342 s, 2.54 GiB; `pm-append-fm6` 355 s, 2.65 GiB, 7197
  masks; `fm8` (`clamped_beyond_6`) 356 s, 2.55 GiB. Baseline `off-r1280-sched` (Sep 18) 348 s,
  2.43 GiB. Two frame memories cost ~14 s and 0.02 GiB; the 32-entry prompt bank 0.11 GiB.
- **Anchor IoU** (`battle-anchor-iou`; full table in the run README and in
  `runs/anchor-scoreboard-20260919/anchor_iou.md`). Overall / `[279,408)` / `[573,722)` /
  `[1020,1172)` / outside: `pm-append` **0.743** / 0.853 / 0.687 / 0.616 / 0.800;
  `pm-append-keepfm` **0.743** / 0.854 / 0.687 / 0.618 / 0.800; `pm-append-fm6` 0.726 / 0.858 /
  0.603 / 0.628 / 0.799; baseline `off-r1280-sched` 0.724 / 0.861 / 0.603 / 0.659 / 0.764; `fm6`
  0.722 / 0.891 / 0.602 / 0.625 / 0.762; `fm8` 0.708 / 0.892 / 0.606 / 0.560 / 0.762; `drop-900`
  0.691 / 0.861 / 0.603 / 0.628 / 0.677. 0 missing everywhere; hidden FP on (1700, rear_body)
  1224-1265 px in every arm (the screwdriver). rear_body and cabin move by <= 0.02 in every arm.
- **What moved, per cell.** `pm-append` (and `pm-append-keepfm`, within 0.01 of it on every
  cell): chassis 700 0.00 -> 0.71 and 1150 0.51 -> 0.60, interior 650 0.46 -> 0.62, 700
  0.20 -> 0.41 and 1200 0.00 -> 0.55; against chassis 1050 0.54 -> 0.00 and interior 370
  0.64 -> 0.55. Keeping the frame memory at a correction changed nothing the anchors see. `fm6`
  gains 370 (chassis 0.72 -> 0.88, interior 0.64 -> 0.85) and loses 1150 chassis 0.51 -> 0.15;
  `fm8` adds chassis 1100 / 1150 0.44 / 0.51 -> 0.00 / 0.00 to that (the clamped arm empties the
  window). `drop-900` is the baseline stream up to 900, then chassis 0.97 -> 0.32 and interior
  0.84 -> 0.20 at 900 itself, chassis 1050 0.54 -> 0.07, 1100 0.44 -> 0.56.
- **Reading.** Appending corrections to the prompt bank is the one memory change that moves
  the mean past the 1280 baseline (+0.019, from `[573,722)` +0.08 and outside +0.04, at -0.04
  inside `[1020,1172)`); the two append arms are tied. The 900-correction hypothesis is **not
  confirmed at 1280**: dropping 900 lowers `[1020,1172)` (0.659 -> 0.628) and does not recover
  the 720 seed-only number (~0.69) there. Frame memory 6 is neutral, 8 is worse. Nothing here
  is adopted; the arms enter the scoreboard for the ensemble-v2 worker.

### Sep 19: DAM4SAM and SAMURAI arms under the SAM3 run conditions (plan step 2)

- **Claim boundary first.** Runtime, VRAM and frame coverage are measurements; the IoU numbers
  are against the 13-frame human review anchors on one view (review evidence for ranking arms,
  not a dataset, not ground truth; means within ~0.02 do not rank). Frame-0 seeds are the
  reviewed human masks; the corrections are the Sep 18 agent-swap schedule. CC BY-NC 4.0. GPU:
  nine queue jobs across four passes (three smokes, three DAM4SAM arms, two SAMURAI attempts,
  one SAMURAI substitute), 04:07-04:44 UTC, one at a time, no `NVRM`/`Xid`; a 60 s `nvidia-smi`
  watchdog with a 13 GiB per-process kill threshold ran throughout and never fired
  (`runs/dam4sam-arms-20260919/nvidia_smi_vram.log`; highest sample 7.48 GiB process, 9.04 GiB
  whole GPU).
- **Wrapper knobs (commit `340f1a5`, the knobs worker; unit tests there, first GPU use here).**
  `battle-four-part-segmentation` / `four_part_video_worker.py`: one shared SAM2 predictor for
  the four DAM4SAM trackers (`src/battle/dam4sam_streaming.py` mixes a caller-supplied predictor
  and a `correct(image, mask)` method into the upstream tracker class; the DAM4SAM checkout is
  untouched); `--sam2-model tiny|large` with the large checkpoint SHA-256 pinned like the tiny
  one; `--input-size 1024|1536` through a battle-owned yaml copy (`sam2_config_source`);
  `--multi-keyframe-correction-schedule` applied through `add_new_mask` (mid-stream for
  DAM4SAM, `correction_timing: mid_stream_after_track`; pre-propagation conditioning frames for
  the offline arms); `--add-correction-to-drm` (default off) decides only whether DAM4SAM's
  `last_added` throttle and `drm_memory_additions` count a correction as an addition. The
  tiny/1024/no-schedule worker command is byte-identical to the one that produced the Sep 18
  60 s run. Manifests gain `sam2_settings`, `vram_probes` (torch counters after 30 and 300
  frames) and `vram_extrapolation` (linear slope to 1800 frames); a 300-frame smoke exits
  non-zero when the projected peak exceeds `--fail-if-extrapolated-vram-over-bytes` (12 GiB).
- **Smokes** (`runs/dam4sam-arms-20260919/smokes/`, 300 frames, correction rebased to frame 150,
  masks not scored). tiny@1024: allocated 0.63 -> 1.44 GiB between frames 30 and 300, 3.07 MiB
  per frame, projected 5.94 / 6.14 GiB (allocated / peak) at 1800, pass, 38 s. large@1024:
  1.33 -> 2.14 GiB, 3.07 MiB/frame, projected 6.64 / 6.91 GiB, pass, 74 s. large@1536 on the
  1080p proxy: SAM2 accepted `image_size` 1536 and ran all 300 frames (seeds and corrections
  resized nearest to 1080p), 1.95 -> 3.81 GiB, **7.07 MiB/frame, projected 14.17 / 14.79 GiB**,
  **gate tripped** (`ExtrapolatedVramOverLimit`, exit 1; the queue stopped as designed), 201 s.
  The full 1536 arm was **dropped: `vram_gate`**. The slope is the same for tiny and large at
  1024 (the memory bank is per frame), and the projections held: tiny 6.03 projected vs 6.07
  measured on 1800 frames, large 6.80 vs 6.80 / 6.85.
- **Full arms** (`runs/dam4sam-arms-20260919/arms/`, 1800/1800 frames each, worker clock, torch
  peak, then the nvidia-smi process maximum). `dam4sam-tiny-1024-sched-60s` 222 s, 6.07 GiB /
  6.35 GiB, 7191 masks (chassis missing from 1469); `dam4sam-large-1024-seed0-60s` 433 s, 6.80 /
  7.02 GiB, 7097 masks (interior missing from 368, rear_body from 1635);
  `dam4sam-large-1024-sched-60s` 433 s, 6.85 / 7.48 GiB, 7172 masks (rear_body from 1635). Every
  scheduled manifest records `scheduled_correction_frame_indices = [327, 900, 1172, 1235]`,
  `correction_api: add_new_mask`, `shared_predictor: true`. The shared predictor took the
  tiny 60 s torch peak from 6.81 GiB (Sep 18, four predictors, no corrections) to 6.07 GiB with
  four corrections.
- **SAMURAI with the schedule: `unsupported`.** The first job died in 5 s: the driver forwarded
  `--sam2-model large` to the worker, which refuses that knob for offline arms (they get the
  large pair through `--sam2-config` / `--checkpoint`, already resolved by the driver). Fixed in
  commit `6eed418` (`worker_flags` forwards `--sam2-model` only for `dam4sam`; test added), new
  snapshot `code-snapshot-6eed418/`. The retry loaded the large checkpoint, conditioned frames
  0 / 327 / 900 / 1172 / 1235 and died at frame 328 inside the SAMURAI checkout
  (`sam2/modeling/sam2_base.py:667`, `output_dict["non_cond_frame_outputs"][i]["best_iou_score"]`,
  `KeyError: 327`): in `samurai_mode` the motion-aware memory selection walks every earlier frame
  in `non_cond_frame_outputs`, and a conditioning frame after index 1 lives in
  `cond_frame_outputs`. Mid-stream corrections are an upstream assumption violation, not a
  wrapper bug; the checkout stays unpatched. Both failed directories are kept
  (`failed-samurai-*`). Substitute run `samurai-large-1024-seed0-60s` (no schedule, four
  independent offline streams): 537 s, 2.64 / 2.99 GiB, 7200 masks.
- **Anchor IoU** (overall / `[279,408)` / `[573,722)` / `[1020,1172)` / outside; full table in
  `runs/anchor-scoreboard-20260919/anchor_iou.md`). `dam4sam-60s` (Sep 18, tiny seed0) 0.717 /
  0.811 / 0.648 / 0.713 / 0.700; `dam4sam-large-1024-sched-60s` 0.715 / 0.808 / 0.677 / 0.684 /
  0.695; `dam4sam-tiny-1024-sched-60s` 0.700 / 0.816 / 0.638 / **0.745** / 0.619;
  `dam4sam-large-1024-seed0-60s` 0.644 / 0.690 / 0.561 / 0.692 / 0.636 (2 missing);
  `samurai-large-1024-seed0-60s` 0.634 / 0.675 / 0.603 / 0.681 / 0.587. Hidden FP on
  (1700, rear_body) 1257-1279 px in every SAM2 arm (the screwdriver).
- **Reading.** For DAM4SAM the schedule matters more than the model size: large seed-only loses
  the interior at 368 and scores 0.644, large with the four corrections 0.715. Tiny with the
  schedule (0.700) is below tiny without it (0.717): the corrections cost the chassis slot from
  1469 on (1500 / 1700 chassis 0.85 / 0.84 -> 0.03 / 0.00). DAM4SAM large with the schedule is
  tied with `off-r1280-sched` (0.715 vs 0.724) and 0.028 below `pm-append` (0.743); it has the
  best chassis mean of any arm (0.697; 1050 / 1100 0.87 / 0.80 against SAM3-1280's 0.54 / 0.44)
  and a weak interior (0.423; 1500 / 1700 0.35 / 0.16 against 0.70 / 0.85). **No DAM4SAM arm
  beats `off-r1280-sched` on `all` by more than 0.01**, so the plan's seven-view DAM4SAM pass is
  not started. Nothing is adopted.

### Sep 19: anchor scoreboard over every first-minute arm (plan step 3, scoring half)

- **Claim boundary first.** `runs/anchor-scoreboard-20260919/anchor_iou.{json,md}`: 19 arms x
  52 human review anchor cells on one view (13 frames), scored with `battle-anchor-iou`. Review
  evidence for ranking arms against each other, not a dataset, not ground truth, no accuracy
  claim; means within ~0.02 are noise and two arms within 0.01 on `all` are tied (the plan's
  rule). The ensemble v2 and any reference decision are **not** made here; README.md and the
  review guide are untouched. CC BY-NC 4.0. CPU only.
- **Table** (`IoU all`, then `[279,408)` / `[573,722)` / `[1020,1172)` / outside; rows grouped and
  sorted by `all`). References: `ensemble-reference-v1` 0.681 (3 missing), `reference` =
  `off-r720-sched` 0.668. SAM3 resolution: `xg-r1280-sched` 0.728 / 0.888 / 0.620 / 0.590 /
  0.796, `off-r1280-sched` 0.724 / 0.861 / 0.603 / 0.659 / 0.764 (tied), `off-r1920-sched` 0.708,
  `off-r1008-sched` 0.705 (tied), `off-r720-sched` 0.668. SAM3 memory arms at 1280:
  `pm-append-keepfm` **0.743** / 0.854 / 0.687 / 0.618 / 0.800, `pm-append` **0.743** / 0.853 /
  0.687 / 0.616 / 0.800 (tied, best overall), `pm-append-fm6` 0.726, `fm6` 0.722 (tied with each
  other and with the 1280 baseline), `fm8` 0.708, `drop-900` 0.691. SAM2 arms: `dam4sam-60s`
  0.717 / 0.811 / 0.648 / 0.713 / 0.700, `dam4sam-large-1024-sched-60s` 0.715 / 0.808 / 0.677 /
  0.684 / 0.695 (tied), `dam4sam-tiny-1024-sched-60s` 0.700 / 0.816 / 0.638 / **0.745** / 0.619,
  `dam4sam-large-1024-seed0-60s` 0.644 (2 missing), `samurai-large-1024-seed0-60s` 0.634. Not
  in the table: `dam4sam-large-1536-sched-1080p-60s` (`vram_gate`), `samurai-large-1024-sched-60s`
  (`unsupported`), `recent-first` (not run). The 1920 masks were downsampled nearest to the
  720p anchors (`resized_run_mask`); `off-r720-sched` and `reference` score identically (the
  scorer's built-in check).
- **Best per window.** `[279,408)` `fm8` 0.892 / `fm6` 0.891 / `xg-r1280-sched` 0.888 (tied);
  `[573,722)` the two `pm-append` arms 0.687; `[1020,1172)` `dam4sam-tiny-1024-sched-60s` 0.745
  (`dam4sam-60s` 0.713 next); outside the two `pm-append` arms 0.800, `pm-append-fm6` 0.799,
  `xg-r1280-sched` 0.796 (tied). No arm leads every window: DAM4SAM leads the `[1020,1172)`
  swap window, SAM3-1280 with the appended prompt bank leads `[573,722)` and the outside frames.
- **Sheet.** `anchors_vs_reference_vs_best.png`: human anchors | old reference (`off-r720-sched`)
  | `pm-append` (the simpler of the two tied best arms), 13 rows, per-part IoU under each run
  tile, `battle-anchor-iou --sheet ... --sheet-reference reference --sheet-best pm-append`.
- **Reading, restated from the run READMEs.** (1) Appending corrections to the prompt bank is
  the one memory change that clears the 0.01 band over `off-r1280-sched` (+0.019; from
  `[573,722)` +0.08 and outside +0.04, at -0.04 inside `[1020,1172)`). (2) `drop-900` does not
  confirm the 900-correction hypothesis at 1280: `[1020,1172)` 0.659 -> 0.628 and frame 900 lost.
  (3) DAM4SAM large with the schedule (0.715) is tied with SAM3-1280 and 0.028 below `pm-append`;
  it has the best chassis of all arms (0.697) and the weakest interior of the scheduled arms
  (0.423). No DAM4SAM arm beats `off-r1280-sched` by more than 0.01, so the seven-view DAM4SAM
  pass was not started. (4) Cabin and rear_body are unchanged across all 19 arms; every arm puts
  1.2-1.3k px on the hidden (1700, rear_body) cell. (5) VRAM: SAM3 1280 2.43-2.65 GiB, DAM4SAM
  shared-predictor 6.07-6.85 GiB torch peak (7.48 GiB nvidia-smi maximum), large@1536 projected
  14.79 GiB and gated, SAMURAI large 2.64 GiB.
- **Deliverables.** Commits `7380a9e` (smoke driver exit 3 on a failed core method), `73040c6`,
  `49b6159`, `8ee1b9f` (ledger), `6eed418` (`--sam2-model` forwarded only to the DAM4SAM worker);
  run roots `runs/sam3-memory-arms-20260919/`, `runs/dam4sam-arms-20260919/`,
  `runs/anchor-scoreboard-20260919/`, each with a README and a `code-snapshot-<commit>/` naming
  the committed code its queue ran. Next, by instruction for a later worker: ensemble v2
  (primary/fallback by anchors, fallback intervals label-free), v5 rebuild, README and review
  guide.

### Sep 20: ensemble reference v2, the named candidate (plan step 3, deciding half; closes the resolution and DAM4SAM follow-up plan)

- **Claim boundary first.** Everything below is CPU work over retained runs. The human review
  anchors are 52 cells on 13 frames of one view (C10379): review evidence for ranking arms
  against each other, not a dataset, not ground truth, no accuracy claim; arm means within
  ~0.02 are noise and two arms within 0.01 on `all` are tied. The ensemble's **arm choice saw the
  anchors**, so its anchor row is selection-biased; its **fallback intervals did not** (cross-view
  consensus only). Nothing is adopted here: ensemble v2 is a *candidate* and adoption stays the
  human's disposition. Assembly101 is CC BY-NC 4.0. No GPU job ran; no viewer was opened.
- **Anchor scoreboard** (`runs/anchor-scoreboard-20260919/anchor_iou.{json,md}`, regenerated
  with the v2 row; 20 arms x 52 cells). Top rows by `IoU all`, then `[279,408)` / `[573,722)` /
  `[1020,1172)` / outside, hidden-FP px at (1700, rear_body): `pm-append-keepfm` **0.743** /
  0.854 / 0.687 / 0.618 / 0.800, 1263; `pm-append` **0.743** / 0.853 / 0.687 / 0.616 / 0.800,
  1265; `ensemble-reference-v2` **0.743** (byte-equal to `pm-append` on every cell, see below);
  `xg-r1280-sched` 0.728; `pm-append-fm6` 0.726; `off-r1280-sched` 0.724; `fm6` 0.722;
  `dam4sam-60s` 0.717; `dam4sam-large-1024-sched-60s` 0.715 / 0.808 / 0.677 / 0.684 / 0.695,
  1279; `off-r1920-sched` 0.708; `fm8` 0.708; `off-r1008-sched` 0.705; `dam4sam-tiny-1024-sched-60s`
  0.700 (best `[1020,1172)`, 0.745); `drop-900` 0.691; `ensemble-reference-v1` 0.681 (3 missing);
  `reference` = `off-r720-sched` 0.668 / 0.824 / 0.628 / 0.533 / 0.684, 1306;
  `dam4sam-large-1024-seed0-60s` 0.644; `samurai-large-1024-seed0-60s` 0.634.
- **Findings restated, once, for the record.** (1) *Resolution:* 720 px had been rejected by the
  label-free proxies (self-consistency, consensus contradiction counts) as the run condition to
  keep; the anchors reverse that: 720 scores 0.668, 1008 0.705, **1280 0.724-0.728** (best), and
  **1920 saturates at 0.708** (tied with 1008; seeds were 1.5x blocky upscales and the encoder
  side was native, both named confounds). 1280 px is the SAM3 run condition. (2) *Memory:*
  appending corrections to the prompt bank (`--prompt-memory-semantics append`, bank 32) is the
  one memory change that clears the 0.01 band over the 1280 baseline (**+0.019**, from
  `[573,722)` +0.08 and outside +0.04, at -0.04 inside `[1020,1172)`); `drop-900` is **refuted**
  (0.691; `[1020,1172)` 0.659 -> 0.628 and frame 900 lost); `--keep-frame-memory-at-correction`
  has **no effect** the anchors see (0.743 both, within 0.01 on every cell); `fm6` / `fm8` bring
  **no gain overall** (0.722 / 0.708; they win `[279,408)` 0.891 / 0.892 and lose 1150).
  (3) *DAM4SAM fairness correction:* with one shared SAM2 predictor for the four trackers, the
  large checkpoint and the SAM3 schedule through `add_new_mask`, DAM4SAM large (0.715) **ties
  SAM3-1280** (0.724) and sits 0.028 below `pm-append`; it has the best chassis mean of all arms
  (0.697) and the weakest interior of the scheduled arms (0.423). The 1536 arm was **VRAM-gated**
  (7.07 MiB/frame slope on the 1080p proxy, **14.79 GiB projected** at 1800 frames against the
  12 GiB limit). SAMURAI with the schedule is **unsupported upstream** (`samurai_mode` indexes
  every earlier frame in `non_cond_frame_outputs`; a conditioning frame after index 1 raises
  `KeyError`), so only its seed-only large arm ran (0.634). (4) *Queue gap:* the smoke driver
  now exits 3 when the run manifest records a failed core method (commit `7380a9e`), so a dead
  worker stops the queue instead of counting as succeeded; every queued job since ran from a
  `code-snapshot-<commit>/` archive.
- **Label-free fallback intervals** (`runs/multiview-part-consensus-first-minute-r1280-pm-append/`,
  74 s; hull `runs/multiview-visual-hull-first-minute-r1280-pm-append/`, 847 s; `summary.md` in
  both compares against the `-r1280` build). The eight-view consensus rebuilt with `pm-append` as
  the C10379 reference contradicts the reference chassis on `[296,313)` `[475,494)` `[508,515)`
  `[1049,1057)` `[1058,1085)` (78 frames; the `-r1280` build had 174, every one inside
  `[573,722)` gone) and rear_body on `[1662,1667)` `[1762,1794)`; cabin never; the interior has no
  consensus (only C10379 tracks it). Rule, recorded in the config and implemented as
  `fallback_intervals_from_contradictions`: merge gaps shorter than 15 frames, then drop runs
  shorter than 10. Result: chassis `[296,313)` `[475,515)` `[1049,1085)`, rear_body `[1762,1794)`,
  interior and cabin none. The anchors were not consulted for any bound. Label-free corroboration
  of the anchor finding: C10379 chassis agreement 0.89 -> 0.94, hull-vs-mask median IoU
  `[573,722)` 0.000 -> 0.675 and `[1020,1172)` 0.466 -> 0.362, the same directions the anchors
  give `pm-append` against `off-r1280-sched`.
- **Policy v2** (`configs/ensemble_reference/first_minute_v2.json`, commit `c02a2d8`). Primary
  `pm-append` (`runs/sam3-memory-arms-20260919/arms/pm-append/...-r1280-pm-append`, 1280 px,
  policy off, schedule `[327, 900, 1172, 1235]`, append prompt bank; chosen over the tied
  `pm-append-keepfm` as the simpler condition). Fallback `dam4sam-large-1024-sched-60s`
  (`runs/dam4sam-arms-20260919/arms/dam4sam-large-1024-sched-60s`; the tiny scheduled arm wins
  `[1020,1172)` but loses the chassis from 1469 and is unsafe as a general fallback). Rules and
  sanity bounds unchanged from v1. **No hidden interval**: the v1 `hidden_agent_label` over the
  interior `[1024,1172)` is withdrawn because the human drew the interior at 1050 / 1100 / 1150.
  **Distractor interval**: rear_body `[1660,1800)` `not_contact_eligible`, `failure_case:
  distractor_confusion`, with the human's rationale from the anchor record (the yellow piece is a
  screwdriver; every arm's rear_body slot sits on it at 1700); the bounds are label-free from the
  primary's own rear_body mask, whose centroid sits on the table piece at (823,615) through 1655,
  jumps 20-50 px/frame over 1661-1668 and settles near (915,400) to the end of the minute
  without returning. No mask is substituted there. New optional schema fields, all defaulting so
  v1 loads unchanged: `selection_provenance` (arm selection by anchors with scoreboard/anchor
  URIs and the tie rule; interval selection with the consensus root, its manifest SHA-256, the
  rule parameters and the raw contradiction runs; a bias statement), interval `provenance`
  literals `multiview_consensus_contradiction` / `primary_mask_trajectory`, `failure_case`, and
  `arm` / `anchor_iou_all` on the source runs; a validator refuses a reviewed fallback interval
  once selection provenance is recorded. Tests: `tests/test_ensemble_reference.py` +4 (17 pass).
- **Build** (`runs/ensemble-reference-first-minute-v2/`, `battle-build-ensemble-reference
  --policy ... --output-root ...`, 2 min 7 s). Fallback frames actually used: **chassis 46**
  (`[311,313)` 2, `[480,494)` 14, `[512,513)` 1, `[514,515)` 1, `[1057,1085)` 28), **rear_body 2**
  (1762, 1776; the fallback's rear_body slot is mostly empty after 1635), interior 0, cabin 0;
  declined: chassis 8 at `[1049,1057)`, rear_body 2 at `[1777,1779)`. Contact-eligible: chassis
  `[0,1049)` `[1057,1200)`, interior / rear_body / cabin `[0,1200)`. Two things the human should
  know: (a) at 1049-1056 the DAM4SAM chassis (7.3k px; anchor IoU 0.87 at 1050 where `pm-append`
  has 0.00) was **declined as discontinuous** (IoU 0.07, centroid jump 73 px) against a
  last-accepted SAM3 chassis that had itself collapsed to 961 px at 1048 and was still recorded
  as sane because rule firings outside an interval are diagnostics only; the grown allowance
  reaches 75 px only at 1057. That is an engine property, not something to tune against the
  anchors; a label-free fix (do not advance the continuity anchor on a primary that fired a
  rule) is a follow-up. (b) Inside the substituted frames the DAM4SAM chassis **overlaps the
  SAM3 interior** (IoU 0.41-0.59 over `[480,494)` / 512 / 514, 0.54-0.82 over `[1057,1085)`):
  the two sources do not agree on where the interior ends, and the sanity bounds test a
  substitute only against its own target's history. The episode check reports it (22 -> 24
  segmentation episodes, two new chassis/interior swap episodes at 480-493 and 512-514) rather
  than hiding it; a symmetric overlap bound was considered and rejected because it would also
  decline `[1057,1085)`, where the anchors say the DAM4SAM chassis is the right one.
- **Anchor score of v2.** `ensemble-reference-v2` **0.743** / 0.853 / 0.687 / 0.616 / 0.800,
  0 missing, hidden FP 1265 px, identical to `pm-append` on all 52 cells: the label-free
  intervals contain exactly one anchor frame (1050, declined), so no substituted frame is scored
  and the anchors cannot tell the ensemble from its primary. Against the old reference (0.668):
  +0.075 overall, `[573,722)` 0.628 -> 0.687, `[1020,1172)` 0.533 -> 0.616, hidden FP 1306 -> 1265
  px (the screwdriver, unchanged in kind). The table states the selection bias in a footnote and
  the README of the scoreboard root carries the addendum; a second sheet
  `anchors_vs_reference_vs_ensemble_v2.png` shows anchors | old reference | v2.
- **v5 review package** (`runs/interaction-review-first-minute-v5/`, 48 s, all layers;
  `battle-build-interaction-review-v4 --output-root ... --reference runs/ensemble-reference-first-minute-v2
  --multiview-consensus runs/multiview-part-consensus-first-minute-r1280-pm-append`; the
  `--multiview-consensus` flag is new so the multiview layer can follow the primary run; the v4
  package is untouched). Against v4 (ensemble v1): contact rows observed 7,598 -> 7,820,
  `invalid_mask` 5,178 -> 4,816, `missing_hand` 1,624 -> 1,764, debounced candidates 3,265 ->
  3,417, contact events 168 -> 185, segmentation triggers 183 -> 206, episodes 35 -> 43, pinned
  moments 56 -> 62, hand disagreements 3,599 both; multiview layer 8 -> 9 views, reference
  contradicted frames 262 -> 115.
- **Review metrics** (`battle-review-metrics --v4-index <package index> --output-root ...`; the
  existing `review-metrics-first-minute-v2` measured a Sep 17 v4 index, so v4 was re-measured
  into `runs/review-metrics-first-minute-v4-current/` and v5 into `runs/review-metrics-first-minute-v5/`,
  ~80 s each). v4 -> v5: episodes 187 -> 185; `mask_area_anomaly_vs_median` 26 -> 18,
  `segmentation_identity_swap` 2 -> 4 (the two fallback overlaps above plus 1048-1084, score 2.7),
  `segmentation_label_crossing` 1 -> 2, `mask_growth_hand_capture_suspect` 34 -> 37, contact
  intervals 138 -> 148 (sub-5-frame 47 both; median 10 both; p90 55.0 -> 53.6; max 297 -> 323),
  hand gaps 13 / 79 frames both. Part-area medians chassis 7369 -> 7054, interior 2503 -> 2872,
  rear_body 2139 -> 2079, cabin 16451 -> 16117. Top of the v5 rank: chassis growth 1057-1071
  (18.6x, the substitution restoring the chassis), rear_body collapse 1748-1761 (11.4x, the
  screwdriver), chassis/interior swap 1048-1084 (2.7x), rear_body appearance leakage 1747-1778,
  interior growth 1044-1047. The chassis collapse anomaly over `[572,695)` persists at 4.0x
  (v4 5.9x): the `pm-append` chassis is still undersized there even though the other views no
  longer contradict it.
- **Candidate, named.** `runs/ensemble-reference-first-minute-v2/` built from
  `configs/ensemble_reference/first_minute_v2.json` is the candidate reference; the v4 builder's
  default stays the v1 ensemble until the human disposes. The review guide
  `docs/review-guide-2026-09-20-ensemble-v2.md` gives the `rerun` command for v5, the three
  numbers and the frames to scrub.
- **Deliverables.** Commits `c02a2d8` (policy v2, schema, tests), `a8362a8` (the
  `--multiview-consensus` flag) and the docs commit that follows (ledger, README, review
  guide). Run roots (ignored):
  `runs/multiview-part-consensus-first-minute-r1280-pm-append/`,
  `runs/multiview-visual-hull-first-minute-r1280-pm-append/`,
  `runs/ensemble-reference-first-minute-v2/`, `runs/interaction-review-first-minute-v5/`,
  `runs/review-metrics-first-minute-v5/`, `runs/review-metrics-first-minute-v4-current/`, and the
  regenerated `runs/anchor-scoreboard-20260919/`. Plan todos `r-ensemble-v2` and `r-docs`
  completed; the extended experiments (E2-E9) stay pending.

### Sep 20: detector scorecard against the 52 review anchors (Track A of the multicam plan)

- **Claim boundary first.** Truth here is the human review anchors: 13 frames x 4 parts of one
  view (C10379), review evidence and not ground truth. A cell is "failed" when the scored run's
  anchor IoU is below 0.5 (a missing run mask is IoU 0) or, on the one hidden cell (1700,
  rear_body), when the run has more than 300 px there (it does in every arm: the screwdriver).
  Failure counts are **9 / 52** on `pm-append`, **14 / 52** on the old reference
  (`off-r720-sched`, Sep 18) and **11 / 52** on `off-r1280-sched`; per class
  (`occlusion_leak` = frames 300/370/400 in `[279,408)`; `rotation_swap` = 600/650/700 and
  1050/1100/1150 in `[573,722)` / `[1020,1172)`; `distractor` = 1200/1500/1700; `outside` = 900)
  the failures are 0 / 7 / 2 / 0, 0 / 9 / 5 / 0 and 0 / 8 / 3 / 0. So **no anchor cell fails
  in the occlusion-leak window or at 900 on any of the three runs**: those classes have no
  positives and every per-class AUROC / precision / recall there is undefined, reported as `-`.
  No anchor frame lies in the 1235 correction region. Counts this small give wide intervals;
  every table carries the counts (TP/FP/FN/TN), not only rates. CPU only; no GPU job ran.
- **Tool.** `battle-detector-scorecard` (`src/battle/detector_scorecard.py`, tests
  `tests/test_detector_scorecard.py`; schemas `DetectorScorecardManifest`,
  `DetectorConfidenceRow`, `ProposedAnchorFrames` appended to `schemas.py`). Ten detectors, each
  a per-frame per-part scalar from existing artifacts only (higher = more suspicious):
  `sam3_score` = 1 - sigmoid(SAM3 `object_score`); `area_jump` = |area_t / median(area over the
  previous 15 frames) - 1|; `area_vs_seed` = |log((area_t+1)/(area_0+1))|;
  `tracker_disagreement_large` / `_tiny` = 1 - IoU with the `dam4sam-large-1024-sched-60s` /
  `dam4sam-tiny-1024-sched-60s` mask on the same frame; `consensus_error_px` (C10379's error in
  the eight-view consensus `per_frame`, 0 inside its own mask) and `consensus_contradicted`
  (inside a majority-contradicting episode); `hull_disagreement` = 1 - hull-vs-mask IoU from
  `hull_series.npz` and `hull_episode`; `overlap_other_parts` = fraction of the part's pixels
  shared with another part's mask. Consensus and hull are undefined for the interior (no other
  view tracks it) and where no hull exists: 13 and 18 of the 52 cells, excluded from those
  detectors' counts. Each run is scored against the consensus / hull built on it
  (`-r1280-pm-append`, `-r1280`, and the Sep 18 `first-minute` pair for the reference). Scoring:
  rank AUROC (Mann-Whitney, ties half), precision / recall at the F1-maximising threshold and at
  the highest threshold with recall >= 0.8, overall and per class; the combination is the mean
  of the top-3 detectors' normalized ranks over the full 1800 x 4 series, scored in-sample and
  leave-one-frame-out (top-3 and both thresholds chosen on 12 frames, the 13th scored, rotated).
  36 s per run, ~21k mask decodes.
- **Which detectors notice a failure** (overall AUROC on 52 cells; R>=0.8 point as
  P / R (TP/FP/FN/TN)). `pm-append` (9 failed): `sam3_score` **0.960**, 0.62 / 0.89
  (8/5/1/38); `area_vs_seed` 0.907, 0.50 / 0.89 (8/8/1/35); `area_jump` 0.866;
  `tracker_disagreement_large` 0.848, 0.36 / 0.89 (8/14/1/29); `consensus_error_px` 0.832 on 39
  cells, 0.50 / 0.80 (4/4/1/30); `tracker_disagreement_tiny` 0.819; `overlap_other_parts`
  0.784; `hull_disagreement` 0.717 on 34 cells; the two episode booleans 0.585 / 0.575. Old
  reference (14 failed): `hull_disagreement` **0.976** on 34 cells, 1.00 / 0.83 (5/0/1/28);
  `sam3_score` 0.942, 0.75 / 0.86 (12/4/2/34); `area_vs_seed` 0.921, 0.75 / 0.86 (12/4/2/34);
  `tracker_disagreement_tiny` 0.883; `consensus_error_px` 0.882 on 39 cells, 0.75 / 0.86
  (6/2/1/30); `area_jump` 0.880; `consensus_contradicted` 0.842; `overlap_other_parts` 0.797;
  `tracker_disagreement_large` 0.776; `hull_episode` 0.708. `off-r1280-sched` (11 failed):
  `hull_disagreement` **0.983** on 34 cells, 0.67 / 1.00 (4/2/0/28); `sam3_score` 0.911,
  0.56 / 0.82 (9/7/2/34); `area_vs_seed` 0.878; `tracker_disagreement_large` 0.831; `_tiny`
  0.830; `hull_episode` 0.825; `consensus_error_px` 0.773; `area_jump` 0.772;
  `consensus_contradicted` 0.735; `overlap_other_parts` 0.694. Reading: **SAM3's own object
  score is the one detector that is top-2 on all three runs** (0.91-0.96) and the only one
  defined on every cell; the seed-area drift is next (0.88-0.92). The hull disagreement is
  near-perfect where it is defined on the two weaker runs but 0.72 on `pm-append`, and it is
  blind to the interior, where 5 of `pm-append`'s 9 failures are. The consensus is a middling
  detector here (0.77-0.88) and its episode boolean is weak (0.59-0.84): it fires on the
  chassis in `[1049,1085)` but not on most anchor-visible failures. The DAM4SAM cross-tracker
  check is 0.78-0.88 with 13-22 false positives at the recall floor: the two trackers disagree
  in many places where the anchors say SAM3 is fine.
- **Per class** (AUROC; classes with positives only). `rotation_swap` (24 cells): `pm-append`
  `sam3_score` 0.98, `consensus_error_px` 0.98 (18 cells), `tracker_disagreement_large` 0.96,
  `area_vs_seed` 0.91, `hull_disagreement` 0.64; reference `area_jump` 0.98, `hull_disagreement`
  0.97, `consensus_error_px` 0.97, `tracker_disagreement_tiny` 0.96, `sam3_score` 0.92;
  `off-r1280-sched` `hull_disagreement` 1.00, `hull_episode` 0.92, `tracker_disagreement_large`
  0.90, `sam3_score` 0.87. `distractor` (12 cells, 2-5 failed): `hull_disagreement` 1.00 on all
  three (8 cells; the rear_body screwdriver contradicts the hull), `sam3_score` 0.90 / 1.00 /
  1.00, `area_vs_seed` 0.95 / 0.89 / 1.00, while the DAM4SAM disagreement is at or below chance
  (0.35-0.80: DAM4SAM sits on the same screwdriver) and `overlap_other_parts` 0.37-0.57.
- **Combined and the honest number.** Top-3 by AUROC: `pm-append` = `sam3_score`,
  `area_vs_seed`, `area_jump`; reference and `off-r1280-sched` = `hull_disagreement`,
  `sam3_score`, `area_vs_seed`. In-sample the rank-average scores AUROC 0.941 / 0.955 / 0.942
  and, at its own R>=0.8 threshold, P / R 0.53 / 0.89 (8/7/1/36), 0.75 / 0.86 (12/4/2/34) and
  0.75 / 0.82 (9/3/2/38). **Leave-one-frame-out** the same procedure gives pooled AUROC
  0.835 / 0.927 / 0.885 and, at the recall-floor threshold chosen on the other 12 frames,
  **P / R 0.36 / 0.44 (4/7/5/36), 0.62 / 0.57 (8/5/6/33) and 0.58 / 0.64 (7/5/4/36)**; at the
  F1-max threshold 0.43 / 0.33, 0.53 / 0.57, 0.55 / 0.55. The ranking transfers (AUROC drops
  0.03-0.11); the **threshold does not**: chosen at the lowest training positive it is brittle on
  13 frames, and the in-sample recall of 0.82-0.89 falls to 0.44-0.64 held out. The detector
  selection itself is stable (the same top-3 set in 9 / 11 / 11 of 13 folds). This is the number
  Track B gates should be set from, not the in-sample one, and it says the detector today is a
  ranker, not yet a calibrated gate.
- **Confidence series.** `runs/detector-scorecard-20260920/<run>/confidence.jsonl` (7,200 rows =
  1800 frames x 4 parts: every detector value, the combined suspicion, `confidence` = 1 -
  suspicion, `abstain` = confidence <= the in-sample R>=0.8 point: 0.187 / 0.213 / 0.159),
  `scorecard.{md,json}` and `confidence_timeline.png` (per part, anchor frames marked, red =
  failed). Abstentions: `pm-append` 1,022 of 7,200 rows (chassis 277, interior 484, rear_body
  261, cabin 0), reference 1,319, `off-r1280-sched` 833. The series is frame-noisy (no
  smoothing); it is the raw input for a gate, not the gate.
- **Proposed anchor frames for the human** (`pm-append/proposed_anchor_frames.json`; multiples
  of 10, none within 20 frames of an existing anchor, proposals at least 20 apart, seed
  20260920). Detector-ranked, by the most suspicious part: **220** interior (combined 0.99,
  seed-area drift 2.06), **1750** rear_body (0.98, the screwdriver span), **480** chassis (0.96,
  area jump 0.61; the consensus also contradicts `[475,494)`), **1540** rear_body (0.96),
  **550** chassis (0.96), **500** interior (0.94), **1650** interior (0.93, SAM3 score 0.01),
  **1730** rear_body (0.92). Random: **40, 80, 860, 1410, 1770**. The ranked frames test the
  detectors' claims (three fall in the never-anchored `[475,560)` region where the consensus
  flagged the chassis); the random ones test the detectors rather than confirm them. Mapping
  these to other views is another worker's step.
- **Not done / undefined.** No per-class threshold is meaningful with 2-9 positives; the per-class
  tables use class-local thresholds and a second table shows every detector's per-class counts at
  the overall threshold. `occlusion_leak` and `outside` have no failed cell on any run, so the
  scorecard says nothing about detecting the 279-408 shape leak the human reported (the anchors
  score those frames >= 0.5 on every arm). The 1235 region has no anchor. Interior cells have no
  consensus or hull signal by construction until the other views track the interior (Track B).
  Deliverables: the module, tests, schemas, this entry; run roots under
  `runs/detector-scorecard-20260920/` (ignored).

### Sep 20: `battle-multiview-reprompt`, the consensus re-prompt loop (B4 of the multicam plan; tool built and unit-tested, GPU arms not run)

- **Claim boundary first.** Every correction this tool writes is agent-authored
  (`selected_by: agent`, provenance `multiview_consensus`); no human reviews any of it. The
  consensus it acts on measures disagreement between views of one tracker seeded by geometry,
  not accuracy, and a candidate it accepts agrees with the other cameras, which does not make
  it right. CPU only today: `plan` ran on the real consensus into a scratch directory (9 s) and
  `decode` ran only against a stub decoder in the tests; the `decode` and `run` commands wait
  for the GPU owner. CC BY-NC 4.0 applies to the dataset assets.
- **Tool** (`src/battle/multiview_reprompt.py`, `battle-multiview-reprompt {plan,decode,run}`;
  schemas appended to `schemas.py`: `MultiviewRepromptDetectorConfig`, `MultiviewRepromptPrompt`,
  `MultiviewContradictionOnset`, `MultiviewRepromptPlan`, `RepromptCandidateScore`,
  `RepromptDecision`, `MultiviewConsensusCorrectionProvenance`, `MultiviewConsensusProvenanceFile`,
  `MultiviewRepromptDecisions`, `MultiviewRepromptRunCommand`). `plan` reads a consensus root
  (`manifest.json`, `consensus_points.npz`) and the target view's tracked run, reuses the
  builder's `episodes_from_errors` on the target's per-frame error, keeps episodes that
  contradict the majority, merges runs closer than 15 frames and drops merged runs shorter than
  10 (the ensemble v2 fallback rule), and gates each onset on >= 3 static views agreeing with the
  consensus at that frame (the builder's `used/` arrays, so ego views enter only behind its pose
  gate). Defaults: error > 40 raw px, >= 5 frames, 30 px agreement, 3 static views;
  `--detector-config <json>` replaces any of those keys with Track A's calibrated values and
  fingerprints the file into the plan. At an onset the consensus point becomes a sphere whose
  radius is the median of the other views' mask equivalent-circle radii at depth, projected into
  the target as two square boxes (margins 0.25 / 0.60) with negative points at the other parts'
  projected consensus centroids (inside the box grown by 0.5) and, with `--hand-negatives`, the
  dataset hand joints. `decode` runs every prompt through one warm calibration-worker
  (`batch_decode` per onset frame), scores each candidate by the seed-transfer rule (area within
  [0.3, 3.0] x the other views' expected area carried by the squared focal/depth ratio, centroid
  ray within 1.5 radii; ranked by ray distance, then decoder IoU), derives a calibration from the
  source run's (`derive_calibration`, hard-linked results, `derived_from_calibration`), appends
  every decoded candidate as `selected_by: agent` (only the accepted one
  `selected_for_correction`), and writes two schedules bound to one manifest hash:
  `multi_keyframe_correction_schedule.json` (human-plus-consensus) and
  `multi_keyframe_correction_schedule.consensus-only.json` (frame-0 seeds plus consensus
  corrections). Source corrections at frames >= 1800 are left out of both (the first-minute run
  never reaches them; the v4 policy's 8-per-target budget would otherwise be spent on 1800 and
  2700) and listed as `source_corrections_dropped_out_of_range`; an accepted candidate that
  would exceed the policy budget is recorded as rejected with `policy_keyframe_limit`.
  `reprompt_decisions.json` holds every accept/reject and reason, `reprompt_provenance.json`
  every correction's consensus fingerprint, onset, views used, target error, accepted score,
  rejected alternatives and iteration; `agent_acceptances.jsonl` logs them with provenance
  `multiview_consensus`, mirroring the visual-review tool. `run` emits the
  `battle-muggled-smoke` command per arm (`--four-part-static-focused --max-frames 1800
  --max-side-length 1280 --prompt-memory-semantics append --checkpoint-every 300`) and writes
  `run_commands.{sh,json}`.
- **Consensus-only arm.** Its schedule simply lacks the human-loop corrections at 327 / 900 /
  1172 / 1235 (frame-0 seeds plus consensus corrections), rather than applying
  `--drop-correction-frame`: a frame drop would also remove a consensus correction landing on a
  human frame and errors when the frame is outside the run's budget, while the schedule's
  fingerprint integrity (`_load_multi_keyframe_correction_schedule`) is identical either way. The
  tracker's own loader was run on both derived schedules in the tests and reports
  `agent_selected_correction_frame_indices` = the onsets.
- **Checkpoint/resume.** A checkpoint at `k` holds the state ready to step `k` (frames `[0, k)`
  and the corrections before `k`), so an arm may resume there only when every source
  correction before `k` is also in its schedule; the consensus-only arm cannot resume past 327.
  Two findings: (a) with the current consensus the earliest onset is 296, before every source
  checkpoint, so both C10379 arms are full runs regardless; (b) the worker's `stream_identity`
  hashes the *whole* `multi_keyframe_schedule_json`, so a resume under a schedule with added
  later corrections would be refused as "a different stream" even when the prefix is identical.
  `run` therefore emits full runs and keeps the qualifying resume in `resume_argv`
  (`--prefer-resume` swaps it in); scoping the identity to the corrections before the resume
  frame is a worker change for the GPU owner, not made here.
- **Reference dependency.** The default consensus has `pm-append`, with the human corrections,
  as a member, so consensus points near frames where C10379 agreed with the majority partly rest
  on the corrected masks; the plan and every provenance record say so
  (`reference_dependency_note`). `battle-build-multiview-part-consensus --exclude-view C10379`
  (new, minimal) builds the consensus from the other eight views, and `plan --consensus-run`
  then computes C10379's error itself with the builder's `distance_to_mask_px`. What remains
  either way: the other views' frame-0 seeds came from the C10379 human masks. The loop removes
  the human corrections, not the human seeds.
- **Plan on the real consensus** (scratch run, 9.4 s; the README commands write under
  `runs/multiview-reprompt-20260920/C10379/iter1/`): chassis onsets 296 `[296,313)` (error 43
  px at onset, 6 static views, radius 44 mm, expected area 10.4k px), 475 `[475,515)` (42 px, 7
  views, 51 mm, 11.0k px) and 1049 `[1049,1085)` (121 px, 7 views, 50 mm, 12.3k px), two prompts
  and 2-3 negatives each; rear_body 1762 `[1762,1794)` blocked because only C10119, C10390 and e4
  agree there (2 static views). The same three chassis runs are the ensemble v2 fallback
  intervals. Two cautions for reading the arms: the C10379 chassis mask is ~5k px against an
  expected 10-12k (the sphere model over-predicts a grazing camera's footprint; ratio ~0.5,
  inside the band but worth remembering), and at 296 and 475 the onset error is 42-43 px, barely
  over 40, with the consensus centroid ~45 proxy px above the mask centroid, which is about one
  part radius and could be grazing-view geometry rather than a tracking failure; the detector
  scorecard's threshold is what should decide, through `--detector-config`.
- **Iteration.** `plan --previous-plan <iter k> --source-run <corrected run>` on a consensus
  rebuilt with the corrected run as reference gives iteration `k+1`; the index is in the plan,
  the decisions and every provenance record, the derived calibration carries the earlier
  consensus corrections forward through `reprompt_provenance.json`, and the loop refuses
  iteration 4.
- **C10119 as target.** `plan` and `decode` work (decisions and masks are written); no schedule
  can be derived because the C10119 run is a `--four-part-multiview-first-minute` run, whose
  profile takes only its geometric seed manifest, the policy and schedule `view_id` contracts
  still name the three original views, and there is no calibration record to derive from.
  `run` emits its command as `BLOCKED` with that reason; the README lists it.
- **Tests.** `tests/test_multiview_reprompt.py`, 12 default-tier tests: onset detection on a
  consensus fixture (gap merge, short-run drop, static gate), detector file override, sphere
  projection on the synthetic rig (radius recovered within 10 %, boxes contain the centroid,
  negatives placed), the acceptance rule (area band and centroid ray beat a higher decoder
  score), decode with a stub decoder writing real PNGs (decisions, derived calibration, both
  schedules rebound to the manifest hash, out-of-range source correction dropped, the tracker's
  own schedule loader accepting both, provenance sidecar and acceptance log), the policy budget,
  the no-schedule path, resume-frame choice per arm, command emission for both arms with resume
  variants, and the iteration cap; one `real_data` test runs `plan` on the pm-append consensus
  and checks the three onsets. Default tier 566 passed / 9 skipped; ruff clean.
- **Deliverables.** The module, schemas, tests, the `--exclude-view` flag, the CLI entry, this
  entry, and `runs/multiview-reprompt-20260920/README.md` (ignored) with the exact plan / decode /
  run / anchor-iou commands for C10379 and the blocked C10119 set. Nothing on the GPU was run.

### Sep 20: Track C1 of the multicam plan, a second recording fetched and prepared (`nusar_9061`)

- **Claim boundary first.** Everything here is acquisition and dataset context: the second
  recording's videos (trimmed windows only), the dataset's own poses, extrinsics, coarse and
  fine-grained labels, fitted intrinsics that reproduce the dataset's internal projection, and
  measured video-vs-pose clock offsets. No model ran, no GPU was used (`CUDA_VISIBLE_DEVICES=""`
  throughout), nothing here is ground truth for any method and no accuracy claim rests on it.
  CC BY-NC 4.0 attribution applies to every dataset asset named below.
- **Choice** (`configs/assembly101/nusar_9061/recording_selection.json`). Toy id `c02a` comes
  from the fine-grained CSV `toy_id` column of recording 1 (and the recording name). The
  coarse-label set lists four assembly recordings of `c02a`; `nusar-2021_action_both_9061-c02a_9061_user_id_2021-02-09_141537`
  (subject 9061) is the only different-subject recording whose coarse actions contain the
  recording-1 chain attach interior -> screw chassis -> attach body -> screw chassis inside
  one 60 s span (388.1-415.2 s) and whose ego cameras are the same headset serials as
  recording 1. Rejected: `9033-..._2021-02-18` (same subject), `9084-...` (attach interior and
  attach body 52 s apart with cabin work between; different headset serials), and the earlier
  9061 span 300-360 s (body only "attempted", roof attach in the middle). Window fetched:
  source **374.000-454.000 s** (raw frames `[22440, 27240)`, 4,800 trim / 2,400 proxy frames),
  core span **384.000-444.000 s = proxy frames `[300, 2100)`**, 10 s margin each side. Named
  differences from recording 1 that make it a real test: the subject is about twice as fast
  (the chain takes 27 s), a fifth part (rear bumper) is handled and attached inside the window,
  the roof was attached to the cabin before the window, and proxy frame 0 sits inside
  `unscrew chassis` (all four parts are separate from about proxy frame 75, fine-grained `put
  down chassis` 56-75, until `pick up interior` at 422).
- **Registry** (new `assembly101_recordings.py`, tracked `configs/assembly101/recordings.json`).
  One typed `Assembly101Recording` per recording carries the window, core span, views, and
  every path the Track 0 tools used to hard-code; `RECORDING_1` is defined in code so all
  existing constants (`RECORDING_ID`, `FOCUSED_START_SECONDS`, `WINDOW_START_POSE_FRAME`,
  `CLOCK_RULES_CONFIG`, `CONFIG_ROOT`, ...) are unchanged and every module keeps its
  recording-1 default; `--recording <label|id>` selects another. `assembly101_fetch_view`,
  `assembly101_clock_offset`, `assembly101_camera_fit`, `assembly101_reference` and
  `multiview_geometry.CameraRig.load` / `run_rig_check` took `recording=` parameters; a
  clock-rule file or scan for the wrong recording is refused (`load_clock_rules_for`,
  `write_clock_rules`).
- **Acquisition.** Videos by `battle-fetch-assembly101-view --recording nusar_9061` (Track 0
  recipe: signed CDN URL, counting proxy, HTTP Range, one decode -> 60 fps trim + 1280x720 /
  954x720 CFR-30 proxy, libx264 crf 18, ffmpeg 8.1.2): eight static views plus HMC_21110305
  (e3) and HMC_21179183 (e4), chosen by camera id because the Track 6 visibility audit needs
  hulls that do not exist yet and recording 2 carries the same headset serials.
  **1,870,659,584 B** for the videos (14-20 % of each file; per view 155-332 MB static, 17-20
  MB ego; 22-68 s each, three in parallel, about 4 min wall). Poses and annotations by the new
  `battle-fetch-assembly101-poses` (port of the Sep 17 scripts): ten `AssemblyPoses.zip`
  members range-extracted with CRC-32 checks (281,494,021 compressed B; landmarks2D 682 MB
  uncompressed), the three fine-grained split CSVs streamed with the full-file SHA-256 equal to
  the LFS etag (3,516 rows for the recording = 293 segments x 12 views, all in `train`), the
  coarse labels and four lookup tables with git-blob etags verified: **465,555,825 B**. Total
  **2,336,215,409 B** (budget 2-3 GB). Video full-file checksums are the LFS etags recorded in
  the clip configs (the files were never downloaded whole, so they cannot be re-hashed here).
  The 2D-landmark window `assembly101_landmarks2D_60fps_frames_22440_27240.npz` covers all 12
  views with no missing pose frame. On disk: 0.83 GB raw, 1.8 GB derived (ignored).
- **Clip configs** (`G2PreprocessingManifest`, built from the acquisition records by
  `write_recording_clip_configs`): `configs/clips/assembly101_nusar_9061_four_part_reassembly_focused_all_static_g2.json`
  (eight views, asset C10379, source 374.000-454.000 s, raw `[22440, 27240)`, analysis
  `[11220, 13620)`, proxy `[0, 2400)`, raw sources as `hf://` paths with the LFS etag) and
  `..._ego_hmc_21110305_g2.json` / `..._ego_hmc_21179183_g2.json` (954x720). Gates read
  `approved_by: user (Track C plan of Sep 20 ...)`: the plan, not a per-clip human look.
- **Clock rules** (`battle-assembly101-clock-offset --recording nusar_9061`, three chunks of
  1,600 frames since the trim is 80 s, otherwise the Sep 18 method; per-view curves under
  `runs/assembly101-clock-offsets-nusar_9061/`, tracked summary
  `configs/assembly101/clock_rules_nusar_9061.json`, 10/10 views with a rule). C10095 +5
  (+-1, sub-frame +4.79), C10115 +6 (+-1, +5.66), **C10118 +5 (+-2, +4.94)**, C10119 +4 (+-1,
  +4.48), **C10379 +6 (+-2, +5.81)**, C10390 +6 (+-1, +5.56), **C10395 +4 (+-2, +4.22)**,
  C10404 +5 (+-1, +5.13), HMC_21110305 0 (+-1, +0.20), HMC_21179183 0 (+-1, -0.08). The three
  bold views were **ambiguous under the Sep 18 rule**: their gradient metric sat 2.1-7 frames
  from skin-hit and motion, which agreed with each other within 0.2-0.8 frame (C10379: skin
  +5.41, motion +6.20, gradient +1.55 with one flat chunk; C10395: +4.90 / +3.53 / -2.12;
  C10118: +5.05 / +4.82 / +6.96). One documented rule change, `decide_offset` majority
  fallback: when three metrics vote and exactly one is the clear odd one out (the remaining
  pair agrees within 2 frames and beats any alternative pair by more than 0.5 frame), the pair
  decides, the dropped metric is named (`dropped_metric`, in the scan, the rule file and the
  evidence text) and the uncertainty is floored at 2 frames. Two-metric disagreement and evenly
  spread metrics stay ambiguous; no recording-1 decision changes (none was ambiguous). The
  offsets differ from recording 1's (C10379 +9 -> +6, C10119 +7 -> +4): the lag is a
  per-session start-time fact, which is why it is measured per recording.
- **Cameras** (`battle-fit-assembly101-camera --recording nusar_9061 --all`, 96 frame groups
  every 50 pose frames; `configs/assembly101/nusar_9061/<view>_camera_estimate.json`). Static
  Brown fits reproduce the shipped 2D through the shipped pose at 0.0004-0.0015 px RMS
  (C10095 0.00071, C10115 0.00039, C10118 0.00039, C10119 0.00104, C10379 0.00154, C10390
  0.00113, C10395 0.00135, C10404 0.00066; max 0.0021), ego rational fits 0.165 px
  (HMC_21110305, max 0.39) and 0.398 px (HMC_21179183, max 0.60). **Same physical rig as
  recording 1, re-calibrated per session:** relative poses between static cameras agree with
  recording 1 to 0.6-4.2 mm and 0.17-0.27 deg over 1.0-2.0 m baselines, the world frame moved
  by one rigid transform (0.52 deg, 59 mm; per-camera residual 0.5-2.0 mm), focal lengths agree
  within 1.7 px and principal points within 1-3 px (C10395 5.4 / 3.7 px), k1 within 0.003. The
  ego intrinsics agree in focal length (0.6-0.7 px) but the rational coefficients differ
  substantially (the model is ill-conditioned; only the projection is comparable). Consequence:
  recording 1's camera files must not be reused for recording 2 (extrinsics differ); the
  per-recording files are what the rig loads. Rig check (`battle-multiview-rig-check
  --recording nusar_9061`, `runs/assembly101-multiview-rig-check-nusar_9061/report.json`,
  1.9 s): eight-static triangulation median 0.0003 mm / p95 0.0004 mm over 8,064 points (rec 1:
  0.0004 / 0.0004); C10379+C10395 0.0016 / 0.0044 mm; C10115+C10404 0.0007 / 0.0008 mm; statics
  + e3 0.034 / 0.58 mm, max 9.1 mm (rec 1: 0.062 / 1.94, max 16.0); table plane residual RMS
  12.9 mm (p95 25.6) over 2,400 of 48,000 fingertips, normal 8.2 deg from camera-down (rec 1:
  10.0 mm, 3.5 deg).
- **Reference window** (`battle-build-assembly101-reference --recording nusar_9061`,
  `runs/assembly101-reference-nusar_9061-v1/{manifest.json,hands.jsonl}`, 2,400 frames on
  C10379 with the +6 rule): dataset hands in 2400/2400 frames, both hands throughout, 4,781 of
  4,800 hands fully inside the image; projection check vs shipped 2D 0.0010 px RMS (max 0.0016)
  over 100,674 points (the last three proxy frames map past the fetched 2D window because of
  the +6 offset and are skipped by the check, not by the window); 61 fine-grained segments,
  among them `position interior` 438-462 and 1884-1949, `screw chassis with screwdriver`
  0-52, 581-719 and 1119-1216, `position rear body` 752-809 and 940-1009, `position cabin`
  1949-2027, and the rear-bumper steps that recording 1's window never had.
- **Contact sheet** (new `battle-assembly101-contact-sheet`,
  `data/derived/assembly101/<recording>/contact_sheet_374.000-454.000.png`, 2400x3228): one
  row per fetched view, frames 0 / 600 / 1200 / 1800 / 2399, each labelled with view, proxy
  frame, source time, core-or-margin and the coarse action. Same rig layout as recording 1 by
  eye; C10379 is again the grazing camera closest to the operator and stays the primary view.
- **Tests.** `tests/test_assembly101_recordings.py`, 13 default-tier tests (registry resolution
  by label and id, recording-1 record reproduces the historical constants, recording-2 window
  arithmetic, core-span validator, fetch and camera paths keyed by recording, clock rules by
  recording with both refusals, chunk plan, the majority fallback and its two ambiguous
  counter-cases, tracked recording-2 camera files and clip configs, contact-sheet frame
  spacing, etag checks) and one `real_data` test over the fetched windows, scans, rig and
  reference. Default tier 578 passed / 9 skipped; ruff clean on every touched file.
- **Open.** The visibility audit for the ego choice was not run (no hulls yet); e1/e2 exist on
  the Hub and can be fetched with the same command if C2 wants them. The +-2 rules of C10118,
  C10379 and C10395 rest on two metrics; a zoomed overlay check like the Sep 17 one has not
  been made for recording 2. The seed frame for C2 should be chosen from the contact sheet
  (proxy frames about 75-420 show the four parts apart), not assumed to be frame 0.

### Sep 20: B0 tooling of the multicam plan, anchors mapped to every view (commit `6c6cd66`)

- **Claim boundary first.** Nothing here is a label. The tool prepares labelling sessions on
  the other views by mapping the 13 C10379 anchor frames through the measured clock rules; the
  per-view configs hold no masks, and `battle-anchor-iou --view <view>` refuses to score until a
  human record exists for that view. CC BY-NC 4.0 applies to the frames. No GPU work: the
  workspace check ran with `--no-worker`.
- **Mapping convention** (`battle-anchor-frames-for-view`, `anchor_frames_for_view.py`). C10379
  analysis frame `p` shows pose frame `17649 + 2p` (clock rule offset +9); the target view's
  frame is `floor((pose - 17640 - offset_target) / 2)`, the half frame rounded *down* so the
  mapped frame is never later than the C10379 frame, the same integer-timeline convention as
  the ego-exo correspondence (`egoexo_correspondence.mapped_analysis_frame`), with the
  residual (target pose frame minus source pose frame) recorded per frame. Because every rule
  has the same raw-frame step the mapping is a constant shift per view: C10095 (+5) shift +2,
  residual 0; C10115 / C10118 / C10395 / C10404 (+6) shift +1, residual -1; C10119 / C10390
  (+7) shift +1, residual 0; e4 (0) shift +4, residual -1. Anchor 300 -> 302 / 301 / 301 / 304.
  Windows and the hidden-prompt interval shift with the frames.
- **Written.** `configs/qa/first_minute_review_anchors_<view_id>.json` for the seven other
  statics and e4 (`ReviewAnchorConfig` with `view_id`, `frame_mapping` (source config and
  clock-rules fingerprints, offsets, shift, convention), per-frame `origin` /
  `source_analysis_frame_index` / `residual_pose_frames`, and `workspace_timestamps`, the
  `--timestamps` string the calibration workspace accepts) and a per-view four-part target
  policy `configs/muggledsam_{static,ego}_four_part_reassembly_focused_manual_seed_<view_id>.json`
  bound to the all-static (or e4) clip config. Extra frames: the tool appends Track A's
  `runs/detector-scorecard-20260920/*/proposed_anchor_frames.json` when it exists (bare list,
  `frames`, or records with `analysis_frame_index`); it did not exist when the configs were
  written, so every config says `extra_frames_pending: true` and the tool prints the rerun
  instruction.
- **C10379 assumptions removed.** `build_first_minute_config` takes the view / clip config /
  target config (defaults unchanged); `battle-anchor-export prepare|export` and
  `battle-anchor-iou` gained `--view` with per-view default paths
  (`runs/human-review-anchors-first-minute-<view_id>`, `docs/qa/first-minute-review-anchors-<view_id>.human-record.json`)
  and a plain failure naming the next step when the record is missing; the calibration
  manifest and the manual-seed target policy accept any `static-*` / `ego-*` view id (they
  were `Literal` over three ids); `muggled_calibration.build_manifest` and the web CLI resolve
  the view from the config (a single-proxy config needs no `--view`, a multi-proxy config
  requires it; the legacy e4 default survives only for the original e4 screen config).
- **Verified headless.** `battle-anchor-export prepare --view static-c10119` and
  `--view ego-hmc21179183` built workspaces on the all-static and e4 configs;
  `battle-muggled-calibration-web --no-worker` served both (`/api/state` returned the
  manifest with the right `base_g2_config` and proxy); the all-static config without `--view`
  is refused with the list of proxies. `battle-anchor-iou --view static-c10119` fails with
  "no human review anchors exist for static-c10119 ... prepare / label / export".
- **Tests.** `tests/test_anchor_frames_for_view.py` (9): the mapping on synthetic rules (+9 ->
  +9/+7/+6/+5/0), the refusal before the target proxy, a full per-view config (frames,
  windows, hidden interval, extra-frame de-duplication, round trip), the pending flag, the
  extra-frame reader, `view_paths`, the missing-record message, and that every committed
  per-view config equals what the mapper writes from the committed clock rules. Existing
  suites for the anchors, the calibration and the web workspace pass (173 in the affected
  files).

### Sep 20: B2 of the multicam plan, seeding strategy search on the C10379 human masks and transfer (commits `2861d2f`, `df1edff`, `afb8475`)

- **Claim boundary first.** Every IoU below is against one person's choice of SAM3
  image-decoder mask on 16 frames of one view (the 13 anchor frames plus the human
  correction frames 0 / 327 / 900 / 1235); it ranks seeding strategies against each other and
  is not accuracy, not a dataset, not ground truth. The prompts are built from the *other*
  views' tracker masks (the agent-seeded 1280 runs and, for the interior prior, the
  human-corrected C10379 run) on the calibrated rig, never from the human mask being scored.
  Seeds written for the other views are agent-selected (`selected_by: agent`); "accepted"
  means the candidate is consistent with the other views' masks, not right. GPU: two queue
  jobs on code snapshots (`runs/seed-search-20260920/code-snapshot-{20c843b,df1edff}`), 69 s
  and 45 s, no `NVRM`/`Xid`. CC BY-NC 4.0.
- **Truth set** (`battle-seed-truth-set`, `runs/seed-search-20260920/truth_set.json`): 68
  human masks (64 on C10379: 51 anchor cells, 13 correction masks at 0 x4 / 327 x4 / 900 x2 /
  1235 x3; 4 e3 frame-0 masks), 1 hidden mark (rear_body 1700), 8 rejected decoder candidates
  (the two chassis prompts rejected at 300 and 650) and 204 unchosen alternatives of accepted
  prompts as labelled negatives; 24 excluded (the agent-selected 1172 / 1100 masks, the 1800 /
  2700 frames). The manifests keep every alternative, so the negatives exist. The anchor
  export's SHA-256s were checked against the workspace candidates.
- **Search** (`battle-seed-search plan|decode|score`, `search/`). Per truth frame and part the
  other views' centroids at the same pose frame (clock-rule shifts) are triangulated (DLT,
  30 raw px drop-worst, 6-9 views used) into a sphere whose radius is the median
  equivalent-circle radius at depth, projected into C10379 as a square box; the interior,
  which no other view tracks, uses the run's own mask one frame earlier (`self_prior`,
  labelled on every cell). Grid: margins {0.15, 0.25, 0.40, 0.60} x negatives {none, other
  parts' centroids, chassis rim, cabin edge, dataset hand joints} x {one box, margin box +
  0.60 box pooled} x pick {highest decoder score, largest, smallest, DINOv2-small exemplar
  from the other truth frames}: 160 strategies, 659 unique prompts (a negative set that adds
  no point inside the box collapses into `none`), decoded with one warm image decoder in
  69 s. Leave-frames-out: fit on all frames but 3 anchor frames, score the fit winner on
  those, rotate over the 13 anchor frames, correction frames always in the fitting set.

  | part | winner (margin / negatives / boxes / pick) | mean IoU all 16 frames | runner-up gap | held-out mean | held-out min | frame-0 IoU | gate >= 0.6 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | chassis | 0.25 / cabin edge / two / highest score | 0.581 | 0.001 | **0.525** | 0.018 (300) | 0.81 | FAIL |
  | interior | 0.25 / other centroids / one / highest score | 0.547 | 0.000 | **0.465** | 0.085 (1050) | 0.94 | FAIL |
  | rear_body | 0.15 / other centroids / two / highest score | 0.786 | 0.003 | **0.795** | 0.238 (1500) | 0.72 | pass |
  | cabin | 0.15 / other centroids / one / highest score | 0.954 | 0.000 | **0.949** | 0.805 (1500) | 0.95 | pass |

  By axis: the decoder's own top score beats the exemplar re-ranker on every part (0.581 vs
  0.496, 0.547 vs 0.524, 0.786 vs 0.712, 0.954 vs 0.877) and `largest` / `smallest` by more;
  `none` / other centroids / chassis rim / cabin edge are within 0.004 of each other on every
  part (the winners above are third-decimal ties) and hand joints cost 0.1-0.3; margin
  0.15-0.25 beats 0.60 by ~0.1; two boxes vs one within 0.03. The exact human mask is never
  returned (0 / 16 on every part: the human drew different boxes). Chassis fails in
  `[279,408)` and `[573,722)` (0.30-0.35 at 300 / 600 / 650: the other views' chassis
  centroid sits where the hand and interior are) and the interior fails where its own prior
  is wrong (1050 0.08, 1100 0.31). At frame 0 all four are 0.72-0.95.
  `search_contact_sheet.png`: truth | geometric prompt | winner pick for six frames.
- **Transfer** (`battle-seed-search transfer-plan|transfer-decode|transfer-accept`,
  `transfer/`, `seeds/`, `proposals/`). Frame 0 of each of the seven other statics and e4:
  the sphere from every *other* view's frame-0 mask (C10379 and e3 human seeds, the seven
  1280 runs, e4; the target view left out), the winner strategy decoded (7-9 prompts per
  view, 5-6 s each), accepted when the candidate's own centroid triangulates with the other
  observations with >= 3 views inside the 30 raw px filter and its area is within
  [0.3, 3.0] x the projected sphere. **rear_body and cabin accepted on all 8 views (16 / 16)**,
  every time with all 10 observations agreeing, reprojection 0.4-13.8 raw px; IoU against the
  Sep 18 agent seeds rear_body median 0.91 (0.68-0.97), cabin median 0.85 (0.39-0.99; C10390
  0.39 and C10395 area x0.45 are the low / corner cameras where the tight box picks a
  smaller cabin surface). Chassis candidates are consistent everywhere (10 views, 0.6-9.5 px,
  IoU vs Sep 18 median 0.97) but the part is below the gate, so they are **proposals only**
  and B3 keeps the Sep 18 chassis seed (`acceptance_basis` note `carried over from Sep 18`).
  **Interior rest frame:** the rule (hands >= N mm from the eight-view chassis consensus
  centroid over 15 frames with the centroid still, N = 100 then 60 then 40) finds nothing at
  100 or 60 mm, because the nearest hand joint is never farther than 63 mm from the chassis
  after frame 323; N = 40 gives C10379 frame **426** (view frames 427-430), so "rest" means
  hands clear by 40 mm, not put down. The interior triangulated there from the C10379 and e3
  run masks is consistent with its candidates in all 8 views (3 observations, the minimum,
  2-16 px) and is **proposals only** (gate 0.465); it is **blocked in the B3 seeds** because the
  multiview run profile seeds every slot at frame 0 and the interior is hand-held then, and a
  mid-minute slot start was not built. Proposals: 24 (3 per view: chassis f0, interior
  f427-430, rear_body or cabin f0) ranked by cross-strategy disagreement (1 - mean pairwise
  IoU of the five best strategies' picks; cabin's always agree), each with the distinct
  candidate PNGs, an overlay with the Sep 18 seed, and `proposal.json` (`human_decision: null`).
- **Schema.** `multiview_schemas.SeedCandidate.acceptance_basis` gains
  `multiview_consistency` and `carried_over_sep18_seed`, plus `consistency_views_used`,
  `consistency_reprojection_px`, `iou_vs_sep18_seed` (all optional, Sep 18 manifests load
  unchanged). The eight `seeds/<view>/seed_manifest.json` load through
  `_load_geometric_seed_manifest` (3 seeds each, interior blocked).
- **Tests.** `tests/test_seed_search.py` (10 default + 1 `real_data`): strategy grid and name
  round trip, pool rules (margin box, the 0.60 box for `two`, negative-set fallback), the four
  pick rules including the exemplar arm, leave-frames-out on synthetic cells (fit winner per
  split, held-out means, the gate), decode-request normalisation, boundary / crop helpers,
  calibration collection (positive, rejected, unchosen, agent excluded); `real_data`: the
  written truth set, search report and transfer report are consistent (68 / 64 positives,
  gates as above, every view's B3 seeds = rear_body + cabin from the search, chassis carried
  over, interior blocked). Ruff clean.
- **Open.** Interior unseeded on every other view; chassis re-seeding rejected by the gate;
  exemplar arm tried in one configuration only; the interior consistency test has exactly the
  minimum three observations. The plan's stop rule ("automatic interior seed fails on >= 4
  views -> run B3 with the human seed") applies: there is no human interior seed on the other
  views, so B3 runs three parts and records the interior as the open problem.

### Sep 20: B3 of the multicam plan, eight views rerun at 1280 / append with the search seeds

- **Claim boundary first.** Label-free. Eight runs of the same tracker (MuggledSAM SAM3,
  `--max-side-length 1280`, policy off, `--prompt-memory-semantics append`; with no later
  corrections the 32-entry bank holds only the seed, so this is the reference's pm-append
  condition and nothing more) whose only change against the Sep 19 `-r1280` runs is the
  seeds: rear_body and cabin from the B2 search (accepted by cross-view consistency), chassis
  the Sep 18 agent seed carried over, interior omitted (B2 gate). Consensus and hull numbers
  are cross-view disagreement of one tracker, not accuracy; nothing is adopted. GPU: one
  queue pass of 8 jobs on `code-snapshot-df1edff`, 15:15-16:04 UTC, all `succeeded`, every
  `gpu_check` ok, no `NVRM`/`Xid`. CC BY-NC 4.0.
- **Runs** (`runs/sam3-views-r1280-pmappend-seeded-20260920/views/<VIEW>/...-r1280-pm-append`,
  `README.md` / `views_table.md`). 1800/1800 frames each, 5305-5400 masks (3 parts), worker
  elapsed 325-331 s (Sep 19: 326-352 s), first output 4.2-4.4 s, peak VRAM 2.39 GiB on every
  view (the bank costs nothing without corrections). Presence unchanged to two decimals
  except C10395 (rear_body 1.00 -> 0.99, cabin 0.99 -> 0.98, chassis now missing from 594
  as at 720 px).
- **Consensus** (`runs/multiview-part-consensus-first-minute-r1280-pm-append-seeded/`, 72 s,
  reference `pm-append`, `--ego-view HMC_21179183`; `summary.md` = `battle-compare-multiview-builds`
  against the Sep 19 `-r1280-pm-append` root). C10379 chassis contradicted by the majority
  78 -> 81 frames (`[621,627)` new, the rest within 2 frames); **rear_body 37 -> 0 frames**
  (`[1662,1667)` `[1762,1794)` gone: the seven re-seeded views now sit on the same yellow
  piece as the reference's rear_body slot after 1660, the screwdriver the human named
  `distractor_confusion`); cabin none in both; C10379 agreement 0.94 / 0.97 / 1.00 -> 0.95 /
  0.99 / 1.00; **episodes over all views 105 (7) -> 80 (6)**; mean views per consensus frame
  7.38 / 7.61 / 8.12 -> 7.45 / 7.88 / 8.15. Per view, rear_body C10095 / C10115 / C10404 0.97 ->
  1.00, cabin C10390 0.96 -> 1.00 (the seed with IoU 0.39 to its Sep 18 predecessor), C10395
  cabin 0.58 -> 0.56, e4 rear_body 0.63 -> 0.60.
- **Hull** (`runs/multiview-visual-hull-first-minute-r1280-pm-append-seeded/`, 844 s). Frames
  with hull 1509 / 1209 / 1726 -> 1535 / 1262 / 1737, but median voxels **408 / 208 / 780 ->
  350 / 176 / 619**: the tighter seeds carve a smaller hull, and the larger C10379 masks
  exceed it more: hull-vs-mask median IoU on C10379 chassis 0.450 -> 0.397, **rear_body 0.447
  -> 0.343**, cabin 0.576 -> 0.513; hull episodes 250 (29) -> 233 (**45**), the new C10379
  rear_body episodes covering `[504,701)` and `[732,966)`; the other views' own rear_body
  hull-vs-mask IoU is unchanged (0.60-0.79, C10118 0.34, C10395 0.37, e4 0.03).
- **Reading.** The search seeds buy centroid agreement among the eight views (fewer
  episodes, no late rear_body contradiction) at the price of a smaller hull that the
  reference's masks disagree with more. Both are one tracker agreeing or disagreeing with
  itself across cameras; which seed is right needs the anchors on the other views (B1). The
  Sep 19 `-r1280-pm-append` roots stay canonical; these roots sit beside them. The interior is
  the open problem it was: no other view tracks it, its search prior was the run's own mask,
  and it failed the 0.6 gate.

### Sep 20: v6 review surface, one recording with three blueprint presets (Track D of the multicam plan)

- **Claim boundary first.** CPU only over retained runs; no model ran, the GPU was not touched
  (`CUDA_VISIBLE_DEVICES=""`), and no viewer was opened by the agent. Everything logged here is
  display of existing artifacts: the candidate arms are one tracker's output each on the same
  C10379 proxy and none is a second reference; the human anchors stay 13 frames on one view;
  the other eight views' seeds are agent-authored and unreviewed; consensus, hull and detector
  series measure disagreement between estimates. Ensemble v2 is still a candidate. CC BY-NC 4.0.
- **What the human asked** (three complaints about v5): which analyses are multi-camera and why
  the recording showed one camera; how to compare SAM3, DAM4SAM and the human label on one
  frame; whether hands and segmentation could be separate layouts that switch every panel at
  once (decision: one `.rrd`, several `.rbl` presets). The v5 provenance legend also still
  listed a `3 hidden` code that ensemble v2 never emits.
- **Builder** (`battle-build-interaction-review-v4`, `interaction_review_v4.py` /
  `interaction_review.py`): `--candidate-arm NAME=RUN_DIR` (repeatable) validates the run like
  every other input (same clip, clock and source fingerprint; every frame not required), logs
  its four part masks as RGBA cut-outs under `comparison/segmentation/NAME/<part>` at every
  frame from the mask cache and its per-part area under `diagnostics/segmentation/NAME/<part>`;
  `--confidence RUN_DIR` logs `confidence.jsonl` of the detector scorecard as
  `diagnostics/confidence/<part>/confidence` and `.../abstain` (points where it fires); anchor
  marks are on by default from `configs/qa/first_minute_review_anchors.json` and the exported
  mask set (`metadata/anchors/anchor_frame`, `failed_cells` from the scorecard's truth column,
  a `TextLog` at `metadata/anchors/log` naming the failed cells; the human masks as outlines at
  `primary/human_anchor_outlines/<part>` and as the arm `human_anchors`, present on the anchor
  frame only and cleared on the next); `--application-id` / `--recording-id`; the provenance
  legend is now `provenance_legend(sidecar, policy)`: the codes whose count is non-zero plus the
  policy's named `not_contact_eligible` intervals (`1 sam3 primary, 2 dam4sam fallback;
  rear_body [1660,1800) not_contact_eligible: distractor_confusion` for v2), used in the panel
  name, the series names and the guide, and the guide text no longer speaks of hidden intervals
  when the policy has none. `InteractionReviewIndexManifest` gained optional `candidate_arms`
  (name -> manifest fingerprint), `confidence_series`, `anchor_marks`, `application_id`,
  `recording_id` (schemas.py; older indexes load unchanged). `_prepare_output_root` tolerates
  the `.rbl` / merged `.rrd` / `presets_check.json` side files.
- **Nine-view recording** (`battle-build-multiview-static-comparison`, `multiview_review.py`)
  rebuilt on the Sep 19 canonical set: consensus `runs/multiview-part-consensus-first-minute-r1280-pm-append/`
  (reference `pm-append`, seven 1280 statics, e4), hull `runs/multiview-visual-hull-first-minute-r1280-pm-append/`,
  `--mask-every 2`, into `runs/multiview-static-comparison-first-minute-r1280-pm-append/`
  (**253.9 MB, 14.5 s**; the Sep 18 720 px recording is left as it was). New: the ego view is
  projected through its pose at the view's frame (the builder had only ever seen static
  sources), `--anchor-masks` draws the C10379 human masks as outlines on that tile at the 13
  anchor frames (`views/C10379/human_anchor_outlines/<part>`), `--proposals-root` draws the
  seed-search candidates on each of the eight other tiles at their frames
  (`views/<VIEW>/proposals/<part>`, one colour per candidate, label `proposal: accept/reject
  pending` with the strategies; 24 cells: chassis f0, rear_body f0, interior f427-430),
  `--application-id` / `--recording-id` / `--no-blueprint`, tile names say `(human seeds +
  corrections, anchor outlines)` or `(agent seeds, unreviewed, proposals)`, and the index records
  ids, roots, outlines, proposals and size.
- **One file.** Both recordings were written with the same ids
  (`battle-interaction-review-v6` / `interaction_review_first_minute_v6`) and merged by the
  new `battle-review-presets --merge` (`review_presets.py`): `rerun rrd merge`, then the
  recording store alone re-written through `rerun.experimental.LazyStore.write_rrd` so the
  merged file carries **no embedded blueprint** (a leftover default layout and a `.rbl` on the
  command line would race for activation). `runs/interaction-review-first-minute-v6/`:
  `interaction_review_first_minute_v4.rrd` **154.5 MB** (the v5-style package with five
  candidate arms, 47.4 s: validate 2.8 / geometry 13.1 / export 31.5; the v5 package is
  untouched at 100 MB), `interaction_review_combined.rrd` **406.4 MB** (one recording store, 384
  entity paths; merge + presets 7 s), the three presets, `presets_check.json`, the index, guide
  and sheet. Total build 69 s from the caches; `battle-cache-masks` was run once for the three
  arms without a sidecar (DAM4SAM large 47 s, ensemble v2 38 s, the 001210z run 67 s).
- **Candidate arms logged** (`comparison/segmentation/<arm>`): `pm-append` (v2 primary),
  `dam4sam-large-1024-sched-60s` (v2 fallback), `off-r1280-sched`,
  `old-reference-off-r720-sched` (`...20260918t001210z`, the run policy v1 calls primary),
  `ensemble-v1`, plus `human_anchors` on the 13 anchor frames; the reference tile is ensemble v2.
  Anchor log, from the scorecard's truth column: 9 failed cells on `pm-append` (600 / 650
  chassis, 700 interior, 1050 chassis + interior, 1100 / 1150 interior, 1500 rear_body, 1700
  rear_body hidden with 1,265 px), none in `[279,408)` or at 900. Abstain marks: 1,022 of 7,200
  rows (chassis 277, interior 484, rear_body 261, cabin 0).
- **Presets** (`segmentation.rbl` 15 views / 25 queries, `hands.rbl` 10 / 21, `multiview.rbl`
  18 / 47; `rerun.blueprint.Blueprint.save(application_id, path)`, Rerun 0.37.1): segmentation =
  reference (provenance overlay, anchor outlines) + six same-frame arm tiles + provenance,
  consensus-contradiction, confidence/abstain, anchor and arm-area series + anchor log, per-frame
  document, guide; hands = stabilized WiLoR, raw WiLoR, MediaPipe, dataset hands 2D, world-mm 3D
  with dataset hands and both ATHENA arms (consensus and hull excluded), WiLoR 3D, disagreement /
  dataset / ATHENA / contact series, no masks; multiview = 3 x 3 camera tiles (masks, consensus
  markers, hull projections, anchor outlines on C10379, proposals elsewhere), world-mm rig with
  hull voxels, the contradiction and per-view error tabs, anchor marks, both episode documents.
  The preset writer reads the recording's entity tree and only declares views for what exists.
- **Validation without a viewer** (`rerun --headless` does not exist in 0.37.1): each `.rbl`
  read back through `RrdReader` (one blueprint store, application id equal to the
  recording's); `rerun rrd verify` passes on the combined file and every preset; every view's
  `space_origin` and `ViewContents` queries are expanded (`$origin`, `/**` = the path or any
  descendant, exclusions ignored) and must match a logged entity: all matched
  (`presets_check.json`). Not proven: the look of the layout, video decoding on the client.
- **Labeling sessions prepared, none started** (`docs/labeling-sessions-2026-09-20.md`):
  (a) C10119, 26 frames, workspace `runs/human-review-anchors-first-minute-static-c10119/`
  written by `battle-anchor-export prepare --view static-c10119`, the exact
  `battle-muggled-calibration-web ... --resume --tailscale` command, export and `anchor-iou`
  commands, and the brief (chassis / rear_body / cabin, interior where visible, hidden and
  distractor vocabulary, what the 13 mapped / 8 detector-selected / 5 random frames test);
  (b) e4 likewise (`runs/human-review-anchors-first-minute-ego-hmc21179183/`) plus a one-off
  seed workspace at e4 frames 4 and 430 (`--timestamps 0.133333,14.333333`) for the human
  interior seed the automatic seed is to be compared against; (c) the 24 seed-search proposals
  as one contact sheet per view (`battle-seed-proposal-sheets`, `seed_proposal_sheets.py`;
  `runs/labeling-sessions-20260920/proposal_sheets/<VIEW>.png`: full frame with every candidate
  outlined plus a zoomed filled crop per candidate with strategy, area and decoder IoU
  estimate) and the decisions template `configs/qa/seed_proposal_decisions.template.json` (24
  cells, `accept` / `reject` / `unsure`, `accepted_candidate` index); (d) recording 2 is the
  GPU worker's; its config `configs/qa/nusar_9061_review_anchors_c10379.json` did not exist
  when the document was written, so it is referenced, not commanded.
- **Docs.** `docs/review-guide-2026-09-20-multiview-presets.md` (commands per preset, which
  entities are multi-camera and which single-camera, how to read confidence / abstain and the
  anchor marks, frames to scrub, how the presets were validated), README intro and the v6
  paragraph under "First-minute v4 review" with the three build commands.
- **Tests.** `tests/test_review_presets.py`: 13 default-tier (arm spec parsing, the legend from
  a sidecar/policy, confidence loader and duplicate refusal, anchor marks logging on the anchor
  frame and clearing on the next, candidate arm frame logging, the v4 blueprint's new roots,
  proposal outlines by frame, anchor outlines by view, tile names, query resolution with globs
  and exclusions, presets written / read back / resolved on a synthetic recording, an unmatched
  query reported, merge yielding one store with no blueprint) and one `real_data` test over the
  built v6 package. Ruff clean on every touched file.
- **Open.** The layouts were not seen by a human or the agent; a 406 MB single file may be
  heavy for a small client (the two component files take the same presets). The hidden / not
  visible vocabulary in the workspace is a button plus a note, not a typed distractor field.
  The arm anchor numbers quoted in the guide are the Sep 19 scoreboard's; nothing was re-scored.

### Sep 20: B4 arms on C10379, the consensus re-prompt loop run (multicam plan headline; commits `78f2eca`, `21da378`, `cf8007e`)

- **Claim boundary first.** Anchor IoU is against one person's choice of SAM3 decoder masks on
  13 frames of one view (51 labelled cells + 1 hidden): review evidence that ranks arms against
  each other, not accuracy, not a dataset, not ground truth. Every correction the loop wrote is
  agent-authored (`selected_by: agent`, provenance `multiview_consensus`); "accepted" means the
  other cameras agree with the mask, not that it is right. GPU: five queue passes on
  `runs/multiview-reprompt-20260920/code-snapshot-0c62799/` (decode 7 s, three arms 329-357 s,
  iteration-2 arm 359 s, two more decodes 7-8 s), every `gpu_check` ok, no `NVRM`/`Xid`.
  CC BY-NC 4.0.
- **Others-only consensus** (`runs/multiview-part-consensus-first-minute-r1280-others-only/`,
  `--exclude-view C10379`, reference `pm-append` named but excluded, seven Sep 19 `-r1280` statics
  + e4 behind the pose gate, 60 s): chassis / rear_body / cabin on 1800 frames each with 7.05 /
  6.70 / 7.13 mean views, 25 / 59 / 17 episodes, interior none (no other view tracks it). This
  removes the human-corrected reference from the consensus; what it does not remove is that
  the seven statics and e4 were seeded at frame 0 by geometric transfer of the C10379 human
  frame-0 masks. The loop is free of human *corrections*, not of human *seeds*.
- **Plan** (`C10379/iter1/reprompt_plan.json`, default detector: error > 40 raw px for >= 5
  frames, >= 3 statics agree within 30 px, merge 15 / min run 10): the same three chassis
  onsets as with the pm-append consensus, 296 `[296,313)` (43 px at onset, 6 statics), 475
  `[475,515)` (42 px, 7), 1049 `[1049,1085)` (121 px, 7); rear_body 1762 blocked (2 statics).
  The 60 px sensitivity variant (`variants/thr60/`, `detector_thr60.json`) keeps only chassis
  1067 `[1067,1081)`: 296 and 475 are 40-60 px onsets, about one part radius on a grazing
  camera. Arms were run for the default plan only; the scorecard said thresholds do not
  transfer leave-one-out, so none was taken from it.
- **Decode** (one warm worker, 6 prompts): all three onsets **accepted**, no rejections (8/8,
  6/8, 7/8 candidates pass the area band and centroid-ray rule; best by ray distance 16 / 10 /
  18 mm = 0.37 / 0.19 / 0.36 radii, area 1.48 / 1.55 / 1.89 x expected, decoder IoU 0.59 / 0.68
  / 0.60). The accepted chassis masks are **15,373 / 17,084 / 23,288 px** where the human's
  chassis on this view is about 5k px: each covers the chassis *and the hand holding it*
  (`calibration/results/t000296-b02_candidate-02_review.png`). The expected area the sphere
  model predicts for the grazing camera is 10-12k px, so a hand+chassis blob sits inside the
  [0.3, 3.0] band and its centroid lies near the consensus ray; the rule cannot separate part
  from hand. That is the mechanism behind every number below.
- **Arms** (all `--max-side-length 1280 --prompt-memory-semantics append --checkpoint-every 300`,
  full runs; `anchor_iou.md`): IoU all / chassis / interior / rear_body / cabin, windows
  279-408 / 573-722 / 1020-1172 / outside, hidden FP px, worker elapsed, peak VRAM.

  | arm | later corrections | all | ch | int | rb | cab | 279-408 | 573-722 | 1020-1172 | outside | hidden FP | s | GiB |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | pm-append (human) | 327 900 1172 1235 | **0.743** | 0.637 | 0.585 | 0.790 | 0.962 | 0.853 | 0.687 | 0.616 | 0.800 | 1265 | 341 | 2.54 |
  | off-r1280-sched | 327 900 1172 1235 | 0.724 | 0.608 | 0.540 | 0.789 | 0.964 | 0.861 | 0.603 | 0.659 | 0.764 | 1228 | - | - |
  | seed-only-720 (Sep 18) | none | 0.632 | 0.427 | 0.384 | 0.766 | 0.960 | 0.624 | 0.618 | 0.695 | 0.598 | 1310 | - | - |
  | **seed-only-1280** (new) | none | **0.591** | 0.404 | 0.228 | 0.789 | 0.959 | 0.648 | 0.628 | 0.579 | 0.527 | 1252 | 329 | 2.39 |
  | **consensus-only** iter 1 | 296 475 1049 (agent) | **0.590** | 0.291 | 0.316 | 0.810 | 0.959 | 0.573 | 0.651 | 0.637 | 0.517 | 1226 | 343 | 2.48 |
  | consensus-only iter 2 | + 333 653 983 1326 1590 | 0.663 | 0.519 | 0.397 | 0.787 | 0.959 | 0.653 | 0.661 | 0.747 | 0.606 | 1251 | 359 | 2.77 |
  | human-plus-consensus | 296 327 475 900 1049 1172 1235 | 0.730 | 0.591 | 0.585 | 0.791 | 0.960 | 0.823 | 0.679 | 0.560 | 0.834 | 1207 | 357 | 2.71 |

  Consensus corrections came from an others-only consensus whose member views were seeded by
  geometric transfer of the C10379 human frame-0 masks (the remaining human dependency).
  **seed-only at 1280 is 0.591**, below the Sep 18 seed-only at 720 (0.632; interior 0.23 vs
  0.38), so the first-minute floor at the run resolution had never been measured and is lower
  than assumed. consensus-only iteration 1 lands exactly on that floor (0.590): three agent
  chassis corrections moved the chassis from 0.404 to 0.291 and the interior from 0.228 to
  0.316 (a chassis blob that includes the hand frees the interior slot), net zero. Adding the
  same corrections to the human ones costs 0.013 (0.743 -> 0.730), all on the chassis.
- **Iterations.** Iteration 2 (`C10379/iter2/`, same others-only consensus, source = the
  iteration-1 run): **five new chassis onsets** 333 `[333,475)`, 653 `[653,682)`, 983
  `[983,1040)`, 1326 `[1326,1565)`, 1590 `[1590,1800)`; the first corrections made the chassis
  contradict the other cameras for 400+ frames instead of 90. All five accepted (masks
  11.6-20.1k px), the arm scores 0.663 (chassis 0.519, interior 0.397). Iteration 3
  (`C10379/iter3/`): two chassis onsets (1605, 1639), both rejected `policy_keyframe_limit`
  (chassis already has 8 later corrections; policy v4 allows 8), nothing to run; the loop's cap
  of 3 was reached anyway.
- **Verdict (plan stop rule).** consensus-only is 0.153 (iteration 1) and 0.080 (iteration 2)
  below pm-append, both far past 0.03, on the one view that has anchors: **multicam is a
  detector, not yet a corrector.** The consensus finds the frames where C10379 disagrees with
  the other cameras (the three onsets are the ensemble v2 fallback intervals), but the
  correction it authors from a sphere + box prompt is a hand+chassis blob the geometric rule
  cannot reject. What a corrector would need, in order of evidence: a hand-aware acceptance
  (dataset hand joints as negatives were available, `--hand-negatives`, not used here; the seed
  search found hand-joint negatives cost 0.1-0.3 IoU at frame 0, so this is not obviously a
  fix), a size prior from the target view's own history rather than the sphere model on a
  grazing camera, or a second decoder opinion. Track C (recording 2) runs with the detector
  framing: the consensus flags, the human scores.
- **Deliverables.** `runs/multiview-reprompt-20260920/` (README results section, `C10379/iter{1,2,3}/`,
  `variants/thr60/`, `jobs_*.json`, `queue.log`, `logs/`); the others-only consensus root; this
  entry. Tooling changes that this run needed are in the B4/C10119 entry below.

### Sep 20: the multiview profile takes agent corrections; C10119 as re-prompt target (plan B4 second target; commits `78f2eca`, `cf8007e`)

- **Claim boundary first.** No anchors exist for C10119: nothing here is scored. The one number
  is cross-view agreement of one tracker with the other cameras (one of which, C10379, carried
  the human corrections), and after frame 1660 the majority's rear_body slot sits on the
  screwdriver the recording-1 human named a distractor, so agreeing more with the majority late
  in the minute is not evidence of being right. The correction is agent-authored
  (`selected_by: agent`, provenance `multiview_consensus`). GPU: two queue jobs (decode 6.5 s,
  arm 370 s incl. export) on `code-snapshot-21da378`, `gpu_check` ok. CC BY-NC 4.0.
- **Why a change was needed.** The B4 tool's `run` emitted the C10119 command as `BLOCKED`: the
  `--four-part-multiview-first-minute` profile took only its geometric seed manifest, the human
  schedule/policy contracts name three view ids, and the calibration-manifest contract forbids
  agent seeds by design (`agent-selected masks are later-frame corrections only, never seeds`),
  so a geometry-seeded run cannot carry a `MuggledSAMMultiKeyframeCorrectionSchedule` at all.
  Widening the Literals would not have helped.
- **Change (minimal, tested).** A separate record for exactly this case:
  `MultiviewAgentCorrectionSchedule` (`schemas.py`; view id by pattern, bound by fingerprint to
  the clip config, the proxy, the geometric seed manifest and the plan; one
  `MultiviewAgentCorrection` per target x frame with slot, candidate id, mask fingerprint,
  iteration, `selected_by: agent`, provenance `multiview_consensus`; a `provenance_file` naming the
  loop's `reprompt_provenance.json`). `battle-muggled-smoke --agent-correction-schedule <json>`
  (multiview profile only): the seeds still come from `--geometric-seed-manifest`; the loader
  checks the schedule's fingerprints, that every correction names a slot the manifest seeded and
  lies inside the frame budget, re-hashes every mask, and hands the worker the same
  `{seeds, corrections, memory_semantics}` payload a human schedule produces (no worker change).
  Run metadata: `FourPartMultiviewRunMetadata.later_corrections` gains `multiview_consensus`
  with `agent_correction_schedule_fingerprint` and `agent_correction_frames` (the profile's
  `requested_seconds`/frame-count rule and `recording_label` were relaxed in the same hunk for
  recording 2, see the C2 entry). `battle-multiview-reprompt decode` writes this schedule for a
  geometry-seeded source instead of a blocked reason (consensus-only arm only; a geometry-seeded
  view has no human corrections to add, so `human-plus-consensus` is reported as blocked with
  that reason), carries the previous iteration's agent schedule forward, applies the policy
  budget, and `run` emits `--agent-correction-schedule`. The human `view_id` Literals on the
  policy/schedule contracts are untouched. Tests: `tests/test_multiview_reprompt.py` (agent
  schedule derivation with slot assignment and mask hashes, carry-forward across iterations,
  emitted command) and `tests/test_exemplar_seed.py` (the smoke loader: payload, slot/budget/
  manifest-drift refusals).
- **C10119, iteration 1** (`runs/multiview-reprompt-20260920/C10119/iter1/`). Consensus
  `runs/multiview-part-consensus-first-minute-r1280-excl-c10119/` (`--exclude-view C10119`,
  reference `pm-append` *included* as a member: the human-corrected C10379 run is in this
  majority, unlike the C10379 arms). Plan: one onset, **rear_body 1533 `[1533,1800)`**, error 281
  px, 6 statics agree, radius 25 mm, expected area 1,280 px. Decode: accepted (8/8 pass; ray 2 mm
  = 0.10 radii, area x1.41, decoder IoU 0.93). Run: the r1280 C10119 run's command plus the
  agent schedule, 1800 frames, 327 s, 2.42 GiB, presence 1.00 / 1.00 / 1.00 (chassis, rear_body,
  cabin).
- **Agreement before/after** (recording-1 consensus rebuilt with the corrected C10119 run
  replacing the Sep 19 one, `runs/multiview-part-consensus-first-minute-r1280-c10119-reprompt-iter1/`,
  73 s): C10119 rear_body agreement **0.872 -> 0.997**, mean error 20.5 -> 0.7 px; its 13
  contradicting rear_body episodes over `[1525,1785)` reduce to one, `[1525,1530)`; episodes over
  all views 105 -> 90; chassis / cabin agreement 1.00 / 1.00 unchanged. Held for human scoring:
  `configs/qa/first_minute_review_anchors_static_c10119.json` (26 frames) is the session; once
  its record exists the command is
  `uv run battle-anchor-iou --view static-c10119 --run r1280=runs/sam3-views-r1280-20260919/views/C10119 --run consensus-only=runs/multiview-reprompt-20260920/C10119/iter1/arms/consensus-only --output runs/multiview-reprompt-20260920/C10119/iter1/anchor_iou.json`.
  What the anchors must decide: whether the rear_body C10119 now agrees with the other six on is
  the rear body or the screwdriver.

### Sep 20: Track C2/C3, recording 2 (`nusar_9061`) seeded and run with zero human input (commits `78f2eca`, `21da378`, `cf8007e`)

- **Claim boundary first.** No human touched this recording: no seed, no correction, no anchor.
  Upstream human input that remains and is named: the DINOv2 exemplar library is recording 1's
  64 human masks on C10379, and the gate that allowed `rear_body` and `cabin` (and refused
  `chassis` and `interior`) is the seed search's held-out IoU on recording 1 (0.795 / 0.949 pass,
  0.525 / 0.465 fail the 0.6 rule), so **chassis and interior seeding is not automatic** and was
  not attempted as seeds. Every mask is agent-selected; consensus, contradiction and confidence
  numbers are one tracker disagreeing with itself across cameras, not accuracy. GPU: three
  queue passes (seed decode 317 s; 7 views 331-340 s each, 17:25-18:05 UTC; re-prompt decode 7 s
  and arm 356 s) on `code-snapshot-78f2eca` / `-21da378`, every `gpu_check` ok, no `NVRM`/`Xid`.
  CC BY-NC 4.0. Full detail: `runs/rec2-automatic-20260920/README.md`.
- **Tool** (`battle-exemplar-seed plan|window|decode|accept|label-session|sheet`,
  `src/battle/exemplar_seed.py`, tests `tests/test_exemplar_seed.py`). Seed frame by the dataset
  hand joints (highest lowest-joint height above the fitted table plane over 5 frames inside
  [300, 422)): **383**, 57 mm. Table region = the hand joints of the window dropped onto the
  plane, grown 120 mm; 208 grid points at 55 mm x 2 box radii (32 / 55 mm) projected per view =
  **3,234 SAM3 image-decoder prompts**, one warm worker per view. Candidates filtered to an
  18-95 mm implied radius on the plane and de-duplicated (<= 220 per view), ranked per part by
  max cosine to that part's exemplars. Consistency: top-6 per view, every pair of views proposes
  a point, supporters reproject within 30 raw px, the largest summed similarity over >= 3
  supporters with radii within [0.5, 2.0] x median wins, parts placed in gate order with a 60 mm
  exclusion; remaining views completed by the nearest in-band candidate to the reprojected
  point (`acceptance_basis: centroid_ray`, recorded as a completion). New seed provenance
  `exemplar_multiview_consistency` across the seed / run / consensus schemas. `window` writes a
  seed-window clip (`configs/clips/..._all_static_seed383_g2.json`: proxy frames [383, 2100)
  re-encoded frame-exact with the proxy recipe, offsets shifted, frame 0 = proxy 383) because
  the tracker seeds at frame 0 of the video it is given. `--recording` / `--analysis-frame-offset`
  on the consensus builder, `--recording` on `battle-muggled-smoke` (multiview profile: any
  static view, budget from `--max-frames`) and on `battle-multiview-reprompt plan`;
  `battle-detector-scorecard --no-truth <scorecard.json>` writes only `confidence.jsonl` /
  timeline / `confidence_only.json` with the detector set and abstain threshold carried from a
  scored run.
- **Seeding outcome** (`runs/rec2-seed-proposals-20260920/seed_table.md`). All four parts reached
  consistency: chassis 5 supporting views + 3 completed (reprojection 5.6-28.3 px, radii 21-60 mm,
  similarity 0.49-0.73, margin -0.10..+0.15), interior 7 + 0 (2.3-18.0 px, 18-48 mm, 0.54-0.69,
  -0.13..+0.05), rear_body 5 + 2 (4.9-29.5 px, 19-66 mm, 0.39-0.70, -0.09..+0.18), cabin 6 + 2
  (2.0-24.6 px, 19-55 mm, 0.39-0.74, -0.17..+0.06). **The margins are small and often negative**:
  the appearance ranking barely separates the parts; geometry makes the picks consistent but
  only says "an object is here". By eye (`seed_picks_contact_sheet.png`) the rear_body pick is the
  green-sticker piece and the cabin pick the roofed cabin at the arm's base in the views
  inspected, while the interior pick sits on the black block that is most likely the chassis and
  the chassis pick on the small piece in the hands: the gate refused exactly the two parts whose
  picks look wrong. Used as seeds: rear_body (7 views) and cabin (8); C10119 got no in-band
  rear_body candidate and, with one part, was **not run** (2-part minimum). Chassis and interior
  (and every part's alternatives) are proposals for the human: 32 cells under `proposals/`.
- **Runs** (`runs/rec2-automatic-20260920/views/`): 7 static views, 1,717 frames each, worker
  309-310 s, peak VRAM 2.39 GiB, presence rear_body 0.77-1.00 (C10404 0.77), cabin 0.83-1.00
  (C10379 0.83).
- **Consensus** (`consensus/`, reference C10379, seven statics, 46 s; wrist check at the offset
  frames median 0.0005 mm): rear_body on 1717 frames with 3.55 mean views, 87 episodes,
  agreement C10379 1.00 / C10095 0.69 / C10115 0.52 / C10118 0.76 / C10390 0.76 / C10395 0.84 /
  C10404 0.56; cabin 3.32 mean views, 119 episodes, C10379 contradicted 78 frames, agreement
  C10379 0.90 / 0.66 / 0.74 / **C10118 0.09** / 0.49 / 0.93 / 0.81; 206 episodes in all.
  Recording 1's eight views agreed on 7.4-8.1 views per frame at 0.86-1.00; here 3.3-3.6 of 7 at
  0.09-0.93. The automatic seeds are consistent at one frame and diverge over the minute.
- **Re-prompt on C10379** (`reprompt/C10379/iter1/`, others-only consensus): 12 onsets, 4 planned
  (cabin 74; rear_body 771, 867, 977), 8 blocked (only 2 statics agree). All 4 accepted (masks
  2.3-4.8k px, decoder IoU 0.13-0.74). Arm: 1717 frames, 326 s, 2.54 GiB, cabin presence 0.83 ->
  0.93; against the rebuilt consensus C10379's cabin contradicts the majority on **144 frames
  (from 78)**, agreement 0.90 -> 0.87. Same reading as recording 1 (detector, not corrector).
  Contact sheet `c10379_consensus_only_contact_sheet.png` (proxy 383 / 683 / ... / 2099): rear_body
  stays on the green-sticker piece; the cabin slot spends most of the minute on the hands.
- **Confidence** (`detector/`, `--no-truth`, thresholds from recording 1's pm-append scorecard:
  `sam3_score` + `area_vs_seed` + `area_jump`, abstain <= 0.187): seeded run 519 / 3,434 rows
  abstain (15.1 %; cabin 519, rear_body 0), consensus-only run 345 / 3,434 (10.0 %; cabin 344,
  rear_body 1). Nothing scored; fewer abstentions is not evidence the cabin improved.
- **Labelling session prepared** (`configs/qa/nusar_9061_review_anchors_c10379.json`, target
  policy `configs/muggledsam_static_four_part_reassembly_focused_manual_seed_nusar_9061_static_c10379.json`):
  13 C10379 frames on the original 80 s proxy timeline, 8 detector-selected from the
  consensus-only run's confidence (456, 986, 1616, 1810, 1878, 1958, 2018, 2078; >= 60 apart) and
  5 seeded random (483, 552, 1191, 1636, 2045); the workspace command is in the README and was
  verified headless (`--no-worker`, `/api/state` returns the recording-2 config, proxy and 13
  timestamps). Open: scoring maps proxy frames to the seed-window run (minus 383) or needs a
  run on the untrimmed clip; `battle-anchor-iou --view static-c10379` needs `--anchors` pointed
  at the recording-2 export.
- **Open / not done.** e4 was not seeded or run (mono ego view; the exemplars are RGB and the
  trimmed timeline would need the ego pose offset plumbed through the ego gate, which
  `--analysis-frame-offset` does but was not exercised). The chassis and interior seeds wait for
  the human's proposal review, then the run resumes from the seed frame for those slots (not
  built). The consensus re-prompt was not iterated past 1 on this recording. Whether any seeded
  part is right waits for the 13 anchors.

### Sep 20: acceptance-rule search for the consensus corrections, and the multicam plan closed (commits `12f48e9`, `833343d`)

- **Claim boundary first.** Every IoU below is against one person's choice of SAM3 decoder mask
  on 16 frames of one view (C10379; the 13 anchor frames and the human correction frames 0 /
  327 / 900 / 1235): it ranks acceptance rules against each other and is not accuracy. The
  prompts come from the others-only consensus, whose member views were seeded by geometric
  transfer of the C10379 human frame-0 masks. GPU: one queue job on
  `runs/correction-acceptance-search-20260920/code-snapshot-12f48e9/` (decode, 27.6 s,
  `gpu_check` ok, no `NVRM`/`Xid`); the consensus-only v2 arm was **not run** (its condition
  was not met, below). CC BY-NC 4.0.
- **Why.** The B4 arms located the chassis correctly from the other cameras and accepted the
  wrong pixels: every accepted correction (296 / 475 / 1049, iteration 2 added 333 / 653 / 983
  / 1326 / 1590) was a chassis+hand blob of 15-23k px against a ~5k px human chassis. Before
  closing the plan the question was whether a different acceptance rule on the same candidates
  would pick the human's mask, or abstain.
- **Tool** (`battle-correction-acceptance-search plan|decode|score|sheet`,
  `src/battle/correction_acceptance_search.py`, tests `tests/test_correction_acceptance_search.py`,
  7 default-tier). Every truth cell (63 unique: 64 positives minus the duplicate at 900, plus
  the hidden rear_body at 1700) is treated as a re-prompt onset; the geometric prompt is built
  from the others-only consensus at that frame exactly as `battle-multiview-reprompt plan`
  builds it (`onset_geometry` + `build_prompts`, margins 0.25 / 0.60, negatives at the other
  parts' consensus centroids). The interior has no consensus, so its 16 cells are unprompted:
  **47 cells** (chassis 16, rear_body 16, cabin 15), 214 prompts. Pools: the tool's `base` set;
  the same boxes with negatives at the dataset hand joints or at the stabilized WiLoR landmarks
  inside the box; every base mask minus the dilated (12 px) convex hull of the dataset / WiLoR
  joints, largest component kept; `all`. Rule grid: area band vs the sphere's expected area
  {[0.3, 3.0], [0.5, 2.0], [0.6, 1.5], [0.7, 1.3]} or vs the pm-append frame-0 seed area
  (|log ratio| < {0.4, 0.7}) x decoder IoU-estimate floor {none, 0.5, 0.7, 0.9} (the image
  decoder exposes no object score) x hand-hull overlap rule {none, dataset, WiLoR; reject > 20 %}
  x other-part disc overlap {off, reject > 30 %} x ranking {centroid ray then score (the tool's),
  decoder score then ray}: 1,728 rules. Metric per cell: IoU of the accepted candidate vs the
  human mask; abstention neutral; harm = accepted with IoU < 0.4. Leave-frames-out as the seed
  search (fit on all but 3 anchor frames, correction frames always in the fit, 13 rotations);
  the fit objective is the highest mean accepted IoU among rules with fit harm <= 0.15 and fit
  acceptance >= 0.25.
- **Chassis (16 cells).** Current rule 0.438 mean IoU, everything accepted, **harm 0.50 (8/16)**.
  Tightened bands 0.421-0.521 with harm 0.27-0.50; seed-relative bands 0.410 / 0.453, harm 0.44;
  decoder floor 0.7: 0.510, harm 0.27 (0.9 abstains everywhere); dataset hand-hull overlap
  rule **0.182, harm 0.71** (the hand is on the chassis: the human's own mask overlaps the joints'
  hull by 0.5-0.85, so the rule keeps the candidates that missed the part); WiLoR hull 0.427,
  harm 0.36 (gentler only where WiLoR sees no hand); other-part disc never fires; extra hand
  negatives and hand subtraction leave the pick unchanged under ray ranking. **Ranking by the
  decoder's own IoU estimate instead of the closest centroid ray: 0.573, everything accepted,
  harm 0.19 (3/16)**, within 0.02 of the oracle on 12 of 16 cells; the three harmful picks
  (300 / 600 / 650, 0.29-0.33) are cells where nothing in the pool exceeds 0.33. Best grid rule
  in-sample (`score >= 0.5 + WiLoR overlap <= 0.2 + rank by score`) 0.633 / 50 % accepted / 0
  harm on all cells, **leave-frames-out 0.470 / 56 % / harm 0.20 (4/20)**; five different
  split winners. **Oracle ceiling** (best candidate in the pool under any rule): 0.596 mean, a
  >= 0.6 candidate on 8 of 16 cells (10 with hand subtraction, which adds 300: 0.33 -> 0.77 and
  1200: 0.57 -> 0.60). **Bar (held-out mean IoU >= 0.6, harm <= 0.15) not met**; the v2 arm was
  not run, per the task's condition.
- **rear_body and cabin clear the bar.** rear_body: current 0.640 / harm 0.19; chosen
  `|log(area/seed)| < 0.4 + score >= 0.7 + dataset hand overlap <= 0.2` 0.836 / 75 % / 0 harm,
  held-out **0.840 / 75 % / 0**, stable winner (10 of 13 splits). cabin: current 0.889 / 0 harm;
  chosen `+WiLoR negatives, score >= 0.9, WiLoR overlap <= 0.2` 0.946 / 67 % / 0, held-out
  **0.961 / 31 % / 0** (the 0.9 floor abstains on a third of frames). Neither part is where the
  consensus-only arm lost its points.
- **Reading.** The acceptance rule was the wrong place to look for 0.15 of anchor IoU. Two
  things are true at once: (1) the tool's ranking is the mechanism behind the blobs, and a
  fixed, unfitted change (decoder score first) removes half the harm; (2) the decoder does not
  return the human's chassis from the consensus box in `[279,408)`, `[573,722)` and at 1700
  (the hand covers the part and the other views' centroid sits on hand + interior), so no rule
  on these candidates can reach the bar, and every filter that reaches 0 harm in-sample does
  so by abstaining on half the cells and does not hold up leave-frames-out. What a corrector
  needs is a better candidate, not a better gate: a prompt that separates the hand (the
  sphere model over-predicts this grazing camera's footprint 1.2-4x on every frame), or a
  second decoder opinion. Files: `runs/correction-acceptance-search-20260920/` (README with
  every table, `search_report.json`, `v2_rule.json` with `meets_bar_held_out: false` for the
  chassis, `acceptance_contact_sheet.png` at 300 / 327 / 600 / 1050 / 1235 / 1500).

#### Closing the multicam plan: the two goals, answered with numbers

**(a) Can we detect loss of confidence?** Partly, as a ranker; not yet as a gate. On the 52
anchor cells (13 frames, one view, 9 / 14 / 11 failures on the three scored runs) SAM3's own
object score is the one detector top-2 on every run, **AUROC 0.91-0.96 in-sample**, and the
only one defined on every cell. The combined rank-average detector (top-3 by AUROC) reaches
0.94-0.96 AUROC in-sample but, chosen and thresholded on 12 frames and scored on the 13th,
**precision / recall 0.36-0.62 / 0.44-0.64** (4/7/5/36, 8/5/6/33, 7/5/4/36 TP/FP/FN/TN): the
ranking transfers, the threshold does not. The consensus and hull detectors are **undefined on
a third of the cells** (13 and 18 of 52: no other view tracks the interior, no hull where views
disagree), and **nothing is scored on the 279-408 shape leak** the human reported, because no
anchor cell in that window fails on any run (0 of 12). Track A's confidence series therefore
runs as review context (abstain marks, 1,022 of 7,200 rows on `pm-append`), not as a gate;
the B4 loop kept the 40 px consensus rule instead of a calibrated threshold.

**(b) Can we stop relying on human segmentation?** For two of four parts' seeds, yes; for
corrections, no. **Seeds:** a tight box at the centroid triangulated from the other cameras
with the decoder's top-scored mask reproduces the human's rear_body / cabin at **0.80 / 0.95**
held-out IoU (13 anchor frames, leave-frames-out) and was used to seed those parts on the
seven other statics and e4 with all 10 observations agreeing (16 / 16 accepted); chassis /
interior score **0.53 / 0.47** and stay human seeds (chassis: the Sep 18 agent seed carried
over; interior: proposals only, unseeded on any other view). **Corrections:** consensus-only
re-prompting on C10379 scores **0.590 (iteration 1) / 0.663 (iteration 2)** against **0.743**
with the four human corrections and a **0.591** seed-only floor at 1280; the corrections it
accepted were hand+chassis blobs. The acceptance-rule search above finds **no rule that
reaches 0.6 held-out IoU with harm <= 0.15 on the chassis** (best 0.470 / harm 0.20; oracle
ceiling 0.596), so the v2 arm was not run; ranking by the decoder score is the one fixed
improvement (0.438 -> 0.573, harm 0.50 -> 0.19). **Recording 2** (`nusar_9061`, different
subject) was seeded and run with **zero human input** for rear_body (7 views) and cabin (8)
by exemplar ranking plus >= 3-view consistency, but the exemplar margins are near zero (-0.17
to +0.18) and the seven views agree on only **3.3-3.6 of 7** per frame (recording 1: 7.4-8.1
of 8); chassis / interior are 32 proposals; nothing is scored until the human labels its 13
frames. **C10119**'s one consensus correction (rear_body 1533) raised its agreement with the
other cameras from 0.872 to 0.997 and is held for scoring on 26 prepared frames; agreement
late in the minute is with a majority sitting on the screwdriver, so it is not evidence yet.

**What remains human:** the frame-0 seeds for the chassis and the interior (rear_body and
cabin seed automatically on recording 1; on recording 2 the same two parts seeded but are
unverified), every correction (four on C10379; the consensus writes corrections but they
score below the seed-only floor), and the anchors that score all of it (52 cells on one view;
three further sessions prepared, none labelled). The plan's stop rule applies verbatim:
**multicam is a detector, not yet a corrector**.

**Next experiment (one pick).** Re-run consensus-only on C10379 and C10119 with the tool's
ranking changed to decoder-score-first (rule (g)), the current filters untouched, and score
C10379 on the existing anchors and C10119 on its 26 frames as soon as they are labelled: one
GPU hour, no fitted knob. Reason: it is the only change the search found that is both large
(0.135 of mean IoU, harm halved at 100 % acceptance) and unfitted (a fixed rule, not a
grid winner), it acts on the exact mechanism the arms exhibited (at 1050 the ray-nearest pick
is a 24k px blob at IoU 0.24 and the decoder's top pick is 8.7k px at 0.76), and the result is
diagnostic either way: if consensus-only leaves the 0.591 floor the corrector's remaining
limit is the candidate set in the occlusion windows (oracle 0.596), which points at prompts
(a size prior from the target view's own history rather than the sphere model, or a second
decoder) rather than at more acceptance rules; if it does not move, the corrections were never
the binding constraint and the consensus stays a detector. The alternatives have lower expected
value now: more acceptance filters are selection artefacts on 16 cells; the interior needs the
human e4 seed before any automatic seed can be compared; recording 2 needs its 13 labels before
any knob is turned, and turning one first would make those labels confirmatory.

### Sep 20, evening: the named next experiment run, decoder-score ranking in the consensus re-prompt loop (commit `625321a` tooling; this entry's commit)

- **Claim boundary first.** Every IoU below is against one person's choice of SAM3 decoder
  masks on **13 frames of one view** (C10379; 51 labelled cells + 1 hidden): review evidence
  that ranks arms against each other, not accuracy, not a dataset, not ground truth. C10119 has
  no anchors and nothing there is scored; its one number is agreement of one tracker with the
  other cameras. Every correction is agent-authored (`selected_by: agent`, provenance
  `multiview_consensus`, now also `candidate_ranking: decoder_score`). The consensus that
  authored them was built without C10379 but its members were seeded by geometric transfer of
  the C10379 human frame-0 masks. GPU: four queue passes on
  `runs/multiview-reprompt-20260920/code-snapshot-625321a/` (`git archive 625321a src/battle`,
  94 files): decode 6.7 s, C10379 arm 403 s (worker 341 s), decode 5.9 s, C10119 arm 370 s
  (worker 327 s), 19:01-19:18 UTC; every `gpu_check` ok, no `NVRM`/`Xid`, one GPU process at a
  time. CC BY-NC 4.0.
- **Change (fixed, unfitted; the closing section's one pick).** `battle-multiview-reprompt
  decode --candidate-ranking ray|decoder_score` (`src/battle/multiview_reprompt.py`,
  `schemas.py`). `ray` is the default and the byte-identical seed-transfer pick (closest
  centroid ray, ties by decoder IoU); `decoder_score` keeps the same acceptance filters (area
  within [0.3, 3.0] x expected, ray <= 1.5 radii, policy budget) and takes the decoder's own
  IoU estimate first, ties by ray (rule (g) of the acceptance search). The ranking is written
  on every `RepromptDecision`, the decisions file, every provenance record, every
  `MultiviewAgentCorrection` (kept per correction because schedules carry earlier iterations
  forward) and the acceptance log; the human-schema `MuggledSAMMultiKeyframeCorrectionSchedule`
  is untouched (it is a human contract and hashed into the tracker's stream identity), so for
  the derived C10379 schedules the ranking lives in the bound `reprompt_provenance.json`.
  Tests: `tests/test_multiview_reprompt.py` +2 (stub decoder; a fixture where two candidates
  pass and the two rankings pick different ones; the ranking recorded on decisions, provenance,
  agent schedule and log under both source profiles); 617 default-tier tests pass, ruff clean.
- **C10379, iteration 1** (`runs/multiview-reprompt-20260920/variants/decoder-score/C10379/iter1/`;
  same others-only consensus, pm-append source, default gate). The plan is identical to the ray
  plan (same three chassis onsets 296 / 475 / 1049, same prompts, rear_body 1762 blocked) and
  the decode passes the same 8 / 6 / 7 of 8 candidates; only the pick changes:

  | onset | accepted: px, decoder IoU, ray mm | ray pick: px, IoU, ray mm | expected px | anchor <= 5 frames | human px | IoU vs human: accepted / ray pick / best in pool |
  | --- | --- | --- | --- | --- | --- | --- |
  | chassis 296 | 8,103, 0.82, 21 | 15,373, 0.59, 16 | 10,381 | 300 | 4,156 | 0.33 / 0.07 / 0.34 |
  | chassis 475 | 9,507, 0.86, 17 | 17,084, 0.68, 10 | 10,991 | none | - | - |
  | chassis 1049 | 8,726, 0.83, 28 | 23,288, 0.60, 18 | 12,319 | 1050 | 6,631 | 0.74 / 0.24 / 0.75 |

  The accepted masks are 0.53 / 0.56 / 0.37 of the ray picks' areas and 0.71-0.86 x the
  sphere's expected area (the ray picks were 1.5-1.9 x); at 1049 the accepted mask is within
  0.01 of the best candidate in the pool, at 296 nothing in the pool exceeds 0.34. Arm (full
  run, 1800 frames, `--max-side-length 1280 --prompt-memory-semantics append`): 341 s, 2.48 GiB.
- **Arm table** (`anchor_iou_arms.md`, regenerated with every row; all / ch / int / rb / cab,
  windows 279-408 / 573-722 / 1020-1172 / outside, hidden FP px, worker s, GiB):

  | arm | later corrections | all | ch | int | rb | cab | 279-408 | 573-722 | 1020-1172 | outside | hidden FP | s | GiB |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | pm-append (human) | 327 900 1172 1235 | **0.743** | 0.637 | 0.585 | 0.790 | 0.962 | 0.853 | 0.687 | 0.616 | 0.800 | 1265 | 341 | 2.54 |
  | seed-only-1280 | none | 0.591 | 0.404 | 0.228 | 0.789 | 0.959 | 0.648 | 0.628 | 0.579 | 0.527 | 1252 | 329 | 2.39 |
  | consensus-only iter 1 (ray) | 296 475 1049 | 0.590 | 0.291 | 0.316 | 0.810 | 0.959 | 0.573 | 0.651 | 0.637 | 0.517 | 1226 | 343 | 2.48 |
  | consensus-only iter 2 (ray) | + 333 653 983 1326 1590 | 0.663 | 0.519 | 0.397 | 0.787 | 0.959 | 0.653 | 0.661 | 0.747 | 0.606 | 1251 | 359 | 2.77 |
  | human-plus-consensus (ray) | 296 327 475 900 1049 1172 1235 | 0.730 | 0.591 | 0.585 | 0.791 | 0.960 | 0.823 | 0.679 | 0.560 | 0.834 | 1207 | 357 | 2.71 |
  | **consensus-only-ds iter 1** | 296 475 1049 | **0.659** | **0.609** | 0.288 | 0.788 | 0.959 | 0.720 | 0.694 | 0.677 | 0.566 | 1266 | 341 | 2.48 |
  | consensus-only-ds iter 2 | no onset | = iter 1 | | | | | | | | | | - | - |

  Chassis per anchor frame, ray -> ds: 300 0.08 -> 0.41, 370 0.00 -> 0.64, 400 0.00 -> 0.85,
  600 0.38 -> 0.36, 650 0.32 -> 0.39, 700 0.19 -> 0.60, 900 0.59 -> 0.60, 1050 0.30 -> 0.75,
  1100 0.78 -> 0.82, 1150 0.52 -> 0.51, 1200 0.60 -> 0.66, 1500 0.00 -> 0.76, 1700 0.00 -> 0.58
  (pm-append at 600 / 650: 0.25 / 0.20; at 1050: 0.00).
- **C10379, iteration 2: no onset.** Consensus rebuilt with the ds run as the C10379 member
  (`runs/multiview-part-consensus-first-minute-r1280-reprompt-ds-iter1/`, 70 s; the README's
  iteration-2 recipe): chassis consensus on 1800 frames, 7.78 mean views, **C10379 contradicts
  on 0 frames**; the only contradictions are rear_body `[1662,1667)` (5 frames, under the
  10-frame minimum) and `[1762,1794)` (2 statics agree). `plan --previous-plan`: 0 planned, 1
  blocked. The same plan against the others-only consensus (the recipe the ray iteration 2
  used; CPU-only diagnostic into `/tmp`, not kept): also 0 planned, 1 blocked. Under ray ranking
  this step had produced five new chassis onsets over 400+ frames (the blobs made C10379
  contradict the other cameras more, not less); under decoder-score ranking the loop is at a
  fixed point after three corrections. Nothing was decoded or run for iteration 2 because
  there was nothing to decode; the iteration-2 row equals iteration 1 by construction.
- **C10119, iteration 1** (`variants/decoder-score/C10119/iter1/`; consensus excluding C10119,
  reference pm-append included as a member, source the Sep 19 r1280 run via the
  agent-correction-schedule path). Plan identical to the ray plan (rear_body 1533
  `[1533,1800)`, 281 px, 6 statics, 8/8 pass). Decode: accepted `t001533-b01#3`, 1,652 px,
  decoder IoU 0.95, ray 4 mm = 0.18 radii, area x1.29 (ray pick `t001533-b02#3`: 1,802 px,
  0.93, 2 mm). Arm: 1800 frames, 327 s, 2.42 GiB. Agreement rebuilt with this run replacing the
  Sep 19 C10119 (`runs/multiview-part-consensus-first-minute-r1280-c10119-reprompt-ds-iter1/`,
  71 s): rear_body agreement **0.872 -> 0.997**, mean error 20.5 -> 0.7 px, 13 contradicting
  episodes -> 1 (`[1525,1530)`), episodes over all views 105 -> 90: identical to the ray arm to
  three decimals; the two runs' rear_body masks agree at 0.958 mean IoU over `[1533,1800)`
  (min 0.887) and are identical before 1533. Held for human scoring on the 26 prepared frames;
  `docs/labeling-sessions-2026-09-20.md` now scores both runs (`consensus-only`,
  `consensus-only-ds`). The caveat stands: after 1660 the majority sits on the screwdriver.
- **Reading, against the pre-registered diagnostic.** consensus-only-ds did **not** stay at the
  0.591 floor: it moved **0.069 of the 0.153 gap** (0.590 -> 0.659 against 0.743), on every
  window, most in 279-408 (0.573 -> 0.720, pm-append 0.853), then outside (0.517 -> 0.566, vs
  0.800), 573-722 (0.651 -> 0.694, vs 0.687) and 1020-1172 (0.637 -> 0.677, vs 0.616); in the
  last two windows it is level with or above the human-corrected run. The chassis, the only
  part the corrections touch, goes from 0.291 to **0.609, within 0.03 of the human's 0.637**,
  and with three corrections rather than the eight the ray loop accumulated. So on the chassis
  the acceptance *ranking* was the binding constraint, as the search predicted, and the
  ceiling it predicted holds where it said: at 600 / 650 (the `[573,722)` occlusion window)
  every arm including pm-append sits at 0.20-0.39 and the pool's best candidate was 0.32, and
  at 300 the accepted 0.33 is the pool's 0.34; those cells are the candidate set, not the rule,
  and the lever there is still prompts (a size prior from the target view's own history rather
  than the sphere model, or a second decoder). What the experiment exposes as the **remaining
  gap is the interior**: 0.288 (seed-only 0.228, pm-append 0.585), untouched because no other
  view tracks it and the consensus therefore never proposes a correction for it; of the 0.084
  still separating consensus-only-ds from pm-append, 0.074 is the interior and 0.007 the
  chassis. The plan's stop rule reads verbatim: 0.084 > 0.03 on the one view with anchors, so
  **multicam is a detector and, on the chassis, now a corrector at the human's level; on the
  interior it is neither**, because it does not see the part. What remains human is unchanged
  (chassis and interior frame-0 seeds, the four C10379 corrections as the reference arm, every
  anchor); the interior needs the human e4 seed before any view other than C10379 can observe
  it, which is the labelling session already ordered (b). No further GPU work was run beyond
  the two targets; the `decoder_score` ranking is a flag, not the default.
- **Deliverables.** `runs/multiview-reprompt-20260920/variants/decoder-score/{C10379/iter1,
  C10379/iter2,C10119/iter1}/`, `anchor_iou_arms.{json,md}`, `jobs_7..10_*.json`, `queue.log`,
  `logs/`, `code-snapshot-625321a/`; the two rebuilt consensus roots named above; README results
  section; `docs/labeling-sessions-2026-09-20.md` (a); this entry.

### Sep 21: FineBio first look (SAM3 zero-shot text prompts on one first-person clip)

- **What.** The first FineBio footage handled in this repository: a smoke to see what the
  existing SAM3 toolchain does on wet-lab first-person video, visualised in Rerun. Not an
  experiment; nothing is measured against ground truth and no accuracy is claimed.
- **Source and licence.** FineBio (Yagi et al., IJCV 2025), access by signed licence
  agreement, non-commercial research only, credentials received by email; the `fpv_test`,
  `fpv_all_w640`, `tpv_test`, `tpv_valid` video archives and the object-detection image set
  were downloaded and extracted under `data/raw/finebio/` (gitignored). Frames, videos, masks
  and the RRD are never committed or redistributed. Entries in `docs/SOURCES.md` and
  `docs/LICENSES.md`.
- **Clip.** `P03_01_01.mp4` (1920x1440, 29.97 fps, 168 s, head-mounted), source 60.000-80.000 s
  where the subject pipettes into a 6-well plate. Proxy 1280x960, 30 fps CFR, exactly 600
  frames, same libx264 arguments as the Assembly101 G2 proxies (checksums in the run
  manifest).
- **Route.** Approach A (reuse `battle-muggled-smoke`) was rejected in under ten minutes of
  reading: its text-prompt path is bound by `require_smoke_range` to exactly the approved
  first 10 s, and `G2PreprocessingManifest` is `assembly101_g2_preprocessing` with pinned
  Hugging Face provenance fields; forging that for a FineBio clip would be dishonest and
  loosening it is not a smoke-sized change. Approach B instead: `scripts/finebio_sam3_smoke.py`
  (committed) with a `track` phase under the MuggledSAM interpreter that copies
  `muggled_worker.py`'s calls (`get_detector_context`, `encode_image` at max side 1280,
  `encode_exemplars(text=...)`, `generate_detections` at threshold 0.40, top score per prompt,
  `encode_prompt_memory_from_mask`, `step_video_masking_multiplex` with 1 prompt / 4 frame
  memory entries, `encode_frame_memory`), the worker's GPU guard with
  `--allow-gpu-neighbour 2071175`, and an `export` phase under the Battle interpreter that
  writes the RRD with the exporter's archetypes (AssetVideo + VideoFrameReference, RGBA
  EncodedImage masks, labelled Boxes2D, Scalars series, pinned blueprint), a six-frame contact
  sheet and `manifest.json`.
- **Prompts and frame-0 result.** `pipette` 0.863 (4 candidates), `centrifuge` 0.746 (1),
  `pipette tip box` 0.867 (8; the top pick is a red-lidded box at the left of the bench, not
  verifiable as a tip box from the frames); `6-well plate` no detection (presence 0.108),
  `tube rack` no detection (presence 0.113). Three slots tracked.
- **Runtime.** 139.8 s wall for 600 frames (load 4.4 s, detection 0.6 s, about 22.5 s per 100
  frames), peak VRAM 2.51 GiB allocated (3.05 GiB reserved), GPU busy 22:41:24-22:43:45 local.
  GPU coordination with the concurrent multiview chain: checked at 22:35, 22:38, 22:40, 22:41
  (hull in its CPU stage, no tracker process, only the calibration worker on the GPU); the
  chain's hull finished at 22:44:04 and its stage 2 stopped on its own error at 22:44:17, so
  no tracker job was ever blocked.
- **Observations on this clip** (details in the run README): the transparent 6-well plate,
  central and hand-held at frame 0, has no mask on any frame; `tube rack` found none of the
  three visible racks; the pipette mask excludes the gripping glove and survives that
  occlusion on every tracked frame, but the slot drops 29 frames (106-111, 305-310, 330-346)
  each time the pipette is carried to the right frame edge during a pan, reacquiring by
  itself; the centrifuge and tip-box slots have output on all 600 frames through the pans
  (neither is occluded in this clip); the two manual pipettes on the bench are never tracked
  (one slot per prompt).
- **Claim boundary.** Detector scores, object scores and predicted IoU are the model's own
  numbers; masks were looked at on six frames and a handful more, not reviewed; nothing here
  says how SAM3 performs on FineBio.
- **Deliverables.** `runs/finebio-sam3-smoke-20260921/` (README, `recording.rrd`,
  `contact_sheet.png`, `observations.jsonl`, `masks/`, `manifest.json`, phase logs),
  `scripts/finebio_sam3_smoke.py`, `docs/SOURCES.md` and `docs/LICENSES.md` entries, this
  entry. View: `uv run rerun runs/finebio-sam3-smoke-20260921/recording.rrd`.

### Sep 22: VRAM-aware GPU guard (`battle.gpu_guard`; worker, `battle-muggled-smoke`, overnight queue)

- **Defect.** The guard was process-name based: any non-whitelisted python-ish process on the
  card refused the start, so the `showtime` video player (389 MiB, `python3` to nvidia-smi)
  blocked queue jobs 6-8 on the night of Sep 21 (`runs/sam3-views-r1280-4part-20260921/
  README_guard_note.md`); a 1.4 GiB game or a browser would do the same, while the risk the
  guard exists for (this machine hard-crashed once under GPU load: two model processes, or too
  little headroom) was not what it measured.
- **Change.** Stdlib-only `src/battle/gpu_guard.py`, imported by the SAM3 worker as a sibling
  module under the MuggledSAM interpreter and by the driver and queue as `battle.gpu_guard`.
  `--gpu-guard vram` (default) reads `nvidia-smi --query-gpu=memory.total,memory.used` and
  `--query-compute-apps`, classifies each neighbour by `/proc/<pid>/cmdline`: `own_repo_model`
  (a Battle worker by script basename, the calibration workspace included since it holds a
  SAM3 model), `known_benign` (compositor, browser, player, game or Wine program, the Rerun
  viewer, or anything under 512 MiB that is not a Battle worker), else `unknown`. It refuses
  beside another `own_repo_model` whose PID was not named by `--allow-gpu-neighbour`, beside an
  `unknown` neighbour above 2048 MiB, or when `total - used - sum(reservations)` (half a model
  neighbour's current usage, zero for a benign one) is below 1.5 x the expected peak
  (`--expected-peak-vram-bytes`, else a profile from recorded manifests: `sam3_1280` 2.6 GiB,
  `sam3_1080p` 3.4 GiB, `four_part` 3.4 GiB, `dam4sam_large` 7 GiB, `unknown` 4 GiB; the driver
  picks the SAM3 profile from `--max-side-length`, the queue infers it from the job command or
  takes `gpu_profile`). Every neighbour (pid, class, MiB, cmdline basename, reservation), the
  headroom arithmetic and the decision are recorded under `runtime_settings.gpu_guard` and in
  the queue's `gpu_check` event (schema `battle-gpu-guard/1`). `--gpu-guard strict` is the
  Sep 21 rule byte for byte. The queue re-evaluates the guard 30 s into each job (the job's own
  process tree excluded) and logs a `gpu_recheck` event naming any neighbour that appeared; it
  kills nothing.
- **What changed for the known cases.** The calibration worker (PID 2071175) still needs
  `--allow-gpu-neighbour 2071175` and now also charges a 599 MiB reservation against headroom;
  the Rerun viewer is benign by basename `rerun` or the `/rerun_sdk/rerun_cli/` path whatever
  its size, as before; a video player, game or browser no longer blocks. New: the queue gates
  on the guard before every job, so a job with no worker-level guard (e.g.
  `battle-multiview-reprompt decode`) is now refused beside a foreign Battle worker unless its
  PID is allowed on the queue command line or in the job's argv. Tonight's card (16303 MiB,
  3064 used, compositor + calibration worker): a `sam3_1280` job sees headroom 12640 MiB
  against 3995 required.
- **Tests.** `tests/test_gpu_guard.py` and queue tests (classification, headroom arithmetic,
  the four decisions, own-process exclusion, strict parity against a verbatim copy of the
  88ba96e function, provenance round trip, `gpu_check`/`gpu_recheck` events); default tier 646
  passed, ruff clean. No GPU job was run.

### Sep 21, night: the three labelling sessions acted on (human-accepted seeds, an interior seed from two human masks, the four-part rerun, the C10119 rear_body finding and a distractor guard)

- **Claim boundary first.** Everything scored below is against one person's choice of SAM3
  image-decoder masks on a handful of frames of one camera at a time (C10379 51 + 1 hidden
  cells on 13 frames; C10119 63 + 1 hidden on 26; e4 52 + 23 hidden on 26): **review evidence
  that ranks arms against each other, not ground truth**, not a dataset, no accuracy claim. The
  human's 24 seed-proposal decisions are one person's choice among agent decoder candidates on
  one frame per cell. The interior seeds on six views are agent-selected from geometry and held
  until the human confirms them (session (e)); every consensus correction is agent-authored.
  GPU: `runs/multiview-seeds-human-accepted-20260921/` decode 33 s; eight view runs 332-352 s
  worker each (`runs/sam3-views-r1280-4part-20260921/`, queue 52 min incl. export, 2.39 GiB peak
  every view); five C10379 arms and one C10119 arm 390-432 s worker (2.54-2.99 GiB); every job
  ran beside the human's calibration workspace worker (PID 2071175, 1.2 GiB), named to the guard,
  never touched; no `NVRM`/`Xid`. Code snapshots `code-snapshot-4551b73`, `-eceae60`, `-259b4f7`.
  CC BY-NC 4.0.
- **The three sessions (numbers; records committed in `4066572`).**
  (a) C10119: 63 labelled / 1 hidden (interior 401) / 40 skipped (rear_body on 17 frames, cabin
  on 23); chassis and interior on all 26 frames. (b) e4: 52 / 23 hidden / 29 skipped; interior
  labelled on 10 frames (224, 304, 484, 604, 654, 1104, 1154, 1204, 1414, 1654) and hidden on 14;
  rear_body hidden on 9 (304, 374, 484, 504, 554, 864, 1054, 1504, 1544). (c) seed proposals,
  `configs/qa/seed_proposal_decisions_2026-09-21.json` (edited in the template, copied out,
  template restored): **chassis 7 / 8 accepted** (C10390 undecided) despite the part's 0.525
  held-out IoU on C10379, **rear_body 8 / 8** (candidate 0 everywhere = the B3 seed itself),
  **interior 1 / 8** (C10390 at 427; the seven rejections name the hand, the chassis or the
  boundary). Scoreboards as labelled (`docs/qa/anchor-scoreboard-{c10119,e4}-20260920.md`):
  C10119 r1280 0.474 all (chassis 0.758, rear_body 0.839, cabin 0.859, interior 0.000 with 25
  cells missing: no slot); e4 r1280 chassis 0.845, rear_body 0.262, cabin 0.831, interior 0.000
  (10 missing), **hidden FP 48,705 px over 8 rear_body cells** (1.2-20.4k px each).
- **The C10119 rear_body finding, verbatim from the anchors.** The one consensus correction
  written for C10119 on Sep 20 (rear_body 1533, both rankings) raised its agreement with the
  other cameras from 0.872 to 0.997 and dropped its rear_body anchor IoU from **0.839 to 0.270**:
  IoU **0.00 on all six labelled rear_body frames 1541-1771** (r1280 there 0.82-0.88), the
  corrected slot being 1.0-1.7k px on the yellow screwdriver where the top-down human sees the
  rear body. The majority the correction joined sits on the screwdriver: the recording-1 human
  marked C10379 rear_body hidden at 1700 with `distractor_confusion`, and the e4 human marked
  rear_body hidden at 1504 and 1544 (and on 7 more frames) with the r1280 e4 run putting
  1.2-20.4k px there. Agreement with the majority late in the minute measured agreement with the
  distractor. **Did the decoder know?** No: the accepted candidate at 1533 (`t001533-b01#3`) had
  the pool's top decoder IoU estimate 0.95 (pool 0.66-0.95; the ray pick 0.93), and the
  tracker's own rear_body object score on C10119 does not fall after 1533 (median 8.31 over
  `[1533,1800)` vs 8.81 before; the corrected runs 7.97 / 8.31, predicted IoU 0.90-0.93); on
  C10379 pm-append the rear_body score dips only from 8.00 to 7.14 over `[1660,1800)`. The
  acceptance-search pool at C10379 1700 (hidden cell, 6 prompts) held 24 candidates with decoder
  IoU 0.26-0.61, so the "rule" that would have abstained there (score >= 0.7) accepts the C10119
  1533 pick at 0.95. Neither the decoder nor the tracker separates the screwdriver from the
  rear body; only the human marks do.
- **Distractor guard (`battle-multiview-reprompt plan --distractor-guard`, default on; commit
  `42a9a14`).** An onset of a part is blocked when a human anchor record on **any** view marks
  that part hidden within +-60 frames of it, frames mapped through the clock rules (e4 1504 ->
  C10379 1500 -> C10119 1501); every mark, its record fingerprint and the suppressed onsets are
  written on the plan (`distractor_guard`), the reason on the onset says "review evidence ...
  not ground truth". Re-planned on the Sep 20 excl-C10119 consensus the guard **suppresses
  rear_body 1533** (e4 marks at 1504 / 1544, 32 frames away; `--no-distractor-guard` plans it);
  on C10379's Sep 20 plan it touches nothing (no human marks the chassis hidden). Its cost shows
  on the ego camera: e4's interior `hidden` marks at 704 / 864 / 1054 / 1504 are out-of-frame
  marks (the C10119 human sees the interior at 701 / 861 / 1051 / 1501), and on the four-part
  C10379 plan below they suppress four of the five interior onsets; a per-view `hidden` reason
  (out of frame vs distractor) is the missing field, so both arms were run.
- **Task 1, human-accepted seeds (`battle-human-accepted-seeds apply`, commit `4551b73`).**
  `runs/multiview-seeds-human-accepted-20260921/<VIEW>/seed_manifest.json` (8): every accepted
  proposal candidate is the view's seed with provenance `agent_proposed_human_accepted`
  (decisions-file fingerprint and the human's note on the seed; `acceptance_basis:
  human_accepted_proposal`); C10390 chassis undecided keeps the B3 (Sep 18) seed; each decision
  is copied into its `proposal.json`; per-part provenance now flows into the run metadata
  (`mixed_per_part` at the manifest level). IoU of the accepted chassis with the B3 chassis it
  replaces: 0.98 on C10119 and e4 (the human picked what the Sep 18 transfer had); rear_body
  identical (1.00) on all eight.
- **Task 2, interior from two human masks (`interior-plan|decode|accept`, commits `4551b73`,
  `eceae60`).** The interior is *not* on the table at frame 0: it is in the subject's left hand
  on every camera (293 mm above the fitted plane). Triangulated from the C10379 human frame-0 seed
  and the C10119 human anchor at frame 41 (the earliest human interior mask on a second camera,
  40 analysis frames later; the C10119 f41 -> f81 centroid motion is **2.4 px** < 5, the C10379
  run's own interior centroid drifts 17 px over 0-40): centre (-115.9, 280.5, -33.2) mm,
  reprojection 11.3 / 11.1 raw px, radius 26.1 mm (28.3 / 24.0 per view); the same-instant
  variant (C10379 run mask f40 x human f41) lands 12 mm away at 3.7 / 3.4 px. Check in hand at
  C10379 300 x C10119 301 x e4 304 (three human masks, one pose instant): 10.3 / 3.0 / 2.1 raw
  px, and the two-static point projects 9.4 px from the human e4 centroid: the three sessions'
  masks agree geometrically. The frame-0 sphere projected into the seven statics as the search
  winner prompt (m0.25, one box, other-part centroid negatives: none fell inside any box) and
  decoded (33 s). Acceptance: decoder top score among candidates inside [0.3, 3.0] x the two human
  masks' area carried by focal/depth **and with median luminance <= 2 x the human masks' (31 on
  both cameras)**, then joint triangulation of the picks with the two human observations (>= 3
  views inside 30 raw px). The luminance rule was added after the first decode: on C10395 and
  C10404 the hand holding the part (luminance 95-124) passed the centroid and area rules.
  Outcome per view (`interior/interior_report.md`): **C10095, C10115, C10119, C10390, C10404
  accepted** (6 views agree, 1.6-13.0 raw px, areas 780-1632 px, luminance 24-35); **C10118 not
  accepted** (its pick reprojects 39 px, a dark blob of 3.7k px); **C10395 not accepted** (all
  four candidates are the hand); **e4 blocked** (the interior is behind the head camera at
  frames 0 and 4, depth -18 / -16 mm; the first human e4 interior mask is 224, which a per-slot
  start frame would take: not built, the B3 report's gap stands). The accepted C10119 candidate
  vs the human's C10119 mask at 41: IoU 0.08, centroid 28.7 px (the hand moved between 0 and
  41); the four-part run's interior mask at 41 vs that human mask: **0.65** (below). Proposal
  sheets for the human, one row per view: `runs/labeling-sessions-20260921/interior_proposal_sheets/`,
  template `configs/qa/interior_seed_decisions.template.json`; session (e) in
  `docs/labeling-sessions-2026-09-20.md`. Provenance `geometric_from_two_human_views` until then.
- **Task 3, eight views at 1280 pm-append with four parts** (`runs/sam3-views-r1280-4part-20260921/`;
  the seeds above; C10118, C10395 and e4 ran with three parts). 1800 / 1800 frames each, worker
  332-352 s, first output 4.3-4.6 s, 2.39 GiB; presence 0.96-1.00 everywhere. Three jobs were
  first refused by the worker guard because the human opened a video player (`showtime`, 389 MiB,
  reported by nvidia-smi as python3) and were re-queued with its PID tolerated
  (`README_guard_note.md`); nothing was killed. **Consensus** (reference pm-append,
  `runs/multiview-part-consensus-first-minute-r1280-4part-20260921/`, 100 s; others-only and
  excl-C10119 roots beside it): the interior now has a consensus on **1800 frames from 3.7 mean
  views** (Sep 20: none), which the human-corrected C10379 agrees with on **0.97** of frames
  (C10404 1.00, C10115 0.92, C10119 0.90, C10095 0.83, **C10390 0.22**: its interior drifts, mean
  error 106 px); C10379 rear_body is now contradicted on **139 frames** (`[1524,1533)`
  `[1664,1800)`; Sep 20 seeded build: 0), i.e. the seven other views no longer sit where C10379's
  rear_body slot sits after 1664, which is the screwdriver; chassis agreement of **C10115 fell
  0.95 -> 0.12** and C10404 0.96 -> 0.79; episodes over all views 80 -> 145. **Hull**
  (`runs/multiview-visual-hull-first-minute-r1280-4part-20260921/`, 918 s): interior hull on 675
  frames, median 27 voxels, hull-vs-mask IoU 0.05-0.25 on the views that carry it; chassis hull
  frames 1535 -> 978 and C10379 chassis hull-vs-mask 0.397 -> 0.226 (the smaller, disagreeing
  chassis set carves less). `summary_vs_seeded_with_hull.md` has every row.
- **What the four-part seeds did on the two views with anchors** (`docs/qa/anchor-scoreboard-{c10119,e4}-20260921.md`;
  a run without a slot scores 0 on that part's labelled cells, so `all` is comparable):

  | view / run | ch | int | rb | cab | all | hidden FP px |
  | --- | --- | --- | --- | --- | --- | --- |
  | C10119 r1280 (Sep 19 seeds, 3 parts) | 0.758 | 0.000 (25 missing) | 0.839 | 0.859 | 0.474 | 0 |
  | C10119 seeded-3part (B3) | 0.758 | 0.000 | 0.842 | 0.948 | 0.478 | 0 |
  | C10119 consensus-only-ds (Sep 20, rear_body 1533) | 0.758 | 0.000 | **0.270** | 0.862 | 0.392 | 0 |
  | C10119 human-accepted-4part | **0.339** | 0.114 | 0.838 | 0.949 | 0.350 | 5,133 |
  | C10119 4part + guarded consensus-only-ds (6 corrections) | 0.667 | **0.290** | 0.833 | 0.945 | **0.554** | 3,795 |
  | e4 r1280 | 0.845 | 0.000 (10 missing) | 0.262 | 0.831 | 0.558 | 48,705 (8 cells) |
  | e4 seeded-3part | 0.842 | 0.000 | 0.262 | 0.940 | 0.567 | 20,083 |
  | e4 human-accepted-4part | 0.851 | 0.000 | **0.349** | 0.938 | 0.589 | **16,474** (7 cells) |

  C10119, per cell: the geometric interior seed **is on the interior while it is in the hand**
  (IoU vs the human 0.65 / 0.67 / 0.79 / 0.35 at 41 / 81 / 221 / 301) and is lost when the part
  is placed inside the chassis (0.00-0.08 from 371 to 1201); worse, from 371 the **interior slot
  takes the chassis** (interior run masks 4.9-6.0k px where the chassis anchor is 4.6-5.6k px;
  the chassis slot shrinks to 0.3-1.8k px, IoU 0.00-0.15 through 1051): the human-accepted
  chassis seed is 0.98 IoU with the B3 seed that held 0.758, so the loss is the fourth slot, not
  the seed. The guarded re-prompt on the 4-part excl-C10119 consensus (plan: chassis 367, 495,
  704, 1484, 1557 and interior 1163 accepted, 5.1-7.1k px, decoder 0.54-0.86; interior 1081
  suppressed by the e4 mark at 1054, rear_body 1529 blocked by the 2-static-view gate) restores
  the chassis to 0.667 (371-1201 back to 0.60-0.95) and puts the interior at **0.74-0.87 on
  1651-1771** after the 1163 correction (0.39 at 1501 / 1541): the first interior tracked on a
  second camera. e4: rear_body 0.262 -> 0.349 and hidden FP 48.7k -> 16.5k px with the same
  rear_body seed (IoU 1.00), so again the change is the slot set; the cabin seed 0.83 -> 0.94.
- **Task 3, C10379 arms from the four-part others-only consensus** (`runs/multiview-reprompt-20260921/`;
  plan: chassis 477, 604, 1049 and interior 1124 planned; interior 665 / 719 / 804 / 1466 and
  rear_body 1525 suppressed by the guard; the 296 chassis onset of Sep 20 no longer exists because
  the other views' chassis moved; `variants/no-guard/` plans and accepts all nine, interior 1466
  rejected by the policy budget). Accepted interior masks are **4.9k / 2.5k / 1.2k / 9.4k px**
  against a human interior of ~1-2k px on this view: the sphere radius from the other views'
  interior slots is 49-50 mm where the human masks give 26 mm, because those slots leak onto the
  chassis once the part is placed. All / ch / int / rb / cab, windows, hidden FP, worker s, GiB
  (`docs/qa/anchor-scoreboard-c10379-arms-20260921.md`):

  | arm | later corrections | all | ch | int | rb | cab | 279-408 | 573-722 | 1020-1172 | outside | hidden FP | s | GiB |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | pm-append (human) | 327 900 1172 1235 | **0.743** | 0.637 | 0.585 | 0.790 | 0.962 | 0.853 | 0.687 | 0.616 | 0.800 | 1265 | 341 | 2.54 |
  | seed-only-1280 | none | 0.591 | 0.404 | 0.228 | 0.789 | 0.959 | 0.648 | 0.628 | 0.579 | 0.527 | 1252 | 329 | 2.39 |
  | consensus-only-ds iter 1 (Sep 20, 3-part consensus) | 296 475 1049 | 0.659 | 0.609 | 0.288 | 0.788 | 0.959 | 0.720 | 0.694 | 0.677 | 0.566 | 1266 | 341 | 2.48 |
  | **consensus-only-ds 4-part, guard** | 477 604 1049, int 1124 | **0.659** | 0.534 | **0.363** | 0.790 | 0.959 | 0.648 | 0.649 | 0.749 | 0.603 | 1219 | 424 | 2.54 |
  | consensus-only-ds 4-part, no guard | + int 665 719 804, rb 1525 | 0.659 | 0.543 | 0.356 | 0.789 | 0.958 | 0.648 | 0.649 | 0.686 | 0.656 | 1241 | 432 | 2.76 |
  | **human-plus-consensus-ds 4-part, guard** | human + 477 604 1049 1124 | **0.742** | 0.636 | 0.583 | 0.790 | 0.961 | 0.853 | 0.663 | 0.609 | 0.822 | 1261 | 427 | 2.76 |
  | human-plus-consensus-ds 4-part, no guard | human + all nine | 0.741 | 0.638 | 0.580 | 0.790 | 0.960 | 0.853 | 0.663 | 0.607 | 0.821 | 1274 | 418 | 2.99 |

  Interior per anchor frame, seed-only / ds Sep 20 / ds 4-part guard / pm-append: 900 0.00 /
  0.14 / 0.15 / 0.84; 1050 0.00 / 0.10 / 0.09 / 0.09; 1100 0.00 / 0.14 / **0.84** / 0.26; 1150
  0.00 / 0.43 / 0.43 / 0.48; 1200 0.00 / 0.28 / 0.26 / 0.55; 1500 0.00 / 0.00 / 0.17 / 0.68;
  1700 0.34 / 0.00 / 0.17 / 0.87. The 1100 gain precedes the 1124 interior correction and comes
  from the chassis corrections (477 / 604 instead of 296 / 475) freeing the slot; the frames
  after 1124 do not move (1150 0.43 -> 0.43, 1200 0.28 -> 0.26), and the no-guard arm's three
  extra interior corrections move 700 not at all (0.20) and 900 to 0.78. Chassis 0.609 -> 0.534
  because the 4-part consensus no longer flags 296 (370 / 400 fall back to 0.14 / 0.08 from
  0.64 / 0.85).
- **Answer to the question asked.** The interior on C10379 moves from 0.288 to **0.363** under
  consensus-only-ds (0.585 human); the gap to pm-append stays **0.084** (0.659 vs 0.743) and is
  still the interior (0.222 of it) plus 0.10 of chassis lost with the 296 onset. The other views
  now *see* the interior, but what they track after the part is placed is interior-plus-chassis
  (radius 49 mm vs 26), so the correction the consensus authors is the wrong size and the
  decoder's top pick at that box is a blob; the one place the interior is tracked well on a
  second camera is C10119 after its own 1163 correction (0.74-0.87 over 1651-1771). Adding the
  four-part consensus corrections to the human's costs nothing (0.742 vs 0.743). Verdict
  unchanged in kind, moved in detail: multicam detects; on the chassis it corrects at the
  human's level when the 296 window is flagged (Sep 20) and 0.10 below when it is not (today);
  on the interior it now proposes but the proposals are chassis-sized. What remains human:
  chassis and interior frame-0 seeds (the human confirmed the chassis ones today and rejected 7
  of 8 interior rest-frame ones), the four C10379 corrections as the reference, every anchor,
  and the `hidden` marks that the guard now reads.
- **Blocked / open.** Per-slot start frame in the multiview profile: not built (e4 interior at
  224 and a direct human C10119 interior seed at 41 both need it; the interior is behind e4's
  camera at 0-4). The guard's `hidden` has no out-of-frame vs distractor field; e4's interior
  marks are the former and suppress interior onsets. The four-part multiplex destabilises the
  chassis slot on C10119 (0.758 -> 0.339 with a seed 0.98 IoU to the old one) and C10115
  (agreement 0.95 -> 0.12): slot interplay, not seeds, and the re-prompt loop repairs it on the
  one view that has anchors. The other agent's VRAM-aware queue guard (`8a3ee2a`) landed during
  this pass; the last decode was re-run with the queue-level `--allow-gpu-neighbour`. Recording
  2's 13 frames are being labelled (the server on 8765 stayed up throughout).
- **Deliverables.** Commits `4066572` (records, decisions, tables), `88ba96e` (worker
  `--allow-gpu-neighbour`), `4551b73` + `eceae60` (`battle-human-accepted-seeds`), `42a9a14`
  (distractor guard), `259b4f7` (tests), this entry's commit (docs/qa tables, sessions doc, this
  entry). Roots: `runs/multiview-seeds-human-accepted-20260921/` (8 manifests, `decisions_report.md`,
  `interior/interior_report.md`, code snapshot), `runs/labeling-sessions-20260921/`,
  `configs/qa/interior_seed_decisions.template.json`, `runs/sam3-views-r1280-4part-20260921/`
  (8 runs, queue logs, `README_guard_note.md`), the three consensus roots and the hull root
  named above, `runs/multiview-reprompt-20260921/` (`C10379/iter1/`, `variants/no-guard/C10379/iter1/`,
  `C10119/iter1/`, `anchor_iou_{arms,c10119,e4}.{json,md}`, stage scripts, queue logs).

### Sep 21: FineBio shipped detector, first run (MMDetection DINO and Deformable DETR on the CPU beside the SAM3 smoke)

- **What.** The detector FineBio ships with its paper, the authors' MMDetection checkpoints
  fine-tuned on their 35 wet-lab classes, run on the same 600-frame `P03_01_01` 60-80 s proxy
  that the SAM3 zero-shot smoke tracked, and both methods put in one Rerun recording side by
  side. The question was what a supervised in-domain detector finds that five SAM3 text prompts
  missed (the transparent cell-culture plate, the tube racks). Qualitative: the FineBio COCO
  annotations are not on this machine, nothing is scored, no accuracy claim for either method.
- **Source and licence.** Same FineBio clip and terms as the first look (non-commercial
  research, nothing redistributed). Code: MMDetection v3.3.0 (`44ebd17b`, Apache-2.0), mmcv
  2.1.0, mmengine 0.10.7; the two FineBio configs from `aistairc/FineBio/object_detection` at
  `cb8d16ef` (MIT repository; `_base_` = the MMDetection COCO configs, `num_classes=35`).
  Weights: the authors' `dino.pth` (SHA-256 `e6399531…29d94`, 579 MB, epoch 12, full training
  checkpoint with optimizer state and `dataset_meta`) and `deformable-detr.pth`
  (`35982a45…bd6c3`, 515 MB, epoch 50). The `finebio.s3.abci.ai/ckpts/` URLs in the
  object_detection README no longer resolve (NXDOMAIN at every resolver tried); the main README's
  2025-12-15 update points to Google Drive, fetched with `gdown`. Entries added to
  `docs/SOURCES.md` and `docs/LICENSES.md`.
- **Environment.** Own venv at `/home/nick/src/finebio-detector` (Python 3.10, torch
  2.1.2+cpu, torchvision 0.16.2+cpu), built by the idempotent
  `scripts/install_finebio_detector.sh` in about 8 minutes wall time. mmcv did not have to be
  compiled: OpenMMLab publishes a CPU wheel for torch 2.1 (`cpu/torch2.1.0/mmcv-2.1.0-cp310`),
  and its `mmcv.ops` (deformable attention, NMS) import; the script keeps the source build
  (`MMCV_WITH_OPS=1`, no CUDA) as the fallback path, not exercised. Two fixes on the way:
  mmdet's editable install needs `--no-build-isolation` and `setuptools<80` (its `setup.py`
  imports `torch.utils.cpp_extension`, which on torch 2.1 imports `pkg_resources`), and gdown
  6.4 dropped `--fuzzy`. Verification: `init_detector` + `inference_detector` on a blank
  1333x800 image, 35 classes from the checkpoint's `dataset_meta` matching the configs, 2.3 s
  cold. The battle env is untouched; the export phase runs under it (Rerun 0.37.1).
- **GPU coordination.** At 23:18 local the other session's queue had a tracker job running
  (`human-plus-consensus`, worker pid 2261783, 3.6 GB on the GPU), so the CPU default held and
  `CUDA_VISIBLE_DEVICES=""` was set on every detect call. The detect phase snapshots `pgrep`,
  `nvidia-smi` and the newest `queue.log` into `runtime_settings.gpu_coordination`; by the first
  sampled pass (23:28) the tracker list was empty and the queue log ended with `queue_end`
  (03:20:57 UTC), so the plan's rule would have allowed the GPU, but the CPU rate was already
  under the 2 s/frame bar and the env has no CUDA build, so the GPU was never used. The
  script's `--device cuda:0` refuses unless trackers are absent, no other model process holds
  the GPU and the newest queue log ends with `queue_end`.
- **Run.** `scripts/finebio_dino_detect.py detect` (detector interpreter): mmdet's default
  test pipeline (`Resize` to fit 1333x800, so 1066x800 for the 1280x960 proxy), every
  detection >= 0.05 recorded, 0.30 used for display and counts. DINO sampled pass (every 10th
  frame plus the six contact-sheet frames, 63 frames) at 1.47-1.53 s/frame on 16 CPU threads,
  then all 600 frames in 978 s (median 1.64 s/frame): 74,411 boxes >= 0.05, 21,803 >= 0.30
  (22-46 per frame, median 37). Deformable DETR: sampled pass at 1.36 s/frame, then all 600
  frames in 832 s (median 1.38 s/frame), 58,708 boxes >= 0.05, 20,695 >= 0.30; 98.4% of
  DINO's boxes >= 0.5 have a same-class DDETR box at IoU >= 0.5, with DDETR's scores higher
  for nearly every class (plate hand-held median 0.70 vs 0.48). `export` (Battle interpreter,
  11 s) writes `recording.rrd` (27 MB: video asset, `detector/<model>/boxes` with class ids 1-35 and family colours,
  per-class count series, the SAM3 smoke's boxes and RGBA masks re-logged under `sam3/`, a
  pinned blueprint with one 2D view per method over the same video and the count series
  below), two contact sheets, `class_counts.md` and `manifest.json`.
- **Observations on this clip** (DINO unless stated; details and numbers in the run README).
  The transparent 6-well plate is `cell_culture_plate` on all 600 frames: 0.32-0.60 while held
  in the hands (frames 0-38), 0.74-0.94 (median 0.90) once on the bench, through both pans;
  SAM3's `6-well plate` prompt found nothing. The racks SAM3's `tube rack` missed get boxes on
  every frame under `50ml_tube_rack`, `micro_tube_rack` (with about 13 `micro_tube` boxes per
  frame on the tubes in it), `8_tube_stripes_rack` (the red rack the SAM3 `pipette tip box`
  prompt picked at frame 0, same box to within 4 px), `15ml_tube_rack` (one box per blue-capped
  tube), `magnetic_rack` (the blue-and-white rack) and the four tip-rack classes on the
  transparent tip boxes, often two classes on one box. 3-4 pipette-family boxes per frame: the
  hand-held pipette (`blue_pipette` on 459 of the 499 frames where a DINO box overlaps SAM3's
  pipette box at IoU >= 0.3, median IoU 0.70, about equal area) plus the two manual pipettes
  and the blue-bodied pipette lying on the bench, each its own box; SAM3 tracked one slot.
  `left_hand` on the left glove on all 600 frames; `right_hand` absent on 100-144, 299-328,
  331-374, the pans that carry the right hand out of the right edge, the same intervals in
  which the SAM3 pipette slot dropped; on frames 0-40 the right-hand box (up to 17% of the
  frame) encloses 89% of the upright pipette's box. No detection at any score >= 0.05 is
  centred on a fiducial marker on the six contact-sheet frames; the odd boxes are a
  `micro_tube` on the horizontal pipette's tip cone, `cell_culture_plate_lid` duplicating the
  plate box on 54 frames (<= 0.45), and `pcr_machine` 0.82-0.90 on every frame for a white block
  with a grid of wells whose identity the frames do not settle. Bench objects that stay in view
  hold their scores through the pans (`centrifuge` 0.75-0.89, `vortex_mixer` 0.72-0.90,
  `8_tube_stripes_rack` 0.84-0.92, never below 0.70); objects that leave the frame drop out and
  return; the hand-held pipette is the least stable (`blue_pipette` median 0.61, 0.30-0.84).
  Eight classes never reach 0.30 (tips other than `blue_tip`, the two lids, spin columns,
  `tube_without_lid`).
- **Claim boundary.** Scores are the detectors' softmax outputs; box labels were read on about
  a dozen frames and the per-frame numbers, not reviewed against truth; the class names are
  the detector's, and where an object's identity could not be told from the frames (the
  "PCR machine", the tip boxes' colours) the README says so. Nothing here says how well either
  method does on FineBio.
- **Deliverables.** `runs/finebio-dino-20260921/` (README, `recording.rrd`, `contact_sheet.png`,
  `contact_sheet.deformable-detr.png`, `detections*.jsonl`, `detect_result*.json`,
  `class_counts.md`, `manifest.json`, logs; ignored), `scripts/finebio_dino_detect.py`,
  `scripts/install_finebio_detector.sh`, `docs/SOURCES.md` and `docs/LICENSES.md` entries, this
  entry. View: `uv run rerun runs/finebio-dino-20260921/recording.rrd`.

### Sep 22: dedup pass 1 (hashing, paths, fingerprints, masks, observation loaders; commits `41052ac`, `b6d84af`, `ad2b3f3`, `39f4b3c`, `d813829`, `3f19d31`)

- **What.** The first of the three deduplication passes over `src/battle` and `scripts`: the
  byte-identical helper copies (SHA-256 loops, `relative_uri`, `ArtifactFingerprint`
  factories, mask decode / IoU / centroid / overlap, `observations.jsonl` readers) replaced by
  four shared modules, one commit per family, every family proven equivalent by rebuilding
  three CPU artifacts before and after. No GPU was used (`CUDA_VISIBLE_DEVICES=""` on every
  command; the human's calibration worker stayed on the card). Nothing in a manifest, digest
  or URI moved.
- **Harness** (`scripts/dedup_equivalence.py`, commit `41052ac`; tests
  `tests/test_dedup_equivalence.py`). `snapshot --label before|after` rebuilds into
  `runs/dedup-equivalence/<label>/`: the eight-view r1280 `pm-append` consensus
  (`battle-build-multiview-part-consensus`, 73-101 s), the 19-arm anchor scoreboard
  (`battle-anchor-iou` plus `runs/anchor-scoreboard-20260919/scoreboard_table.py`, 6 s) and the
  `reference_masks`-only interaction review v4 with `--verify-fingerprints` (22 s) followed by
  `rerun rrd verify`; it also records `pytest -q`, `pytest -q -m real_data` and `ruff check` in
  `summary.json`. The `before` snapshot at `20fdffd` reproduces the committed consensus and
  scoreboard runs exactly (manifest and `anchor_iou.json` equal after the volatile keys, the
  `per_frame.jsonl` and the scoreboard markdown byte for byte). `compare` deep-compares
  JSON/JSONL after dropping `generated_at`, `created_at`, `elapsed_seconds`, `duration_s`,
  `runtime_seconds`, `run_id`, every key ending in `_at` and a `comparison_id` that embeds a
  timestamp, normalising `runs/dedup-equivalence/<label>` inside strings; Markdown as text
  with the same substitution; `.npz` array by array (the zip container carries a write time);
  PNG and everything else by bytes. **RRD determinism.** Two builds of the same code do not
  give the same `.rrd` bytes: the recording carries `RecordingInfo:start_time` and a
  `log_time` per row, the SDK batches rows into chunks of varying size (`num_rows_max` 579 vs
  593), and the blueprint store gets fresh view/container UUIDs, so `rerun rrd compare
  --unordered` fails too. The recording is therefore compared by a chunk-independent content
  digest through `rerun.experimental.RrdReader`: for every entity path and column the rows of
  all chunks are concatenated, sorted on the recording's own timelines and hashed (`RowId`,
  `log_time`, `RecordingInfo:start_time` dropped, string columns label-normalised); the
  blueprint as a UUID-normalised multiset of rows per entity kind. 282 columns; two builds of
  unchanged code digest identically and a one-character change to a logged text is caught. A
  `{uri, sha256}` whose `uri` points into the scratch root (the index's own recording, guide
  and contact sheet; the consensus's `per_frame.jsonl`) is compared as a placeholder and
  re-hashed against its file on each side, because the review guide embeds its own absolute
  path and its digest legitimately differs between labels.
- **Family 1, hashing** (`b6d84af`; 29 files, +230/-261 in src+scripts). New stdlib-only,
  Python-3.10-compatible `src/battle/fs_common.py` (`sha256_file`, `write_json` with each
  caller's `indent`/`sort_keys` and an `atomic` tmp+replace form, `run_timestamp`
  `%Y%m%dt%H%M%Sz`, `relative_uri` with the resolving default and a `resolve=False` form),
  imported as `battle.fs_common` or as a sibling module by the workers (`muggled_worker`,
  `muggled_calibration_worker`, `four_part_video_worker`, `ego_diagnostic`,
  `dam4sam_streaming`, the finebio and LM-EEC scripts; the samurai/dam4sam/wilor/LM-EEC/
  FineBio interpreters are 3.10, so no `datetime.UTC`). `digest_cache` hashes through it. The
  17 local SHA-256 loops and thin wrappers and the 6 `_write_json` copies and 12 run-id
  timestamp sites now call `digest_cache.sha256_file` (battle venv), `fs_common.sha256_file`
  (workers) or `fs_common.write_json` / `run_timestamp`. Harness equivalent (13 files).
- **Family 2, paths** (`ad2b3f3`; 16 files, +43/-101). The 15 `relative_uri` /
  `_relative_uri` / `_relative` definitions become imports of `fs_common.relative_uri`; no
  caller caught the strict forms' `ValueError`. `four_part_contract.relative_uri` and
  `multiview_consensus.relative_uri` stay as re-exports for their importers;
  `muggled_smoke.relative_uri` stays a one-line wrapper with `resolve=False` because the seed
  / score / calibration manifests written through it have always compared paths as given
  (identical whenever the root is `Path.cwd()`, which every CLI uses, but not proven for other
  roots, so the form is kept); `human_qa.repository_relative_uri` keeps its
  must-be-inside-the-repository error contract. Harness equivalent.
- **Family 3, fingerprints** (`39f4b3c`; 22 files, +139/-270). `schemas.fingerprint(path,
  repository_root, *, source="measured", verify=False)` beside `ArtifactFingerprint`
  (`source` admits the model's two literals) replaces 19 per-module factories: the 12
  identical ones plus the public `fingerprint()` of `egoexo_correspondence` and
  `multiview_reprompt`; `athena_hands` and `kineo_multiview` (uncached `hashlib` loop, same
  digest), `multiview_seed_transfer` and `calibration_rescale` (a `source` parameter) call it
  directly; `assembly101_reference` (with its `verify` flag), `assembly101_clock_offset` and
  `exploratory_comparison._file_fingerprint` keep their existence check and call it inside.
  Harness equivalent. **Note.** This commit swept in one hunk that was not part of the pass:
  another worker's uncommitted `api: Literal[..., "muggledsam_sam3_exemplar_detector"]` line
  in `schemas.py`, staged because `git add schemas.py` takes the whole file; that worker's
  `0ef1868` records it. Later commits were checked hunk by hunk before staging.
- **Family 4, masks** (`d813829`; 13 files, +161/-102, of which `mask_ops.py` is 120). New
  `src/battle/mask_ops.py`: `decode_mask_png` (re-exporting `mask_cache`), `mask_area`,
  `mask_centroid(mask, *, pixel_center=False, scale=1.0)`, `mask_iou(a, b, *,
  empty_union=0.0)` with a shape check, `overlap_fraction(mask, region, *, empty=0.0)` and
  `overlap_fraction_union`. The six IoU copies agreed on the arithmetic and differed at the
  edges, so the edge is the parameter and each former copy is one call:

  | former copy | missing (`None`) mask | empty union | shapes differ |
  |---|---|---|---|
  | `interaction_review._mask_iou` | not accepted | 0.0 | numpy error |
  | `ensemble_reference.mask_iou` | None | 0.0 | numpy error |
  | `segmentation_disagreement.mask_iou` | `(0.0, "missing_in_a|b")`; both: `(None, "both_missing")` | `(None, "both_missing")` | numpy error |
  | `detector_scorecard._iou` | both: NaN; one: 0.0 (an all-False mask counts as missing) | NaN | resize b nearest |
  | `egoexo_correspondence.iou` | not accepted | None | ValueError |
  | `multiview_seed_transfer.iou` | not accepted | 0.0 | numpy error |

  `segmentation_disagreement` keeps its reason labels and `detector_scorecard` its presence
  test and nearest resize as wrappers; the others import the shared function under their old
  names. Centroids: `ensemble_reference` `(x, y)` or None; `multiview_seed_transfer` the
  pixel centre as an array, raising on empty; `multiview_consensus.mask_centroid_raw` the
  pixel centre through `cv2.moments` scaled to raw pixels. Overlap: two single-region copies
  with 0.0 on an empty mask, one union-of-others copy with NaN. **Evidence.** All 7200
  `pm-append` masks decode identically through `cv2.IMREAD_GRAYSCALE > 0`, PIL `L > 0` and
  `mask_cache.decode_mask_png`, and `cv2.moments` and the numpy mean agree bit for bit on
  all 7200 (every mask non-empty), so one arithmetic serves the three centroids and the PIL
  copies in `external_smoke_import`, `exploratory_comparison`, `four_part_comparison` and the
  cv2 reader in `multiview_seed_transfer` call `decode_mask_png`. `tests/test_mask_ops.py`
  pins every caller's edge contract; the tests were run green against the copies before the
  copies were deleted. Harness equivalent (the consensus centroids and the review's
  reference masks are in it). Left in place: the worker-side readers (`four_part_video_worker`,
  `dam4sam_streaming`, `muggled_calibration_worker`, `ego_diagnostic`) and the uint8 overlay
  readers that resize or blend (`g3_contact_sheet`, `four_part_segmentation`,
  `calibration_rescale`); `seed_search._mask_area(path)` already reads through `mask_cache`.
- **Family 5, observations** (`3f19d31`; 15 files, +145/-220, of which `observations.py`
  is 103). New `src/battle/observations.py`: `load_observations` (schema-validated, the
  `muggled_smoke` reader with its `path:line` error), `observations_by_frame`,
  `rebuild_tracker_observations` (the tolerant known-fields rebuild of the external-worker
  drivers: `view_id` / `analysis_frame_index` / `source_seconds` and per object `object_id` /
  `label` / `confidence` / `box` / `mask`, everything else a worker wrote ignored, with a
  `hand_builder` hook) and `object_for_label(frame, label, *, require_mask=True)`. The two
  readers stay distinct on purpose: `VersionedModel` forbids extra keys, so the strict reader
  cannot read a worker's file that carries diagnostics the schema does not declare, and the
  rebuild must not be used on exporter output whose hands or scores it would drop.
  `dam4sam_video`, `samurai_video`, `grounding_dino_sam2_video`, `boxmot_track` (no mask key
  in its records, so the shared mask branch is inert) and `four_part_segmentation` (now also
  skips blank lines) bind `_load_observations` to the rebuild; `wilor_hands` keeps only its
  `PerFrameHand` builder; `muggled_smoke` and `fine_substep_pipeline` re-export;
  `egoexo_correspondence.load_observations` wraps the run directory. Six `next(...)` label
  lookups (`interaction_review`, `interaction_review_v4` x2, `ensemble_reference`,
  `multiview_review`, `multiview_consensus`) call `object_for_label`. Harness equivalent.
- **Totals.** Over the five dedup commits, src+scripts: 954 lines deleted, 718 inserted
  (312 of them the three new modules with their docstrings), net -236; inside the existing
  modules -548. Tests: `pytest -q` 646 -> 696 passed (50 new: harness 19, `fs_common` 11,
  `schemas.fingerprint` 1, `mask_ops` 14, `observations` 5), 9 skipped, 54 deselected;
  `-m real_data` 52 passed before and after; `ruff check` / `ruff format --check` clean on
  every tracked file of this pass. Final harness `compare` at `3f19d31`: 13 files, no
  non-volatile difference; the RRD by content digest as above. Worker imports were
  smoke-checked under their own interpreters (`muggled_worker`, `muggled_calibration_worker`,
  `ego_diagnostic`, `finebio_sam3_smoke` under MuggledSAM 3.14; `four_part_video_worker`,
  `dam4sam_streaming` under samurai 3.10; `lm_eec_driver` under LM-EEC 3.10;
  `finebio_dino_detect` under the detector venv 3.10), `--help` or import only.
- **Not done here** (later passes per the plan): the six cloned video drivers and worker
  `_extract_frames` (pass 2, needs GPU smokes), Rerun logging helpers, queue writers,
  contact-sheet grids and `metrics.py` (pass 3), the README pointers. Test counts above
  exclude the other worker's `test_sam3_appearance.py`, `test_exemplar_pool.py` and
  `test_detector_scorecard_v2.py`, which landed in the same window.

### Sep 22: SAM3 appearance detectors on three cameras (Track A of the exemplar plan; commits `0ef1868`, `fb426c2`, `d914bef`, `d5298a8`, `7f608e5`, `329eea1`, `bc794a1`)

- **Claim boundary first.** Truth is the human review anchors on **13 frames of C10379 (52
  cells), 26 of C10119 (64) and 26 of e4 (75, 23 hidden)**: one person's choice of SAM3
  decoder masks, review evidence that ranks detectors against each other, not ground truth, not
  a dataset, no accuracy claim. Six tracked runs are scored, each against its own camera's
  anchors (`pm-append`; C10119 `r1280`, `4part`, Sep 20 `consensus-only-ds` with the 1533
  correction; e4 `r1280`, `4part`); a cell is failed when its anchor IoU is below 0.5 or a
  hidden cell carries more than 300 px. The reference masks are human masks too (C10379 frame-0
  seeds and corrections at 327 / 900 / 1235, the agent-selected 1172 left out; C10119 and e4
  human-accepted frame-0 seeds plus every human anchor on their earliest interior frame, 41 and
  224), so the cells on those frames are scored **leave-reference-out** (the row records
  `reference_frames_used`). Every rate carries its counts. GPU: six offline passes of 966-1094 s
  each (4.1-4.6 GiB peak) and one 14 s zero-shot pass, all through the queue on code snapshots,
  beside the human's calibration worker (PID 2356892, named to the guard, never touched); no
  `NVRM`/`Xid`. CC BY-NC 4.0.
- **Tool.** `src/battle/sam3_appearance.py pass` runs under the MuggledSAM interpreter (stdlib,
  numpy, torch, cv2; `gpu_guard` as a sibling) and, per frame at 1280 (square sizing), (a)
  mask-pools the 1024-channel ViT token map, bilinearly upsampled to the 4x grid, under each
  part's tracked mask (`native/embeddings.npz`, L2-normalised, plus the reference and distractor
  embeddings) and (b) runs SAM3's visual-exemplar detector: exemplar tokens cut from the human
  masks' boxes on their own frames (`encode_exemplars`, `include_coordinate_encodings=False`,
  tokens of several reference frames concatenated), positives only and positives plus negative
  boxes at the other parts' masks on the same frame, from a same-view set and a cross-view set
  (the 13 C10379 masks applied on C10119 and e4), batched eight sets at a time with a padding
  mask, the segmentation head run on the top-10 tokens, areas / centroids / IoU with the tracked
  mask on the device (`detections.jsonl`; top-10 masks written at the anchor frames). A frame
  costs 0.55-0.61 s for 8-16 exemplar sets. `battle-detector-scorecard-v2 --spec records.json`
  (`src/battle/detector_scorecard_v2.py`, imports the v1 module) scores several records with the
  ten v1 detectors plus `emb_self` (1 - max cosine to own-part references), `emb_swap` (max
  other-part cosine - own), `emb_distractor` (screwdriver reference = pm-append rear_body at
  1690, a non-anchor frame, minus own), `det_iou` (1 - best IoU of the tracked mask with a top-10
  detection), `det_top_iou`, `det_presence` (1 - presence), `det_disagree` (top detection's
  centroid > 40 px from the tracked mask), `det_top_score`, with `_pos` and `_xv` (cross-view)
  variants; failure classes from the view's mapped windows, `distractor` for anchor frames >=
  1195 outside them, `out_of_frame` for the e4 hidden cells; in-sample and leave-one-frame-out
  per record, pooled tables, leave-one-record-out (top-3 and the R>=0.8 threshold chosen on the
  other cameras; only detectors defined on half the training cells are eligible), named tests,
  presence-as-visibility, `confidence_v2/<record>/confidence.jsonl`. Reference specs and the
  decoder client live in `src/battle/exemplar_pool.py`. Tests with a numpy stub model
  (`tests/test_sam3_appearance.py` 10, `tests/test_detector_scorecard_v2.py` 4,
  `tests/test_exemplar_pool.py` 6); default tier 733 passed with the concurrent refactor's
  `tests/test_fine_substep_pipeline.py` ignored (its working tree broke the import at the time);
  ruff clean on these files.
- **C10379 pm-append (52 cells, 9 failed), overall AUROC / R>=0.8 P / R (TP/FP/FN/TN).**
  `sam3_score` **0.960**, 0.62 / 0.89 (8/5/1/38) as on Sep 20; **`emb_self` 0.943**, 0.57 / 0.89
  (8/6/1/37); **`det_top_iou` 0.925**, 0.67 / 0.89 (8/4/1/39); `emb_self_xv` (C10119 + e4
  references) 0.912; `area_vs_seed` 0.907; `det_iou` 0.889, 0.47 / 0.89 (8/9/1/34);
  **`emb_distractor` 0.873**; `det_iou_pos` 0.822; `emb_swap` 0.767; `det_top_score` 0.760;
  `det_disagree` 0.676; **`det_presence` 0.376** (presence is 0.99-1.00 on every frame of this
  camera: the concept is always somewhere in view). Per class: `rotation_swap` (24 cells, 7
  failed) `emb_self` 0.98, `det_top_iou` 0.94, `sam3_score` 0.98; `distractor` (12, 2 failed:
  1500 rear_body 0.00 and the 1700 hidden screwdriver) `emb_self`, `emb_distractor`,
  `det_top_iou`, `det_presence` all **1.000** (2/0/0/10) where the Sep 20 detectors had
  `sam3_score` 0.90 and the DAM4SAM disagreement 0.35-0.45; `occlusion_leak` and `outside` still
  have no failed cell. Top-3 becomes `sam3_score`, `emb_self`, `det_top_iou`: in-sample R>=0.8
  **0.667 / 0.889 (8/4/1/39)** against Sep 20's 0.533 / 0.889 (8/7/1/36); leave-one-frame-out
  pooled AUROC **0.897** (Sep 20 0.835) but the threshold still does not transfer: **0.429 /
  0.333 (3/4/6/39)** at the R>=0.8 point chosen on 12 frames (Sep 20 0.364 / 0.444). Named
  cells: **1050 chassis** (IoU 0.00, the 273 px collapsed slot) is in the record's top decile
  for `det_iou` 0.91, `det_disagree`, `emb_self` 0.43, `emb_self_xv`, `sam3_score`,
  `area_vs_seed`, both DAM4SAM disagreements; **1700 rear_body** (hidden, 1265 px on the
  screwdriver) for `emb_distractor` (+0.18: the pooled embedding is nearer the screwdriver
  reference, cosine 0.91, than any rear_body reference, 0.73), `det_top_iou` (the top detection
  does not overlap the tracked mask), `det_disagree` (top centroid 414 px away), `emb_self`,
  `det_presence` (0.988, the second-lowest presence on the camera) while `sam3_score` is 0.001 (the tracker is
  confident, as the Sep 21 entry found).
- **C10119 (26 frames).** `4part` (64 cells, **43 failed**: the interior slot takes the chassis
  from 371, the chassis slot shrinks): **`emb_swap` 0.944**, `emb_swap_xv` 0.874, `area_vs_seed`
  0.792, `sam3_score` 0.696, `emb_self` 0.663, every `det_*` 0.42-0.50 (the exemplar detector
  finds chassis pixels under both slots, so overlap says nothing); combined in-sample 0.97 / 0.81
  (35/1/8/20), leave-one-frame-out **0.971 / 0.791 (34/1/9/20)**. `r1280` (39 cells with a
  mask, 2 failed; 25 interior cells have no slot and are excluded as structural): too few
  failures to rank (`emb_swap_xv` 0.90, `area_vs_seed` 0.73). **`consensus-only-ds` with the
  1533 correction** (39 cells with a mask, 8 failed, six of them rear_body 1541-1771 at IoU 0.00
  on the screwdriver): `area_vs_seed` 0.952, `emb_self_xv` 0.904, `sam3_score` 0.898,
  `emb_self` 0.879, `det_iou_xv` 0.863, `det_top_iou_xv` 0.858, `det_disagree_xv` 0.825,
  `det_iou` 0.815, `det_top_iou` 0.806, `det_disagree` 0.806, `det_presence_xv` 0.800; combined
  in-sample 0.78 / 0.88 (7/2/1/29), leave-one-frame-out **0.750 / 0.750 (6/2/2/29)**. **The
  1533 named test:** on the six corrected rear_body cells `det_disagree` = 1 and `det_top_iou`
  = 1 on all six (the top rear_body detection is elsewhere than the tracked screwdriver),
  `det_iou` 0.42-1.00, `emb_self` 0.27-0.35 (own-reference cosine 0.65-0.73) against 0.23-0.29
  on the uncorrected `r1280` at the same frames (a shift of 0.03-0.06, small); in the record's own top decile: `emb_self_xv`
  at 1541 / 1651 / 1751 / 1771, `emb_self` at 1701-1771, `det_iou` at 1541 / 1771, `det_iou_xv`
  at 1541 / 1731, `sam3_score` at 1701-1771 (1 - sigmoid 0.000-0.001: on this record the whole
  camera is confident so a small dip ranks high; at 1541 and 1651 nothing but the embedding and
  detection rows flag). `det_presence` is 1.000 on every one of these cells: the rear body *is*
  in view, so presence cannot see a localisation failure.
- **e4 (26 frames, 65 cells with a mask; 10 interior anchors have no slot on either run).**
  `r1280` (16 failed, 8 of them rear_body on hidden cells with 1.2-20.4k px): **`emb_self`
  0.979**, 0.87 / 0.81 (13/2/3/31); `sam3_score` 0.903 on 50 cells; `emb_self_xv` 0.895;
  **`det_presence_xv` 0.882**, 0.68 / 0.94 (15/7/1/42); `emb_swap_xv` 0.875; `area_vs_seed`
  0.750; `det_presence` 0.763; `det_iou` 0.662. `4part` (12 failed): `emb_self` 0.928,
  `emb_self_xv` 0.869, `area_vs_seed` 0.838, `det_presence_xv` 0.826, `sam3_score` 0.800.
  Leave-one-frame-out 0.857 / 0.750 (12/2/4/32) and 0.562 / 0.750 (9/7/3/46). **The 23 hidden
  cells** (`out_of_frame` class): the 14 interior cells have no slot on either run (0 px,
  correct by construction, nothing to flag); of the 9 rear_body cells, `r1280` puts >300 px on
  8, and something in the record's top decile flags 7 of them (864 with 20.4k px: eight
  detectors; 1054: `emb_swap`, `emb_self`, `det_presence_xv`; 1504 / 1544: only `det_iou_xv` /
  `det_iou_pos`; **504 with 3.4k px: nothing**; 304 with 1.8k px: `det_iou_xv` only). **Presence
  as a visibility detector is a negative result**: mean presence 0.999 on the 23 hidden cells and
  0.999 on the 41 visible ones (AUROC 0.46 same-view, 0.53 cross-view; `det_top_score` 0.55);
  with these exemplars SAM3 says "present" on every e4 frame, out of frame or not.
- **Pooled (324 cells with a mask, 90 failed, six records; raw values pooled).** **`emb_self`
  0.894** on 285 cells, 0.69 / 0.80 (72/32/18/163); `area_vs_seed` 0.852 (324); `sam3_score`
  0.806 (291); `emb_swap_xv` 0.802; `emb_self_xv` 0.762; `emb_swap` 0.760; `det_iou` 0.702;
  `det_iou_xv` 0.677; `det_top_iou` 0.665; `det_presence_xv` 0.590; `det_disagree` 0.576;
  `det_presence` 0.535. Per class: `rotation_swap` (62 / 18) `emb_self` **0.955**, `emb_swap_xv`
  0.899, `emb_swap` 0.870, `emb_self_xv` 0.861, `area_vs_seed` 0.856, `sam3_score` 0.853,
  `det_iou` 0.831; `occlusion_leak` (32 / 8, now with positives from C10119 and e4)
  `det_iou_xv` **0.988** on 18 cells, `area_vs_seed` 0.958, `emb_swap_xv` 0.938, `emb_self`
  0.915, `sam3_score` 0.739; `distractor` (110 / 36) `emb_self` **0.911**, `area_vs_seed` 0.827,
  `emb_self_xv` 0.814, `sam3_score` 0.794, `det_iou` 0.663; `out_of_frame` (46 / 13)
  `det_presence_xv` **0.927**, `area_vs_seed` 0.848, `det_presence` 0.570 (the embedding rows are
  defined on 15 of these cells only); `outside` (74 / 15) `emb_self` 0.936, `area_vs_seed`
  0.894, `emb_swap` 0.834. **Leave-one-record-out** (top-3 and the threshold chosen on the other
  five cameras' records): the chosen set is **`emb_self` + `area_vs_seed` + `sam3_score` on four
  of six folds** (`emb_self_xv` replaces `area_vs_seed` for C10119-4part, `emb_swap_xv`
  replaces `sam3_score` for consensus-only-ds); held-out pooled AUROC 0.97 (pm-append), 0.93 /
  0.89 (e4), 0.68 / 0.65 / 0.63 (C10119); at the transferred threshold recall 0.88-1.00 on five
  records with precision 0.08-0.67 (pm-append 9/14/0/29, e4-r1280 14/7/2/42, e4-4part
  11/10/1/43, C10119-r1280 2/22/0/15, consensus-only-ds 8/23/0/8) and the slot-swap record the
  exception (C10119-4part 14/6/29/15: recall 0.33, its failures need `emb_swap`). The reading
  stands as on Sep 20: the ranking transfers across cameras, a fixed threshold does not.
- **Cross-view transfer, answered on this evidence.** The C10379 human masks used as exemplars
  on another camera work as *detectors*: `emb_self_xv` 0.90-0.91 on pm-append (with C10119 + e4
  references) and consensus-only-ds, 0.87-0.90 on e4, `emb_swap_xv` 0.87-0.90 on the C10119
  swap records, `det_iou_xv` 0.99 on the pooled occlusion-leak cells and 0.86 on the 1533
  record, `det_presence_xv` 0.88-0.93 where the part leaves the ego frame; on the same-camera
  swap record (C10119-4part) `emb_self_xv` is at chance (0.47) while `emb_swap_xv` is 0.87. The
  same set as a *candidate pool* on C10119 is the one pool that clears the correction bar for the
  chassis (Track B entry below).
- **Confidence series v2.** `runs/sam3-exemplar-20260922/scorecard/with-mask/confidence_v2/<record>/confidence.jsonl`
  (7,200 rows per record, every detector, the combined rank-average of the record's top-3,
  confidence, abstain at the record's in-sample R>=0.8 point). Not logged into a v6 rebuild:
  `interaction_review_v4.py` is the concurrent refactor's dirty file tonight; the command is the
  v6 build in the README with `--confidence runs/sam3-exemplar-20260922/scorecard/with-mask/confidence_v2/C10379-pm-append`
  in place of `--confidence runs/detector-scorecard-20260920/pm-append`.
- **Deliverables.** `runs/sam3-exemplar-20260922/` (README, `passes/`, `scorecard/{all-cells,with-mask}/`,
  reference specs, queue logs, code snapshots), the three modules and their tests, the
  `--candidate-source` / `--target` hooks in `multiview_reprompt.py`, the `api` literal in
  `schemas.py` (it landed inside the refactor's `39f4b3c` sweep), `sam3_appearance.py` named an
  own-repo worker in `gpu_guard.py`, CLI entries in `pyproject.toml`. README lines for the two
  new CLIs are below rather than in `README.md` (the user's uncommitted edit is there tonight):

  ```text
  - `battle-exemplar-pool references|pass|truth-plan|decode-control|compare|rec2-references|zero-shot`
    (`src/battle/exemplar_pool.py`): human-mask reference specs per view, the GPU appearance pass
    (`src/battle/sam3_appearance.py` under the MuggledSAM interpreter: mask-pooled ViT embeddings
    and visual-exemplar detections), exemplar-vs-decoder candidate pools on a view's anchor
    cells, and recording-1 exemplars applied zero-shot on recording 2.
  - `battle-detector-scorecard-v2 --spec records.json --output DIR [--exclude-missing]`: the
    detector scorecard over several cameras' runs with the appearance detectors, leave-one-frame
    and leave-one-record-out, named tests, `confidence_v2/<record>/confidence.jsonl`.
  - `battle-multiview-reprompt decode --candidate-source image_decoder|exemplar_detector|both
    --exemplar-references SPEC [--exemplar-set same_view|cross_view] [--target PART ...]`.
  ```

### Sep 22: exemplar detections as correction candidates (Track B of the exemplar plan; the C10119 chassis arm)

- **Claim boundary first.** Every IoU is against one person's choice of SAM3 decoder masks: 63
  C10379 truth cells (51 anchors + 12 human corrections at 0 / 327 / 900 / 1235; the 1700 hidden
  cell prompted but not scored) and 64 C10119 anchor cells, both prompted from a 4-part consensus
  that excludes the scored view (others-only for C10379, excl-C10119 for C10119), two boxes with
  margins 0.25 / 0.60 and other-part negatives exactly as `battle-multiview-reprompt plan` builds
  them. The candidate pools compared: the image decoder's base pool (the Sep 20 control, decoded
  again on these plans: 20-22 s each) and the exemplar detector's top-10 detections whose
  centroid lies inside the prompt boxes, same-view and cross-view references, positives-only and
  positives+negatives, and a `+size` variant (B4: area within [0.5, 2] x the part's frame-0
  seed). The acceptance rule is the tool's (area within [0.3, 3.0] x the consensus expected area,
  centroid within 1.5 projected radii, highest score wins: the decoder's IoU estimate for the
  decoder pool, the detection score for the exemplar pools) with the ray test taken in the image
  plane; harm = accepted with IoU < 0.4; leave-frames-out rotates 3 held-out anchor frames and
  picks a score floor on the rest (floor 0 was chosen in every fold). **The bar (chassis and
  interior held-out >= 0.6 with harm <= 0.15) is met by one pool on one camera**, so one arm
  ran. GPU: the pool masks come from the Track A passes; five exemplar decodes of 8-11 s, one
  arm of 424 s (`code-snapshot-7f608e5`), through the queue beside the calibration worker.
  CC BY-NC 4.0.
- **Tool.** `battle-exemplar-pool truth-plan` (the acceptance-search plan for any view from its
  anchor mask set; `correction_acceptance_search.py` is the refactor's dirty file, so the C10119
  extension lives in `exemplar_pool.py` and imports the harness's schema and `run_decode`),
  `decode-control`, `compare`; `battle-multiview-reprompt decode --candidate-source
  exemplar_detector|both --exemplar-references SPEC --exemplar-set same_view|cross_view
  [--target PART]`: a `decoder_factory` hook on `decode_plan` hands in
  `exemplar_pool.ExemplarDecoderClient`, which starts `sam3_appearance.py serve-jsonl`, a
  `batch_decode` server in the calibration worker's protocol whose candidates are the gated
  exemplar detections (`api: muggledsam_sam3_exemplar_detector`, `iou_score` = detection score,
  one empty mask when nothing lies inside the box so the rule rejects; `both` merges the image
  decoder's candidates from the same loaded model, one GPU process). `--target` keeps only the
  named parts' onsets (the dropped ones are listed in `decode_result.json`); `candidate_source.json`
  is written beside the decisions because the decision and schedule schemas are hashed human
  contracts.
- **Pool comparison, C10379 (per part: oracle mean IoU, cells with a >= 0.6 candidate, accepted
  mean IoU / harm, held-out mean IoU / harm).** Chassis (16): decoder 0.579, 6 / 16, 0.487 / 5
  of 16, **held-out 0.519 / 0.27**; exemplar same-view posneg **0.701, 12 / 16**, 0.611 / 3 of
  16, **held-out 0.594 / 0.23** (pos 0.601 / 0.23; +size 0.588 / 0.23): a higher ceiling and
  fewer harmful picks than the decoder, **0.006 below the bar on IoU and 0.08 above it on harm**;
  the three harmful cells are 300 / 600 / 650 (0.31 / 0.35 / 0.37), the occlusion windows where
  no pool has a candidate above 0.38, exactly where the Sep 20 search put the ceiling. Elsewhere
  the exemplar pick is 0.50-0.92 (1050: 0.85 where the decoder-score arm reached 0.75 and
  pm-append 0.00; 1500: 0.67; 1700: 0.57). Interior (16): decoder 0.326, 3 / 16, held-out
  **0.185 / 0.92**; exemplar 0.547, 7 / 16, held-out **0.323 / 0.50** (pos 0.321 / 0.50); the
  B4 size prior does not help (**0.336 / 0.60**): the candidates are 2.6-4.7k px against 0.7-4.2k
  human interiors, the wrong pixels rather than the wrong size (IoU 0.00-0.01 at 327 / 400 /
  1100), the concept-not-instance failure the plan predicted. rear_body (15): decoder 0.792 /
  0.08, exemplar **0.858 / 0.00** (pos 0.839 / 0.00), both clear the bar; cabin (15): 0.935 /
  0.00 vs 0.940 / 0.00, both clear.
- **Pool comparison, C10119.** Chassis (26): decoder 0.722, 23 / 26, held-out **0.573 / 0.27**;
  exemplar same-view (references: the human-accepted frame-0 seed and the anchor at 41) 0.671,
  held-out **0.583 / 0.24**; **exemplar cross-view (the 13 C10379 human masks) 0.751, 22 / 26,
  accepted 0.713 with 1 harmful pick of 25, held-out 0.713 / 0.04: clears the bar**, the only
  pool that does. Interior (25): decoder 0.571 (11 / 25), held-out 0.412 / 0.56; exemplar
  same-view 0.177 (one reference frame), cross-view 0.412, held-out 0.371 / 0.58: nobody clears.
  rear_body (9 cells, 1541-1771 where the other cameras sit on the screwdriver): decoder 0.661,
  held-out 0.587 / 0.33; exemplar 0.10-0.14 oracle, 2 of 9 accepted: inside the consensus box the
  exemplar detector returns almost nothing that overlaps the human rear body. cabin (3 cells):
  decoder 0.949, exemplar cross-view 0.856, too few for a held-out number.
- **Exemplar decodes on the existing plans** (decisions, decoder-score ranking). C10379, the
  Sep 21 no-guard 4-part plan (9 planned onsets), same-view exemplars: chassis 477 / 604 / 1049
  accepted (8.4k / 7.9k / **6.6k px** at 1049 against a 6.6k human chassis at 1050, scores
  0.80-0.92), interior 665 and 1124 accepted (4.3k / 3.9k px, scores 0.38 / 0.62), interior
  719 / 804 / 1466 **rejected: no detection inside the box**, **rear_body 1525 rejected**
  (both prompts empty) where the decoder pool would have accepted the screwdriver. C10119, the
  Sep 20 plan with the single **rear_body 1533** onset: **rejected by the same-view and by the
  cross-view exemplars** (no detection centroid inside the consensus box; the decoder pool had
  accepted `t001533-b01#3` at 0.95 and dropped rear_body from 0.839 to 0.270). The exemplar
  pool abstains on the two screwdriver onsets by not finding the concept where the cameras
  agree.
- **The arm.** Parts clearing the bar: rear_body and cabin on C10379 (rear_body's only onset,
  1525, was rejected; cabin has none), chassis on C10119 with cross-view exemplars. So
  **`C10119 4part consensus-only-exemplar (chassis)`**: the Sep 21 guarded plan decoded with
  `--candidate-source exemplar_detector --exemplar-set cross_view --target chassis`
  (interior 1163 dropped by `--target`), chassis **367 / 495 / 704 / 1484 / 1557 accepted**
  (6.1-8.2k px, scores 0.87-0.90 except 1484 at 0.28), agent schedule = the human-accepted
  seeds + the five exemplar corrections, full run 424 s. `battle-anchor-iou` on the 64 C10119
  cells (all / ch / int / rb / cab, windows 280-409 / 574-723 / 1021-1173 / outside, hidden FP):

  | run | corrections | all | ch | int | rb | cab | 280-409 | 574-723 | 1021-1173 | outside | hidden FP |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | r1280 (Sep 19, 3 parts) | none | 0.474 | 0.758 | 0.000 (no slot) | 0.839 | 0.859 | 0.436 | 0.364 | 0.335 | 0.510 | 0 |
  | human-accepted-4part | none | 0.350 | 0.339 | 0.114 | 0.838 | 0.949 | 0.155 | 0.028 | 0.176 | 0.436 | 5,133 |
  | 4part + guarded consensus-only-ds (decoder pool) | ch 367 495 704 1484 1557, int 1163 | 0.554 | 0.667 | 0.290 | 0.833 | 0.945 | 0.487 | 0.361 | 0.330 | 0.616 | 3,795 |
  | consensus-only-ds Sep 20 (rear_body 1533) | rb 1533 | 0.392 | 0.758 | 0.000 | 0.270 | 0.862 | 0.436 | 0.364 | 0.335 | 0.399 | 0 |
  | **4part consensus-only-exemplar (chassis, cross-view)** | ch 367 495 704 1484 1557 | 0.513 | **0.740** | 0.106 | 0.849 | 0.945 | 0.484 | 0.363 | 0.348 | 0.558 | 4,829 |

  The chassis goes from 0.339 (the 4-part seeds alone) to **0.740 with five exemplar
  corrections at the same onsets where the decoder pool reached 0.667**, within 0.02 of the
  3-part run that never lost the slot (0.758); rear_body is unchanged (0.849), cabin unchanged.
  `all` is 0.513 against the decoder arm's 0.554 because that arm also carried an interior
  correction (0.290 vs 0.106) and the interior pool did not clear the bar here, so none was
  applied. Iteration 2 was not run (the plan caps at two; the budget went to the passes).
  No C10379 arm ran: chassis 0.594 / 0.23 and interior 0.323 / 0.50 are under the bar, and the
  one rear_body onset was rejected.
- **Recording 2, zero-shot (optional step).** Recording 1's 13 C10379 human masks as the only
  reference set on recording 2's C10379 proxy at the 13 calibration frames (14 s), scored
  against the **7 masks the human has accepted so far** (chassis 456 / 483 / 552 / 1191, cabin
  456 / 1191, rear_body 1191) and the **5 hidden marks** (interior 456 / 552 / 1191, rear_body
  456 / 552): chassis top detection IoU **0.79 / 0.93 / 0.91 / 0.72** (posneg; pos 0.00 at 456
  where the top pick is a 12k px blob, else 0.95 / 0.92 / 0.75), cabin **0.00** at both frames
  (the top pick is 13k px; a 0.86-0.89 candidate sits lower in the top-10), rear_body 1191
  **0.00** (best in the top-10 0.15); mean top IoU 0.48 (posneg) / 0.37 (pos), best-in-pool
  0.62 / 0.77, 4 of 7 top picks >= 0.5; presence 0.99-1.00 on every accepted and every hidden
  cell. The chassis transfers zero-shot to a second subject and recording; the cabin and rear
  body do not at the top pick.
- **Reading.** (1) The exemplar pool is a better *candidate set* than the box-prompted image
  decoder for the chassis (ceiling 0.70 vs 0.58 on C10379, 0.75 vs 0.72 on C10119 with
  cross-view references) and the first pool to clear the bar on any camera (C10119 chassis,
  cross-view: 0.713 / 0.04), and the arm confirms it on the anchors (0.740 vs 0.667 with the
  decoder pool at the same onsets). (2) It does not clear the bar on C10379 (0.594 / 0.23) and
  its three harmful cells are the occlusion windows where no pool has a candidate: the ceiling
  moved from 0.58 to 0.70, not to 1. (3) The interior stays out of reach for both pools
  (0.19-0.37 held-out), and the size prior changes nothing because the candidates are the wrong
  black plastic, not the wrong size. (4) On the two screwdriver onsets the exemplar decoder
  abstains where the decoder accepted (C10379 1525, C10119 1533), which is the behaviour the
  distractor guard had to supply from human marks. (5) Cross-view references beat same-view
  ones on C10119 because there are 13 of them on 4 frames against 6 on 2; the concept
  transfers between these two static cameras and, for the chassis, to recording 2. What remains
  human is unchanged: the reference masks themselves, the interior, and the anchors that score
  all of this.
- **Deliverables.** `runs/sam3-exemplar-20260922/poolcmp/{C10379,C10119}/` (plans, control
  decodes, `compare.{json,md}`), `reprompt/*/iter1/` (five exemplar decodes with
  `candidate_source.json`, the C10119 chassis arm), `anchor_iou_c10119.{json,md}`,
  `rec2_zero_shot.{json,md}`, `references_rec2_C10379.json`; the hooks and tests named above;
  this entry. The plan file's Outcome section records the todo states.

### Sep 22: dedup pass 2 (the video-driver family; commits `d550d1a`, `cc56797`, `512a61e`, `2a90ed2`, `08c24cb`)

- **What.** The second deduplication pass: the six bounded external-worker video drivers
  (`dam4sam_video`, `samurai_video`, `grounding_dino_sam2_video`, `boxmot_track`,
  `wilor_hands`, `mediapipe_hands`; 0.73-1.00 similar) and their workers, plus the FineBio
  scripts' helper copies, folded onto two new modules, `src/battle/video_driver.py` (battle
  venv) and `src/battle/worker_common.py` (stdlib-only, Python 3.10, sibling-importable like
  `fs_common` / `gpu_guard`). One commit per helper group, the CPU harness after each, and one
  10 s GPU smoke per driver before and after with masks compared byte for byte and manifests
  modulo volatile fields. Method-specific argv, metadata models, notes and constants stay in
  each driver; no worker model code moved.
- **GPU sharing.** The other session's SAM3 exemplar queue
  (`runs/sam3-exemplar-20260922/jobs_5_remaining.json`, 8 jobs, `sam3_appearance.py` at
  6.0-6.6 GiB) held the card from before this pass started (05:20 UTC) until 06:51:23 UTC;
  the human's calibration worker (PID 2356892, 1.8 GiB) stayed up throughout. A poll loop
  (`pgrep` for the queue and its workers every 60 s) waited 82 min (05:29:28 -> 06:51:30) while
  the CPU refactor was written and committed; nothing was killed or paused. The smokes then
  ran through `scripts/overnight_queue.py` with the VRAM guard (`--gpu-guard vram`,
  `--allow-gpu-neighbour 2356892` for the calibration worker only), first from a `git archive`
  code snapshot of the pre-refactor commit `6e7f1df` (`PYTHONPATH` on the snapshot's `src`, so
  the drivers and the workers they locate beside themselves are the committed pre-refactor
  code; the four commits between `6e7f1df` and this pass's start `aa7cb70` touch only the
  exemplar work), then from a snapshot of the post-refactor HEAD `2a90ed2`. Six before-runs
  06:51:31-06:53:19, six after-runs 06:53:22-06:55:03, every job `succeeded`, every worker
  `succeeded`, 300 frames each. No `runs/*-10s-*` smoke from Sep 16 was reused as a baseline:
  their manifests predate schema fields added since (below), and a fresh pre-refactor run
  from the same code costs under two minutes for all six.
- **Group a, `video_driver.py`** (`d550d1a`; src+scripts +1236/-1581 over 10 files, of which
  `video_driver.py` is 522; `tests/test_video_driver.py` 775).
  `verify_inputs(*, repository_root, config_path, view_id, seconds, min_seconds, max_seconds,
  checkpoints=(), min_frames=None)`: the SAM2 trackers' `[1, 20]` range with the checkpoint
  and 30-frame rules, the hands / BoxMOT `(0, 60]` range (`min_seconds=None`), each
  `(path, sha256, label)` checked as before (`"<label> checksum mismatch for <path>"`).
  `run_external_worker(python, argv, *, run_directory, env=None, cwd=None,
  record_command=False)`: `CUDA_VISIBLE_DEVICES=0`, `worker.stdout.log` / `worker.stderr.log`,
  `worker_command.txt` for the three SAM2 trackers, the failed-state record when no
  `worker_result.json`; `prepend_pythonpath(root)` for WiLoR's and the fine-substep worker's
  `PYTHONPATH`. `load_worker_observations(..., exact=)` (DAM4SAM and SAMURAI demand the exact
  count, the others at least). `bounded_video` is the one ffmpeg argv (libx264 crf 18, no
  audio) for the eight identical copies and `four_part_segmentation`'s near-identical one
  (its `frame_count=FRAME_COUNT` default was never used: the single call site passes the
  count). `contact_sheet(draw=...)` with `tracker_contact_sheet` (box + `object_id: label`)
  and `hands_contact_sheet` (skeleton, landmarks, box, `hand_id: side`; `HAND_CONNECTIONS`
  now imported from `exporter`, whose tuple the two hands drivers duplicated). `worker_status`,
  `worker_measurements`, `worker_runtime_settings` (scalars pass, anything else
  `json.dumps(sort_keys=True)`, then `analysis_fps`, then the driver's extra worker keys in
  order), `common_metadata_fields` (frame range, requested seconds, the raw-source / proxy /
  config fingerprints, the observations / Rerun / QA URIs), `assemble_run_manifest` (clip cut
  to the seconds, one covered interval, the zero-overlap `ChunkContinuityPolicy` that the
  two-argument copies produced through the model defaults, the method status beside its
  export status) and `export_manifest` (`export_run` plus the mask sidecar when a mask root is
  given). The five worker drivers gain a `_build_manifest(...)` that `run()` and the tests
  call; `mediapipe_hands` keeps its own `_verify_inputs` (different error texts, unresolved
  proxy path, over-budget rather than minimum-frame rule) and its in-process detection.
  **Tests.** `tests/test_video_driver.py` carries the former inline manifest code of
  `samurai_video` and `boxmot_track` and the two contact-sheet drawers verbatim from
  `6e7f1df`, and asserts the new path reproduces the `model_dump_json(indent=2)` text
  (succeeded and failed worker states; the flattening of a nested setting, the `analysis_fps`
  int overwrite, the key order) and the PNG bytes; a `real_data` test rebuilds every pass-2
  smoke manifest through the current drivers from `worker_result.json` +
  `observations.jsonl` + the config and requires the JSON text to equal the recorded
  `manifest.json` byte for byte: 10 of 10 (five worker drivers x before/after; MediaPipe has
  no worker result). The Sep 16 `runs/*-10s-*` smokes rebuild identically except for schema
  fields added since (`nlf_body_2d` null -> [], WiLoR's `external_source_*` settings).
- **Group b, `worker_common.py`** (`cc56797`; +122/-93 over 5 files, of which the module is 88).
  `extract_frames(video, directory, frame_count, *, exact=True, check_open=True)` is the cv2
  decode-to-`native/frames/<index:05d>.jpg` loop the four workers copied; the two behaviours
  they differed in are parameters (the Grounding DINO worker tolerates a short video and errors
  only on zero frames, `exact=False`; the four-part worker never checked `isOpened()`,
  `check_open=False`), the three error texts collapse onto the DAM4SAM wording, and the count
  callers take `len()`. `cuda_peak_bytes(device=None)` is `torch.cuda.max_memory_allocated`
  (torch imported inside), the number every worker records as `gpu_peak_vram_bytes`.
  `dam4sam_video_worker`, `samurai_video_worker`, `grounding_dino_sam2_video_worker` (try the
  relative import, else the `sys.path` sibling) and `four_part_video_worker` (its `_sibling`)
  call them. Import smokes: `worker_common` under samurai 3.10.19, grounded_sam2 3.10.19,
  wilor 3.10.19 and muggled_sam 3.14.7; each touched worker answers `--help` under its own
  interpreter; `extract_frames` decodes a synthetic clip under samurai and grounded_sam2.
- **Group c, FineBio scripts** (`512a61e`; +65/-104 over 4 files). `finebio_sam3_smoke.py` and
  `finebio_dino_detect.py` drop their `gpu_processes` / `looks_like_model_process` / blocking-
  tolerated partition and `git_revision` copies: `gpu_guard.legacy_gpu_processes()` (the Sep 21
  `{pid, process_name, memory}` records, `"N MiB"` / `"[N/A]"` as before) with
  `partition_neighbours_strict`, and `fs_common.git_revision(path)` (`{revision, dirty}`,
  `check=False`, stdout only). `gpu_guard` became Python 3.10 compatible
  (`datetime.timezone.utc`) because the detector script's detect phase runs under the 3.10
  MMDetection venv; both scripts import `gpu_guard` beside `fs_common` as `battle.*` or by
  path. Left per script: `tracker_processes()` (the pgrep pattern and the 200-character
  truncation differ) and the SAM3 track loop, the deliberate copy of `muggled_worker`, now
  saying so in a comment. Smoke: `--help` and a module import with the guard evaluated under
  muggled_sam 3.14.7 (sam3 smoke) and the detector venv 3.10.20 (dino detect: the calibration
  worker tolerated, the exemplar worker blocking); `gpu_guard` and `fs_common` import under
  samurai 3.10.19.
- **Harness** (`2a90ed2`, `08c24cb`): `scripts/dedup_equivalence.py compare-runs A B
  [--report] [--replace OLD=NEW]` compares two run directories of one driver: every PNG under
  `native/masks/` or `masks/` by SHA-256, then the `compare` walk with both run directories
  (absolute, repository-relative, the run id) folded onto `<run>`,
  `time_to_first_usable_output_seconds` and `gpu_peak_vram_bytes` volatile in addition, the
  mask cache's `stamp/<mask>` arrays (`[st_size, st_mtime_ns]`; the first comparison reported
  exactly these 300 per tracker, mtime only, sizes equal) skipped while its bits / rgba / shape
  arrays are compared, and `--replace` for the two code-snapshot directories named in
  `worker_command.txt`. CPU harness: `pass2-before` at `aa7cb70`, `pass2-a` / `pass2-b` /
  `pass2-c` after each group and `pass2-after` at `2a90ed2` all compare equivalent to
  `pass2-before` (13 files, RRD by content digest); the harness artifacts do not exercise
  these drivers, so the GPU smokes below are the evidence for this pass.
- **Smokes.** Six 10 s runs per side under `runs/dedup-pass2-smokes-20260922/{before,after}/`
  (`jobs_before.json` / `jobs_after.json`, `queue_before.log` / `queue_after.log`, `logs/`,
  `compare/<driver>.json` from `compare-runs`, `make_jobs.py`, the two code snapshots). Masks
  are the worker PNGs under `native/masks/` (DAM4SAM, SAMURAI, Grounding DINO + SAM2); BoxMOT,
  WiLoR and MediaPipe write no masks, so for them the byte-identical `observations.jsonl` and
  `contact_sheet.png` are the equivalent check (those two files and `input.mp4` are
  byte-identical for all six pairs; the RRDs equal by content digest). `diff -rq` on the mask
  directories agrees with the per-PNG SHA-256. The three trackers' before-masks are also byte
  for byte the masks of the Sep 16 smokes (`runs/dam4sam_video_smoke-10s-20260916t052430z`,
  `runs/samurai_sam2_video_smoke-10s-20260916t052328z`,
  `runs/transformers_grounding_dino_plus_sam2_video_smoke-10s-20260916t0518z`), so the
  workers are deterministic across days and a byte comparison is a meaningful test.

  | driver | before (`runs/dedup-pass2-smokes-20260922/before/`) | after (`.../after/`) | masks identical | manifest equal (modulo volatile) |
  |---|---|---|---|---|
  | dam4sam_video | `dam4sam_video_smoke-10s-20260922t065131z` | `dam4sam_video_smoke-10s-20260922t065322z` | yes (300/300 PNG) | yes (609 files compared) |
  | samurai_video | `samurai_sam2_video_smoke-10s-20260922t065149z` | `samurai_sam2_video_smoke-10s-20260922t065339z` | yes (300/300 PNG) | yes (609) |
  | grounding_dino_sam2_video | `transformers_grounding_dino_plus_sam2_video_smoke-10s-20260922t065209z` | `transformers_grounding_dino_plus_sam2_video_smoke-10s-20260922t065359z` | yes (300/300 PNG) | yes (609) |
  | boxmot_track | `boxmot-yolo-static-10s-20260922t065232z` | `boxmot-yolo-static-10s-20260922t065420z` | n/a (no masks; observations and sheet byte-identical) | yes (6) |
  | wilor_hands | `wilor-hands-static-10s-20260922t065241z` | `wilor-hands-static-10s-20260922t065426z` | n/a (no masks; observations and sheet byte-identical) | yes (6) |
  | mediapipe_hands | `mediapipe-hands-static-10s-20260922t065315z` | `mediapipe-hands-static-10s-20260922t065458z` | n/a (no masks; observations and sheet byte-identical) | yes (5) |

  Volatile fields dropped from the manifests: `run_id`, `elapsed_seconds`,
  `time_to_first_usable_output_seconds`, `gpu_peak_vram_bytes` (identical here anyway:
  DAM4SAM 814398464, SAMURAI 818672640, Grounding DINO + SAM2 6372003328, WiLoR 2819373056,
  BoxMOT 74796544 bytes on both sides) and the `*_at` keys; the run directory path inside every
  URI is the same string on both sides once folded.
- **Left in place, with reasons.** `kineo_fusion._run_worker` (pixi, `check=True`, no logs,
  no result JSON: not the shape); `muggled_smoke._run_worker` (its failed-state record carries
  six more keys and the six smokes do not exercise it; only a drop-in was to be adopted);
  `mediapipe_hands._verify_inputs` (above); the finebio `tracker_processes()` pair and the
  SAM3 track loop (above); `muggled_worker._gpu_processes` (the same records
  `legacy_gpu_processes` now returns, but the SAM3 worker is outside this pass's smokes);
  the `_normalize_box` copies in the workers and in `samurai_video` /
  `grounding_dino_sam2_video` (the driver copies are imported by tests, the workers are
  foreign-interpreter model code). Behaviour changes, all on error paths: WiLoR's missing
  checkpoint or detector now raises `FileNotFoundError(path)` before hashing (same type as
  before, different message); the fine-substep driver reports a worker that wrote no
  `worker_result.json` as `RuntimeError(reason)` instead of `FileNotFoundError`, and a
  `PYTHONPATH` with leading or trailing separators is no longer stripped there; the three
  worker frame-extraction error texts are unified.
- **Totals.** `git diff --shortstat aa7cb70..HEAD -- src scripts` less the other session's
  two files in the range: 20 files, +1666/-1810 (net -144); the two new modules are +610
  and the harness +246/-35, so inside the 17 existing modules -1778/+813 (net -965). Tests:
  `pytest -q` 716 -> 740 passed (new: `video_driver` 11, `worker_common` 5, `fs_common` 1,
  `gpu_guard` 2, harness 4; one from the other session), 9 skipped; `-m real_data` 52 -> 62
  passed (the 10 smoke-manifest rebuilds); `ruff check` / `ruff format --check` clean on every
  file of this pass. The `pass2-after` harness snapshot recorded one default-tier failure
  (`test_exemplar_seed`) from the other session's then-uncommitted `exemplar_seed` edit; it
  passes at their `7f608e5` and in the final run above.
- **Not done here** (pass 3 per the plan): Rerun logging helpers, `media_probe` in
  `exploratory_comparison`, the queue writers and a `battle-code-snapshot` helper (the
  `git archive` + `PYTHONPATH` step this pass hand-ran again), `cli_common`, the remaining
  contact-sheet grids, `metrics.py`, the README pointers to the new modules.

### Sep 22: dedup pass 3 (Rerun helpers, media probe, queue writers, CLI fragments, contact-sheet grid, dead code; commits `7ddfe88`, `84bbdc7`, `b0aa69a`, `3749a0d`, `b6bbb5f`, `6d7e5bc`)

- **What.** The third and last deduplication pass: the `rr.*` idioms the review, comparison
  and export builders repeated, `exploratory_comparison`'s private ffprobe, the two queue-job
  writers, the argparse fragments every CLI wrote the same way, the contact-sheet grid loops
  and the production-dead `metrics.py`, folded onto three new modules (`rerun_logging.py`,
  `cli_common.py`, `contact_sheet.py`) and onto `media_probe` / `overnight_queue`. CPU only
  (`CUDA_VISIBLE_DEVICES=""` on every command; the human's calibration worker, PID 2356892,
  stayed on the card and was never touched). One commit per group, the harness after each,
  the v6 combined package rebuilt before and after group 1. No digest, URI or manifest field
  moved; the blueprints stay with their products.
- **Group 1, Rerun helpers** (`7ddfe88`; 16 existing files +305/-416, `rerun_logging.py` 127,
  `tests/test_rerun_logging.py` 170). `init_and_save(application_id, path, *,
  recording_id=None, default_blueprint=None)` replaces the 15 `rr.init` + `rr.save` openings
  (`athena_triangulation`, `exploratory_comparison`, `interaction_review`,
  `interaction_review_v4`, `egoexo_correspondence`, `drop_dtw_align`, `fine_substep_align`,
  `exporter` x2, `review_metrics`, `multiview_review`, `athena_hands_review`,
  `policy_ablation`, `four_part_comparison`, `kineo_multiview_review`; recording ids and
  default blueprints pass through, `spawn` stays False; `policy_ablation` now creates the
  output directory before the init rather than between init and save, no output effect).
  `log_rgba_mask` / `log_rgba_masks` are the RGBA cut-out `EncodedImage` (`media_type`
  `image/png`, the caller's opacity and draw order) that `exporter._log_masks`,
  `exploratory_comparison._log_masks` and the same block in `interaction_review`,
  `interaction_review_v4` x2, `multiview_review`, `policy_ablation`, `four_part_comparison`
  and `egoexo_correspondence` x5 wrote; `exploratory_comparison` keeps its
  decode-validate-log order through a lazy generator, its two error texts unchanged.
  `log_boxes_from_observation(entity, items, dimensions, *, labels, colors, **boxes)` is the
  `Boxes2D` scaled from normalised boxes (`exporter` objects x2 and hands,
  `exploratory_comparison` objects and hands, `four_part_comparison`, `interaction_review`
  person boxes, `interaction_review_v4` provenance overlay); the label formats and colours
  stay with each caller because every builder formats them differently.
  `time_series_view(origin, name, contents="$origin/**")` and `time_series_stack(root,
  entries)` replace every `rrb.TimeSeriesView(origin=..., name=..., contents=...)` in
  `review_presets`, `interaction_review`, `exporter`, `multiview_review`, `review_metrics`,
  `athena_hands_review`, `kineo_multiview_review`, `egoexo_correspondence`, `policy_ablation`
  and `fine_substep_align`. **Evidence.** Harness `pass3-before` (`ae0df0b`) vs `pass3-g1`
  equivalent (13 files; the review-v4 RRD by content digest, `rerun rrd verify` ok). The v6
  combined package (`runs/interaction-review-first-minute-v6`'s README command: ensemble-v2
  reference, r1280 pm-append consensus, the five `--candidate-arm`s, `--confidence
  runs/detector-scorecard-20260920/pm-append`, `--application-id battle-interaction-review-v6
  --recording-id interaction_review_first_minute_v6`) rebuilt before the change into
  `runs/dedup-equivalence/pass3-v6-before/` and after it into `pass3-v6/`: **1221
  content-digest columns, 0 differing**; the index, review guide and contact sheet
  equivalent through `compare`; `rerun rrd verify` "1 file verified without error" on both
  (49.6 s and 51.4 s builds). `tests/test_rerun_logging.py` (8) asserts the component
  batches each helper logs equal the hand-written archetype, the lazy decode order, and the
  ids `init_and_save` records.
- **Group 2, media probe** (`84bbdc7`; +3/-23). `exploratory_comparison._video_info` calls
  `media_probe.video_info`; its ffprobe argv was byte for byte the probe's (`-count_frames`,
  `nb_read_frames`, fps = `round(avg_frame_rate)`, width x height), so the frame count has
  the same semantics (decoded frames, not the container's `nb_frames`) and is now cached in
  the `input.mp4.probe.json` sidecar as `interaction_review`'s already was. Checked once on
  the actual 20 s proxy
  (`runs/mediapipe-hands-static-20s-fused-dedup-th035-20260916t0428z/input.mp4`):
  `(600, 30, (1280, 720))` both ways. Harness `pass3-g2` equivalent.
- **Group 3, queue writers and `battle-code-snapshot`** (`b0aa69a`; `overnight_queue.py`
  +206/-1, `egoexo_correspondence` and `kineo_multiview` +26/-34, `pyproject.toml` +1, tests
  +193). `overnight_queue.queue_job(*, name, argv, cwd, timeout_s, env=None, interpreter=(),
  gpu_profile=None, expected_peak_vram_bytes=None)` builds one job record in the key order
  the files have always carried (the two guard fields only when set), `job_list(*jobs,
  log_path=None)` the `{"jobs": [...]}` document and `write_jobs(path, spec, *, indent=2)`
  validates it as a `QueueSpec` before writing through `fs_common.write_json`.
  `egoexo_correspondence.queue_job` (LM-EEC cwd, `CUDA_VISIBLE_DEVICES` +
  `PYTHONUNBUFFERED`, the LM-EEC interpreter; written with indent 1) and
  `kineo_multiview.queue_job` (resolved Kineo root, `CUDA_VISIBLE_DEVICES`, no interpreter;
  indent 2, `jobs_t4_kineo_<arm>.json` and the run's `queue_job.json`) keep their signatures
  as thin wrappers. **Evidence.** Both callers' documents were captured from the pre-change
  functions with fixed argv and are byte-equal after; the test pins those bytes.
  `battle-code-snapshot ROOT [--commit HEAD] [--path src/battle ...] [--allow-dirty]`
  (`overnight_queue.code_snapshot` / `code_snapshot_main`) is the `git archive <commit>
  src/battle` step every queue README documents and pass 2 hand-ran twice: the archive
  extracted into `<root>/code-snapshot-<short sha>/`, `snapshot.json` beside it with the full
  sha, the paths, the dirty flag and the `PYTHONPATH`, the `# every job:
  env.PYTHONPATH=<root>/code-snapshot-<sha>/src` and `# built with: git archive ...` lines
  printed. Uncommitted changes under the archived paths are refused unless `--allow-dirty`
  (the archive is of the commit, so the snapshot would be named after a commit the tree no
  longer matches; edits elsewhere, the README or an untracked script, do not count); an
  existing snapshot of the same sha is reused, any other content at the target is an error.
  Smoke against this repository: refused on the dirty `src/battle` mid-pass, then archived
  `7ddfe88` under `--allow-dirty` with `rerun_logging.py` identical to `git show`. Harness
  `pass3-g3` equivalent.
- **Group 4, CLI fragments** (`3749a0d`; `cli_common.py` 67, 49 existing modules
  +154/-104, tests +53). `add_repository_root(parser)` (the `--repository-root` option
  defaulting to the current directory, now with one help text) replaces the 54 identical
  `add_argument` lines; `add_output_root(parser, default, *, help=None)` the 35
  `--output-root` lines (the three that derive their default keep their help);
  `add_output_flags(parser, *, overwrite_help, quiet=True, quiet_help)` the
  `--overwrite`/`--quiet` pair of `interaction_review_v4` and the `--overwrite` of
  `review_metrics` and `ensemble_reference` with their own help texts;
  `open_output_directory(path, *, overwrite, message=None)` the mkdir-or-refuse-when-non-empty
  rule in `review_metrics` (which used to raise the bare path) and `ensemble_reference`
  (whose text is the default). Every touched CLI answers `--help`; the harness runs three of
  them, `pass3-g4` equivalent. **Accounting.** This group is line-neutral per site and adds
  one import per module (+49); what it buys is one definition of each option.
- **Group 5, contact-sheet grid** (`b6bbb5f`; `contact_sheet.py` 109, 5 existing files
  +39/-79, tests +79). `render_grid(cells, *, columns, labels=None, title=None, gap=0,
  fill=0)`: cells in rows of `columns`, each cell padded at the bottom to its row's tallest,
  each row padded on the right to the widest, a short last row filled the way
  `np.zeros_like` filled it, an optional 30 px label bar per cell and 44 px title bar
  (`label_bar` / `title_bar` / `text_bar`, the dataset sheet's strips). Callers:
  `seed_search.render_search_sheet` and `correction_acceptance_search.render_sheet` (three
  captioned tiles per frame, `np.pad` to the widest row), `exemplar_seed.render_run_sheet`
  (two columns, `zeros_like` fill), `assembly101_contact_sheet.render_contact_sheet` (five
  labelled panels per view, the title, `copyMakeBorder` padding; its `_label` and the two
  height constants moved) and `seed_proposal_sheets.write_view_sheet` (one PIL row per cell
  with 6 px gaps; the TrueType title stays PIL-drawn above the grid). Each drawer keeps its
  own cell content. `human_seeds` has no grid of its own (`render_interior_sheets` shells
  out to `battle-seed-proposal-sheets`). **Evidence.** One sheet per caller rendered from its
  existing inputs before and after into `runs/dedup-equivalence/pass3-sheets/{before,after}/`
  (the seed-search sheet, the acceptance-search sheet, the rec2 consensus-only exemplar
  sheet at frames 0-1716, the recording-1 dataset sheet, the C10095 proposal sheet): the
  before-renders equal the committed sheets byte for byte where one exists
  (`runs/seed-search-20260920/search_contact_sheet.png`,
  `runs/correction-acceptance-search-20260920/acceptance_contact_sheet.png`,
  `runs/rec2-automatic-20260920/c10379_consensus_only_contact_sheet.png`,
  `runs/labeling-sessions-20260920/proposal_sheets/C10095.png`; 4 of 4), and **all five
  after-renders equal their before-render byte for byte** (pixel identity was not required,
  the layout arithmetic is the same). `side_by_side/<caller>.png` beside them for a reader.
  The fill test caught that a scalar `cv2.copyMakeBorder` value fills one channel only, so
  the pad passes a triple (no committed sheet was affected: every caller fills with 0).
  Harness `pass3-g5` equivalent.
- **Group 6, dead code** (`6d7e5bc`; -33, test -7/+6). `src/battle/metrics.py` removed.
  **Decision.** Its `calculate_success_measure` (coverage ratio and completed-method ratio
  averaged into a `combined_ratio`) was imported by `tests/test_schemas.py` only; it is not
  one of the five documented pre-accuracy success measures (coverage, time to first usable
  output, peak VRAM, runtime, ID resets; those live in `RuntimeMeasurements` and the
  manifests), and neither `docs/` nor the README names the module, the function or
  `combined_ratio` (the Sep 8 line "Success measure: full fixture coverage and completed
  exporter contract" above is prose), so the plan's default applies rather than a move into
  `schemas.py`. The fixture test keeps the contract it asserted (coverage 1.0, method states
  `succeeded` / `not_run`) without the helper.
- **Left in place, with reasons.** `interaction_review_v4._prepare_output_root` (deletes the
  package's own files, its test pins two messages); `interaction_review` and
  `exploratory_comparison` (`--no-overwrite` and a file-level check); `athena_hands`
  (refuses any existing directory, empty or not); `assembly101_reference` (a different
  message and no mkdir at that point); the plain `--overwrite` flags whose meaning is not
  "replace a package" (fetch-view trim, camera fit, human QA, kineo prepare, reprompt plan);
  `ego_diagnostic` (`--repository-root` required, foreign interpreter) and
  `g3_contact_sheet` (no `battle` imports); `fine_substep_align`'s pixel-box `Boxes2D` and the
  JPEG / by-path `EncodedImage`s (`egoexo_correspondence` frames, `multiview_review` hull
  projections); the mask decode-and-validate pair in `exploratory_comparison` and
  `four_part_comparison` (four different error texts for two checks); `finebio_*` scripts'
  `rr.init`/`rr.save` (deliberate dual-interpreter copies). The ~55 `--repository-root`
  sites were adopted although the line count does not fall (above).
- **Totals.** `git diff --shortstat ae0df0b..6d7e5bc -- src scripts`: 56 files,
  +1035/-689 (net +346). The three new modules are +303 (`rerun_logging` 127,
  `cli_common` 67, `contact_sheet` 109) and `overnight_queue` +206/-1 (the job writers and the
  new `battle-code-snapshot` command, roughly 150 lines of new function); inside the other
  52 existing modules -688/+526 (net -162), of which the CLI sweep's imports are +49 and
  group 1 alone -416/+305. Tests: `pytest -q` 740 -> 762 passed (new: `rerun_logging` 8,
  `overnight_queue` 5, `cli_common` 3, `contact_sheet` 6), 9 skipped; `-m real_data` 62
  passed before and after; `ruff check` / `ruff format --check` clean on every file of this
  pass (60 files). Final harness `pass3-after` at `6d7e5bc`: equivalent to `pass3-before`
  and to pass 2's `pass2-after` (13 files, RRD by content digest, `rerun rrd verify` ok).
  One default-tier flake was seen once in the middle of the pass
  (`test_exemplar_seed::test_consistency_needs_three_views_and_rejects_off_size_and_neighbours`
  failed in a full run and passed alone and in every later full run; not touched here).
  **Note on the working tree.** A `ruff format` run without file arguments during group 4
  reformatted the Python block inside `README.md`; that hunk was reverted with `git apply -R`
  before anything was staged, and the user's own uncommitted README hunk (the Tailscale
  section) was never staged.

### Sep 22: dedup passes 1-3, summary

Three passes over `src/battle` and `scripts` (plan: "Deduplicate src/battle and scripts"),
each proven equivalent by `scripts/dedup_equivalence.py` on three CPU artifacts (the r1280
pm-append consensus, the 19-arm anchor scoreboard, the `reference_masks` review v4 with
`rerun rrd verify`) plus, per pass, the evidence its helpers needed: byte-identical GPU smoke
masks for the six drivers (pass 2), the v6 combined package's 1221-column content digest and
five byte-identical contact sheets (pass 3). Totals in src+scripts: pass 1 -954/+718 (net
-236; -548 inside the existing modules), pass 2 -1810/+1666 (net -144; -965 inside the
existing modules, the harness excluded), pass 3 -689/+1035 (net +346; -162 inside the existing
modules once the three new modules and the new `battle-code-snapshot` command in
`overnight_queue` are set aside, +43 with it). Over the three passes: 3453 lines deleted, 3419
inserted, net -34, with **-1675 inside the pre-existing modules** (-1470 counting the queue's
new command against them) and the difference in twelve shared modules
(`fs_common`, `mask_ops`, `observations`, `video_driver`, `worker_common`, `rerun_logging`,
`cli_common`, `contact_sheet`, plus the extended `digest_cache`, `media_probe`, `mask_cache`,
`schemas.fingerprint`, `overnight_queue`) and the harness. Tests 646 -> 762 in the default
tier, 52 -> 62 in `real_data`. Deliberately not merged, as the plan said: the seed and score
module families, the per-product blueprints, the worker model code, the FineBio track loop
and the 83 `main()` entry points beyond their shared fragments.

### Sep 24: Assembly101 phase closed

- **What this is.** The decision was "close now": nothing else runs on Assembly101 unless a
  FineBio result sends us back. This entry records the dispositions as they stand on Sep 24
  (nothing inferred, nothing labelled, no run re-scored), finalizes the goals scorecard at the
  top of this file, puts the final numbers in one table with their counts and sources, lists
  what `runs/` holds for the user's deletion decision (nothing deleted), and names the tag.
  CPU only; no GPU job; the calibration web server was not touched. Every IoU below is against
  one person's choice of SAM3 image-decoder masks on a handful of frames of one camera at a
  time: review evidence that ranks arms against each other, not ground truth, not a dataset, no
  accuracy claim. CC BY-NC 4.0 on every frame and everything derived from it.
- **Ensemble v2.** Ensemble v2 remains the named candidate; **no adoption decision was recorded
  by the human.** The v4 builder's default reference stays v1 (also unchanged):
  `configs/ensemble_reference/first_minute_v2.json` builds
  `runs/ensemble-reference-first-minute-v2/` (0.743 on the 52 anchor cells, arm choice
  selection-biased), and `battle-build-interaction-review-v4` still defaults to the v1 ensemble.
  Record: [ensemble v2](#sep-20-ensemble-reference-v2-the-named-candidate-plan-step-3-deciding-half-closes-the-resolution-and-dam4sam-follow-up-plan).
- **Recording 2 (`nusar_9061`) C10379 anchors: not labelled to completion; recording 2 stays
  unscored; the generalization claim stays open.** The workspace
  `runs/human-review-anchors-nusar_9061-static-c10379/` is kept as-is. Its
  `calibration_manifest.json` on Sep 24 holds 7 decoded candidates, every one with
  `human_accepted: true`, `selected_by: human` (chassis at 456 / 483 / 552 / 1191, cabin at
  456 / 1191, rear_body at 1191, all decoder candidate 0), and 5 human `hidden` marks (interior
  456 / 552 / 1191, rear_body 456 / 552): 12 of the 52 cells on 4 of the 13 frames, the other 9
  frames untouched, `final_proposal_uri` null, no `battle-anchor-export` run, no human record
  under `docs/qa/`, `battle-anchor-iou` never run on it. (The Sep 22 zero-shot exemplar check
  used those 7 masks and 5 marks as they stood; the close-out brief described the workspace as
  "7 candidates, 0 accepted", and the manifest says otherwise, so the manifest's state is what
  is written here.) The zero-human-input run `runs/rec2-automatic-20260920/` and its
  consensus-only arm therefore have no anchor score, and "proven on a second recording" is not
  claimed. `configs/qa/nusar_9061_review_anchors_c10379.json` has no status field (its
  `provenance` block carries `author`, `reviewed_at`, `tool`, `notes`, the first two and the last
  null) and is left unchanged; this paragraph is the status record. Records:
  [C2/C3](#sep-20-track-c2c3-recording-2-nusar_9061-seeded-and-run-with-zero-human-input-commits-78f2eca-21da378-cf8007e),
  session (d) in [`docs/labeling-sessions-2026-09-20.md`](labeling-sessions-2026-09-20.md).
- **Interior seeds on five statics: unconfirmed.** The frame-0 interior seeds on C10095, C10115,
  C10119, C10390 and C10404 keep provenance `geometric_from_two_human_views` (triangulated from
  the human's C10379 frame-0 mask and C10119 anchor at 41, decoded and filtered by the agent);
  C10118 and C10395 have no accepted candidate and ran with three parts.
  `configs/qa/interior_seed_decisions.template.json` has 0 of 7 decisions (`decision` and
  `accepted_candidate` null on every cell, `author` / `reviewed_at` null); session (e) was not
  held; the template is left in place. Every four-part number on those views (the Sep 21 rerun,
  the four-part consensus, the C10119 interior 0.106-0.290) rests on unconfirmed interior seeds
  and is labelled so where it appears. Record:
  [Sep 21, night](#sep-21-night-the-three-labelling-sessions-acted-on-human-accepted-seeds-an-interior-seed-from-two-human-masks-the-four-part-rerun-the-c10119-rear_body-finding-and-a-distractor-guard).
- **Extended experiments E2-E9 of the resolution plan: not run** (the plan file's todos
  `e-cutie`, `e-bidir`, `e-negprompt`, `e-reprompt`, `e-appearance`, `e-roi1080`, `e-anchors2`,
  `e-matrix` stay `pending`; the plan has no Outcome section, its main track closed in the
  Sep 20 ensemble v2 entry). Per item, with what later work touched instead: E2 Cutie / XMem,
  not tried. E3 bidirectional SAM3 from the 1700 anchors, not tried. E4 automatic mutual
  negative prompts in the worker, not built. E5 consensus/hull-driven re-prompting, not run as
  specified; the multicam plan's B4 loop (`battle-multiview-reprompt`, Sep 20) is the same idea
  with a different design and is recorded there. E6 DINOv2 appearance templates, not built;
  the exemplar plan's Track A (Sep 22) used SAM3's own backbone embeddings and exemplar
  detector instead, and the Sep 19-20 seed search found DINOv2 exemplar re-ranking no better
  than the decoder score. E7 tracked-ROI SAM3 at native 1080p, not run (1920 full-frame
  saturated at 0.708, so the premise lapsed). E8 a second anchor set on C10115 plus extra C10379
  frames in 1230-1300, around 1700 and inside 573-722, not done; anchors were labelled on
  C10119 and e4 instead (Sep 21), so the 1235 screwdriver correction on C10379 is still unscored.
  E9 method x view matrix, not run.
- **Remaining items of the multicam and exemplar plans: not run.** Multicam plan (Outcome,
  Sep 20; `g-label` partly done Sep 21): recording 2's 13 frames (above); the per-slot start
  frame for e4's interior (first human e4 interior mask at 224); the 13 consensus and 37 hull
  `not_contact_eligible` proposals of Sep 18 (never dispositioned); the confidence series as a
  gate (kept as review context). Exemplar plan (Outcome, Sep 22): iteration 2 of the C10119
  chassis arm (capped at two, budget spent on the passes); a C10379 consensus-only-exemplar arm
  (chassis 0.594 / harm 0.23 and interior 0.323 / 0.50 below the bar, the one rear_body onset
  rejected); logging `confidence_v2` into a v6 rebuild (command written in the Track A entry,
  not run); the README lines for `battle-exemplar-pool` and `battle-detector-scorecard-v2`
  (kept in the Track A entry because `README.md` holds the user's uncommitted edit).
- **Older open items closed as not dispositioned** (they stay listed under "Open items" as the
  record of what was never decided): the fixed two-timestamp QA records under `docs/qa/` (all
  `pending`; superseded by the anchors as the yardstick on Sep 19); the hidden-interior
  semantics for [1024, 1172); the late reassembly beyond the first minute (target loss at
  1946 / 2381 / 2578 / 2684); the G5 claims gate including the CC BY-NC question for any demo
  use; the LM-EEC checkpoint direction; seeding e1 mid-minute.
- **The user's working tree.** The uncommitted `README.md` hunk (the Tailscale review-serving
  subsection removed) and the untracked `test.sh` are the user's; neither was staged, reverted
  or committed in this pass. The README edits of this entry were staged by hunk.
- **Final numbers.** One table; each row names its truth set and where the number is on disk.
  Truth sets: C10379 = 51 labelled + 1 hidden cells on 13 frames (`runs/human-review-anchors-first-minute/`);
  C10119 = 63 labelled + 1 hidden on 26 frames (40 cells skipped); e4 = 52 labelled + 23 hidden
  on 26 frames (29 skipped). Every number was re-read from the artifact named in its row on
  Sep 24; none differed from the ledger.

  | Measure | Value | Truth set / counts | Where |
  | --- | --- | --- | --- |
  | SAM3 1280 `pm-append` (human frame-0 seeds + corrections 327 / 900 / 1172 / 1235), C10379 | **0.743** all (chassis 0.637, interior 0.585, rear_body 0.790, cabin 0.962); 1265 px on the hidden 1700 cell | C10379, 51 / 1 | `runs/anchor-scoreboard-20260919/anchor_iou.md`; [memory arms](#sep-19-sam3-correction-memory-arms-on-c10379-at-1280-plan-step-1b) |
  | Ensemble reference v2 (candidate; `pm-append` primary, DAM4SAM-large fallback in label-free consensus intervals) | **0.743** (arm choice saw these anchors: selection-biased, not out-of-sample) | C10379, 51 / 1 | same file, row `ensemble-reference-v2`; [ensemble v2](#sep-20-ensemble-reference-v2-the-named-candidate-plan-step-3-deciding-half-closes-the-resolution-and-dam4sam-follow-up-plan) |
  | `off-r1280-sched` (1280 px, replace-memory corrections; the run condition before `pm-append`) | 0.724 | C10379, 51 / 1 | same file |
  | Old reference (Sep 18 ensemble v1 primary = `off-r720-sched`, byte-equal on the first minute) | 0.668 (ensemble v1 0.681 with 3 missing cells) | C10379, 51 / 1 | same file, rows `reference`, `ensemble-reference-v1` |
  | `seed-only-1280` (human frame-0 seeds, no corrections: the floor) | 0.591 (chassis 0.404, interior 0.228) | C10379, 51 / 1 | `runs/multiview-reprompt-20260920/anchor_iou_arms.md` |
  | `consensus-only-ds` (frame-0 seeds + agent corrections from the cross-view consensus, decoder-score ranking) | 0.659 (chassis 0.609, interior 0.288); 4-part variant 0.659 (chassis 0.534, interior 0.363) | C10379, 51 / 1 | `runs/multiview-reprompt-20260920/anchor_iou_arms.md`, `runs/multiview-reprompt-20260921/anchor_iou_arms.md` |
  | `human-plus-consensus-ds-4part` (human corrections + agent corrections) | 0.742 (vs 0.743 human only) | C10379, 51 / 1 | `runs/multiview-reprompt-20260921/anchor_iou_arms.md` |
  | `dam4sam-large-1024-sched-60s` (shared predictor, large checkpoint, the SAM3 schedule) | 0.715 (chassis 0.697, the highest chassis on the C10379 scoreboard; tied with SAM3-1280's 0.724 under the 0.01 rule) | C10379, 51 / 1 | `runs/anchor-scoreboard-20260919/anchor_iou.md`; [DAM4SAM arms](#sep-19-dam4sam-and-samurai-arms-under-the-sam3-run-conditions-plan-step-2) |
  | C10119 `r1280` (Sep 19 agent seeds, 3 parts, no interior slot) | 0.474 all (chassis 0.758, rear_body 0.839, cabin 0.859, interior 0.000 with 25 cells missing) | C10119, 63 / 1 (38 scored) | `runs/anchor-scoreboard-c10119-20260920/anchor_iou.md` |
  | C10119 4-part + guarded `consensus-only-ds` (human-accepted seeds, decoder pool, distractor guard) | 0.554 all (chassis 0.667, interior 0.290, rear_body 0.833, cabin 0.945); 3,795 px hidden FP | C10119, 63 / 1 | `runs/multiview-reprompt-20260921/anchor_iou_c10119.md` |
  | C10119 4-part `consensus-only-exemplar` (chassis only, cross-view exemplars, 5 corrections) | chassis **0.740** (all 0.513: no interior correction applied, 0.106) | C10119, 63 / 1 | `runs/sam3-exemplar-20260922/anchor_iou_c10119.md`; [exemplar Track B](#sep-22-exemplar-detections-as-correction-candidates-track-b-of-the-exemplar-plan-the-c10119-chassis-arm) |
  | e4 `r1280` (agent seeds, 3 parts) -> `human-accepted-4part` | 0.558 -> 0.589 all (rear_body 0.262 -> 0.349; hidden FP 48,705 px on 8 cells -> 16,474 px on 7) | e4, 52 / 23 (42 scored; 10 interior cells have no slot) | `runs/anchor-scoreboard-e4-20260920/anchor_iou.md`, `runs/multiview-reprompt-20260921/anchor_iou_e4.md` |
  | Failure detector, `sam3_score` (SAM3's own object score) | AUROC 0.91-0.96 in-sample, top-2 on every scored run | C10379, 52 cells, three runs (9 / 14 / 11 failures) | [multicam closing](#closing-the-multicam-plan-the-two-goals-answered-with-numbers); `runs/detector-scorecard-20260920/` |
  | Combined detector, leave-one-frame-out (top-3 rank average, threshold at R >= 0.8 on 12 frames) | precision / recall 0.36-0.62 / 0.44-0.64 (4/7/5/36, 8/5/6/33, 7/5/4/36 TP/FP/FN/TN): the ranking transfers, the threshold does not | C10379, 52 cells | same |
  | Seeding acceptance by the human (session (c), Sep 21) | chassis 7 / 8 accepted (C10390 undecided), rear_body 8 / 8, interior 1 / 8 | 24 cells, one frame each, 8 views | `configs/qa/seed_proposal_decisions_2026-09-21.json` |
  | Seed transfer, held-out IoU of the winning strategy (leave-frames-out on the C10379 human masks; 0.6 gate) | chassis 0.53 / rear_body 0.80 / interior 0.47 / cabin 0.95 (rear_body and cabin pass; the 0.525 / 0.795 / 0.465 / 0.949 of the search rounded) | C10379 truth set, 13 anchor frames | [multicam closing](#closing-the-multicam-plan-the-two-goals-answered-with-numbers); `runs/seed-search-20260920/` |
  | Dedup passes 1-3 (Sep 22) | 3,453 lines deleted / 3,419 inserted in `src` + `scripts`, net -34; **-1,675 inside the pre-existing modules**; tests 646 -> 762 default tier, 52 -> 62 `real_data`; equivalence on three CPU artifacts per pass | repository | [dedup summary](#sep-22-dedup-passes-1-3-summary) |
  | Test tiers and lint at close (Sep 24) | default `uv run pytest -q`: **762 passed**, 9 skipped, 64 deselected, 16 s; `-m real_data`: **62 passed**; `-m gpu`: 2 collected, not run (CPU day); `uv run ruff check src tests scripts`: all checks passed | repository | this entry |

- **Pointers.** The review surface is the v6 package `runs/interaction-review-first-minute-v6/`
  (`interaction_review_combined.rrd` with `segmentation.rbl`, `hands.rbl`, `multiview.rbl`,
  `review_guide.md`, `presets_check.json`); guides
  [`docs/review-guide-2026-09-20-multiview-presets.md`](review-guide-2026-09-20-multiview-presets.md),
  [`docs/review-guide-2026-09-20-ensemble-v2.md`](review-guide-2026-09-20-ensemble-v2.md),
  [`docs/review-guide-2026-09-18-multicam.md`](review-guide-2026-09-18-multicam.md). The three
  plan files' closing sections, by path: `/home/nick/.cursor/plans/resolution_and_dam4sam_follow-up_038860f0.plan.md`
  (no Outcome section; the todo states are the record, E2-E9 `pending`),
  `/home/nick/.cursor/plans/multicam_segmentation_transfer_8116dfb2.plan.md` ("Outcome (Sep 20,
  closing)" with the Sep 21 addendum), `/home/nick/.cursor/plans/sam3_exemplar_detection_correction_87b15462.plan.md`
  ("Outcome (2026-09-22 ...)"); the close-out and FineBio plan itself is
  `/home/nick/.cursor/plans/finebio_pivot_and_assembly101_closeout.plan.md`, whose Part 2 is
  copied into the repository as
  [`docs/plan-2026-09-24-finebio-detector-seeded-lab.md`](plan-2026-09-24-finebio-detector-seeded-lab.md)
  (status: planned, not started). Human records:
  [`docs/labeling-sessions-2026-09-20.md`](labeling-sessions-2026-09-20.md) (outcome lines for
  all five sessions), `docs/qa/*.human-record.json`, `docs/qa/anchor-scoreboard-*.md`.
- **Disk.** `uv run python scripts/prune_runs.py` (dry run; the script never deletes): 255 run
  directories, 161 cited by tracked files, 18 cited only by another run's manifest, **76
  unreferenced, 1.94 GB**. The listing with sizes, grouped as the plan asked, is
  [`docs/qa/runs-archive-list-2026-09-24.md`](qa/runs-archive-list-2026-09-24.md): unreferenced
  runs 76 / 1,938,450,588 bytes; `blocked-by-gpu-guard/` 1 / 75,514; `failed-worker-edit-*`
  1 / 91,352; `code-snapshot-*` 22 / 75,881,653; `runs/dedup-equivalence/` 1,165,708,288 (its
  ten `pass3-*` roots 773,321,258); `runs/dedup-pass2-smokes-20260922/` 620,400,580; no
  `*-local` root exists any more. De-duplicated total 3,793,709,039 bytes (3.79 GB). **Nothing
  was deleted or moved**; the user decides.
- **Tag and commits.** Annotated tag `assembly101-lab-close` on the last commit of this
  close-out (its message carries the final-numbers summary; `git tag -n20 assembly101-lab-close`
  prints it). The close-out commits, in order: `8921b56` (the FineBio plan document and its
  README link), `ac52fd7` (the archive list), and this entry's commit (the dispositions, the
  goals table, the labeling-sessions outcome lines), which the tag points at.

### Sep 24: cleanup pass

- **What this is.** After the close and the tag, the user approved a cleanup: the `runs/`
  deletion the archive list had been written for, and a small repository tidy-up (dead
  constants, unused configs, one-off scripts, stale defaults, superseded guides). CPU only
  (`CUDA_VISIBLE_DEVICES=""`); no GPU job, no run re-scored, no labeling; no calibration server or
  Rerun viewer was running (`pgrep -af muggled_calibration` empty before and immediately before
  the deletion). Nothing here changes a number in this ledger. Commits: `599c227` (code and
  configs), `fd40794` (scripts and docs), and this entry's commit (the deletion record and this
  entry). The user's own uncommitted README hunk (the Tailscale section) was left unstaged
  throughout; only the cleanup's own README hunks were staged, by patch.
- **Three decisions.** (1) The two Sep 15 policy JSONs the close-out plan named for deletion,
  `configs/muggledsam_static_four_part_reassembly_correction_policy.json` and
  `configs/muggledsam_static_four_part_reassembly_manual_seed.json`, are **kept as provenance**:
  run manifests that survive the deletion fingerprint them by sha256 (`source: measured`): the
  Sep 16 `muggledsam-sam3-four-part-static-full-exploratory-static-c10379-20260916t012945z`
  manifest, the `...static-corrections-frames-0-36-65-162-20260915t2331z` schedule and prompt,
  and the `...frame-zero-source190-calibration-redo-20260915t1749z` prompts. Deleting the files
  would leave those fingerprints pointing at nothing. (2) `configs/methods.yaml` **moved** to
  [`docs/methods-inventory-2026-09.md`](methods-inventory-2026-09.md) (`git mv`, body fenced as
  YAML, still parses: 14 methods): a hand-maintained Sep 8-16 snapshot that no code ever loaded;
  [`SOURCES.md`](SOURCES.md) and the goals table at the top of this file are the maintained
  records. (3) Every `code-snapshot-*` directory **deleted**: each was `git archive <commit>
  src/battle` for a commit still in this repository's history (`git cat-file -t` = commit for all
  20 distinct short shas), so the bytes were reproducible; the full shas are recorded in the
  archive list.
- **Code (commit `599c227`).** Fourteen module constants removed, each shown by `rg` to have
  no reference outside its own definition in `src/`, `scripts/`, `tests/`, `docs/`, README and
  configs, and none exported through an `__all__` or a star import: `C10119_ANCHOR_RECORD`
  (`human_seeds`), `CORRECTION_API` (`dam4sam_streaming`), `GATE_REASONS` (`muggled_worker`),
  `IMAGE_API` (`sam3_appearance`), `NEIGHBOUR_CLASSES` (`gpu_guard`), `OUT_OF_FRAME_CLASS`
  (`detector_scorecard_v2`), `MAX_COMMENT_LENGTH` (`zip_range`), the recording-1 `RAW_ROOT`
  aliases in `assembly101_reference`, `assembly101_clock_offset` and `assembly101_fetch_view`,
  `DERIVED_ROOT` and `LOCAL_RECORDINGS` (`assembly101_fetch_view`), `FINE_GRAINED_CSV` and
  `EGO_PROXY_DIMENSIONS` (`assembly101_reference`). None was kept: no test or document used
  any of them by name as an API. The workers that run under foreign interpreters were smoked
  afterwards: `muggled_worker.py --help` and `sam3_appearance.py --help` under
  `/home/nick/.pyenv/versions/muggled_sam/bin/python`, `dam4sam_video_worker.py --help` under
  the `samurai` pyenv, and `gpu_guard` + `dam4sam_streaming` imported under both; all exit 0.
  `configs/methods.example.yaml` and `configs/clips/assembly101_comparison.example.yaml`
  deleted (cited only by the Sep 8 plan's rename-history line, which stays as written).
  `battle_plan.agent.final.md` untracked (`git rm --cached`, `/battle_plan.agent.final.md` in
  `.gitignore`; the 167,691-byte file stays on disk and its history stays in git); README, this
  file's "original ask" paragraph and the Sep 8 plan's editorial preamble now say so, the plan
  body's own link is preserved as written.
- **Scripts and docs (commit `fd40794`).** `scripts/serve_review_over_tailscale.sh` defaults to
  `runs/interaction-review-first-minute-v6/interaction_review_combined.rrd` plus
  `segmentation.rbl`, passed to `rerun --serve-web` as a second positional path (the same form
  the README's `uv run rerun <rrd> <rbl>` line uses); it accepts `[recording.rrd
  [blueprint.rbl]]` and applies the default preset only when no path is given, since a preset
  fits only the package it was built for; `--dry-run` checked for four argument shapes.
  `scripts/rerun_client/README.md` updated to match; the working-tree README has no Tailscale
  section any more (the user's hunk), so no README line was changed for this. Five one-off
  helpers moved with `git mv` to `scripts/archive/` (`compare_frame_rate_arms.py`,
  `report_e4_candidate.py`, `report_ego_viewpoint_screen.py`, `render_overnight_v3_audits.py`,
  `profile_view_route.py`) with a [`README`](../scripts/archive/README.md) naming what each
  produced and which run root (all those roots are ledger-cited and kept); the two that find the
  repository root by `parents[]` now use `parents[2]`; README and ledger paths updated. The
  [Sep 18 multicam](review-guide-2026-09-18-multicam.md) and
  [ensemble v2](review-guide-2026-09-20-ensemble-v2.md) review guides carry a superseded banner
  pointing at [the presets guide](review-guide-2026-09-20-multiview-presets.md) and the v6
  package; the v4 and v5 recordings they open still exist. The user's untracked `test.sh` was
  pointed at the same v6 pair in place (same `rerun --serve-web` form, other lines kept); it
  stays untracked.
- **Runs deleted (this entry's commit records it).** Re-verification per candidate before
  deleting, stricter than `prune_runs.py`: no tracked file names it (the archive list itself
  excluded from the corpus, because it names every candidate and is now tracked); no
  manifest-like file (`*.json`, `*.md`, `*.yaml`, `*.txt`, `*.log`) anywhere under a surviving
  run names it (the script reads only three top-level files); not on the protected list;
  exists. **72 of the 76 passed and were deleted, 1,814,511,538 bytes.** Four failed the
  surviving-run check and are kept (123,939,050 bytes): `wilor-hands-stabilized-60s-v4`
  (`review-metrics-first-minute-v1/metrics.json`), `kineo-nlf-headless-60s-v4-postreboot`
  (`kineo-nlf-fused-60s-v4/provenance.json`), and the two Sep 21 4-part consensus variants
  `...-r1280-4part-excl-c10119-20260921` and `...-r1280-4part-others-only-20260921`, named by
  `reprompt_plan.json` / `reprompt_provenance.json` / `chain2.log` under the protected
  `multiview-reprompt-20260921` and by `sam3-exemplar-20260922` (`README.md`, `poolcmp/*/plan.json`,
  `reprompt/*/iter1/*`). Also deleted: `runs/dedup-equivalence/` (17 sub-roots, 1,165,708,288
  bytes), `runs/dedup-pass2-smokes-20260922/` (620,400,580), the 22 `code-snapshot-*`
  directories (75,881,653; two of them inside the dedup-pass2 root),
  `runs/sam3-views-r1280-4part-20260921/blocked-by-gpu-guard` (75,514) and
  `runs/sam3-views-r1280-20260919/failed-worker-edit-20260920t0156z` (91,352); the parent runs
  of the last two and of every snapshot stay. De-duplicated sum of deleted file sizes
  **3,669,769,989 bytes (3.67 GB)**; `du -sb runs` 62,554,940,832 -> 58,885,393,099 bytes
  (`du -sh` 60G -> 56G); 96 `rm -rf` targets, each resolved and checked to be a directory under
  `runs/`; top-level run directories 255 -> 182. Every protected root (the v6 package, ensemble
  v2, anchor scoreboards, per-view anchor workspaces, the r1280 and 4-part view passes, the
  pm-append consensus and hull roots, the reprompt and exemplar roots, dam4sam arms, the policy
  ablation, detector scorecard, seed search, finebio, rec2, assembly101, labeling sessions,
  human-accepted seeds, the static comparison) and all 18 section-7 runs are present. The full
  tables (names, bytes, snapshot shas, the skipped four) are the "Deleted Sep 24" section of
  [`docs/qa/runs-archive-list-2026-09-24.md`](qa/runs-archive-list-2026-09-24.md).
- **Tests and lint.** Before the deletion (after the code changes): default tier 762 passed /
  9 skipped (the nine `test_worker_policy` torch-absent skips, as at the close), `-m real_data`
  62 passed, `ruff check src tests scripts` clean. After the deletion: default tier 762 passed /
  9 skipped, unchanged; `-m real_data` **52 passed / 1 skipped**: the ten parametrised cases of
  `tests/test_video_driver.py::test_recorded_smoke_manifests_rebuild_byte_for_byte` over
  `runs/dedup-pass2-smokes-*/*/*` collapsed to one "got empty parameter set" skip. No other test
  newly skips or fails; `-m gpu` not run (CPU only).

### Sep 24, night: FineBio contracts and fixtures (p0-contracts)

- **What this is.** The first todo of the FineBio 3D-tracking plan
  ([`docs/plan-2026-09-25-finebio-3d-tracking.md`](plan-2026-09-25-finebio-3d-tracking.md)):
  the contracts the lanes develop against, the preflight's geometry code moved into the
  package, recorded preflight observations as committed fixtures, and the frame-index contract
  as a test. Evidence base: [`docs/preflight-2026-09-24-finebio.md`](preflight-2026-09-24-finebio.md).
  CPU only, no GPU, no Rerun viewer, nothing under `data/` or `runs/` written except the two
  preflight files re-generated for the byte-identity check. Other lanes committed concurrently
  in the same checkout (trial selection, the detector driver, the worker modes); only the files
  named below were staged, by path. Commits: `9bfbeed` (schemas), `ce9b010` (camera library,
  refactor, config), `835566b` (fixtures), `7436c7a` (frame-index contract test), and this
  entry's commit.
- **Schemas (`src/battle/multiview_schemas.py`, commit `9bfbeed`).** Four `VersionedModel`
  contracts (extra fields forbidden, frozen, `schema_version`), kept small so the lanes can
  extend them. `FineBioObservation`: `view`, raw `frame_index`, `slot` (`<class>#<k>`; for
  detector rows k is the same-class score rank within the frame, not an identity; for SAM3 rows
  the seeded slot), `object_class` (the field is not called `class`, a keyword),
  `detector_score`, `box_xyxy_px`, optional `mask_bbox_px`, `mask_centroid_px`,
  `mask_area_px`, `sam3_object_score`, `pose_valid`, `source` in {`detector`, `sam3_decode`,
  `sam3_video`}, `provenance` dict; a row must carry a box or a mask bbox (detector rows a box,
  SAM3 rows a mask bbox), and `point_px` is the mask centroid when present, else the box centre.
  One deviation from the plan's wording: `box_xyxy_px` is optional, because a SAM3 video-memory
  row has a detector box only on frames where the detector ran. `FineBioCameraConfig`: trial,
  recording day, `fixed` keyed by view (`FineBioFixedCamera`: camera id, `provenance` in
  {`shipped`, `marker_pnp`}, K for the shipped video resolution, five distortion terms, `rvec`,
  `tvec`, image size, the marker-fit residual of the pose in use and of the shipped pose, so
  camera 6's 93.7 px sits beside its 0.7 px), `fpv` (`FineBioFpvCamera`: rescaled K,
  distortion, pose source file, pose frame count, valid fraction, marker-residual and velocity
  gate parameters), `units` = "board centimetres, z into the bench", `frame_index_offset`
  (raw = proxy + offset; 0 for the trial-level config). `Track3D`: frame, track id, class,
  position cm, scalar uncertainty cm, support views, `state` in {`observed`, `single_view`,
  `coasting`, `held`, `contained`, `lost`}, confidence, abstain, `possibly_same_as`.
  `TrackEvent`: frame, track id, `kind` in {`birth`, `lost`, `coasting`, `reacquired`,
  `ambiguous`, `held`, `contained`, `handoff_reseed`, `detector_reseed`}, payload. Plus
  `write_jsonl` / `read_jsonl` (`compact=True` omits `schema_version` and empty provenance,
  restored on read). 13 tests in `tests/test_finebio_contracts.py`.
- **Camera library (`src/battle/finebio_cameras.py`, commit `ce9b010`).** `Camera`,
  `intrinsics(kind)` with the 0.5 / 0.48 rescale, `days`, `fixed_camera`, `marker_points`,
  `fpv_poses`, `fpv_camera`, `detect_markers` (ArUco `DICT_6X6_50`), `match_markers`,
  `rig_cameras` (shipped pose kept within 10 px, else marker PnP) moved verbatim out of
  `scripts/finebio_preflight.py`; new `camera_config_from_mapping(mapping, trial)`,
  `cameras_from_config`, `fpv_camera_from_config`, `write_camera_config`,
  `read_camera_config`, `marker_corner_rms`. `video_path` and `read_frame` went to
  `finebio_frames.py`. The script imports both modules and keeps its own `STATIC_CLASSES`,
  `MOVING_CLASSES`, `draw_markers` and subcommands. **Byte-identity check:** the Sep 24
  `mapping/mapping.json`, `mapping.md`, `rig/rig.json`, `rig.md` were copied to `/tmp`, then
  `uv run python scripts/finebio_preflight.py --trial P03_01_01 mapping --seconds 30,60,90`
  and `... rig --detections runs/preflight-finebio-20260924/detections` were re-run from the
  refactored script; all four files were rewritten (mtimes 23:03) and `cmp` finds no
  difference (`mapping.json` sha256 `d85122db…`, `rig.json` `89fb6928…`). The trial config
  `configs/finebio/cameras/P03_01_01.json` (5,192 bytes, numbers only) was written by
  `camera_config_from_mapping` from that mapping: day 221013; `T1..T5` = cameras 1,2,3,4,6;
  T1-T4 `shipped` at 6.32 / 6.94 / 5.08 / 2.23 px; T5 `marker_pnp` at 0.71 px with the shipped
  93.71 px recorded beside it, centre (-2.87, 5.27, -90.58) cm; fpv K rescaled, 5032 pose
  frames, 97.18% valid, gates 20 px and 5 cm/frame (parameters, not measurements). 5 default
  tests plus 2 `real_data` tests (`camera_config_from_mapping` reproduces the committed config
  field for field; the T4 frame at 60 s fits the committed pose under 10 px).
- **Fixtures (`tests/fixtures/finebio_preflight/`, commit `835566b`).** 4,523,703 bytes:
  `observations.jsonl` 4,448,885 bytes, 20,209 rows (15,695 `detector`: every FineBio DINO box
  at score >= 0.3 on the 78 preflight frames 1798..1858 consecutive plus 1888..2368 every 30th,
  six views; 52 `sam3_decode`: the box-prompt masks at frame 1798, encoder side 1280, six
  views; 4,462 `sam3_video`: the SAM3.1 track series 1799..2097 in fpv, T2, T4, T5 on plate,
  pipette, centrifuge and 50ml tube, with object score, mask bbox, mask area, and the detector
  box and `detector_box_iou` on the 67 frames where the detector ran; the 14 T2 tube frames
  behind the arm and the 9 fpv pipette frame-edge frames are absent rows, listed in the
  reference); `cameras.json` 5,192 bytes, identical to the committed config;
  `fpv_poses.json` 43,686 bytes (the shipped fpv `rvec`/`tvec` and validity for the 310 frames
  that carry an observation, so six-view triangulation needs no `data/`); `rig_reference.json`
  20,237 bytes; `README.md` with the commands, the licence note (numbers only; no frame,
  video, mask or `.rrd` anywhere in the repository) and the centroid approximation: SAM3 rows
  carry the **mask-bbox centre** as `mask_centroid_px`, not the area centroid, because the
  preflight kept only bounding boxes. Boxes rounded to 0.1 px, scores to 4 decimals. Builder
  `scripts/finebio_preflight_fixtures.py`; loader `tests/finebio_fixtures.py`
  (`load_preflight_fixtures()` -> observations, cameras, rig reference, fpv poses, with
  `rows(...)`, `by_frame()`, `fixed_cameras()`, `fpv_camera(frame)`). `tests/test_finebio_fixtures.py`
  (7): the eleven static objects re-triangulated from the fixture boxes and `cameras.json`
  match the preflight points within 0.05 cm and every leave-one-view-out residual within
  0.5 px (LOO summary 10.4 / 25.3 px); the plate hand-off into the fpv reproduces 45 frames,
  median and p90 within 0.5 px, inside the fpv box on 100%; the frame-1798 fpv centre
  (5.96, 36.31, -35.54) cm; 197 fpv rows carry `pose_valid: false`.
- **Frame-index contract (`src/battle/finebio_frames.py`, `tests/test_finebio_frames.py`,
  commit `7436c7a`).** `proxy_ffmpeg_args(raw, out, start_frame, frame_count)` is the one
  proxy recipe for this phase and the contract lane B's `finebio_preprocessing` builds on:
  native resolution, native 30000/1001, **no `fps=` filter**, exact-frame trim
  `select='between(n,start,end)',setpts=N/FRAME_RATE/TB` on a full decode (no `-ss`),
  `-fps_mode passthrough`, `-frames:v count`, libx264 crf 18, yuv420p, faststart, no audio.
  `build_proxy` verifies the counted frame count; `frame_index_contract` reports the mean
  absolute grey difference between proxy frame k and raw frames start+k-1, start+k, start+k+1;
  `pose_length_check`; `proxy_marker_check`. **Numbers** (P03_01_01 fpv, 30-frame proxy from
  raw 1798 built into `tmp_path` in 2.5 s; 1920x1440, `r_frame_rate` 30000/1001, 30 counted
  frames): proxy frames 0 / 14 / 29 vs raw 1798 / 1812 / 1827 differ by **0.71 / 0.69 / 0.91**
  grey levels at offset 0, **1.41 / 1.05 / 4.27** at -1 and **1.42 / 1.19 / 5.24** at +1; the
  minimum is at 0 on every sample (the head barely moves over 1798-1812, so the margin there is
  1.5-2x; at 1827 it is 5x). Pose length: 5032 == fpv, T1, T5 frame counts for P03_01_01;
  8492 for P03_03_01 (fpv 99.0% valid). Markers on the three proxy frames fit the shipped pose
  of raw frame 1798+k at **1.03 / 0.74 / 0.89 px** median corner RMS (2 markers each), the same
  0.9 px the preflight measured on raw frames. Raw frames are read by cv2 seek, the access path
  the preflight's detections and fpv-pose check used, so the seek index and the pose index are
  already known to agree. The default tier checks the recipe's arguments and an exact 2-of-3
  frame trim on the synthetic clip.
- **Tests and lint.** Default tier **789 passed / 9 skipped** (762 + 27 new; the nine skips are
  the `test_worker_policy` torch-absent skips as before); `uv run pytest -q -m real_data -k
  finebio` **7 passed** (2 camera, 5 frame-index); `uv run ruff check src tests scripts` clean
  and `ruff format --check` clean on the ten files touched. `-m gpu` not run.
- **For the next phase.** Lane B (`p0-cameras`, `p1-configs`, `p1-rig`): build proxies with
  `proxy_ffmpeg_args` unchanged and record `frame_index_offset = start_frame` in a per-window
  copy of the config; `camera_config_from_mapping` is the function the `battle-finebio-cameras`
  step should wrap (the mapping report shape is the preflight's); `rig_reference.json` holds the
  numbers the `-rig` regression reproduces. Lane C (`p0-slice`, `p3-tracker`):
  `load_preflight_fixtures()` gives six views, 78 detector frames and 299 SAM3 frames in raw
  pixels with `fixed_cameras()` and `fpv_camera(frame)`; detector `slot` ranks are not
  identities; SAM3 centroids are bbox centres; missing SAM3 masks are missing rows; every
  triangulation goes through `Camera.undistort` then `dlt_triangulate` on the `projection`
  matrices, as in `tests/test_finebio_fixtures.py`.

### Sep 24, night: FineBio detector, CUDA build and battle-finebio-detect (p2-detect tooling)

- **What this is.** The tooling half of `p2-detect` in the FineBio 3D-tracking plan
  ([`docs/plan-2026-09-25-finebio-3d-tracking.md`](plan-2026-09-25-finebio-3d-tracking.md)):
  a CUDA build of the FineBio detector environment, time-boxed to one hour, and the
  `battle-finebio-detect` driver that will detect every frame of the trial window once the
  window is chosen (`p0-trials`, another lane). Evidence base:
  [`docs/preflight-2026-09-24-finebio.md`](preflight-2026-09-24-finebio.md); its
  `runs/preflight-finebio-20260924/detections/*.jsonl` (CPU DINO, 6 views x 78 raw frames) is
  the regression reference for everything below. GPU use: about two minutes in total (blank
  image smokes, six views at one frame, the 78-frame set, a 60-frame strided run, Deformable
  DETR at one frame), never beside another model process (`nvidia-smi` showed only
  `kwin_wayland`, 144 MiB, before every GPU command; the Steam game named in the brief was not
  running). Other lanes committed concurrently in the same checkout; only the files named here
  were staged, by path. Commits: `beec0ae` (venv script, driver, tests, superseded-script
  notes, one `pyproject.toml` line) and this entry's commit. Outputs under
  `runs/finebio-detect-validation-20260925/` (gitignored; FineBio licence) and the venv under
  `/home/nick/src/finebio-detector/.venv-cuda` (outside the repository).
- **CUDA build: option (a) worked, nine minutes of the sixty.** Timeline (UTC): 02:56:38 venv
  created; 02:57:52 torch installed; a first mmcv build failed after 3.5 min on
  `crt/host_config.h:137: unsupported GNU version! gcc versions later than 15 are not
  supported` (Fedora 44's default compiler is GCC 16.2.1; CUDA 13.2 accepts up to 15); the
  second build with `CC=gcc-15 CXX=g++-15` (installed on this machine; `torch.utils.cpp_extension`
  turns `$CC` into `nvcc -ccbin`) ran 03:01:24-03:03:51 (**2 min 27 s**, `MAX_JOBS=16` on 32
  cores); mmdetection installed editable by 03:05. **No mmcv source patch was needed**: the
  two legacy APIs mmcv 2.1.0's ops use, `THC/THCAtomics.cuh` and `c10::optional`, are still
  shipped by torch 2.13 (`c10::optional` as an alias of `std::optional`). Exact versions in
  `.venv-cuda`: Python 3.10.20 (`uv venv --python 3.10`), **torch 2.13.0+cu130** and
  **torchvision 0.28.0+cu130** from `download.pytorch.org/whl/cu130` (the torch line the
  MuggledSAM env runs; `torch.cuda.get_arch_list()` = sm_75, 80, 86, 90, 100, **120**),
  mmengine 0.10.7, numpy 1.26.4 (`numpy<2` as in the CPU venv), setuptools 78.1.0
  (`setuptools<80`, mmcv's `setup.py` imports `pkg_resources`), ninja 1.13.2, **mmcv 2.1.0**
  built from the PyPI sdist `mmcv-2.1.0.tar.gz` (sha256
  `d387bcab66b467479b6660310e23746cfc79c6e57acf04094680adb499a5cd3f`, kept with its `build/`
  tree under `/home/nick/src/finebio-detector/mmcv-2.1.0/`) with `MMCV_WITH_OPS=1 FORCE_CUDA=1
  TORCH_CUDA_ARCH_LIST="12.0" CUDA_HOME=/usr/local/cuda` (nvcc 13.2.86, driver 610.57.04;
  `mmcv.ops.get_compiling_cuda_version()` = 13.2, compiler GCC 15.3), **mmdet 3.3.0**
  editable from the shared `mmdetection` checkout `44ebd17b` (v3.3.0), the same configs and
  checkpoints as the CPU venv. `mmcv.ops.nms` on CUDA tensors and DINO's
  `MultiScaleDeformableAttention` CUDA kernel both run. Two runtime accommodations, both in
  the driver, neither in the environment: (1) torch >= 2.6 loads checkpoints weights-only by
  default and the authors' `.pth` files carry mmengine `HistoryBuffer`s (numpy arrays pickled
  through `getattr`) in their message hub, which the safe unpickler refuses even after
  allow-listing `HistoryBuffer`, `numpy.core.multiarray._reconstruct`, `ndarray`, `dtype`,
  `Float64DType` (it then asks for `getattr` itself); the worker sets
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for its own process only, and the manifest records the
  checkpoint's sha256 and `torch_load_weights_only: false`. (2) cuDNN's default TF32
  convolutions move DINO's scores by up to 0.03 against the CPU reference (T5 frame 1798,
  top 20: 0.29 px / 0.031 with TF32, **0.010 px / 0.0004** with `allow_tf32 = False`, at
  48.3 vs 52.7 ms/frame; TF32 on both conv and matmul 39.6 ms); the worker turns TF32 off by
  default and `--tf32` allows it. `scripts/install_finebio_detector.sh --cuda` (or
  `FINEBIO_CUDA=1`) is the idempotent recipe with these pins and the host-compiler override
  (`FINEBIO_HOST_CC` / `FINEBIO_HOST_CXX`); re-running it on the built venv skips every step
  and runs the GPU verification (`load=0.5s infer=52ms peak_reserved=696MiB` on a blank
  1333x800 image; skipped automatically while a Battle worker is on the card). The CPU target
  is unchanged and still passes `--skip-verify`. Blank-image smoke before the driver existed:
  load 0.6 s, 43-48 ms/frame at 1920x1080, peak reserved 678 MiB.
- **Driver (`src/battle/finebio_detect.py`, console script `battle-finebio-detect`).** Folds
  `scripts/finebio_dino_detect.py` (Sep 21) and `scripts/finebio_preflight_detect.py`
  (Sep 24) into one CLI; both stay in place with a one-line superseded note because records
  cite them. Reads the **raw** FineBio videos by raw frame index (fixed 1920x1080, fpv
  1920x1440, 30000/1001 fps), one `cv2` seek to the first wanted frame then sequential
  `grab`/`read`, the preflight's access pattern, so pixels and boxes are the same. `run
  --trial --views fpv,T1..T5 --start S --count N` (or `--frames N,A-B,A:B:S` for an explicit
  set such as the preflight's `1798-1857,1798:2398:30`) `--stride-fpv --stride-fixed --model
  dino|deformable-detr --device cpu|cuda --output runs/<run>/detections`; `--device` picks
  `.venv/bin/python` or `.venv-cuda/bin/python` under `/home/nick/src/finebio-detector` and the
  CLI (Battle venv) spawns that interpreter on this file's `worker` subcommand, which imports
  only the standard library plus `fs_common` and `gpu_guard` by path (tested: the file runs
  `--help` under `python -S` with no `battle` package). Outputs: `<view>.jsonl` (per raw frame
  `view`, `frame_index`, `image_hw`, `detections` with `class`, `class_id`, `score`,
  `box_xyxy_px` at score >= 0.05 in raw pixels, sorted by score, `interpolated`), `frames.json`
  (`frames_per_view` plus the flat union), `worker_result.json` (the detector interpreter's own
  record), `worker.log`, `manifest.json` (`battle-finebio-detect/1`: `state`, `model` with
  config and weights sha256 and the 35 classes, `settings` with device / interpreter / tf32 /
  threads / guard options, `frames` per view with stride and first/last, `detection_coverage`
  in {`full`, `strided`, `strided_interpolated`, `explicit`}, `versions`, `timing` and
  `per_frame_seconds` per view, `gpu` with peak reserved and allocated bytes and the full
  `gpu_guard` provenance, `inputs` with each video's sha256, `battle` and `mmdetection` git
  revisions, `claim_boundary`). **Strides and interpolation** (the plan's CPU fallback): with a
  stride the detected frames are `start, start+s, ...` plus always the window's last frame, so
  the fill can reach the end; `interpolate --directory` (or `run --interpolate`) fills every
  skipped frame by linear interpolation of box coordinates and score between detections on the
  two neighbouring detected frames that share a class and are associated greedily one-to-one by
  IoU >= 0.3 (`--interpolation-iou`); a class absent on either side, or whose boxes moved past
  the threshold, is never invented; filled rows carry `interpolated: true` and `source_frames:
  [a, b]`; the operation drops earlier filled rows first (idempotent) and flips the manifest to
  `strided_interpolated` with the fill counts. **GPU coordination** is `battle.gpu_guard` in the
  worker process (the one that holds the card), `vram` mode with the existing `unknown` profile
  (4 GiB expected peak x 1.5 = **6 GiB headroom required**, the plan's rule) unless
  `--gpu-guard-profile` / `--expected-peak-vram-bytes` say otherwise; a benign neighbour (the
  compositor, a browser, a Wine game such as `SYNTHETIK.exe`) never blocks, another Battle worker
  does unless its PID is named with `--allow-gpu-neighbour`; the decision is the manifest's
  `gpu.guard`. A refusal, a missing interpreter (`blocked`, with the install command in the
  reason) or a worker exception (`failed`, traceback in `worker.log`) all leave a manifest. No
  Rerun export: the viewer lane logs boxes from the JSONL.
- **Regression against the preflight (CPU).** `run --trial P03_01_01 --views T5 --start 1798
  --count 5 --device cpu` reproduces `runs/preflight-finebio-20260924/detections/T5.jsonl` rows
  1798..1802 **exactly**: 106 / 109 / 109 / 107 / 103 detections per frame, same classes in the
  same score order, worst box delta 0.0 px, worst score delta 0.0, `image_hw` [1080, 1920]. This
  is the `real_data` test `test_cpu_reproduces_preflight_t5_rows` (11 s; skips when the preflight
  JSONL, the raw video or the CPU venv is absent). CPU speed today 3.5 s/frame with 32 threads
  while a GPU job ran beside it (the preflight measured 1.95 s/frame on an idle machine).
- **GPU against CPU.** Six views at raw 1798 with the driver's first (TF32) build: all 120 top-20
  detections matched by class and IoU (min 0.992), 119/120 boxes within 1 px (worst 1.71 px on a
  pair of overlapping `right_hand` boxes in the fpv whose near-equal scores swapped), 115/120
  scores within 0.02 (worst 0.043). With TF32 off, the **78-frame preflight set on six views**
  (468 frames): elapsed **31.6 s** including model load 0.8 s and video decode, mean inference
  **54.7 ms/frame** (min 40, median 53, max 454 on the first frame's kernel warm-up), steady
  51-59 ms; peak reserved **924,844,032 bytes (925 MB)**, peak allocated 542 MB; **9360/9360**
  top-20 detections matched the preflight's class (min IoU 0.996), **all within 1 px (worst
  0.22 px)** and **all within 0.02 in score (worst 0.0087)**. Deformable DETR on the six views:
  42 ms/frame steady (294 ms on the first frame), peak reserved 736 MB; its CPU run on T5 1798
  matches the GPU top 20 to 0.0004 px / 5e-6. `torch.cuda.is_available()` true in the worker,
  recorded as `versions.cuda_available`; guard headroom at run time 15,033 MiB.
- **The fallback measured.** A strided GPU run over the preflight's 60 consecutive frames
  (`--start 1798 --count 60 --stride-fpv 3 --stride-fixed 5 --interpolate`): 21 fpv + 13 per
  fixed view detected in 8.5 s, 39 + 5 x 47 rows filled, coverage `strided_interpolated`.
  Filled rows against the preflight's real detections at score >= 0.3 on the same frames,
  matched by class and IoU: fixed views IoU **median 0.992-0.995, p10 0.93-0.97**, real
  detections with no interpolated counterpart 3-6% (T1 33/1120, T2 36/1446, T3 85/1726, T4
  101/1762, T5 49/1781; flicker and the moving hands), 19-63 extra interpolated boxes per view;
  the **fpv** at stride 3 is worse, median 0.979, p10 0.884, **21% missed** (287/1377): head
  motion breaks the IoU association. The fallback is therefore adequate for the fixed cameras and
  poor for the moving one, which is one more reason to run the GPU at full coverage.
- **Tests.** `tests/test_finebio_detect.py`: 21 default-tier tests (view and frame-spec
  parsing, strides keep the last window frame, coverage labels, video path layout, IoU,
  same-class one-to-one association by descending IoU, linear fill values and `source_frames`,
  refusal to invent a class absent on one side, idempotence and verbatim detected rows,
  manifest round trip and every validation branch, `interpolate` on a synthetic run directory
  and refusal on a failed manifest, worker command per device and the worker parsing what the
  driver produced, `CUDA_VISIBLE_DEVICES` per device, a blocked run without the interpreter
  writes a manifest, non-empty output refused without `--overwrite`, `run` without a window,
  the worker file importing without `battle`, and the detector's guard settings tolerating a
  1.4 GB `SYNTHETIK.exe` under Wine while refusing another Battle worker unless allowed) plus
  the `real_data` T5 reproduction. Default tier **810 passed / 9 skipped** (the nine
  `test_worker_policy` torch-absent skips as before); `uv run ruff check src tests scripts`
  and `ruff format --check` clean. `-m gpu` not run (no gpu-marked test added: the GPU checks
  above are recorded here, not in the suite, so the suite never holds the card).
- **For the detection phase and the seeding/arms lanes.** Full coverage of a 3600-frame
  window on six views (21,600 frames) is about **25 minutes** of GPU with DINO (67 ms/frame
  end to end, 55 ms inference) and about 20 with Deformable DETR; the CPU fallback at stride
  3 / 5 (4,800 frames) is 2.7-4.7 hours and loses a fifth of the fpv's real detections to
  interpolation, so it is a fallback only. The commands, once the window is fixed:
  `uv run battle-finebio-detect run --trial P03_03_01 --views fpv,T1,T2,T3,T4,T5 --start <S>
  --count 3600 --model dino --device cuda --output runs/<run>/detections` and the same with
  `--model deformable-detr --output runs/<run>/detections-ddetr` for the agreement check;
  `--allow-gpu-neighbour <pid>` only if a SAM3 worker must share the card. Rows are in raw
  pixels at raw frame indices (proxy frame k == raw start+k); `class_id` is the checkpoint's
  index into the 35-class list; a filled row's `interpolated: true` must be honoured by the
  seeder (a `strided_interpolated` manifest means the in-hand object's boxes between detected
  frames are guesses). Scores are the detector's own; nothing here is accuracy.

### Sep 24, night: plan approved, trial windows chosen (p0-trials, p-docs)

- **What this is.** The `p0-trials` todo of the FineBio 3D-tracking plan and the "plan copied
  into docs on approval" half of `p-docs`. The plan
  (`/home/nick/.cursor/plans/finebio_3d_object_tracking_demo_5b2e9c17.plan.md`) was approved on
  the night of Sep 24 after the preflight
  ([`docs/preflight-2026-09-24-finebio.md`](preflight-2026-09-24-finebio.md)) and the build
  started with four agents in one checkout (contracts, detector driver, SAM3 worker, this
  lane); only the files named below were staged, by path. CPU only: the FineBio DINO detector
  in its CPU venv (~2.2 s/frame), ffmpeg/ffprobe, cv2, PIL. No GPU, no viewer, nothing under
  `data/` written. Commits: `83e980a` (plan copy), `aebf229` (trials config and QA doc),
  `a9db73d` (SOURCES / LICENSES), and this entry's commit.
- **Plan copy (`docs/plan-2026-09-25-finebio-3d-tracking.md`, commit `83e980a`).** Header
  with the status (approved Sep 24 night, build started), the 19 todos from the plan's own
  frontmatter as an id / content table (all `pending` at approval), the body copied verbatim
  (checked with `diff` against the plan file). It is the fixed text this phase is measured
  against, as [`plan-2026-09-24-finebio-detector-seeded-lab.md`](plan-2026-09-24-finebio-detector-seeded-lab.md)
  was for the previous draft; that file now carries a one-line superseded note and is otherwise
  as written.
- **How the trials were read (`docs/qa/finebio-trials-2026-09-25.md`, commit `aebf229`).**
  Three series per trial from the top-down `T5` video: (i) every 30th raw frame through
  `scripts/finebio_preflight_detect.py --views T5 --spaced 0:N:30` (284 frames for `P03_03_01`
  in 613 s, 202 for `P20_03_01` in 402 s; the centrifuge box at 0.84-0.87 on every frame, hand
  boxes at score >= 0.3 counted per frame); (ii) mean absolute grey difference between
  consecutive frames on a 192x108 downscale of the lower 75% of the frame, every frame; (iii)
  the **lid state at every frame** as the fraction of pixels in a fixed rotor region inside the
  centrifuge base (`P03_03_01` `(1135, 300, 1270, 445)`, `P20_03_01` `(800, 180, 900, 285)`,
  raw T5 pixels, placed from the DINO box: closed `(1110, 271, 1294, 461)` / open
  `(1128, 162, 1341, 462)` in P03, closed `(765, 129, 930, 332)` / open `(728, 0, 952, 332)` in
  P20) whose OpenCV HSV hue is 75..100 with s > 70, v > 50: the rotor exposed when the lid is
  open reads ~0.02, the teal dome ~0.85-0.97, a gloved hand pressing the dome 0.36-0.62. The
  threshold **0.24** is the midpoint of the widest empty gap in the 1 fps sample of P03
  (nothing between 0.123 and 0.357); the full-rate histogram is bimodal (P03: 7043 frames under
  0.10, 62 between 0.10 and 0.35, 1387 above; P20: 5879 / 25 / 141) and thresholds 0.15, 0.20,
  0.24, 0.30, 0.35 give the same 10 state runs on P03 and 9 on P20, the closed-frame total
  moving by 52 and 17 frames across that range. By eye: 36 distinct 1:1 crops of P03 (every
  10 s over 0-150 s plus the frames on either side of each transition) and 16 of P20 agree with
  the state assigned; P20 frame 5190 (0.25) is the lid halfway down. Claim boundary: a pixel
  heuristic on one camera, verified on a handful of frames; hand and moved counts are detector
  output.
- **`P03_03_01` (trial 1).** Six views x 8492 frames (`ffprobe -count_frames`), pose file
  8492 rows, 8407 valid (99.0%), two invalid runs of >= 10 frames, [4851, 4907) and
  [8287, 8300). Lid: closed [0, 1099), open [1099, 1176) (tubes loaded), **closed [1176, 1228)**,
  open [1228, 3224), **closed [3224, 3311)**, open [3311, 5020), closed [5020, 5111), open
  [5111, 7458), closed [7458, 7545), open [7545, 8492): the lid opens once at 36.7 s and every
  later closure is a spin of 52-91 frames with a hand on the lid, **four cycles**, not the
  preflight's "at least six" (a loose count from a 12-frame sheet). Hands: 1.6-2.4 boxes in
  every sampled frame, so hands do not separate windows; bench motion does: 20-30 s is the
  quietest bin of the trial (0.33, 2 moved instances), 30-80 s and 100-140 s are busy
  (0.66-1.00), micro tubes move 4-12 instances per 10 s over 40-110 s (rack, vortex,
  centrifuge), the blue tip rack moves in the 40, 60, 70 and 80 s bins. No 120 s window that
  contains 916 can hold spin 3 (it ends at 5111, so the window would start at 1511), so two
  cycles is the maximum. Six candidates scored (916 offset, cycles, annotated frames, hands,
  motion, moved, fpv validity). **Window = raw frames [600, 4200)**, 20.02-140.14 s: 916 at
  316 frames (10.5 s) in, both cycles with loading and 29.7 s of unloading after spin 2,
  annotated frames 916 (six-view), 1544 and 3970 inside, motion 0.679 (highest), fpv pose valid
  3587/3600 (longest gap 5 frames). Rejected: [0, 3600) (trial start, quiet 10-30 s, 9 s after
  spin 2, 3970 lost), [900, 4500) (916 sixteen frames in), [750, 4350) (no gain over the
  chosen), [0, 4200) (140 s for a quiet 20 s).
- **`P20_03_01` (trial 2).** Checks passed: fpv and T1..T5 present, 6045 decoded frames in
  each, 1920x1440 / 1920x1080 at 30000/1001, pose `rets (6045,)` with 5716 valid (94.6%). The
  protocol-05 fallbacks were not needed. Lid open from frame 0, four spins of 34-42 frames:
  **[895, 937), [2500, 2539), [3798, 3834)**, [5190, 5224). Room 2 is busier (motion 0.8-1.3 in
  most bins) and its top-down camera sits higher (four markers in view). **Finding:** the fpv
  pose's five long invalid runs, [892, 909) + [915, 992), [2523, 2589), [3812, 3883),
  [5209, 5280), are one per spin, each starting between 3 frames before and 23 after the lid
  closes and ending 49-56 frames after it re-opens: leaning over the centrifuge takes the
  markers out of the head camera's view, so the fpv has no pose for the 2-3 s around every
  containment episode in this trial and the fixed cameras carry it alone; P03 has no such
  coupling. **Proposed window = raw frames [600, 4200)**, the same offset and length as trial
  1 (no per-trial window tuning either): 1442 at 842 frames (28.1 s) in, three full cycles,
  six of the seven annotated frames (695, 1112, 1442, 2368, 2769, 4196), motion 0.994 (highest),
  208 moved instances, fpv valid 93.1% (248 frames, longest gap 77) against 95.3% for
  [0, 3600) which has one cycle fewer; the drop-outs are the case the tracker is meant to
  survive, so the busier window is proposed.
- **`P03_01_01`** stays the smoke and the preflight regression reference: [1798, 1858), six
  views x 5032 frames, pose 4890/5032 valid, detections and SAM3 masks on disk, fixtures
  committed by `p0-contracts`.
- **`configs/finebio/trials.json`.** Per trial: id, role (`trial1` / `trial2` / `smoke`),
  protocol, room, day (P03 221013 from the preflight; P20 `null` until the camera solve), per-view
  frame counts, pose file, length and validity, invalid runs of >= 10 frames, window (start, end
  exclusive, seconds), in-window pose validity, shipped annotated frames and those inside the
  window, centrifuge boxes and rotor region, all lid-closed intervals, cycles in the trial and in
  the window, notes with the rationale and the rejected alternatives. Header comment states the
  frame-index convention (proxy frame k == raw start + k, one window for all six views).
- **SOURCES / LICENSES (commit `a9db73d`).** `docs/SOURCES.md` gains "Approved local source:
  FineBio camera poses and shipped checkpoints (Sep 24, 2026)": the archive
  `misc/finebio_camera_poses.zip` (64,013,894 B, SHA-256 `ee8ee467…217c3a`) and its 291 files
  (two GoPro 9 intrinsic npz with full hashes, ten days 221013..221208 x cameras 1,2,3,4,6 of
  extrinsics plus `marker_points.npy`, 226 fpv pose files, the authors' README and two vis
  scripts with hashes), how Battle uses them, and the seven `ckpts/` files with full SHA-256
  and sizes and what each is per the dataset and benchmark READMEs (`dino.pth` and
  `deformable-detr.pth` identical to the Sep 21 `gdown` copies; `dino_checkpoint_e30.pth` the
  IDEA DINO object detector frozen inside the manipulated/affected benchmark;
  `handobj_checkpoint_e5.pth` the Shan et al. CVPR 2020 hand-object detector re-implemented on
  IDEA DINO; `actionformer.pth.tar`, `asformer.model`, `mstcn.model` the atomic-operation and
  step-segmentation baselines, inventoried, not loaded). The dataset README (read Sep 24) notes
  the licence agreement text was updated 2026-09-10; the signed version is the user's record.
  `docs/LICENSES.md` gains the matching bullet: same terms, no licence of their own, everything
  under `data/` and every derived image outside Git, numbers only committed, no sharing
  determination.
- **Files kept and not kept.** Evidence images under
  `runs/preflight-finebio-20260924/trials/<trial>/` (gitignored): every 30th T5 frame,
  `motion.csv`, `lid_fullrate.csv`, `lid_state.csv`, `detections_T5_every30/T5.jsonl`, the
  contact sheets (5 s / 10 s), the 1:1 centrifuge crops looked at, `detect_every30.log`. The
  analysis was four one-off scripts under `/tmp` (sequential cv2 pass, HSV statistic, PIL
  contact sheets, window scoring); none committed, nothing new under `scripts/`.
- **For lane B (`p1-configs`).** Cut both proxies on exactly [600, 4200) with
  `finebio_frames.proxy_ffmpeg_args(raw, out, 600, 3600)`; proxy frame k == raw 600 + k, pose
  row `rets[600 + k]`; six-view anchor frames at proxy 316 (P03) and 842 (P20); cycles in proxy
  frames [576, 628), [2624, 2711) (P03) and [295, 337), [1900, 1939), [3198, 3234) (P20); the
  T5 centrifuge boxes and rotor regions apply unchanged to a native-resolution proxy. P20 needs
  `p0-cameras` before its rig check; its fpv gaps around the spins are drop-outs of the shipped
  pose, not outliers to gate. For `p5-events`: the lid intervals above are the reference the
  `contained` event's lid state can be checked against on these two trials.

### Sep 24, night: SAM3 worker modes for the FineBio arms (p3-worker)

- **What this is.** The `p3-worker` todo of the FineBio 3D-tracking plan: the two SAM3 worker
  modes the tracking arms (`p4-arms`) need, built into the production worker
  (`src/battle/muggled_worker.py`, MuggledSAM `a004ffc`, `sam3.1_multiplex.pt`) from the two
  paths `scripts/finebio_preflight_sam3.py` measured on Sep 24 (section C of
  [`docs/preflight-2026-09-24-finebio.md`](preflight-2026-09-24-finebio.md)). Four agents worked
  in one checkout; only the files named below were staged, by path (the `pyproject.toml` hunk
  by patch). GPU use: two smokes of 20 frames each, about 20 s in total, peak 2.8 GiB, no
  neighbour other than the compositor; no viewer. Commit `6e3d1db` (code, schemas, driver,
  tests) and this entry's commit. Claim boundary: every IoU below is between a SAM3 mask's
  bounding box and the FineBio detector's box on one frame of one trial, i.e. agreement between
  two models; nothing is measured against ground truth.
- **What already existed, and what was reused.** The worker had: manual seeds and
  multi-keyframe correction schedules with `selected_by: human | agent` provenance
  (`_corrections_by_frame`, `_read_verified_mask`, `_replace_prompt_memory_for_correction`
  rebasing a corrected slot into the predicted multiplex batch and installing it with
  `encode_prompt_memory_from_mask` under `replace` or `append` semantics; Sep 20
  `battle-multiview-reprompt` authored agent corrections through exactly this path); the Sep 18
  memory policy arms `off | exclusivity | gate | exclusivity+gate` with `memory_gate` handing the
  memory encoder score -1 (MuggledSAM's `no_object_embed`) for an untrusted slot; checkpoints
  named by the frame they have not yet stepped, `stream_identity` hashing everything that
  decides the stream, and the GPU guard (`strict` / `vram`, `--allow-gpu-neighbour`). All of it
  is kept and reused; nothing was rewritten.
- **Mode (i), memory-free per-frame box decode** (`--prompt-mode box_stream --box-stream
  <jsonl>`, `run_box_stream`). Input, one JSON record per frame: `{"frame_index": k, "boxes":
  [{"slot": s, "label": "...", "box_xyxy_px": [x0, y0, x1, y1] | "box_xyxy_norm": [...],
  "score": f | null, "source": "..."}]}`; `parse_box_stream` requires slots contiguous from
  zero over the stream, one label per slot (the labels are the run's concepts), exactly one box
  form with finite ordered coordinates; a frame absent from the stream, or with no boxes,
  writes an empty row. Per prompted frame: `interact.encode_image(frame, 1280, True)` once,
  then every box in **one batched decoder call** (`decode_box_prompts`: boxes as a `Bx1x2x2`
  tensor through `encode_prompts`; MuggledSAM's mask decoder expands the single image encoding
  over the prompt batch, `mask_decoder_model.py` lines 152-164), top-IoU candidate per box,
  logits > 0 bilinearly resized to source pixels. `--other-slots-as-negatives` adds the other
  boxes' centres as background points (the same count for every prompt, as batching needs);
  off by default so arm (b) is the plain decode the preflight measured. Output is the worker's
  usual `observations.jsonl` + `masks/<frame>_<slot>.png`, one object per slot per frame, with
  `object_score = iou_prediction = confidence =` the decoder's IoU prediction (0..1, not a
  tracker logit) and per row `prompt_box`, `source: sam3_decode`, `prompt_source`,
  `prompt_score`; the per-slot diagnostics carry the same plus `decoder_candidate_index`.
  Timing after `cuda.synchronize` per prompted frame (image encode, decode of all boxes, per
  box) goes into `runtime_settings.box_decode_timing_ms` with median / mean / p90 / max.
  `--checkpoint-every` and `--checkpoint-at` are a no-op with a printed message and an
  `ignored_checkpoint_flags` record (there is no tracker state; resume is re-running from
  `--start-frame`); `--resume-from-checkpoint` is refused by the parser. `stream_identity` gains
  the box stream's SHA-256, `other_slots_as_negatives` and `start_frame`, each only when used,
  so every earlier identity is unchanged. New for both modes: `--start-frame N` makes source
  frame N analysis frame 0 by decoding and discarding the earlier frames (no seek; the FineBio
  pose is indexed by raw frame).
- **GPU smoke of mode (i)** (`tests/test_muggled_arms_gpu.py::test_box_decode_arm_...`, run
  once): `P03_01_01` fpv raw frames 1798..1817 (20 frames, 1920x1440) with the preflight's four
  seed boxes (`sam3_preflight.json -> track.fpv.seeds`) on every frame, encoder 1280 square.
  Masks written for all four slots on frame 0; mask-bbox IoU vs the prompt box **plate 0.944,
  blue pipette 0.947, centrifuge 0.990, 50 ml tube 0.966** (the preflight's decode table has
  0.94 / 0.95 / 0.99 / 0.96 for the same frame); **152 ms per prompted frame** median, of which
  the image encode is 149 ms and the four decodes together 2 ms (1 ms per box); peak VRAM
  2.15 GiB. Against the preflight's 214-220 ms/step for the four-slot tracker at the same side,
  the memory-free arm is about 30% faster per frame and its cost is the encoder, not the box
  count: adding slots is nearly free. A second run of the same smoke while the detector lane's
  CUDA venv (882 MiB) shared the card gave identical masks (same four IoUs to the third
  decimal) at 225 ms/frame (encode 220, decode 5): the number to plan with is the uncontended
  one, and the arms should not share the GPU with the detector. Both smokes' outputs are kept
  under `runs/p3-worker-gpu-smoke-20260924/` (gitignored, 1.6 MB).
- **Mode (ii), the tau hook.** Checked first whether the Sep 18 `gate` already does it: with
  `--memory-gate on` a slot is memorised as absent when its raw score `<=
  gate_min_object_score` **or** its predicted IoU `< 0.5` **or** its contested fraction `> 0.2`
  **or** its area leaves `[0.5, 2] x` the rolling median of trusted frames; the ablation found
  the area band starving slots (53-67% gated without corrections). So it is not the plan's
  "skip the write when score < tau" and was not reused. Added beside it, with the same policy
  plumbing: `--memory-write-min-score TAU` (`memory_write_allowed(score, tau)`: write iff `score
  >= tau`; unset writes every present slot as before). Inside `memory_gate` the tau test is
  judged first and independently of the gate mode, so with the gate off it is the only test (no
  area history is kept) and with the gate on the two compose (tau, then the Sep 18 order). A
  gated slot is handed score -1 to `encode_frame_memory`, reason `low_object_score`, and its
  mask is still reported. The value joins `tracker_memory_policy` (`memory_write_min_score`),
  the `TrackerMemoryPolicy` schema (`is_default`, `worker_arguments`, run-id suffix `tau0p5`),
  the manifest and `stream_identity` **only when set**, so the default policy record, older
  checkpoints and the Sep 18 byte-identity regression are untouched. `battle-muggled-smoke`
  passes the flag through as well.
- **Mode (ii), box re-prompts and provenance.** A schedule correction may now be a box instead
  of a mask: `{"frame_index": k, "multiplex_slot": s, "target": label, "prompt_box": {"x",
  "y", "width", "height"} | "prompt_box_xyxy_px": [x0, y0, x1, y1], "selected_by":
  "detector_reseed" | "track_reproject"}`; `_validate_prompt_entry` refuses a box with a mask, a
  box in both forms, a malformed box, and a later box correction with any other `selected_by`
  (mask corrections keep `human` / `agent`, and legacy payloads without either field still
  load: the existing schedule-loading tests in `test_muggled_smoke.py` and
  `test_worker_memory_arms.py` pass unchanged). At frame k the worker decodes the box with the
  interactive decoder **on the tracker's own image tokens** (the tracking and interactive
  contexts share `encode_image`, `sam_v3p1_model.py` line 498, so a re-prompt costs one decoder
  call and no second encode), takes the top-IoU candidate, thresholds at source size and hands
  it to `_replace_prompt_memory_for_correction`, i.e. rebased into the multiplex batch and
  installed under the run's `--prompt-memory-semantics` (`append` for the arms) with the frame
  memory cleared unless `--keep-frame-memory-at-correction`. A box that decodes to nothing is
  skipped and recorded (`correction_skipped: decoded_empty`) rather than installed as an absent
  prompt. The corrected row's confidence is the decoder's IoU (1.0 stays the reviewed-mask
  sentinel); diagnostics carry `selected_by`, `prompt_box`, `prompt_decoder_iou`,
  `decoder_candidate_index`. Frame-0 seeds may be boxes too (`prompt_box` /
  `prompt_box_xyxy_px` on a seed, decoded at frame 0). `runtime_settings` records
  `box_prompt_correction_frames`, `correction_selected_by_kinds`, `box_prompt_seed_slots`,
  `box_prompt_api`.
- **Per-slot start frame, and the fixed-slot-count constraint.** The multiplex object count is
  fixed when the first prompt memory is encoded (`num_multiplex_objects`), so a slot cannot be
  added later. A seed with `"start_frame": k > 0` is therefore allocated at frame 0 with an
  all-false mask and its own prompt (box or verified mask) is applied at frame k through the
  correction path (`_corrections_by_frame` synthesises the entry, `seed_start: true`,
  `selected_by` defaulting to `detector_reseed`). Until then the slot is forced absent: no
  object row, `active: false`, memory score -1 (`memory_written: false`, reason `unseeded`),
  so its frame memory is only written once seeded. Two details: MuggledSAM's NumPy path in
  `encode_prompt_memory_from_mask` rescales a mask by `max - min`, a division by zero for an
  empty slot, so when any seed slot is empty the batch is handed over as a tensor of the same
  +/-1024 logits per pixel (`_prompt_mask_batch`; the NumPy path is kept when every slot has
  pixels, for byte-identity); and under `append` the frame-0 prompt memory with the empty slot
  stays in the prompt bank beside the later seed (under `replace` it is dropped). In the smoke
  the tracker itself scored the empty slot -2.2 to -2.3 on frames 1-4 (absent), so the empty
  prompt did not invent an object. `slot_start_frames` is recorded in `runtime_settings` and
  the metadata; a start frame that collides with a correction of the same slot is refused.
- **GPU smoke of mode (ii)** (`test_video_memory_arm_...`, run twice, same numbers): same 20
  fpv frames, three box seeds at frame 0 (plate, pipette, centrifuge; decoder IoU 0.90 / 0.91 /
  0.97 as the frame-0 confidences), the 50 ml tube starting at **frame 5** by `detector_reseed`
  box, a `track_reproject` box on the pipette at frame 10, `--prompt-memory-semantics append`,
  `--memory-write-min-score 0.5`. Tube absent with reason `unseeded` on frames 1-4 (the tracker's
  own raw score for the empty slot -2.2 to -2.5), seeded at 5 with decoder IoU 0.93 (mask-bbox
  IoU vs its box 0.93), tracked with raw score 8.4-10.8 on **14/14** frames after; the pipette
  re-prompt at 10 decoded at IoU 0.91 (mask-bbox IoU vs its box 0.94) and the slot continued at
  9.9-11.1; every active slot scored 8.4-12.6, so no slot fell below tau 0.5 and the gate reasons
  seen are `ok`, `corrected`, `unseeded`; 20 frames, 75 masks, 8.9 s (11.6 s beside the
  detector) including model load, peak 2.81 GiB (the four-slot tracker plus one decoder pass on
  a correction frame). Run id `muggledsam-arm-video-memory-fpv-<ts>-r1280-tau0p5-pm-append`.
- **Driver: `battle-muggled-arms` rather than `battle-muggled-smoke`.** The smoke's manifest
  path assumes an Assembly101 G2 preprocessing manifest (`--config`), a fixed `--view` choice
  list, `view_id` Literals on the schedule / policy / metadata schemas, and a correction
  schedule bound to a calibration workspace manifest; FineBio has none of these (its
  preprocessing manifest kind is `p1-configs`, in progress), so a sibling
  (`src/battle/muggled_arms.py`, two subcommands) spawns the same worker with the same GPU
  guard, `CUDA_VISIBLE_DEVICES`, `PYTHONPATH` and log files, on a raw or trimmed video with
  `--start-frame`, validates the box stream or schedule payload with the worker's own functions
  before launch (mask paths resolved against the schedule file and hashed), and writes a
  `MuggledSAMArmRunManifest` (`manifest_kind: muggledsam_sam3_arm_run`) binding video, prompt
  file, checkpoint and worker source by SHA-256, with the worker's condition record flattened,
  its measurements, `stream_identity` (now also written into `runtime_settings` in both modes),
  the memory policy / settings / schedule metadata, `observation_rows` and the validated
  `observations.jsonl` hash, and the FineBio licence note. Exit 3 when the worker did not
  succeed. Invocations:
  `uv run battle-muggled-arms box-decode --video <mp4> --view-id fpv --box-stream boxes.jsonl
  --start-frame 1798 --max-frames 20 --max-side-length 1280 --run-root runs/<root>` and
  `uv run battle-muggled-arms video-memory --video <mp4> --view-id fpv --schedule schedule.json
  --start-frame 1798 --max-frames 20 --max-side-length 1280 --prompt-memory-semantics append
  --memory-write-min-score 0.5 [--checkpoint-every 300] --run-root runs/<root>`; the schedule
  file is the worker payload (`seeds` with `target`, `initial_multiplex_slot`, `mask_path` |
  `prompt_box` | `prompt_box_xyxy_px`, optional `start_frame` and `selected_by`; `corrections`
  as above; `memory_semantics` optional and checked against the run's flags).
- **Schemas (`schemas.py`, additive hunks only).** `PerFrameObject.prompt_box / source
  ("sam3_decode") / prompt_source / prompt_score`; `TrackerSlotDiagnostic` box-prompt fields and
  `MemoryGateReason` `unseeded`; `PromptSelectedBy`; `TrackerMemoryPolicy.memory_write_min_score`;
  `MuggledSAMMultiKeyframeCorrection.prompt_box` with `candidate_id`,
  `human_selected_candidate_index` and `calibration_mask_fingerprint` optional **only** for a
  box entry and `selected_by` restricted per kind (a mask entry still needs all three and
  `human` / `agent`); the schedule validator skips uniqueness checks on absent candidates and
  masks; `MultiKeyframeCorrectionScheduleMetadata.detector_reseed_correction_frame_indices /
  track_reproject_correction_frame_indices / slot_start_frames`; `MuggledSAMArmRunManifest`.
  Every existing schedule, metadata and policy JSON round-trips unchanged (legacy payloads
  without the new keys are in the tests).
- **Tests.** `tests/test_worker_finebio_modes.py`, worker loaded by file path: 49 default-tier
  tests plus 5 torch cases (skipped in the battle venv, all 54 pass under the MuggledSAM
  interpreter, as do the 12 Sep 18 policy tests): box-stream parsing and 19 named validation
  faults, pixel / normalised box handling with clamping and the unit-square ulp guard, box
  corrections with the two provenance kinds and the refusals, mid-stream seeds and start-frame
  collisions, `memory_write_allowed`, tau alone and composed with the Sep 18 gate on synthetic
  scores, `_mark_unseeded`, the empty-slot seed batch, what joins `stream_identity`, the CLI
  cross-checks, a blocked `box_stream` run's condition record. `tests/test_muggled_arms.py`
  (12): both worker commands, run ids, the schedule payload loader (normalisation, hashing,
  refusals), schema round trips. `tests/test_muggled_arms_gpu.py` (`gpu`, 2): the two smokes
  above. Default tier **871 passed / 14 skipped** (762 / 9 before tonight across all four
  lanes); ruff clean on every file touched.
- **What the arms phase must know.** (1) Arm (b) is `box-decode` fed by the detector's
  per-view, per-frame boxes as a stream whose slots are the tracker's or seed tool's instance
  ids (slot labels are the concepts; a slot may be absent on any frame); identity is entirely
  the caller's. (2) Its cost is the encoder (~150 ms/frame at 1280 on this card), so a 3600-frame
  window on six views is about 55 min of GPU regardless of box count. (3) Arms (c)/(d) are
  `video-memory` with `--prompt-memory-semantics append`; seeds can be boxes, so no seed-mask
  PNGs are needed, and a slot that first appears mid-window is a seed with `start_frame`; the
  slot set must be known when the run starts (fixed multiplex count), so a new object that
  appears late needs either a pre-allocated slot or a second run. (4) Re-seeds are box
  corrections with `detector_reseed` / `track_reproject`; a box that decodes to nothing is
  skipped and visible as `correction_skipped`. (5) Tau is off unless `--memory-write-min-score`
  is given; the plan leaves its value to the data. (6) Checkpoint/resume works as before in
  `video-memory` (the schedule JSON is hashed whole into the identity, so a resume needs the
  same schedule); `box-decode` has no state. (7) The worker records `stream_identity` in
  `runtime_settings` now; nothing else in the observations or manifests of earlier runs changed.
### Sep 25: thin slice on the preflight window (p0-slice)

- **What this is.** The `p0-slice` todo of the FineBio 3D-tracking plan
  ([`docs/plan-2026-09-25-finebio-3d-tracking.md`](plan-2026-09-25-finebio-3d-tracking.md)):
  the whole chain, observations -> per-frame same-class triangulation -> 3D points, residuals
  and the seven preflight cross-checks in one Rerun recording, run end to end on the 60-frame
  preflight window of `P03_01_01` before any phase scales up. Built against the `p0-contracts`
  fixtures only (`tests/fixtures/finebio_preflight`, no `data/` needed except for the optional
  frame images and the marker corners), CPU, 3 s, no GPU, no viewer opened. Other lanes
  committed concurrently in the same checkout (camera CLI, worker modes); only the files named
  here were staged, by path. Commit `c23ba6b` and this entry's commit.
- **Tool (`src/battle/finebio_slice.py`, console script `battle-finebio-slice`).**
  `--fixtures <dir>` or `--observations <jsonl> --cameras <config.json>` (plus `--fpv-poses`
  in the fixtures' JSON shape, else the shipped pose file named in the config when it is on
  disk, and `--rig-reference` for the static comparison), `--frames start:count | a-b | list |
  all` (default: the consecutive frames present), `--min-score 0.3`, `--min-fixed-views 3`,
  `--fixed-views-with-fpv 2`, `--video-root data/raw/finebio --image-every N` (JPEG frames
  logged into the recording, guarded on the files existing; never committed), `--no-rerun`.
  Two observation sets go through one code path: `detector` (the top-scoring box per class per
  view at score >= 0.3) and `sam3` (`sam3_decode` and `sam3_video` rows, one per class per
  view by object score, `point_px` = the mask centroid, which in these fixtures is the
  mask-bbox centre). A class is triangulated when it is seen in >= 3 fixed views, or in 2 fixed
  views plus the fpv with a valid pose (`fpv_used: true`, the plan's birth rule for the raised
  in-hand object). Geometry as the preflight and `tests/test_finebio_fixtures.py`:
  `Camera.undistort` then DLT on the projection matrices (`triangulate_pixels`, which also
  accepts per-view weights for the tracker; equal weights reproduce `dlt_triangulate`
  exactly), reprojection with distortion, leave-one-view-out residual per view, height `-z`,
  fpv reprojection against the fpv box (skipped when the fpv is in the triangulation or the
  point is behind it). Outputs: `points3d.jsonl` (`SlicePoint`: frame, class, set, point cm,
  height, views used, fpv used, per-view residual and LOO px, fpv residual and inside flag,
  scores), `summary.json`, `summary.md` (per class: frames, residual and LOO median/p90,
  height, fpv residual and inside fraction, a **one-object flag** = median all-view residual
  within 30 px, P03's association gate; static classes against `rig_reference.json`; what the
  floor still needs), `slice.rrd`.
- **Recording (`runs/finebio-slice-20260925/slice.rrd`, 11.8 MB, 262 entity paths, `rerun rrd
  verify` clean, gitignored).** `rr.init(spawn=False)` + `rr.save` through
  `rerun_logging.init_and_save` with the blueprint (3D view left, six camera tiles in a 2x3
  grid right, time series and text tabs below). World `RIGHT_HAND_Z_DOWN`: bench outline,
  board origin, the day's three markers, five static frusta (`Transform3D` + `Pinhole`), the
  fpv frustum and trail per frame, `world/points3d/detector` and `/sam3` coloured by class
  with labels, `world/static_reference` (the preflight's eleven points, magenta) beside
  `world/static_objects` (the slice's window medians, green). The seven cross-checks as
  standing entities: (1) `world/<view>/markers_projected` (static for the fixed views, per
  frame for the fpv); (2) `world/<view>/fpv_camera_centre` in every fixed view per frame; (3)
  `world/static_objects` + `world/<view>/static_reprojected`; (4) `world/left_hand`,
  `world/right_hand` + `world/<view>/<hand>_triangulated`; (5) `checks/clock_scan`, the
  preflight's table as a `TextDocument` (not recomputed; on a window without a reference it
  says so); (6) `checks/loo/<class>/<view>` `Scalars` per frame; (7)
  `checks/handoff/cell_culture_plate` and `_inside` `Scalars` per frame. Also per view and
  frame: `detector` boxes (the top box per class), `sam3_masks` boxes, `points_reprojected_*`,
  `image` every 10th frame (36 JPEGs), `checks/support/<class>` (fixed views used) and
  `checks/summary` (the markdown).
- **Numbers, frames 1798..1857 (60 frames; the fixtures' 61st consecutive frame 1858 came
  from the spaced set and is left out).** Static classes, detector set, window median vs the
  preflight point: **0.00-0.01 cm on all eleven** (per-frame distance median 0.01-0.02 cm,
  p90 0.02-0.27), heights equal to the preflight's to 0.01 cm. `cell_culture_plate`: 60/60
  frames from 5 views, all-view residual **16.9 px median / 33.7 p90**, LOO per view
  **34.9 / 17.1 / 44.5 / 17.2 / 33.7 px** (T1..T5; preflight 34.8 / 17.2 / 43.7 / 17.1 /
  33.5 at score 0.4), height 3.25 cm; fixed -> fpv hand-off **36.8 px median / 43.1 p90 on
  60/60 frames, inside the fpv box on 100%** (preflight 36.5 / 51.9 on 45 frames at score
  0.4). `left_hand` 60/60 at 7.8 px (LOO 13.1), 3.25 cm; `right_hand` 51/60 frames (score
  0.3 vs the preflight's 0.5 and 14/61 at >= 3 fixed views; here 2 fixed + fpv fills in),
  17.6 px, 20.5 cm; `blue_pipette` 60/60, 16.1 px, 18.1 cm; `centrifuge` 8.6 px, 4.08 cm;
  `pcr_machine` 4.5 px; `trash_can` 16.6 px. SAM3 set: plate 60/60 from T2/T4/T5 at 11.2 px
  (LOO 26.9), hand-off 13.9 px median into the fpv mask, inside 100%; centrifuge 3.6 px;
  pipette from T2 + T5 + fpv on 59 frames. **Four classes fail the one-object flag**:
  `50ml_tube` (154 px), `micro_tube` (58 px), `pen` (304 px), `50ml_tube_rack` (389 px), and
  the SAM3 `50ml_tube` (121 px, height 21 cm): several instances on the bench and a different
  instance as the top box in each view, and the preflight seeded each view's SAM3 tube slot
  from its own top box, so the four SAM3 tube tracks are not one tube. That is the identity
  problem the tracker's pairwise-assignment birth exists for, seen on real rows.
- **Tests (`tests/test_finebio_slice.py`, 8, default tier).** Weighted DLT equals
  `dlt_triangulate` at equal weights and recovers a synthetic point to 1e-6; frame parsing;
  top-per-class selection (detector top box; SAM3 rows for the four seeded classes); the plate
  triangulates on every window frame with LOO medians within 1.5 px of the reference per view
  and the hand-off median within 1 px, inside 100%; the eleven static classes within 0.05 cm
  of the preflight and the one-object flag false for `50ml_tube` and `micro_tube`; the SAM3
  set uses the 2-fixed-plus-fpv rule for the pipette; settings change the rule; the CLI writes
  points, summaries and a recording into `tmp_path`. Default tier after this commit: 8 new
  tests; `uv run ruff check src tests scripts` clean.
- **What the floor still needs (also in `summary.md`).** SAM3 per-frame masks over the whole
  window in every view (arm b) instead of the preflight's four-view video-memory run with
  bbox-centre centroids; identity (a point here is one frame's agreement between views under
  the top-box rule; nothing links frames or survives an occlusion, and two instances collapse
  onto one box); detections on every frame of the trial-1 window, lane B's per-trial camera
  solve and rig gates. The tool already runs on real window outputs through `--observations`
  and `--cameras`.
### Sep 25: battle-multiview-tracks core (p3-tracker)

- **What this is.** The core of the `p3-tracker` todo of the FineBio 3D-tracking plan
  ([`docs/plan-2026-09-25-finebio-3d-tracking.md`](plan-2026-09-25-finebio-3d-tracking.md)):
  a 3D object with a persistent id as the tracked entity, every camera's box or mask an
  observation of it. Developed against the `p0-contracts` fixtures only, CPU, 1 s for the
  60-frame window and 3 s for 300 frames, no GPU, no viewer. The plan's Phase 3 "Core" list
  and nothing more: the `held` / `contained` / group extensions (`p3-tracker-ext`) wait for
  the occlusion inventory from arms (a)/(b). Other lanes committed concurrently (camera solve,
  worker modes); only the files named here were staged, by path. Commit `77d40f4` and this
  entry's commit. Time: about 2.5 h of the ~4 h box.
- **Tool (`src/battle/multiview_tracks.py`, console script `battle-multiview-tracks`).**
  `--fixtures <dir>` or `--observations <jsonl> --cameras <config.json> [--fpv-poses]`;
  `--gates <rig.json>` reads lane B's `gates` block (`association_px`, `handoff_px`,
  `birth_min_fixed_views`, `birth_fixed_views_with_fpv`), the four CLI flags of the same names
  override it, and without either the P03 preflight values apply (30 px association, 55 px
  hand-off at 1920, 3 fixed views, 2 fixed + fpv). Parameters: `--coast-timeout 30` frames,
  `--handoff-after 5` (K), `--reacquire-min-views 2`, `--min-score 0.3`,
  `--process-noise-cm 2`, `--coast-growth-cm 1`, `--fpv-weight 0.5`, `--source auto |
  detector | sam3`, `--frames`, `--reference-labels`; in code also `handoff_repeat_frames`
  30, `duplicate_distance_cm` 3, `base_uncertainty_cm` 1, `max_uncertainty_cm` 30. Outputs
  under `--output`: `tracks.jsonl` (`Track3D` rows per live track and frame, plus one row at
  the lost frame), `events.jsonl` (`TrackEvent`), `residuals.jsonl` (per track, view and
  frame: residual, gate, slot, source, confirmed), `identity_metrics.json`.
- **What the core does, per frame.** *Observations:* `point_px` (mask centroid, else box
  centre), class, detector and SAM3 scores, per-view slot; the fpv observes only with a
  valid pose. Source rule `auto`: per view, frame and class the SAM3 rows when present, else
  the detector rows (what arms (c)/(d) produce: masks for the seeded slots, boxes for the
  rest); `detector` and `sam3` force one. An observation is *class-confirmed* when it is a
  detector row, carries a detector score, or a same-class detector box lies within the
  association gate in that view and frame. *Predict:* stationary prior, the scalar
  uncertainty grows by `process_noise_cm` per frame (constant-position Kalman; velocity not
  modelled, no test asked for it). *Update:* every localised track is projected into every
  view with a valid pose; the gate is the association gate (hand-off gate for the fpv) plus
  the uncertainty projected at the track's depth (`f / depth` px per cm, ~10 px/cm on the
  fixed cameras); nearest same-class observation, conflicts between tracks by distance;
  with >= 2 views a weighted DLT (fixed 1, fpv 0.5; `finebio_slice.triangulate_pixels`),
  one prune pass over the gate, scalar Kalman blend with the prior. A view whose SAM3 slot
  differs from the slot the track had there before is a **slot disagreement**, written to
  the row (`slot_disagreement_views`), halving the confidence and setting `abstain`, never
  resolved. *Occlusion:* support 0 -> `coasting` (event), uncertainty + `coast_growth_cm` per
  frame, > timeout -> `lost` (event; the class is free for new births). *Birth:* per class,
  every fixed-view pair triangulated, pair cost = the larger of the two residuals, gated by
  the association gate, Hungarian per view pair (a numpy Kuhn-Munkres; `scipy` is not in the
  battle env), accepted pairs merged into cliques (one observation per view, every pair
  accepted), clique re-triangulated with the worst member dropped while over the gate, the
  fpv attached when its nearest unassigned same-class observation is within the hand-off
  gate; born with >= 3 fixed views or 2 fixed + fpv (`fpv_rule` in the payload).
  *Re-acquisition:* a candidate inside a coasting track's inflated gate in >= 2 of its views;
  exactly one candidate for exactly one track and confirmed in >= 2 views -> the id resumes
  (`reacquired`, latency); unconfirmed -> a new id with `possibly_same_as` and
  `unconfirmed_reacquisition_of` in the birth payload; two candidates for a track or two
  tracks for a candidate -> nobody resumes, every candidate born with `possibly_same_as` and
  an `ambiguous` event, one ambiguity counted per coasting track. *Hand-off re-seed:* for an
  `observed` track, a view with a valid pose and no associated observation for K frames
  (repeat every 30) whose projection lies inside the image emits `handoff_reseed` with the
  reprojected box (the view's last extent, else 120x80, centred on the projection;
  `provenance: track_reproject`, `box_from`, `frames_missing`, `uncertainty_cm`, `gate_px`),
  or `detector_reseed` with the detector box when a same-class box sits within the hand-off
  gate + uncertainty there. A view that never had the object is offered it the same way (the
  cross-camera case). *Duplicates:* live same-class tracks within 3 cm are marked
  `possibly_same_as` on both rows and abstain, counted, never merged.
- **Schema (`multiview_schemas.py`).** `Track3D` gains four optional fields with defaults, so
  the `p0-contracts` rows still validate: `residual_px` per view, `support_slots` per view,
  `slot_disagreement_views`, `frames_unobserved`. `confidence` is a heuristic (support
  fraction x residual term x slot term; coasting decays to 0; `abstain` on anything but a
  clean `observed`) to be replaced by `p5-confidence`.
- **Identity metrics (`identity_metrics.json`).** Tracks born / lost / live, re-acquisitions
  with latency median and max, ambiguities, duplicate pair frames, fragmentation per class
  (tracks born beyond the maximum simultaneously live; a lower bound), slot disagreements,
  **id switches** against `--reference-labels` (`{"view/slot": identity}`) when given, else
  against the SAM3 per-view slots as a proxy labelled `sam3_slots_proxy` (detector slots are
  score ranks and are excluded), event counts, the parameters, residual medians per class
  and view.
- **Tests (`tests/test_multiview_tracks.py`, 12, default tier; synthetic rig = the P03
  cameras of `cameras.json` plus the fixture fpv pose of frame 1798).** Hungarian equals
  brute force on 3x3, 2x4, 4x2, 5x5; gates from a rig block with defaults; birth from three
  fixed views (position within 0.3 cm); birth from two fixed + a valid fpv (refused with no
  pose, with `pose_valid: false`, or without the fpv); a false pair (A in T1/T3, B in T2/T4)
  rejected and two objects in four views assigned apart; coasting keeps the position, grows
  the uncertainty, abstains, is lost at 4 + timeout + 1 and a fresh id is born afterwards
  (fragmentation 1); re-acquisition resumes with detector confirmation (latency 5) and refuses
  SAM3 rows without a detector box (new id, `possibly_same_as`, the old track keeps
  coasting), a detector box within the gate confirming them; two candidates 1.5 cm apart ->
  nobody resumes, one ambiguity, both flagged and abstaining; hand-off re-seed with the
  last-extent 90x60 box centred on the reprojection in the view that lost the object and a
  default box in the view that never had it, `detector_reseed` at 48 px with a tighter prior;
  the source rule and confirmation; **the preflight fixtures** (detector source, 1798..1857):
  every static class's track median within 1 cm of the slice's point (all eleven under
  1 cm), one plate id `observed` on all 60 frames with >= 5 views and per-view LOO of its own
  support at or under the preflight's medians (T1 35 / T2 8 / T3 44 / T4 17 / T5 34 px vs
  34.8 / 17.2 / 43.7 / 17.1 / 33.5: in T2 the tracker's nearest-in-gate box is not the
  preflight's top-scoring one and fits better); CLI end to end with a gates file and an
  override. Default tier **899 passed / 14 skipped** (other lanes' tests included), `uv run
  ruff check src tests scripts` clean.
- **Runs on the fixtures (`runs/finebio-tracks-20260925/`, gitignored).** *`window`*
  (1798..1857, `auto`): 45 tracks born, 0 lost, 0 ambiguities, 2 re-acquisitions (the
  right hand coasting 2 and 3 frames), 2 id switches (a second pipette id born at 1857).
  Plate and centrifuge one id each on 60/60 frames, plate residuals T1..T5 24 / 3 / 11 / 12
  / 23 px and 41 px in the fpv (the preflight's hand-off 37); **the two 50 ml tubes the
  slice collapsed at 154 px are two tracks at 1.5-4 px** (T1/T2/T3/fpv and T1/T3/T5); 13
  micro tubes born, all flagged as near duplicates of their rack neighbours (3 cm) and
  abstaining; 96 `handoff_reseed` + 60 `detector_reseed` events, mostly bench objects
  offered to views that do not detect them. *`sam3-300`* (1798..2097, `auto`; after 1858 the
  detector rows exist every 30th frame only): plate and centrifuge still one id each on
  300/300 frames (plate 24 / 5 / 11 / 6 / 3 px, fpv 20); 172 born / 133 lost, 116
  re-acquisitions all at latency 30 (bench objects coast between detector frames and resume
  at each, timeout 30 being exactly the gap), 124 ambiguities of which 85 are micro tubes
  (identical instances, out of scope by the plan), the in-hand pipette fragmented into 5
  ids (it moves, and its SAM3 mask-bbox centre differs from the detector box centre; the
  centroid approximation the fixture README warned about), the T4/T5 50 ml tube emitting a
  `handoff_reseed` box for T2 every 30 frames because the preflight seeded T2's tube slot on
  a different tube: the event the worker's `track_reproject` correction is meant to consume.
  Slot disagreements 0 in both runs.
- **For the arms (what the tracker expects from the worker and the detector).**
  `FineBioObservation` JSONL in raw pixels and raw frame indices: detector rows with
  `box_xyxy_px`, `detector_score`, `slot = <class>#<rank>`; SAM3 rows with `mask_bbox_px`,
  the **area centroid** as `mask_centroid_px` (the fixtures carry the bbox centre), `slot`
  = the seeded slot id, `sam3_object_score`, and the detector box and score on the row when
  the mask came from one (arm b) or matched one; fpv rows with `pose_valid` from the config's
  gate. No converter was written here: `battle-finebio-detect` output and the worker's
  `observations.jsonl` each need a short adapter to these rows (a `scripts/` job for the arms
  phase; the fixture builder `scripts/finebio_preflight_fixtures.py` shows the mapping for
  the preflight's formats). Gates: pass lane B's `rig.json` through `--gates`. Open, to be
  read off the arms: K and the repeat cadence, the timeout (30 equals the detector stride
  here), the process noise for the in-hand object, whether the pipette needs velocity.

### Sep 25: battle-finebio-cameras, the per-trial camera solve (p0-cameras)

- **What this is.** The `p0-cameras` todo of the FineBio 3D-tracking plan
  ([`docs/plan-2026-09-25-finebio-3d-tracking.md`](plan-2026-09-25-finebio-3d-tracking.md)):
  the preflight's `mapping` subcommand as a standing step with a committed config per trial,
  its regression on `P03_01_01`, and the first two new trials. Evidence base:
  [`docs/preflight-2026-09-24-finebio.md`](preflight-2026-09-24-finebio.md) section A. CPU
  only (25-60 s per trial), no viewer; frames are read from the raw videos, nothing under
  `data/` or `runs/` is committed. Other lanes committed concurrently (slice, tracker, worker,
  detector, trials); only the files named here were staged, by path. Commit `45abe78`.
- **Library (`src/battle/finebio_cameras.py`).** `solve_mapping(trial, frames, read)` is the
  preflight's per-view logic as library calls: ArUco `DICT_6X6_50` on the N frames of every
  fixed view, `solve_view` ranks every (recording day, camera id) shipped pose by the median
  corner RMS against the detected corners (`_residual_table`, nearest projected centroid,
  best cyclic corner order), keeps the best, the best other camera and the same camera on
  other days, and runs one PnP over all corners against the best day's markers
  (`marker_pnp`; this block is the regression reference, unchanged); `decide_day` picks the
  day that minimises the summed residual over the five chosen cameras (the residual-weighted
  vote recorded beside it); then a **`chosen_day` block per view** records the chosen day's
  shipped residual and a PnP against the chosen day's markers, which is the pose a
  `marker_pnp` view actually uses (the preflight PnP'd against each view's *best* day, and on
  P03 T4's best day is 221109, not 221013). `fpv_pose_check(trial, day, read, step)` is the
  `fpv-pose` subcommand: shipped pose vs markers every `step`-th valid frame, the marker-PnP
  centre distance where >= 2 markers are seen, plus `fpv_velocity_gate_fraction` (consecutive
  valid frames whose centre steps more than 5 cm). `camera_config_from_mapping` now **gates on
  the residual of the pose in use**: the chosen day's shipped residual (from the block, or
  from the report's `table` for the preflight's `mapping.json`), shipped if <= 10 px, else
  the chosen-day PnP if its RMS <= 10 px, else the view is **dropped** (absent from `fixed`,
  listed with its numbers under `provenance.dropped_views`: the plan's stop rule, never a
  faked camera); per-view numbers under `provenance.views`. `rig_cameras` is unchanged for
  the preflight script, which still runs.
- **CLI (`src/battle/finebio_cameras_cli.py`, console script `battle-finebio-cameras`).**
  `solve --trial T --seconds a,b,c | --frames N,A-B,A:B:S [--output configs/finebio/cameras/T.json]
  [--evidence runs/finebio-cameras-T-<date>] [--fpv-step 250] [--max-rms-px 10] [--no-overlays]`
  writes the `FineBioCameraConfig` (numbers only, committed) and, under the gitignored
  evidence dir, `mapping.json` / `mapping.md`, `fpv_pose.json` / `fpv_pose.md`, one marker
  overlay per fixed view (ArUco red, chosen-day shipped pose green, the PnP pose cyan where
  it is the one in use) and a copy of the config. The day gets a **second witness**: the
  shipped fpv pose is the authors' own marker PnP, so it fits only the marker layout of the
  day it was computed against; the CLI measures the fpv corner RMS against the top-3 candidate
  days' `marker_points` (every 300th frame) and records it in `provenance.fpv_day_crosscheck`.
  `show --config` prints the decisions.
- **P03_01_01, the regression (`--seconds 30,60,90`, raw 899 / 1798 / 2697).** Ids 1,2,3,4,6,
  day 221013 (summed residual 116.8 vs 374.8 for 221021 and 474.4 for 221110), residuals
  identical to the preflight `mapping.json` (T1 6.32, T2 6.94, T3 5.08, T4 2.23 on its best
  day 221109, T5 93.71 px; PnP 0.78 / 1.10 / 0.40 / 0.76 / 0.71 px; PnP-vs-shipped 0.28 /
  0.62 / 0.88 / 0.96 / 6.43 cm; PnP centres equal to 0.01 cm). Chosen-day block: T4 fits day
  221013's shipped pose at **4.78 px** (PnP 0.80 px, 0.36 cm); T5's chosen-day PnP is the
  preflight's pose (centre -2.87, 5.27, -90.58). fpv: 0.93 px median / 1.96 p90 on 196/196
  frames, 3 over 20 px; PnP-vs-shipped 0.33 cm median; velocity gate trips on 16/4877 pairs
  (p99 step 3.19 cm). Cross-check: 221013 0.9 px vs 221021 22.0 / 221110 14.6. **The committed
  config changes in one number**: T4's `shipped_marker_residual_px` / `marker_fit_residual_px`
  2.234 -> 4.783 (the fit of the pose in the config, day 221013's, rather than day 221109's
  best fit; rvec / tvec unchanged), and `tests/fixtures/finebio_preflight/cameras.json` was
  refreshed to stay byte-identical (its README line updated). Everything else outside
  `provenance` is identical to the p0-contracts config.
- **P03_03_01 (trial 1; `--seconds 30,120,240`, raw 899 / 3596 / 7193).** Day 221013 (117.7
  vs 389.2 / 472.2 / 549.9; fpv cross-check 1.0 px vs 22.2 / 14.0). T1..T5 = cameras 1,2,3,4,6;
  T1-T4 `shipped` at **6.32 / 6.14 / 6.71 / 4.78 px** (best other camera 69.5 / 62.5 / 38.3 /
  105.2 px), PnP-vs-shipped 0.21 / 0.57 / 0.87 / 0.35 cm; T5 `marker_pnp` at **0.72 px** over
  36 corners, shipped 93.71 px, 6.44 cm apart. Camera 6's solved pose differs from
  P03_01_01's by 0.14 cm, 0.08 degrees and 0.16 px on the bench (same day, same mount:
  consistent). fpv 99.0% valid, 0.97 px median / 1.93 p90 on 169/169 frames, 1 over the
  gate, PnP-vs-shipped 0.29 cm median; velocity gate 16/8400 pairs. No view dropped.
- **P20_03_01 (trial 2, room 2; `--seconds 30,60,90`).** Room-2 days carry **four markers**
  (16 points) whose layouts differ by 2.5-7 cm between days. Day **221124** by summed residual
  (100.0 vs 155.2 for 221125, 213.8 for 221207, 316.1 for 221118); the fpv cross-check settles
  it: **1.1 px on 221124's markers vs 34.9 on 221125 and 46.6 on 221207**. T1..T5 = cameras
  1,2,3,4,6 again (best other camera 87-147 px). **No shipped fixed pose of the day fits
  cameras 1-4**: 29.4 / 16.2 / 14.7 / 32.2 px (T4's best day is 221125 at 17.8 px, still over
  the gate), so all four are **`marker_pnp` at 0.97 / 1.80 / 0.71 / 2.09 px** over 28-32
  corners, 4.77 / 2.97 / 2.81 / 3.89 cm from the shipped centres (the whole rig shifted a few
  centimetres between the calibration day and this recording, or the trial was recorded on an
  uncalibrated day close to 221124). T5 (camera 6) fits its shipped pose at **7.47 px** and is
  kept `shipped` by the rule; its PnP is 1.21 px and 2.92 cm away, so the rig check will say
  whether the rule was right for it. Camera heights 57-63 cm (T1-T4) and 90 cm (T5), the
  asymmetric room-2 layout. fpv 94.6% valid (invalid runs around the centrifuge spins, per
  `trials.json`), 1.15 px median on 114/115 frames but p90 6.2 px and **6 frames over the
  20 px gate** (max 352 px), PnP-vs-shipped p90 14.9 cm, velocity gate 50/5704 pairs (p99
  4.63 cm): the fpv validity gate matters more on this trial. **No view dropped**; the stop
  rule did not fire.
- **Tests.** `tests/test_finebio_cameras_cli.py`: 8 default-tier (the pose decision incl. the
  table fallback and the drop rule, the day vote, parsers, the three committed configs, camera
  6 across the two P03 trials) + 1 `real_data` (`solve_mapping` on P03_01_01 reproduces
  `rig_reference.json` ids, day, residuals within 0.1 px, PnP distances within 0.05 cm and the
  preflight PnP centres, and `camera_config_from_mapping` equals the committed config outside
  `provenance`). The p0-contracts test that rebuilds the config from the preflight
  `mapping.json` still passes (the table fallback gives T4 the same 4.78 px). Default tier
  887 passed / 14 skipped at commit time, `ruff check` clean.

### Sep 25: battle-finebio-rig and the gate formulas (p1-rig)

- **What this is.** The `p1-rig` todo: the preflight's `rig` subcommand generalised into
  `src/battle/finebio_rig.py` (console script `battle-finebio-rig`), run on any camera config
  and detector pass, with the tracker's gates as **formulas** evaluated per trial and written
  into the output. Regression on the preflight detections; the negative control the Evidence
  preset needs. CPU, 0.7 s on the 78-frame preflight set. Commit `15e5257`.
- **Inputs and the seven checks.** `--config <FineBioCameraConfig.json> --detections <dir>
  [--frames N,A-B,A:B:S] [--output runs/finebio-rig-<trial>-<date>] [--clock-max-frames 600]
  [--negative-control]`; the detections directory holds one `<view>.jsonl` per view with
  `frame_index` (raw) and `detections[{class, score, box_xyxy_px}]`, the shape of
  `runs/preflight-finebio-20260924/detections` and of `battle-finebio-detect`; frames come from
  `--frames`, else `frames.json`, else the frames detected in >= 3 fixed views; "consecutive"
  frames are those with a neighbour. Every triangulation is `Camera.undistort` ->
  `dlt_triangulate` on the projection matrices; reprojection uses the distortion. The checks,
  as in the preflight: (1) static objects, per-view median box centre over the frames (score
  >= 0.5), >= 3-view triangulation, all-view and leave-one-view-out residuals, height above
  the bench; (2) hands as probes, per frame from >= 3 fixed views; (3) the clock scan per view
  and moving class over -15..+15 frames, the other views' per-frame triangulation computed once
  and reprojected into the shifted view, >= 10 frames per offset, with an **informative** flag
  when the best offset beats both +/-3 neighbours by 20% (the preflight's "flat for the plate
  and left hand" made a rule; on long windows the consecutive frames are subsampled evenly to
  `--clock-max-frames`); (4) leave-one-view-out on the moving classes per frame with the
  inside-held-out-box fraction; (5) the fixed -> fpv hand-off of the plate and the hands
  through the shipped fpv pose; (6) the fpv camera centre in every fixed view; (7) the marker
  fits, read from the config. `rig.json` (everything, plus `gates` and `negative_control`) and
  `rig.md`.
- **Gates (`rig.json["gates"]`, the keys lane C's `battle-multiview-tracks --gates` reads).**
  `association_px = clamp(3 * static_loo_median_px, floor, cap)`;
  `handoff_px = clamp(moving_loo_p90_px, floor, cap)` where `moving_loo_p90_px` is the **max
  over views of the p90 of the per-frame held-out residuals of the tracked moving objects**
  (plate and pipette; hands are probes and stay out) reprojected into that view, the
  fixed-view LOO cells and the fixed -> fpv hand-off alike, views with fewer than 10 residuals
  skipped. One gate is applied to every view, so the widest view sets it; the *pooled* p90
  (45.5 px on P03) would refuse about a fifth of the correct hand-offs into the head camera,
  whose residuals are the widest (52 px p90). `floor = 15 px` at 1920: 1.5x the static LOO
  median, which is box-centre parallax on true matches, so a tighter gate would reject them;
  `cap = 80 px`: about 4% of the width, under the spacing of neighbouring same-class bench
  objects (tip racks 80-150 px apart in the fixed views) and inside an fpv plate box
  (250-360 px), so a wider gate would start merging identities; both scale with the image
  width. `clock_offset_frames` per view = the median best offset over the informative scans,
  uncertainty +/-1 frame, with a `significant` flag (|offset| > 1) so an offset inside the
  uncertainty is reported and not applied; `birth_min_fixed_views` 3,
  `birth_fixed_views_with_fpv` 2. **P03_01_01 comes out at association 31.1 px (3 x 10.37)
  and hand-off 51.9 px** (per-view p90 T1 35.6, T2 38.1, T3 45.7, T4 34.1, T5 34.4, fpv 51.9;
  94-100 residuals per fixed view, 45 in the fpv), the preflight's 30 / 50-60; clock offsets
  T1 0 (no informative scan), T2 0, T3 0 (left hand -1, right hand +1), T4 +1, T5 -1, none
  significant: no per-view offset is applied.
- **Regression on the preflight.** On `runs/preflight-finebio-20260924/detections` with the
  committed P03_01_01 config the rig reproduces `rig_reference.json`: the eleven static points
  within 0.05 cm (centrifuge 4.1 cm, trash can 9.7 cm, racks 0.2-0.9 cm: half heights), every
  LOO and all-view residual within 0.5 px (49 cells, median 10.4, p90 25.3), hands 61/61 at
  7.7 px and 3.3 cm / 14/61 at 14.0 px and 26.6 cm, every clock best offset, the moving LOO
  cells (plate inside the held-out box on 98-100%), the plate hand-off 45 frames at 36.5 px
  median / 51.9 p90 inside 100%, left hand 31.6 px. The same run on the committed fixture
  observations (boxes rounded to 0.1 px) reproduces it in the default tier, association
  31.2 px there.
- **Negative control (`--negative-control`).** Every `marker_pnp` view re-evaluated with its
  **shipped** pose, other views unchanged. T5 (camera 6, centres 6.4 cm apart): markers
  **0.71 vs 93.7 px**; static LOO median **6.7 vs 94.3 px** (every object 83-113 px under the
  shipped pose: centrifuge 2.4 vs 97.9, vortex 1.4 vs 94.8, micro-tube rack 1.5 vs 91.7);
  left hand **7.6 vs 47.8 px**, right hand 20.3 vs 67.7; heights move by up to 6 cm (red tip
  rack 5.9 -> 11.7 cm, 8-channel rack 3.6 -> 9.6). These are the side-by-side numbers for the
  Evidence preset's control.
- **For the GPU phase.** The trial windows need the detections `battle-finebio-detect` will
  produce on the proxies; the exact commands are recorded in each clip config's
  `downstream.rig`:
  `uv run battle-finebio-rig --config configs/finebio/cameras/P03_03_01_600-4200.json
  --detections runs/finebio-detect-P03_03_01-600-4200 --frames 600-4199
  --output runs/finebio-rig-P03_03_01-600-4200 --negative-control` and the same with
  `P20_03_01_600-4200` (`--frames 600-4199`). On P20 the negative control covers T1-T4 (all
  four are `marker_pnp`, so each is reported against its shipped pose), and T5, kept
  `shipped` at 7.47 px, is the camera to watch in the static LOO column. On 3600 frames the
  clock scan runs on 600 subsampled frames (~30 s); the rest takes seconds.
- **Tests.** `tests/test_finebio_rig.py`: 4 default-tier (frame helpers; the formulas with
  floor, cap, width scaling, the per-view minimum and the significance flag on synthetic
  inputs; the rig on the committed fixtures reproduces the reference; P03's gates at 30-32 /
  50-60 px with the fpv the widest view) + 1 `real_data` (the preflight detections with the
  shipped fpv poses and the negative control). Default tier green, `ruff check` clean.

### Sep 25: finebio_preprocessing and the trial proxies (p1-configs)

- **What this is.** The `p1-configs` todo: the `finebio_preprocessing` manifest and the
  `battle-finebio-preprocess` step that cuts a trial window into six proxies with the
  p0-contracts recipe unchanged and proves the frame-index contract on each; the smoke window
  and both trial windows cut; the Assembly101 `clip.views == ("static-c10379",)` asserts made
  config-driven. CPU; 26 s for the 600-frame smoke window, about 2 min per 3600-frame trial
  window (six libx264 encodes in parallel). Commit `426bd00`.
- **Manifest (`src/battle/finebio_preprocessing.py`, `FineBioPreprocessingManifest`, kind
  `finebio_preprocessing`; kept in the module, `schemas.py` untouched).** Trial, recording
  day, `window_start_frame` / `window_end_frame_exclusive` / `frame_count`,
  **`frame_index_offset = start`** (raw frame = proxy frame + offset, every view), `fps`
  `30000/1001`; `views[view]` = `FineBioProxyRecord`: raw uri + sha256 + container frame
  count, proxy uri + sha256, width / height, `r_frame_rate` / `avg_frame_rate`, counted
  frames, the `media_probe` fps (and its `.probe.json` sidecar beside the proxy), the
  **`proxy_ffmpeg_args` command verbatim**, encode seconds, the `frame_index_contract` report
  on three sample offsets (first, middle, last frame: proxy frame k vs raw start+k-1 / +k /
  +k+1) and on the fpv the `proxy_marker_check` against the shipped pose; `proxy_recipe`
  (function, resolution, rate, trim, codec, template); `camera_config` and
  `window_camera_config` refs with sha256; `intrinsics_rescale` 0.5 / 0.48;
  `pose_length_check`; `window_source` (trials.json ref + its entry window + the requested
  window); `clip_config_uri`; `checks_passed`; a licence line; `created`; `command`.
- **CLI.** `battle-finebio-preprocess --trial T [--window-from configs/finebio/trials.json |
  --start S --end E] [--output data/derived/finebio/T/S-E] [--camera-config] [--jobs 6]
  [--skip-existing] [--dry-run] [--no-configs]`. Cuts the six proxies
  (`<trial>_<view>_<start>-<end>.mp4`) with `finebio_frames.build_proxy` (which verifies the
  counted frame count), probes them, hashes proxy and raw, runs the contract per view (the
  minimum must sit at offset 0 on every sample, the rate must be the native one, the fpv
  markers under 10 px) and `pose_length_check` on all six videos, then writes the manifest,
  **a per-window copy of the camera config at
  `configs/finebio/cameras/<trial>_<start>-<end>.json` with `frame_index_offset = start`**
  (the choice: one committed file per window, poses unchanged, provenance names the trial
  config and the window) and the clip config
  `configs/clips/finebio_<trial>_<start>-<end>.json` (JSON, matching `configs/clips`):
  `config_kind: finebio_clip_config`, `clip_id`, source and licence lines, trial / role /
  room / day from `trials.json`, `fps`, `window` (frames and seconds), `frame_index_offset`,
  `views` (`T1..T5, fpv`), `fixed_views`, `fpv_view`, `view_sizes`, **`targets`** (the
  tracked classes: plate, the four pipettes, 50 ml / 15 ml / micro tubes, tube strips, the
  four tip racks), `containers` (centrifuge, vortex, PCR machine, the racks, the trash can),
  `probes` (hands), `camera_config`, `window_camera_config`, `preprocessing_manifest`,
  `proxies` and `proxy_sha256` per view, annotated frames and centrifuge cycles inside the
  window, `trials_source` (uri + sha256), and `downstream` with the detections directory the
  GPU phase should write and the exact `battle-finebio-rig` command. A failed check exits 1
  and writes no clip config.
- **Proxies cut (`data/derived/finebio/`, gitignored).** *Smoke* `P03_01_01/1798-2398`
  (600 frames per view, 13-19 s each, 73 MB): contract at offset 0 = 0.60 / 0.59 / 0.77 (T1),
  0.64 / 0.68 / 0.86 (T2), 0.63 / 0.69 / 0.84 (T3), 0.66 / 0.72 / 0.87 (T4), 0.61 / 0.69 /
  0.79 (T5), 0.70 / 0.79 / 0.81 (fpv) grey levels with the minimum at 0 on every sample (the
  fixed cameras hardly move, so a one-frame offset is only 1.1-1.3x worse there; on the fpv
  sample at 2098 it is 22.5 vs 0.79); fpv markers on 3/3 samples at 1.03 px. *Trial 1*
  `P03_03_01/600-4200` (3600 frames per view, 68-97 s each, 489 MB): at 0 = 0.63-1.01 on the
  fixed views and 0.85 / 0.89 / 1.10 on the fpv, off-zero minimum 15.2 on the fpv; markers
  0.76 px; pose length 8492 == every raw frame count. *Trial 2* `P20_03_01/600-4200` (3600
  frames, 69-98 s, 459 MB; its camera solve dropped nothing): at 0 = 0.54-0.82 fixed, 0.75 /
  0.75 / 1.01 fpv, off-zero minimum 1.84; markers 0.97 px on day 221124; pose length 6045 ==
  raw, 94.6% valid. Six manifests' worth of `checks_passed: true`. The smoke window is the
  user's 1798..2398 (600 frames), wider than `trials.json`'s 60-frame preflight window; the
  manifest records both.
- **Committed configs.** `configs/finebio/cameras/P03_01_01_1798-2398.json`,
  `P03_03_01_600-4200.json`, `P20_03_01_600-4200.json` (offset 1798 / 600 / 600) and
  `configs/clips/finebio_P03_01_01_1798-2398.json`, `finebio_P03_03_01_600-4200.json`,
  `finebio_P20_03_01_600-4200.json` (`trials_source` sha256 `63ef49e1…`, the committed
  `trials.json`).
- **Config-driven asserts.** `four_part_contract.py` gains `DEFAULT_TARGETS`,
  `DEFAULT_APPROVED_VIEWS = ("static-c10379",)`, a `ReviewScope(views, targets, source)` and
  `review_scope(clip_config)` that reads `clip.views` from an Assembly101 preprocessing
  manifest (targets stay the four parts) or `views` + `targets` from a FineBio clip config,
  and returns the Assembly101 defaults when no config is given. `interaction_review._validate_run`
  takes `approved_views`, `build_interaction_review` takes `clip_config` (CLI `--clip-config`)
  and checks the reference's target order against the scope; `exploratory_comparison._load_method`
  / `build_exploratory_comparison` likewise, with `view_id = approved_views[0]`;
  `kineo_fusion.verify_source_alignment(approved_views=...)` reads the scope from its existing
  `--config` clip config. The literal `"static-c10379"` remains only in Kineo's pkl payload
  field and as the default; the per-part loops inside `interaction_review` keep the four-part
  `TARGETS` default (the plan asked for the asserts, not a refactor). Every existing test passes
  unchanged.
- **Tests.** `tests/test_finebio_preprocessing.py`: 5 default-tier (manifest round trip with
  the recipe verbatim and the kind rejected when wrong; `checks_pass` on a shifted proxy, a bad
  marker fit and a wrong rate; the window helpers on `trials.json`; the clip config contents
  and its `downstream.rig` command; the committed smoke window config and clip config agree)
  + 1 `real_data` (the 60-frame smoke window built into `tmp_path` passes the contract in six
  views, offsets 0 / 30 / 59, fpv markers under 10 px). `tests/test_review_scope.py`: 5
  (defaults, the Assembly101 configs incl. the eight-view one, the FineBio config, rejections,
  `verify_source_alignment` with and without a scope). **Default tier 913 passed / 14 skipped**
  (other lanes' tests included), `uv run pytest -q -m real_data -k finebio` **11 passed**
  (2 cameras, 1 cameras CLI, 5 frames, 1 rig, 1 preprocessing, 1 detector), `uv run ruff check
  src tests scripts` and `ruff format --check` clean.
- **Left for the GPU phase.** `battle-finebio-detect` on the six proxies of each trial window
  (raw `frame_index = proxy + offset`, per the clip config), then the `downstream.rig` command
  above; lane C's tracker takes the resulting `rig.json` through `--gates`.

### Sep 25: battle-detector-seed, observation adapters, gate-1 sheets (p2-seeds, p2-gate1)

- **What this is.** The `p2-seeds` todo of the FineBio 3D-tracking plan
  ([`docs/plan-2026-09-25-finebio-3d-tracking.md`](plan-2026-09-25-finebio-3d-tracking.md)) and
  the sheets half of `p2-gate1`: which objects get a SAM3 slot in which view, where each slot
  starts, whether the SAM3 image decoder accepts the detector box as its seed, the worker-ready
  files for the arms, and the contact sheets the human decides on
  ([`docs/labeling-sessions-2026-09-25-finebio.md`](labeling-sessions-2026-09-25-finebio.md)).
  Plus the two adapters the tracker needed (detector JSONL and SAM3 worker run ->
  `FineBioObservation` rows). Developed on the preflight detections of `P03_01_01` (78 frames)
  and the committed fixtures, then run on trial 1 `P03_03_01` (raw frames [600, 4200), the DINO
  pass `runs/finebio-detect-P03_03_01-600-4200-20260925/dino`, `state: succeeded`, coverage
  `full`, 47 ms/frame, landed at 23:45). GPU: two seed decodes of 11 and 13 s, peak 2.41 GiB,
  beside the detector lane's P20 worker (784-1170 MiB, named with `--allow-gpu-neighbour`; the
  guard's record is in `seeds.json`), no viewer. Five agents committed concurrently in one
  checkout; only the files named here were staged, by path. Commits: `7124ea1` (adapters),
  `a07b19a` (the tool), `6e310fd` (identity over the window, the trial-1 slot set), `97e25fb`
  (the brief), and this entry's commit. Outputs under `runs/finebio-seeds-P03_03_01-20260925/`
  (gitignored; FineBio licence: sheets, masks and instance files never committed).
- **Adapters (`src/battle/finebio_observations.py`, commit `7124ea1`).**
  `detections_to_observations(detections_dir, views, min_score, pose_valid=, frames=)`:
  `battle-finebio-detect` JSONL (raw `frame_index`, `detections[{class, score,
  box_xyxy_px}]`, `interpolated` + `source_frames` on filled rows) -> detector rows with
  `slot = <class>#<rank>` (same-class score rank per frame), boxes rounded to 0.1 px and
  scores to 4 decimals, the interpolation flag kept in `provenance`; on the preflight
  detections it reproduces the **15,695 committed fixture rows exactly**. The fpv needs a pose
  validity lookup (`fpv_pose_validity(trial)` from the shipped pose file,
  `pose_validity_from_fixture(fpv_poses.json)`; `fill_pose_valid` applies it).
  `worker_to_observations(run_dir, view, start_frame, slot_labels=, pose_valid=,
  detector_rows=, match_iou=0.3)`: the SAM3 worker's `observations.jsonl` + mask PNGs -> SAM3
  rows in raw frames (`start_frame + k`) with the **area centroid** (pixel centres, +0.5; the
  fixtures carry the bbox centre), mask bbox `[xmin, ymin, xmax+1, ymax+1]`, area,
  `sam3_object_score` = the row's `object_score` (the decoder's IoU in box-decode, the tracker
  logit in video memory), `source` `sam3_decode` from the row's own `source` else `sam3_video`,
  the slot label from the multiplex index (`slot_labels`) or the worker label when it carries
  `#`, the prompt box + `prompt_score` as the detector box on decode rows, the best same-class
  detector box by IoU >= 0.3 on video rows (`detector_box_iou`, `detector_slot` in
  provenance), a normalised-box fallback with `mask: absent` when a PNG is missing. On the Sep
  24 smoke runs: 80 box-decode rows and 75 video rows, centroids inside their bboxes, the plate's
  `detector_box_iou` 0.944 at 1798. **For the arms**: `slot_labels` are the seed tool's labels
  (the worker's concepts are `<class>#<k>` already, so the labels pass through), pass the
  detector rows of the same view so video rows carry the detector box the tracker confirms
  against, `write_observations` sorts and writes compactly.
- **Tool (`src/battle/detector_seed.py`, console script `battle-detector-seed`; commits
  `a07b19a`, `6e310fd`).** Subcommands `instances`, `select`, `decode` (spawns
  `decode-worker` under the MuggledSAM interpreter), `sheets`, `apply-decisions`, `run`. One
  run directory; each step reads the previous one's files and the parameters it recorded.
- **`instances`: per-view identity over the window.** Same-class boxes at score >= 0.2 on
  consecutive detected frames associated by IoU >= 0.3 (greedy one-to-one, numpy IoU matrix,
  a tracklet survives 30 unmatched frames); hands tracked as probes. The first run on the
  3600-frame window showed why that is not enough: **1,400-11,400 tracklets per view, 13
  centrifuge and 26 PCR-machine instances in T1** (the lid changes the centrifuge box past IoU
  0.3, the operator's body hides the PCR machine for seconds, the head camera looks away for
  minutes), and the slot cap filled with fragments of two objects. Three joins follow: (1)
  `merge_fragments`, same-class fragments across gaps up to 300 frames whose last and first
  boxes overlap at IoU >= 0.3 (the earlier box carried by the scene's motion in the fpv), best
  overlap then smallest gap; (2) `merge_singletons`, every instance of a class the bench holds
  once is one object (centrifuge, vortex_mixer, pcr_machine, magnetic_rack, trash_can,
  cell_culture_plate and its lid, blue / yellow / red / 8_channel_pipette; **a bench assumption
  on the record**, `--singleton-classes`); (3) `merge_footprints` in the fixed views,
  same-class container instances whose median boxes overlap (IoU >= 0.3 or containment >=
  0.5) are one rack seen with and without its tubes (identical racks side by side have disjoint
  footprints and stay apart). Trial 1: T1 1433 fragments -> 468 instances (731 gap joins, 155
  singleton, 79 footprint), T2 1464 -> 587, T3 1527 -> 633, T4 1982 -> 804, T5 1461 -> 606,
  fpv 11443 -> 2893 (8058 gap joins, 492 singleton; no footprint merge under head motion).
  16 s for six views; `instances/<view>.jsonl` + `summary.json` with per-class counts and
  lifetimes. A known limit: a long-gap join at the same place can chain two different tubes
  that used the same rack hole in turn; the 3D tracker owns identity, these are per-view SAM3
  slots.
- **`select`: the plan's seed rule, and what the data forced.** A slot opens for a persistent
  instance (>= 15 frames detected at score >= 0.3; start frame = the first detected frame of a
  dense run) that **moves** or **sits in a hand**; the named containers open once as static
  volumes; groups; everything else `detector_only` with its reason. *Moves*: the first
  version (smoothed centre displacement over a 30-frame window) fired on every rack a hand
  passed over (a box shrinking under an occluder moves its centre); the rule is now
  **sustained**: the median box centre over `[t-30, t)` differs from the median over
  `[t, t+30)` by > 20 px at 1920, evaluated at every 5th frame of a grid shared by all
  instances, boxes cut by the frame border excluded. *The head camera* moves everything: in the
  fpv the displacement is measured against the **scene's own motion**, an affine map fitted
  by least squares to the other non-hand, border-free instance centres shared by the two
  frames (trimmed 25%, one 5 px inlier pass; composed from per-step fits when fewer than six
  are shared; a translation model left 40-320 px residuals on static bench objects in the
  preflight window, the affine leaves 2-15 px), with a 40 px threshold, **and only for a class
  some fixed camera saw move or be held** (fixed views are selected first; `uncorroborated`
  instances are recorded: 68 in trial 1's fpv, among them the plate at 118 px and the magnetic
  rack at 330 px of pure head motion). Fixed cameras get the identity map: a 6-instance affine
  fit on a fixed view was absorbing the movers themselves. *In hand*: centre inside a
  `left_hand` / `right_hand` box at score >= 0.3 on > 10% of detected frames. *Containers*:
  centrifuge, vortex_mixer, pcr_machine, magnetic_rack and the eight rack classes,
  `role: container`; a container that moved keeps `rule: moves` on the record. *Groups*:
  identical-instance classes (tubes, strips, tips, spin columns) with >= 3 members whose
  centres sit in one rack box on >= 50% of their frames become one `<class>_group#<k>` slot
  per rack (union box per frame, `members` listed) unless a member moves, which is its own
  slot. *Suppression*: nested same-class boxes (the plate and the plate-with-lid box,
  containment >= 0.7 on >= 50% of shared frames) and the same box under two class names (IoU
  >= 0.8; the held pipette read as blue and as yellow) are dropped and recorded; tips never
  open a slot (`excluded_classes`, no attachment observable). *Cap* 10 per view, ordered:
  the landmark containers (centrifuge, vortex_mixer, pcr_machine, whatever rule opened them),
  the objects that moved (by persistence, then movement), the tube groups, the objects only
  held, the racks (in use first); containers never compete with objects for the dynamic
  slots; `capped` slots are recorded with their numbers. Labels `<class>#<k>`, slots 0..N-1,
  seed candidates = the start frame and every 15th detected frame after it (3 attempts).
  20 s for six views on trial 1.
- **`decode`: the SAM3 image decoder on the seed frames (GPU).** `decode/requests.json` (per
  slot the attempts in order: for every candidate frame the tight box, then the same box with
  a 0.15 margin; other slots' centres as negatives with `--other-instances-as-negatives`, off
  by default as in arm (b)) -> `decode-worker` under
  `/home/nick/.pyenv/versions/muggled_sam/bin/python` with the same guard as the SAM3 worker
  (`vram`, profile `sam3_1280`, `--allow-gpu-neighbour`, `CUDA_VISIBLE_DEVICES=0`,
  `PYTHONPATH=/home/nick/src/muggled_sam`; the file imports only `fs_common`, `gpu_guard` and
  `finebio_detect.box_iou` by path there): `sam3.1_multiplex.pt`, bfloat16, one
  `encode_image(frame, 1280, square)` per distinct seed frame per view, per prompt
  `encode_prompts([box], [], negatives)` -> `generate_masks` -> the top-IoU candidate,
  logits > 0 at source size; **accepted by mask-bbox IoU vs the detector box >= 0.6** (fill
  ratio reported, not tested: the plate fills its box to 0.5 when correct); a rejected tight
  box is retried with the margin, then on the next candidate frames; else `unseeded` with the
  reason. Per attempt: the four candidate IoUs, the chosen index, the decoder IoU, mask area,
  bbox, IoU, fill; the accepted mask as a PNG under `decode/masks/<view>/`. Preflight
  (P03_01_01, six views): 60 slots, **60/60 accepted** (mask-bbox IoU 0.71-0.99; the T1 plate
  0.76 against the preflight's 0.77), 13 image encodes, 60 decodes, 11 s wall including the
  4 s model load, 2.41 GiB. Trial 1: 60 slots, **60/60 accepted at 0.63-0.99**, 24 image
  encodes, 62 decodes (two slots needed the margin box or a second frame:
  `50ml_tube_group#1` in T1 seeded at 1158 rather than 1142), 13 s, 2.41 GiB (guard headroom
  13,567 MiB against 3,995 required with the detector's 990 MiB reserved beside it).
- **Files for the arms (written by `decode`, validated with the worker's own parsers
  `parse_box_stream`, `_corrections_by_frame`, `slot_start_frames`).** `seeds.json`
  (`selected_by: detector`, `provenance: auto`, per slot label / class / role / rule /
  instance / members / start frame / persistence / movement / hand fraction / candidates /
  attempts / accepted seed / `schedule_slot`, the decode record with the guard, the parameters,
  the claim boundary) + `seeds.md`. **Mode (i)** `box_streams/<view>.jsonl`: per analysis frame
  (`frame_index = raw - 600`) the boxes of every selected slot from its tracklet from its start
  frame on (`{"slot", "label", "box_xyxy_px", "score", "source": finebio_dino |
  finebio_dino_interpolated | finebio_dino_group}`; a group's box is the union of its members'),
  3600 frames with boxes per fixed view (fpv 3561), 23,000-31,000 boxes per view;
  `battle-muggled-arms box-decode --video <raw mp4> --view-id <view> --box-stream
  box_streams/<view>.jsonl --start-frame 600 --max-frames 3600`. **Mode (ii)**
  `schedules/<view>.json`: the accepted seeds as `{"target": <label>, "initial_multiplex_slot":
  0..N-1, "prompt_box_xyxy_px": <seed box>, "start_frame": <seed frame - 600 when > 0>,
  "selected_by": "detector" at frame 0 | "detector_reseed" later}` (the worker's vocabulary
  for a mid-stream box prompt; the plan's `selected_by: detector` lives in `seeds.json`),
  `corrections: []`; 43 of the 60 trial-1 slots start at analysis frame 0, the rest where the
  object first persists (T2 `50ml_tube#0` at 1443, T5's held `micro_tube#0` at 2217, ...);
  `battle-muggled-arms video-memory --video <raw mp4> --view-id <view> --schedule
  schedules/<view>.json --start-frame 600 --max-frames 3600 --prompt-memory-semantics append`.
  The worker reads a raw video with `--start-frame 600` or a lane-B proxy with `--start-frame
  0`; the box stream and schedule are in analysis frames either way.
- **Trial-1 slot set (`seeds.md`; the brief has the per-view table).** Every view: the three
  machines (`centrifuge` `moves` 53-161 px in every view because its lid opens; the vortex
  25 px in T2, 74 in the fpv), the pipettes in use (`blue_pipette` moves 400-1000 px, yellow
  131-650, red 75-142, 8-channel 60-143), the 15 / 50 ml tubes that move (20-410 px), micro-tube
  groups of 29 (T1), 38 (T3), 54 (T5) tubes plus 15 ml and 50 ml groups of 3-5, one held micro
  tube in T5 (frames 2817-3004, hand 100%), `trash_can` in T4 and the fpv (20 / 60 px).
  Rules per view: T1 4 moves / 2 container / 4 group; T2 9 moves / 1 group (the PCR machine is
  never detected in T2); T3 7 / 2 / 1; T4 8 moves / 1 in_hand / 1 container; T5 5 / 1 / 2 / 2;
  fpv 9 moves / 1 container. **`detector_only`**: the plate in every fixed view (static in
  protocol 03: `cell_culture_plate` moves nowhere), pens, 8-tube strips, most micro tubes that
  are not in the racks' footprint, the 8-tube-strip rack lid; per view 12-27 instances, 111 in
  the fpv. **Capped** (would be slots beyond 10): the racks (`micro_tube_rack` and
  `50ml_tube_rack` moved 25-107 px in T1/T3; the tip racks 25-40 px in T1/T5), the
  `magnetic_rack`, further held micro tubes, further 50 ml tube groups, and in the fpv 310
  instances, mostly tube fragments. Suppressed 4-19 duplicates per view. Runtime: 16 + 20 +
  13 + 2 s for the four steps.
- **`sheets` and the gate (`p2-gate1`).** `sheets/<view>.jpg`: one tile per accepted seed
  (crop 2.5x the box, >= 320 px, the mask filled in the class family's colour with its
  contour, the box, `<label> slot <n>` / `<rule> / <role> f<frame>` / `det dec bbox IoU
  kind`), `<view>_unseeded.jpg` when a slot failed; `decisions.template.json` (one entry per
  slot, `decision: accept | reject | null`, `note`); `apply-decisions --decisions <file>`
  removes the rejected slots, renumbers the rest and writes `filtered/box_streams`,
  `filtered/schedules`, `filtered/seeds.json` (`provenance: human_filtered` with the
  decisions' SHA-256; per slot `human_accepted` or `auto`). The brief asks for 15 minutes on
  60 tiles: accept when the mask is on the labelled object, reject the wrong object, a mask on
  the glove or bench, an occluded seed frame, or two slots on one object; the default when
  nothing is done is every seed kept with `provenance: auto`. Looked at by the agent (not a
  human): the T5 sheet shows the four pipettes, the three machines, the micro-tube group as
  the rack with its tubes, the 50 ml group and the held tube each on the right object; the T3
  preflight sheet shows the held pipette twice under two colour names, which is what the
  suppression now removes when the boxes coincide and what the human removes when they do
  not.
- **Tests.** `tests/test_finebio_observations.py`: 8 default (ranks and interpolation, the fpv
  pose requirement and the frame filter, pixel-centre measurements, box-decode rows with the
  prompt box, video rows matched to detector boxes, the missing-mask fallback, the fixture
  pose fill, the ordered writer) + 2 `real_data` (preflight detections == fixture detector
  rows; the smoke runs convert). `tests/test_detector_seed.py`: 13 default (IoU matrix and
  greedy pairs; association across a gap and by class; the dense start frame and seed
  candidates; the rules on synthetic tracklets: moves / in_hand / container / group with a
  member that walks out / detector_only / the cap order; the start frame from the score
  history; scene motion holding 12 static objects under a 3 px/frame drift + zoom at < 2 px
  while the mover keeps 60-100 px and border-clipped boxes are excluded; fpv corroboration;
  the cap; nested and same-box suppression; box stream and schedule through the worker's
  parsers; decode requests, margins and mask metrics; the CLI end to end on synthetic
  detections with the decode blocked without the interpreter and `apply-decisions`; the
  worker command and environment) + 1 `real_data` (the preflight: the plate opens in T1-T4
  and the held pipette in T1/T2/T3/T5 by `moves` / `in_hand`, the PCR machine is a container
  in every fixed view and never a moving object, the fpv's plate is corroborated). Default tier
  **934 passed / 14 skipped** (899 before this lane); `uv run ruff check src tests scripts`
  and `ruff format --check` clean. No `gpu`-marked test: the two decodes are recorded here.
- **What the arms phase must know.** (1) Arm (b) takes `box_streams/<view>.jsonl` as is
  (all selected slots, accepted or not; the worker decodes every frame anyway); arms (c)/(d)
  take `schedules/<view>.json` (accepted seeds only; a slot that starts late is a seed with
  `start_frame`). (2) Prefer `filtered/` when the human has run `apply-decisions`. (3) The
  slot labels are the worker concepts, so `worker_to_observations(run, view, 600)` gives the
  tracker's rows with no label map; pass the same view's detector rows for confirmation. (4)
  The per-view tracklets are detector identity: a tube that leaves the window and returns is
  the same slot only if its box overlapped within 300 frames; the 3D tracker resolves the
  rest. (5) The plate is not a slot in trial 1 because it does not move there; if the demo
  needs it, `--container-classes` with the plate added opens it as a static volume. (6) Trial
  2 runs the same command on `runs/finebio-detect-P20_03_01-600-4200-20260925/dino` once it
  lands; nothing in the parameters is P03-specific except the bench's singleton list, which
  holds for room 2 as well (one machine of each kind, one pipette of each colour).

### Sep 25: detector runs on the trial windows (p2-detect, runs)

- **What this is.** The runs half of `p2-detect` in the FineBio 3D-tracking plan
  ([`docs/plan-2026-09-25-finebio-3d-tracking.md`](plan-2026-09-25-finebio-3d-tracking.md)):
  FineBio's shipped detectors on **every frame of both trial windows in all six views**, with
  the tooling of the previous night (`battle-finebio-detect`, the CUDA venv
  `/home/nick/src/finebio-detector/.venv-cuda`; entry "Sep 24, night: FineBio detector, CUDA
  build and battle-finebio-detect"). Windows from `configs/finebio/trials.json`: trial 1
  `P03_03_01` and trial 2 `P20_03_01`, both raw frames **[600, 4200)** (3600 frames,
  20.02-140.14 s). Four runs in order, DINO then Deformable DETR per trial, each a background
  process watched to completion, each preceded by `nvidia-smi --query-compute-apps`: the card
  held only `kwin_wayland` (144-145 MiB) at every start, so no `--allow-gpu-neighbour` was
  passed and the guard (`vram` mode, `unknown` profile, 6 GiB required, 14,954-14,992 MiB
  available) accepted every time; the seeds lane's two SAM3 decodes ran *beside* the trial-2
  DINO worker with that worker named as their neighbour (their record), nothing was killed, no
  viewer opened, `finebio_detect.py` untouched. Outputs (gitignored, FineBio licence; no frame,
  mask or video anywhere): `runs/finebio-detect-P03_03_01-600-4200-20260925/{dino,ddetr}/` and
  `runs/finebio-detect-P20_03_01-600-4200-20260925/{dino,ddetr}/`, each with `<view>.jsonl`
  x 6, `frames.json`, `worker_command.json`, `worker.log`, `worker_result.json`,
  `manifest.json`, `analysis.json` (this entry's numbers), plus a `README.md` per trial
  directory with the same tables and the driver logs; 451 / 333 / 422 / 308 MB. Relative
  symlinks `runs/finebio-detect-P03_03_01-600-4200 -> …-20260925/dino` and
  `runs/finebio-detect-P20_03_01-600-4200 -> …-20260925/dino` make the `downstream.detections`
  path and the `battle-finebio-rig --detections` command in the committed clip configs
  (`configs/clips/finebio_<trial>_600-4200.json`) resolve as written. The analysis was one
  script under `/tmp` (verification, per-frame counts, class presence, IoU agreement, box-centre
  displacement, hand-box containment), not committed. Claim boundary: the detector was trained
  on FineBio's own objects and cameras; every number below is model output at the detector's own
  score, nothing is scored against the FineBio annotations (treated as unavailable), and the
  move / in-hand tables are heuristics over those boxes.
- **The four runs.** All `state: succeeded`, `detection_coverage: full`, TF32 off (the driver's
  default), record threshold 0.05, `torch 2.13.0+cu130`, `mmcv 2.1.0`, `mmdet 3.3.0` at
  `44ebd17b`, RTX 5070 Ti.

  | run | elapsed (21,600 frames, incl. load + decode) | inference ms/frame mean (min / median / max) | per view fpv, T1..T5 | peak reserved / allocated | created (UTC) |
  |---|---|---|---|---|---|
  | trial 1 DINO | **1065.5 s (17.8 min)** | **47.3** (40 / 47 / 632 warm-up) | 45.8, 49.3, 47.2, 47.0, 47.2, 47.3 | **924,844,032 B (882 MiB)** / 517 MiB | 03:45:14 |
  | trial 1 DDETR | **963.4 s (16.1 min)** | **42.8** (32 / 39 / 541) | 47.5, 52.5, 39.9, 39.3, 38.9, 38.7 | **736,100,352 B (702 MiB)** / 418 MiB | 04:01:30 |
  | trial 2 DINO | **1028.5 s (17.1 min)** | **46.0** (38 / 47 / 326) | 39.6, 47.0, 47.8, 47.0, 47.2, 47.0 | 924,844,032 B (882 MiB) / 517 MiB | 04:19:05 |
  | trial 2 DDETR | **854.3 s (14.2 min)** | **38.1** (32 / 38 / 311) | 32.9, 38.8, 39.0, 39.8, 39.1, 38.7 | 736,100,352 B (702 MiB) / 418 MiB | 04:33:33 |

  Total GPU wall clock 65 min for 86,400 detected frames; the peak-memory figures are the
  tooling night's to the byte. Trial 1's DDETR fpv and T1 means (47.5, 52.5) are host
  contention, not the card: for about two minutes another lane ran `ffmpeg` on ~20 cores plus
  several `ffprobe` and a `pytest` (load average 94 on 32 cores) while `nvidia-smi` showed the
  detector alone at 60% utilisation, SM clock 2880 MHz, no throttle reason; the timer wraps
  `inference_detector`, whose CPU-side resize/normalise slows under contention. Uncontended
  DDETR is 33-40 ms, DINO 40-49. The fpv is the faster view for both models although it is the
  larger frame: mmdet's `Resize (1333, 800) keep_ratio` turns the 4:3 fpv into 1067x800 and the
  16:9 fixed frames into 1333x750 (fewer pixels through the backbone). DDETR's `max_per_img` is
  100, hence its flat 99-100 detections per frame at 0.05.
- **Verification, every run.** Each view's JSONL has exactly **3600 rows, `frame_index` 600..4199**
  in order with no duplicates; `interpolated` is `false` on every row and `source_frames`
  occurs nowhere; `image_hw` [1440, 1920] for the fpv and [1080, 1920] for T1..T5; `frames.json`
  lists 3600 frames per view; the manifests record the window `{start 600, count 3600,
  end_exclusive 4200}`, stride 1 in both view classes, the raw videos' sha256 (8492 frames each
  in P03, 6045 in P20) and the checkpoint sha256s (`dino.pth` `e6399531…`, 579,232,009 B;
  `deformable-detr.pth` `35982a45…`, 515,194,905 B; the SOURCES entries). Detections per frame,
  median at score >= 0.05 / 0.3 / 0.5 (fpv, T1..T5): trial 1 DINO 112/39/36, 129/23/18,
  166/29/21, 103/31/28, 151/38/31, 120/37/28; trial 1 DDETR 99/39/37, 100/26/22, 100/31/27,
  99/31/28, 100/41/34, 100/35/31; trial 2 DINO 101/37/33, 136/23/17, 117/21/14, 142/29/21,
  129/32/24, 105/35/28; trial 2 DDETR 80/37/34, 100/21/16, 100/19/16, 100/27/23, 100/27/24,
  73/36/33. At 0.3 the two models agree on the count per view within 1-4 boxes.
- **Frame presence, trial 1 (`P03_03_01`), DINO, fraction of the 3600 frames with at least one
  box of the class; DDETR in brackets where it differs by >= 0.10.** At score >= 0.3:

  | class | fpv | T1 | T2 | T3 | T4 | T5 |
  |---|---|---|---|---|---|---|
  | cell_culture_plate | 0.99 | 1.00 | 0.98 | 1.00 | 0.96 | 1.00 |
  | blue_pipette | 0.59 | 0.37 (0.17) | 0.93 (0.58) | 0.59 | 0.49 (0.88) | 0.57 |
  | yellow_pipette | 0.90 | 0.00 | 0.69 (0.96) | 0.92 (0.81) | 0.97 (0.78) | 1.00 |
  | red_pipette | 0.79 | 0.00 | 0.01 | 0.01 | 0.92 | 0.98 |
  | 8_channel_pipette | 0.61 | 0.43 (0.85) | 1.00 | 0.75 (0.86) | 1.00 | 0.97 |
  | micro_tube | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 |
  | 50ml_tube | 0.87 | 0.88 | 1.00 | 0.95 | 0.99 | 0.99 |
  | 15ml_tube | 0.89 | 1.00 (0.55) | 0.88 | 1.00 | 0.95 | 0.00 |
  | blue_tip_rack | 0.78 | 0.85 | 1.00 | 0.69 (0.93) | 0.96 | 0.98 |
  | yellow_tip_rack | 0.78 | 0.82 (0.97) | 1.00 | 0.94 | 0.98 | 0.99 |
  | red_tip_rack | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
  | 8_channel_tip_rack | 0.11 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
  | centrifuge | 0.93 | 0.99 (0.78) | 1.00 | 1.00 | 1.00 | 1.00 |
  | vortex_mixer | 0.92 | 1.00 | 0.99 | 0.99 | 0.96 | 1.00 |
  | pcr_machine | 0.86 | 1.00 | 0.00 | 1.00 | 0.97 | 1.00 |
  | left_hand | 0.96 | 0.61 | 0.20 (0.44) | 0.94 (0.81) | 0.94 | 0.94 |
  | right_hand | 0.72 | 0.49 | 0.58 | 0.81 (0.93) | 0.93 (0.63) | 0.93 |

  At score >= 0.5:

  | class | fpv | T1 | T2 | T3 | T4 | T5 |
  |---|---|---|---|---|---|---|
  | cell_culture_plate | 0.99 | 1.00 | 0.97 | 1.00 | 0.92 | 1.00 |
  | blue_pipette | 0.23 (0.37) | 0.11 | 0.12 | 0.20 (0.30) | 0.27 (0.40) | 0.20 (0.42) |
  | yellow_pipette | 0.84 | 0.00 | 0.03 (0.85) | 0.77 (0.58) | 0.94 (0.65) | 0.99 |
  | red_pipette | 0.66 (0.80) | 0.00 | 0.00 | 0.00 | 0.79 | 0.94 |
  | 8_channel_pipette | 0.56 | 0.08 (0.52) | 0.72 | 0.68 | 0.99 | 0.79 (0.93) |
  | micro_tube | 1.00 | 0.82 (0.99) | 0.99 | 1.00 | 1.00 | 1.00 |
  | 50ml_tube | 0.84 | 0.83 | 1.00 | 0.94 | 0.99 | 0.82 |
  | 15ml_tube | 0.87 | 1.00 (0.39) | 0.87 (0.71) | 1.00 | 0.94 (0.82) | 0.00 |
  | blue_tip_rack | 0.76 | 0.78 | 1.00 | 0.60 (0.89) | 0.93 | 0.90 |
  | yellow_tip_rack | 0.74 | 0.13 (0.92) | 1.00 | 0.93 | 0.95 | 0.97 |
  | red_tip_rack | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
  | 8_channel_tip_rack | 0.05 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
  | centrifuge | 0.91 | 0.70 | 0.86 (0.99) | 0.96 | 0.99 | 1.00 |
  | vortex_mixer | 0.89 | 1.00 | 0.97 | 0.97 | 0.93 | 1.00 |
  | pcr_machine | 0.85 | 1.00 | 0.00 | 1.00 | 0.46 (0.98) | 1.00 |
  | left_hand | 0.91 | 0.49 | 0.05 (0.24) | 0.80 | 0.72 (0.94) | 0.90 |
  | right_hand | 0.66 | 0.25 | 0.33 | 0.55 (0.89) | 0.64 (0.48) | 0.87 |

  Reading: the plate, the tubes, the machines and the two tip racks in use are on the bench in
  every fixed view on 88-100% of frames at 0.3 (the plate 0.92-1.00 even at 0.5; the
  preflight's transparent-plate worry does not show up as missed frames); the moving
  `blue_pipette` is the weakest class (0.37-0.93 at 0.3, **0.11-0.27 at 0.5** in the fixed
  views), the preflight's "object in the hand is the weakest detection" on 3600 frames; the
  red pipette and the 15 ml tube are out of some cameras' view entirely (T1/T2/T3 and T5);
  `red_tip_rack` is not on this bench; the `8_channel_tip_rack` is a T4-only detection (100%
  there, 0-11% elsewhere), so either a rack only T4 can see or a T4-specific confusion; the
  PCR machine is outside T2's frame. Hands: 0.49-0.96 at 0.3 per view, T2 sees the left hand
  least (0.20).
- **Frame presence, trial 2 (`P20_03_01`), DINO (DDETR in brackets where |delta| >= 0.10).** At
  score >= 0.3:

  | class | fpv | T1 | T2 | T3 | T4 | T5 |
  |---|---|---|---|---|---|---|
  | cell_culture_plate | 1.00 | 1.00 | **0.00 (0.90)** | 1.00 | **0.56 (0.89)** | 1.00 |
  | blue_pipette | 0.74 (0.87) | 0.30 (0.04) | 0.90 (0.49) | 0.63 (0.49) | 1.00 | 0.92 |
  | yellow_pipette | 0.87 | 0.09 | 0.48 | 0.37 | 0.47 | 0.87 |
  | red_pipette | 0.71 | 0.00 | 0.00 | 0.02 (0.15) | 1.00 | 0.92 |
  | 8_channel_pipette | 0.60 | 0.69 | 1.00 | 0.63 | 1.00 | 1.00 |
  | micro_tube | 1.00 | 0.96 | 0.97 | 0.98 | 0.95 | 1.00 |
  | 50ml_tube | 0.97 | 0.97 | 1.00 | 0.96 | 1.00 | 1.00 |
  | 15ml_tube | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
  | blue_tip_rack | 0.84 | 0.75 (0.15) | 1.00 | 0.90 | 1.00 | 0.97 |
  | yellow_tip_rack | 0.94 | 0.72 | 1.00 | 0.75 | 1.00 | 1.00 |
  | red_tip_rack | 0.54 | 0.99 (0.77) | 1.00 | 0.07 | 1.00 | 1.00 |
  | 8_channel_tip_rack | 0.74 | 1.00 (0.77) | 1.00 | 0.91 (0.80) | 1.00 | 1.00 |
  | centrifuge | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 |
  | vortex_mixer | 0.97 | 0.13 (0.84) | 0.00 (0.28) | 0.96 | 1.00 | 1.00 |
  | pcr_machine | 1.00 | 1.00 | 0.11 (0.95) | 1.00 | 0.82 | 1.00 |
  | left_hand | 0.99 | 0.78 (0.56) | 0.09 | 0.51 | 0.79 (0.92) | 0.97 |
  | right_hand | 1.00 | 0.43 (0.97) | 0.39 (0.51) | 0.91 | 0.71 (0.17) | 0.94 |

  At score >= 0.5:

  | class | fpv | T1 | T2 | T3 | T4 | T5 |
  |---|---|---|---|---|---|---|
  | cell_culture_plate | 1.00 | 1.00 | 0.00 (0.89) | 1.00 | 0.09 (0.88) | 1.00 |
  | blue_pipette | 0.59 (0.82) | 0.03 | 0.00 | 0.09 | 1.00 | 0.88 |
  | yellow_pipette | 0.55 (0.65) | 0.00 | 0.12 (0.23) | 0.05 | 0.32 | 0.62 |
  | red_pipette | 0.65 | 0.00 | 0.00 | 0.00 (0.11) | 0.98 (0.87) | 0.88 |
  | 8_channel_pipette | 0.55 | 0.04 | 0.96 | 0.50 | 1.00 | 1.00 |
  | micro_tube | 1.00 | 0.94 | 0.94 | 0.97 | 0.92 | 1.00 |
  | 50ml_tube | 0.96 | 0.94 | 1.00 | 0.91 | 1.00 | 1.00 |
  | 15ml_tube | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
  | blue_tip_rack | 0.81 | 0.21 (0.03) | 1.00 | 0.84 | 1.00 | 0.97 |
  | yellow_tip_rack | 0.91 | 0.70 | 1.00 | 0.68 | 1.00 | 0.96 |
  | red_tip_rack | 0.53 | 0.96 (0.71) | 1.00 | 0.01 | 1.00 | 1.00 |
  | 8_channel_tip_rack | 0.71 | 0.96 (0.17) | 1.00 | 0.79 (0.60) | 1.00 | 1.00 |
  | centrifuge | 1.00 | 0.98 | 0.65 (0.97) | 1.00 | 0.99 | 1.00 |
  | vortex_mixer | 0.95 | 0.01 (0.64) | 0.00 | 0.95 | 0.96 | 1.00 |
  | pcr_machine | 0.99 | 1.00 | 0.00 (0.89) | 1.00 | 0.17 (0.90) | 1.00 |
  | left_hand | 0.97 | 0.16 (0.39) | 0.01 | 0.31 (0.47) | 0.31 (0.81) | 0.94 |
  | right_hand | 0.98 | 0.17 (0.66) | 0.19 | 0.75 (0.96) | 0.34 (0.09) | 0.89 |

  Reading: room 2's bench holds no 15 ml tube and does hold the red tip rack; the vortex mixer
  is outside T2's frame and at the edge of T1's; the moving `yellow_pipette` is the weak class
  here (0.09-0.87 at 0.3 in the fixed views, 0.00-0.62 at 0.5). **The plate**: DINO never
  detects it in T2 and sees it on 56% of T4 frames at 0.3 (9% at 0.5), while DDETR has it on
  90% / 89% of frames in both. Looked at: DDETR's T2 plate is a 97x84 px box near the top of
  the frame (`[1220, 110, 1317, 194]`) where DINO has **no box of any class at IoU >= 0.5 at
  any score >= 0.05 on 3207 of those 3215 frames** (its best plate score per frame is 0.07
  median); in T4 DDETR's 109x72 px plate (`[595, 154, 704, 226]`) coincides with DINO's
  low-score plate (median 0.32) on 2111 frames and with DINO's `pcr_machine` box on 1045. No
  ground truth to adjudicate a ~100 px far object; recorded as the one class-level
  disagreement that touches a tracked target. Under DINO the plate has three confident fixed
  views in room 2 (T1, T3, T5), the plan's minimum for a birth without the fpv; under DDETR it
  has five.
- **Agreement, DINO vs Deformable DETR** (fraction of DINO boxes at score >= 0.5 with a
  same-class DDETR box at IoU >= 0.5 on the same frame; the DDETR partner at >= 0.3 unless
  stated). Trial 1:

  | view | DINO boxes >= 0.5 | DDETR >= 0.3 | DDETR >= 0.5 | any DDETR box | reverse (DDETR >= 0.5 matched by DINO >= 0.3) |
  |---|---|---|---|---|---|
  | fpv | 119,037 | 0.986 | 0.974 | 0.995 | 0.977 (124,215) |
  | T1 | 63,786 | 0.857 | 0.790 | 0.930 | 0.778 (77,379) |
  | T2 | 73,260 | 0.953 | 0.894 | 0.996 | 0.887 (94,959) |
  | T3 | 96,021 | 0.964 | 0.937 | 0.992 | 0.964 (97,276) |
  | T4 | 109,208 | 0.945 | 0.900 | 0.988 | 0.911 (120,926) |
  | T5 | 99,045 | 0.976 | 0.934 | 0.992 | 0.937 (111,783) |
  | all | 560,357 | **0.954** | 0.915 | - | 0.917 (626,538) |

  Per class (all views, DINO boxes in brackets): cell_culture_plate 1.000 (21,193), pcr_machine
  1.000 (16,069), yellow_tip_rack 0.998, vortex_mixer 0.995, blue_tip_rack 0.994,
  8_channel_tip_rack 0.994, red_pipette 0.991, 8_channel_pipette 0.976, micro_tube 0.967
  (150,228), yellow_pipette 0.927, 50ml_tube 0.927, left_hand 0.887, 15ml_tube 0.873,
  centrifuge 0.814, right_hand 0.789, **blue_pipette 0.732 (4,110)**. Trial 2:

  | view | DINO boxes >= 0.5 | DDETR >= 0.3 | DDETR >= 0.5 | any DDETR box | reverse |
  |---|---|---|---|---|---|
  | fpv | 113,104 | 0.986 | 0.970 | 0.996 | 0.978 (117,838) |
  | T1 | 62,631 | 0.772 | 0.724 | 0.901 | 0.831 (59,455) |
  | T2 | 50,351 | 0.810 | 0.762 | 0.897 | 0.748 (56,623) |
  | T3 | 71,860 | 0.939 | 0.902 | 0.972 | 0.918 (79,269) |
  | T4 | 85,185 | 0.861 | 0.830 | 0.980 | 0.927 (84,060) |
  | T5 | 101,532 | 0.952 | 0.927 | 0.982 | 0.924 (116,734) |
  | all | 484,663 | **0.904** | 0.873 | - | 0.906 (513,979) |

  Per class: cell_culture_plate 1.000 (14,708), pcr_machine 0.998, yellow_tip_rack 0.994,
  vortex_mixer 0.986, blue_pipette 0.973, blue_tip_rack 0.964, red_tip_rack 0.958, red_pipette
  0.956, micro_tube 0.920 (98,635), left_hand 0.891, yellow_pipette 0.881, 8_channel_tip_rack
  0.875, centrifuge 0.786, right_hand 0.785, 50ml_tube 0.771, **8_channel_pipette 0.723**.
  Reading: the two detectors agree on 95% of confident DINO boxes in room 1 and 90% in room 2,
  the fpv at 0.99 in both; the static bench objects agree at 0.96-1.00; the disagreement sits
  on the moving pipette (the blue one in trial 1, the 8-channel and 50 ml tube in trial 2),
  the hands (0.79-0.89) and the centrifuge (0.79-0.81, its box straddles the lid state), and
  in room 2 in the two side views T1 / T2 (0.77 / 0.81) whose rig is asymmetric. Nine
  percent of DINO's confident boxes have a DDETR partner only below 0.3 or not at all; the
  plan's `p5-confidence` flag ("DINO vs DDETR agreement, near-redundant") therefore fires on
  5-10% of boxes, concentrated where the SAM3 score and the cross-view residual will also be
  weakest.
- **What moves and what sits in a hand, trial 1, DINO, fixed views (the input to the
  `p2-seeds` rule).** Method: at score >= 0.3, for every 30-frame step k -> k+30 (3570 steps),
  the top-scoring instance's box-centre displacement (`top`), the top instance at k against the
  nearest same-class box at k+30 (`nearest`, not inflated by the top instance switching
  objects), and the fraction of steps on which *any* instance has no same-class box within
  20 px thirty frames later (`steps with a move`, an upper bound: a vanished box counts);
  `start-to-end` = the top instance's median centre over the first 60 frames vs the last 60;
  `in hand` = fraction of the 3600 frames on which some instance's box centre lies inside a
  `left_hand` / `right_hand` box at >= 0.3 (`centre`) or >= 50% of its box area does
  (`covered`). Labels: **moves** = top median > 20 px in some fixed view; **static** = nearest
  p90 <= 20 px and <= 5% of steps with a move in every fixed view; **intermittent** otherwise.
  A fixed view sets a label only when it sees the class on both frames of >= 900 steps (a
  persistent detection: trial 2's `blue_pipette` in T1, seen on 30% of frames at 0.3 and 2.7%
  at 0.5, moves 56 px median as a flickering box and would otherwise mislabel a static object).
  **The fpv column is excluded from the labels: with the head moving, 86-99% of every class's
  steps exceed 20 px there.** Every column below is the max over the counted fixed views.

  | class | label | inst/frame | top median px | nearest p90 px | steps with a move | start-to-end px | in hand (centre / covered) | > 10% in hand |
  |---|---|---|---|---|---|---|---|---|
  | cell_culture_plate | **static** | 1 | 0.6 | 2.3 | 0.00 | 1 | 0.056 / 0.040 | no |
  | blue_pipette | **moves** | 1 | 36.8 | 324.4 | 0.64 | 146 | 0.524 / 0.424 | **yes** |
  | yellow_pipette | intermittent | 1 | 1.6 | 46.6 | 0.21 | 1 | 0.093 / 0.070 | no |
  | red_pipette | intermittent | 1 | 0.8 | 49.2 | 0.21 | 17 | 0.022 / 0.012 | no |
  | 8_channel_pipette | intermittent | 1 | 4.2 | 24.4 | 0.12 | 5 | 0.053 / 0.052 | no |
  | micro_tube | intermittent | 12 | 2.0 | 3.1 | 0.40 | 272 | 0.440 / 0.439 | **yes** |
  | 50ml_tube | intermittent | 2 | 0.9 | 13.0 | 0.18 | 100 | 0.113 / 0.110 | **yes** |
  | 15ml_tube | static | 1 | 0.6 | 7.7 | 0.02 | 4 | 0.146 / 0.143 | **yes** |
  | blue_tip_rack | intermittent | 1 | 1.3 | 22.6 | 0.14 | 7 | 0.074 / 0.072 | no |
  | yellow_tip_rack | intermittent | 1 | 2.0 | 10.4 | 0.06 | 5 | 0.084 / 0.070 | no |
  | red_tip_rack | absent | - | - | - | - | - | 0.000 / 0.000 | no |
  | 8_channel_tip_rack | static | 1 | 0.6 | 1.2 | 0.00 | 1 | 0.001 / 0.001 | no |
  | centrifuge | intermittent | 2 | 0.9 | 25.1 | 0.12 | 212 | 0.096 / 0.055 | no |
  | vortex_mixer | static | 1 | 0.2 | 5.8 | 0.04 | 0 | 0.139 / 0.130 | **yes** |
  | pcr_machine | static | 1 | 0.6 | 1.7 | 0.00 | 0 | 0.015 / 0.012 | no |
  | left_hand | moves | 1 | 45.1 | 307.7 | 0.62 | 193 | - | - |
  | right_hand | moves | 1 | 104.4 | 482.2 | 0.71 | 307 | - | - |

  Per fixed view (top median px / fraction of steps with a move / in-hand centre fraction; n =
  steps with the class on both frames):

  | class | T1 | T2 | T3 | T4 | T5 |
  |---|---|---|---|---|---|
  | cell_culture_plate | 0.2 / 0.00 / 0.00 (3570) | 0.2 / 0.00 / 0.04 (3421) | 0.2 / 0.00 / 0.00 (3570) | 0.6 / 0.00 / 0.06 (3345) | 0.1 / 0.00 / 0.00 (3570) |
  | blue_pipette | 12.2 / 0.41 / 0.05 (765) | 36.8 / 0.60 / 0.06 (3081) | 35.8 / 0.61 / 0.17 (1576) | 23.2 / 0.53 / 0.43 (1256) | 35.8 / 0.64 / 0.52 (1448) |
  | yellow_pipette | - | 1.5 / 0.21 / 0.01 (1823) | 1.6 / 0.19 / 0.09 (3139) | 1.0 / 0.15 / 0.05 (3340) | 0.3 / 0.05 / 0.02 (3566) |
  | red_pipette | - | - | - | 0.8 / 0.21 / 0.02 (3058) | 0.8 / 0.21 / 0.00 (3444) |
  | 8_channel_pipette | 2.7 / 0.26 / 0.05 (830) | 4.2 / 0.02 / 0.00 (3552) | 1.6 / 0.10 / 0.03 (2381) | 0.4 / 0.07 / 0.01 (3568) | 0.3 / 0.12 / 0.05 (3368) |
  | micro_tube | 0.6 / 0.40 / 0.26 (3200) | 0.8 / 0.34 / 0.20 (3564) | 0.6 / 0.28 / 0.33 (3570) | 0.4 / 0.40 / 0.44 (3554) | 2.0 / 0.36 / 0.26 (3570) |
  | 50ml_tube | 0.4 / 0.09 / 0.06 (2929) | 0.2 / 0.08 / 0.05 (3568) | 0.6 / 0.12 / 0.11 (3337) | 0.3 / 0.12 / 0.11 (3534) | 0.9 / 0.18 / 0.03 (3526) |
  | 15ml_tube | 0.6 / 0.00 / 0.15 (3568) | 0.5 / 0.01 / 0.01 (3001) | 0.3 / 0.00 / 0.00 (3570) | 0.5 / 0.02 / 0.08 (3268) | - |
  | blue_tip_rack | 1.3 / 0.02 / 0.07 (2888) | 0.4 / 0.01 / 0.00 (3570) | 0.6 / 0.02 / 0.07 (2025) | 0.7 / 0.14 / 0.04 (3338) | 0.3 / 0.11 / 0.01 (3444) |
  | yellow_tip_rack | 2.0 / 0.03 / 0.04 (2518) | 0.2 / 0.00 / 0.00 (3570) | 0.3 / 0.01 / 0.08 (3309) | 0.5 / 0.06 / 0.04 (3408) | 0.3 / 0.02 / 0.03 (3534) |
  | 8_channel_tip_rack | - | - | - | 0.6 / 0.00 / 0.00 (3570) | - |
  | centrifuge | 0.6 / 0.10 / 0.04 (3500) | 0.9 / 0.12 / 0.01 (3545) | 0.2 / 0.05 / 0.10 (3539) | 0.3 / 0.12 / 0.08 (3556) | 0.2 / 0.05 / 0.06 (3570) |
  | vortex_mixer | 0.2 / 0.00 / 0.05 (3570) | 0.1 / 0.04 / 0.09 (3491) | 0.2 / 0.00 / 0.06 (3520) | 0.2 / 0.02 / 0.14 (3299) | 0.1 / 0.00 / 0.07 (3570) |
  | pcr_machine | 0.2 / 0.00 / 0.00 (3570) | - | 0.3 / 0.00 / 0.00 (3570) | 0.6 / 0.00 / 0.01 (3414) | 0.3 / 0.00 / 0.00 (3570) |
  | left_hand | 6.5 / 0.32 (1688) | 14.9 / 0.48 (420) | 15.7 / 0.46 (3192) | 45.1 / 0.62 (3231) | 12.7 / 0.44 (3228) |
  | right_hand | 43.3 / 0.60 (1237) | 22.5 / 0.59 (1579) | 104.4 / 0.71 (2545) | 96.2 / 0.71 (3107) | 50.0 / 0.66 (3105) |

  Hand boxes for scale: diagonal median / p90 px fpv 517 / 694, T1 223 / 342, T2 304 / 448,
  T3 288 / 358, T4 305 / 412, T5 248 / 306; a hand is in view on 2272 (T2) to 3597 (T5) of the
  3600 frames. Reading, for the rule "seed what moves or sits in a hand box": (i) **moves**:
  `blue_pipette` only (median 23-37 px in T2/T3/T4/T5, relocates 35-146 px, in a hand on 52% of
  T5 frames): the pipette in use. (ii) **Handled although the top instance is still**:
  `micro_tube` (12 per frame in T4/T5; 28-40% of steps some tube has no same-class box within
  20 px thirty frames later; in a hand on 26-44% of frames; the top instance jumps 272 px in
  T3) and `50ml_tube` (2 per frame, 18% of steps, 11% in hand, the top tube relocates 46-100 px
  in four views): the "or in a hand" clause is what catches them, the displacement clause alone
  would not, which is the plan's reason for having both. (iii) **Static and not in a hand**:
  `cell_culture_plate` (**0.6 px median, 2.3 px p90, no step with any instance moved, 0-1 px
  start-to-end, in a hand on at most 5.6% of frames in any fixed view**), `pcr_machine`,
  `8_channel_tip_rack`, both tip racks in use (7 and 5 px start-to-end; the blue one has 14%
  of steps with a move in T4, hands over it). (iv) **Static but in a hand > 10%**: `15ml_tube`
  (14.6% in T1) and `vortex_mixer` (13.9% in T4, tubes being vortexed on it): the hand test
  fires on contact with a resting object, so the rule would open them; the seeds lane's cap
  order (landmarks first, held-only objects after movers) is what keeps that harmless.
  (v) **Intermittent but not relocated**: `yellow_pipette`, `red_pipette`,
  `8_channel_pipette` (12-21% of steps over 20 px, p90 24-49 px, but 1-17 px start-to-end and
  < 10% in hand): brief handling or flicker, not use; the `centrifuge` (2 instances in T1/T2,
  12% of steps; 155-212 px start-to-end in T1/T2 and 49-54 px in T3..T5) is the **lid**, closed
  at frame 600 and open at 4199 (trials.json's T5 closed vs open centres are 62 px apart), not
  the body. (vi) Deformable DETR, same method, gives the same label to every class of interest
  except `yellow_tip_rack` (static vs intermittent, 5% vs 6% of steps) and the same > 10%
  in-hand set except `15ml_tube` (5.4% vs 14.6% in T1).
- **The plate finding.** In this window of protocol 03 the plate is a bench object: it does not
  move and no hand rests on it in any fixed view, in either trial (trial 2: 0.2 px median,
  1.4 px p90, 1% in hand). The plan's rule therefore does **not** open a slot for it, while the
  storyboard needs it (the fpv pans off the plate and back; the transparent plate masked in six
  views). The seeds lane reached the same conclusion from its tracklets while these runs were
  in flight (its entry above: "the plate never moves in protocol 03 and stays detector_only in
  every fixed view", with `--container-classes` + plate as the switch that opens it as a static
  volume). Both methods agree; the decision is the gate-1 human's or the arms lane's, and it is
  a named-object decision, not a threshold.
- **Trial 2, the same tables in brief (DINO, fixed views; full tables in its README).** The
  pipette in use is the **yellow** one (median 38.8 px in T5, relocates 173 px in T5 / 228 px in
  T3, in a hand on 52.5% of T5 frames), and its `yellow_tip_rack` is the rack that moved
  (39-57 px start-to-end in every fixed view, 12.6% in hand); `blue_pipette` is static where
  it is seen persistently (T4 0.7 px at 100% presence, T5 0.3 px) and `intermittent` overall
  (13.1 px median in T3, 20 px start-to-end, 9.9% in hand); `micro_tube` handled (9-11 per
  frame, 60% of steps with a move in T5, 30% in hand, top instance relocates 207 px in T4);
  `50ml_tube` 25% of steps, the top tube relocates 107 px in T5, 8.5% in hand (under the line
  here); `cell_culture_plate` static (above); `red_tip_rack` static (present in room 2 only);
  `centrifuge` 28% of steps, 0-2 px start-to-end (four spins; in a hand on 10.9% of frames by
  the centre test, 3.7% by coverage: both hands on the lid during each spin, as the trials QA
  saw), `vortex_mixer` and `pcr_machine` static, `8_channel_tip_rack` 44% of steps with 2
  instances per frame and 2 px start-to-end (instance switching), `15ml_tube` absent. The rule
  picks the right pipette in each room with nothing changed between them, which is what "zero
  per-trial tuning" needs to be true. DDETR agrees on the plate, the yellow pipette, the micro
  tubes, the red tip rack and the hands; it puts `blue_pipette` in a hand on 13.8% (DINO 9.9%)
  and the `vortex_mixer` at 14% of steps (DINO 3%).
- **For the arms and the tracker.** Rows are in raw pixels at raw frame indices (proxy frame k
  == raw 600 + k); `class_id` indexes the 35-class list; no row is interpolated, so the seeder
  and the observation adapter never meet a `strided_interpolated` manifest on these trials. The
  fpv's per-frame pose validity is not in these files (`fpv_pose_validity(trial)` in the
  adapters). Per-class agreement above is the prior for the `p5-confidence` flag. The trial-2
  plate in T2/T4 is the first concrete case where the two detectors disagree on a tracked
  target's existence in a view; the tracker's birth rule (>= 3 fixed views) does not need those
  views under either detector.

### Sep 25: tracking arms on trial 1 (p4-arms; p1-rig run)

- **What this is.** The `p4-arms` todo of the FineBio 3D-tracking plan
  ([`docs/plan-2026-09-25-finebio-3d-tracking.md`](plan-2026-09-25-finebio-3d-tracking.md)) and
  the "run on trial 1's window" half of `p1-rig`: the rig check on `P03_03_01` [600, 4200), the
  plate added to the seed slots, then arms (a) boxes only, (b) per-frame box decode, (c) SAM3.1
  video memory seeded once and (d) (c) + re-seeds on the six views at encoder side 1280, each
  through the 3D tracker with the rig's gates, with the label-free measures, identity metrics,
  cross-view residuals and the occlusion inventory per arm. Inputs: the DINO pass
  `runs/finebio-detect-P03_03_01-600-4200-20260925/dino` (every frame, six views), the lane-B
  proxies `data/derived/finebio/P03_03_01/600-4200/` (proxy frame k == raw 600 + k, so every
  worker ran with `--start-frame 0` on a proxy and the seeds' analysis frames matched), the
  window camera config and the seeds run. GPU: **about 4.0 h ((b) 3635 s, (c) 4466 s, (d) 6354 s,
  the two smokes 91 s, the seed decode 12 s)** in four queues (never beside
  another model process: `nvidia-smi` showed only `kwin_wayland`, 144 MiB, before every worker;
  the P20 detector lane had finished), no viewer opened. Outputs under
  `runs/finebio-arms-P03_03_01-20260925/` (README with the scoreboard and the layout;
  gitignored, FineBio licence) and `runs/finebio-rig-P03_03_01-600-4200/`. Commits: `e0bbe1c`
  (the orchestration module and tests), `fa2dacc` (the adapter's speckle rule), `fbc3037`
  (scoreboard columns), and this entry's commit. Other lanes committed concurrently; only the
  files named here were staged, by path. **Claim boundary:** the FineBio detector was trained on
  FineBio's own objects and on frames from these cameras; detector-vs-mask agreement is
  model-vs-model, not accuracy; identity metrics against the SAM3 per-view slots are a proxy; no
  human anchor exists yet.
- **Step 0, the plate slot (plan-driven, not a rule outcome).** The seed rule had left
  `cell_culture_plate` `detector_only` in every view (it never moves in protocol 03); the plan's
  shortlist and storyboard include it. `battle-detector-seed select / decode / sheets` were
  re-run into `runs/finebio-seeds-P03_03_01-20260925/with-plate/` (the `instances/` step copied)
  with the plate appended to `--container-classes` and `--landmark-classes` and `--slot-cap 11`:
  the plate opens as a landmark container in all six views and the previous ten slots per view
  are unchanged (the old slot set is a subset of the new in every view; the plate takes slot 3,
  slot 2 in T2 where the PCR machine is never detected). Decode: 66 slots, **66/66 accepted**,
  24 encodes, 68 decodes, 10 s, 2.41 GiB; the plate's seed mask-bbox IoU vs the box 0.941 /
  0.929 / 0.945 / 0.915 / 0.922 / 0.933 (T1..T5, fpv). `battle-finebio-arms mark-plan-slots` set
  the six plate slots' `rule` to **`landmark_plan_shortlist`** in `seeds.json`, `slots.json` and
  `seeds.md` (the tool's `container` kept as `rule_from_seed_tool`, the note recorded), so the
  record says the plate is there because the plan asked, not because the data did. 51 of the
  66 schedule seeds start at analysis frame 0.
- **Step 1, the rig on the window (`battle-finebio-rig`, 15 s CPU).** The clip config's
  `downstream.rig` command with `--detections` pointed at the DINO sub-directory. Gates:
  static LOO median **10.0 px** over 42 cells (preflight 10.4 over 49) -> `association_px` **30.1
  px** (preflight 31.1); moving-object LOO p90 per view T1 14.1, T2 11.9, T3 4.5, T4 11.7, T5 8.1,
  **fpv 27.0** -> `handoff_px` **27.0 px** (preflight 51.9). The hand-off gate is half the
  preflight's because the objects that define it, the plate and the blue pipette, behave
  differently here: the plate does not move in protocol 03 (LOO 3.7-13.3 px median, inside the
  held-out box on 100% of 3400-3588 frames per view; fixed -> fpv hand-off 9 px median / 27 p90,
  inside the fpv box on 97% of 3556 frames), and the raised blue pipette has >= 3 other views on
  only 108-217 frames per view with residuals of 41-132 px median, too few to move the p90. The
  formulas were applied as written; what the tighter gate does to the in-hand object is in the
  identity metrics below. Clock offsets 0 in every view (T5 the only informative scan, right
  hand), none significant. Static heights (cm above the bench): PCR 1.6, vortex 1.9, 8-tube-strip
  rack 1.9, micro-tube rack 0.0, magnetic rack 1.0, blue tip rack 3.8, yellow tip rack 2.8,
  trash can 10.1, **centrifuge 8.4** (4.1 in the preflight: the lid is open for most of this
  window, so its box centre sits higher and its LOO cells, 35-129 px, are the static column's
  p90 of 40.8). Hands as probes: left 2546/3600 frames from >= 3 fixed views at 10.9 px median,
  2.2 cm; right 1904/3600 at 51.7 px, 22.8 cm (raised). fpv pose valid 3587/3600. Negative
  control T5: markers 93.7 vs 0.72 px, static LOO 97.7 vs 4.8 px, left hand 54.4 vs 9.5 px. One
  degenerate cell on the record: `left_hand` held out of T2 (282 frames, 843 px median).
- **The orchestration (`src/battle/finebio_arms.py`, console script `battle-finebio-arms`,
  commit `e0bbe1c`).** `run --arm a|b|c|d` builds the arm's observation set (every view's
  detector rows at score >= 0.3 through `detections_to_observations` with the fpv pose validity
  from the shipped pose file; for the SAM3 arms the worker run per view under
  `--worker-root/<view>` through `worker_to_observations`, area centroids, the same view's
  detector rows attached for confirmation, six views in a process pool), writes
  `observations.jsonl`, runs `battle-multiview-tracks` in-process with the window camera config
  and `--gates rig.json`, and derives `measures.json` / `.md` and the inventory. Measures, per
  view and slot for the SAM3 arms: frames with a mask against the frames from the slot's start,
  **mask-bbox IoU against the best same-class DINO box (score >= 0.3) of the same view and
  frame** (median, p10, fraction >= 0.5; one reference for every arm, so (b), whose prompt is a
  DINO box of the same tracklet, and (c), whose mask is free, are read on one footing; the 8
  group slots have no single detector box and are outside the IoU pool; rows with no same-class
  detection in the view are counted, not scored), the SAM3 object score (the decoder's IoU in
  (b), the presence logit in (c)/(d): not comparable across arms), mask-area step stability
  (median relative step between consecutive frames, fraction of jumps > 0.5), the fraction of
  the slot's rows the tracker associated; per view pooled and overall pooled IoU; cross-view
  residuals from `residuals.jsonl` per view and per source; the identity metrics plus an
  `objects_only` line (the tracker also tracks the hands, which this plan treats as probes).
  **Occlusion inventory** (`occlusion_inventory.jsonl` / `.md`): every maximal run of `coasting`
  rows of one track, with start, last observed frame, length (a lost episode's length is the
  30-frame coast timeout, not the occlusion's duration), outcome (`reacquired` with latency,
  `lost`, `open_at_window_end`), the last 3D position and support, that position projected into
  every view with a valid pose at the first coasting frame and tested against the hand boxes
  (`held` when inside one in >= 2 views; hand boxes are large in the fixed views, so this is an
  upper bound and `held_in_all_projected_views` is the stricter count), the container classes'
  boxes (`contained`, >= 2 views, the container named) and the same-class detector boxes within
  the association gate (`detector_visible_association_miss`: the detector still saw the object
  there and the tracker did not associate it, usually because another track took the box),
  the successor ids born with `possibly_same_as` and the `ambiguous` events; hand-track episodes
  counted apart; the summary counts the episodes each `p3-tracker-ext` extension would serve.
  `sanity` (first-view check: ms per frame or step within 2x of the preflight's 152 / 220 and
  mask-bbox IoU vs the prompt or detector box on the first N frames >= 0.8), `reseed-schedule`
  (arm (d)'s per-view payload from (c)'s `detector_reseed` / `handoff_reseed` events: the slot
  is the SAM3 slot the track last had in that view per `residuals.jsonl`, one box correction per
  slot per K = 30 frames, `selected_by` `detector_reseed` / `track_reproject`, events for tracks
  the view never masked skipped and counted), `decide` (the (c)-vs-(b) rule), `mark-plan-slots`,
  `scoreboard`; `--fpv-poses` takes the fixtures' pose file so the pipeline runs without
  `data/`, `--reuse-observations` re-runs tracker and measures from a written
  `observations.jsonl`. Tests (`tests/test_finebio_arms.py`, 10, default tier): episodes with
  the three outcomes; held / contained / association-miss inference on the fixture cameras with
  probes apart; slot measures against the detector reference with a group slot and a mask-less
  row; area stability; the reseed schedule with every skip reason, accepted by the worker's own
  `_corrections_by_frame`; the sanity check passing and failing on timing and on IoU; the plan-slot
  annotation (idempotent); the decision rule; the committed clip config; arm (a) end to end on
  the preflight fixtures with observation reuse.
- **One adapter fix on the way (`finebio_observations.py`, commit `fa2dacc`).** The arm (c)
  first-view sanity (fpv, 300 frames) read mask-bbox IoU medians of **0.035 / 0.154 / 0.023**
  for the plate, centrifuge and vortex while the masks were right: SAM3.1 video-memory masks
  carry 1-15 isolated single positive pixels far from the object (the worker's own box already
  ignores them: components under 20% of the largest, `BOX_COMPONENT_KEEP_FRACTION`), and
  `worker_to_observations` measured the bbox over every nonzero pixel. `filter_mask_components`
  now applies the worker's rule before the bbox, area centroid and area are measured; rows that
  lost pixels record `mask_speckle_pixels_dropped` and `mask_components` in their provenance;
  single-component masks are untouched and `mask_measurements` is unchanged. Speckles were
  dropped on **54.5%** of (c)'s rows and **36.2%** of (b)'s; after the fix the same 300 frames
  read median 0.941 (plate 0.937, centrifuge 0.978, vortex 0.951). Test: a mask with three
  stray pixels measures as the object alone with 3 pixels and 3 components on the record; a
  component at >= 20% of the largest is kept. Arm (b)'s observations were rebuilt after the fix
  (its first build had not finished); nothing else re-ran.
- **Arms: what ran and what it cost.** Queue order fpv, T1..T5, one worker at a time
  (`battle-muggled-arms`, MuggledSAM interpreter, `sam3.1_multiplex.pt`, bfloat16, 1280 square).
  **(a)** boxes only: no GPU; observations + tracker + measures 53 s CPU. **(b)** `box-decode`
  on `with-plate/box_streams/<view>.jsonl`, 3600 frames per view: **593-614 s per view**
  (149-151 ms per prompted frame median, of which the image encode 145; 164-171 ms end to end
  with video decode and PNG writes), **2.15-2.22 GiB**, 26,582-34,851 masks per view (181,984 in
  all), adapter + tracker + measures 6.5 min CPU. **(c)** `video-memory` on
  `with-plate/schedules/<view>.json` (11 slots per view), `--prompt-memory-semantics append`, no
  tau, `--checkpoint-every 600`: **708-793 s per view** (195-217 ms/step steady; the preflight's
  220 for 4 slots: 11 slots cost nothing more, the encoder dominates), **3.61-3.64 GiB on the
  fixed views, 4.15 on the fpv** (2.6 for 4 slots), 36,222-37,799 masks per view (222,520: a
  memory slot answers on every frame from its start), pipeline 7.9 min. **(d)** `video-memory` on
  the schedules built from (c)'s events (986 box corrections in all: fpv 426, T1 149, T2 23, T3 177,
  T4 78, T5 133; `track_reproject` 636, `detector_reseed` 350; 8,978 events skipped, 8,917 of them
  for tracks the view never masked), same flags: **895-1162 s per view** (247-322 ms/step: every
  correction is a decoder pass plus a prompt-memory encode, and `append` keeps every prompt in the
  bank), **3.81-4.07 GiB fixed, 4.39 fpv**, 222,740 masks, pipeline 7.7 min.
  Sanity checks before each queue continued: (b) fpv 100 frames **149.6 ms per prompted frame**,
  mask-bbox IoU vs the prompt box **0.936 median, p10 0.879, 100% >= 0.5** on 873 masks, masks on
  the boxes from frame 0; (c) fpv 300 frames **204 ms/step**, **4.13 GiB**, IoU vs the best
  same-class detector box **0.941 median, p10 0.715, 98.7% >= 0.5** on 2428 masks (plate 0.937,
  centrifuge 0.978, vortex 0.951, PCR 0.900, blue pipette 0.731; the preflight's tracks 0.70-0.98)
  after the adapter fix; (d) fpv, first 300 frames of the full run, **321.5 ms/step**, 0.941
  median, 98.2% >= 0.5, 4.39 GiB.
- **Scoreboard (per view and pooled; `scoreboard/scoreboard.md`).** Det-box IoU pooled
  median / p10 / fraction >= 0.5 / n: **(b) 0.926 / 0.842 / 0.991 / 148,362**; **(c) 0.914 /
  0.060 / 0.793 / 175,827**; **(d) 0.911 / 0.000 / 0.757 / 175,997**. Per view (b) 0.921 / 0.920 /
  0.920 / 0.926 / 0.931 /
  0.936 (T1..T5, fpv) with 98.2-99.9% >= 0.5 everywhere; (c) 0.846 / 0.815 / 0.913 / 0.921 /
  0.930 / 0.932 with **65.8 / 61.9 / 68.8 / 97.4 / 93.3 / 82.7%** >= 0.5; (d) 0.843 / 0.799 / 0.912
  / 0.920 / 0.926 / 0.929 with 65.9 / 62.3 / 68.9 / 92.6 / 87.2 / 72.9%. Where
  (c) loses is legible per slot: the landmark and static slots hold in every view at (b)'s level
  (plate 0.917-0.954, vortex 0.918-0.943, PCR 0.846-0.928, trash can 0.941-0.952, yellow and red
  pipettes 0.885-0.960), while **the centrifuge in T1 and T2 reads 0.475 / 0.447 with 22 / 19%
  >= 0.5** (the memory holds the base while the detector box grows to include the open lid; in
  T3-T5 and the fpv 0.965-0.973), **the 50 ml and micro tube slots drift to nothing** (T2
  `50ml_tube#0` 0.000 on 2157 masks after the tube leaves at frame 2043; T1/T3 `50ml_tube#0`
  0.24; `micro_tube#0` 0.00-0.44 in T2/T3/T5; fpv `15ml_tube#1` 0.000), and **the in-hand blue
  pipette drifts** (T1 0.067, T3 0.420, fpv 0.562 vs (b)'s 0.83-0.89). (b) cannot drift by
  construction: its weak cells are the same in-hand objects at p10 0.52-0.63 and the 27-frame
  held micro tube in T5 (0.000 on 92 masks: the box-prompted decode lands on the glove).
  Identity (tracker; (a) / (b) / (c) / (d)): tracks born **285 / 308 / 270 / 264**, objects only
  226 / 249 / 211 / 205; lost 243 / 264 / 234 / 227; re-acquired 129 / 141 / 101 / 79 at latency
  median 10 / 9 / 9 / 11; **ambiguities 126 / 158 / 133 / 173**; **fragmentation 220 / 242 / 207 /
  201** (objects only 168 / 190 / 155 / 149); id switches against the SAM3-slot proxy - / 629 / 264
  /
  358 ((a) has no SAM3 slots); slot disagreements 0 / 64 / 28 / 17; duplicate-pair frames 41,090 /
  40,982 / 24,468 / 24,146. Per class, the arms agree on the objects that do not move (plate,
  vortex,
  PCR, trash can, yellow pipette, 15 ml tube, the micro-tube group: one id each over 3600
  frames in every arm) and differ on the rest: **the centrifuge is one id in (c) and 8-9 ids in
  (a)/(b)** (the box centre jumps when the lid opens and the stationary prior loses it; the mask
  centroid stays on the body), the blue pipette **47 / 80 / 30 / 22** ids, the 8-channel pipette 19
  /
  8 / 15 / 10, the red pipette 9 / 5 / 4 / 4, the 50 ml tubes **5 / 6 / 19 / 8** (the drifted (c)
  slots breed false tracks), micro tubes 96 / 100 / 101 / 118 (identical instances, out of
  scope). Cross-view residual pooled median 5.4 / 5.4 / 6.0 / 6.0 px (SAM3 rows alone 6.4 in (b),
  6.8 in (c), 6.7 in (d); detector rows 5.2-5.8), fpv 8.8-9.0 px.
- **The (c)-vs-(b) decision and arm (d).** The rule written into `decide` before the numbers
  existed: (d) runs if (c) beats (b) by >= 0.02 on the pooled det-box IoU median, or is better on
  all of id switches, fragmentation and ambiguities and worse on none. (c) **loses on IoU** (-0.012
  on the median; 79.3 vs 99.1% >= 0.5) and **wins on all three identity metrics** (264 vs 629,
  207 vs 242, 133 vs 158), so (d) ran (`decision_c_vs_b.json`), with the caveat on the record
  that the id-switch proxy treats a drifted slot that stays on the wrong object as one identity.
  **(d) did not pay for itself**: pooled IoU 0.911 (below (c) and (b)), 75.7% >= 0.5 (below
  (c)'s 79.3%), **ambiguities 173** (worse than both), fragmentation 201 (best by 6), proxy id
  switches 358 (worse than (c), better than (b)), re-acquisitions 79 (fewest), slot disagreements
  17 (fewest). Per slot the corrections cut mostly down: T4 `50ml_tube#1` took **one**
  `track_reproject` box and fell from 97.4% to 0.1% >= 0.5 (the box decoded another object and
  `append` kept it in the prompt bank), T5 `8_channel_pipette#0` (52 corrections) 95.7 -> 48.0%,
  fpv `15ml_tube#0` (59) 52.4 -> 3.3%, fpv `blue_pipette#0` (23) 57.5 -> 31.5%, the fpv trash can
  / centrifuge / PCR machine (49 / 60 / 13) lost 9-14 points; the gains are T1 `blue_pipette#0`
  17.9 -> 34.0% and T3 `micro_tube#0` 10.1 -> 16.4%; per class the 50 ml tubes (19 -> 8 ids), the
  blue pipette (30 -> 22) and the 8-channel (15 -> 10) fragment less, the micro tubes more (101
  -> 118). The mechanism as run (tracker-emitted boxes, one per slot per 30 frames, installed
  under `append` with no acceptance test on the decoded mask) injects a wrong prompt whenever the
  projection lands where the object is not visible; a correction would need the seed tool's
  acceptance rule (decoded mask-bbox IoU vs a same-class detector box >= 0.6) before it is
  installed, a change to the worker's correction path that was not made here. **Verdict for the
  plan's stop rule** ("(b) within 0.02 of (c)/(d): video memory and re-seeding are not
  adopted"): (b) is not within 0.02, it is ahead (0.926 vs 0.914 / 0.911 on the median, 99.1 vs
  79.3 / 75.7% on the fraction >= 0.5), so **the memory-free per-frame decode is the mask source
  for the floor and for `p5-*`**; video memory's one legible advantage is identity through
  appearance change (the centrifuge one id across the lid cycles in (c)/(d), 8-9 ids in (a)/(b))
  and masks on the frames the detector misses (222k vs 182k), its cost is drift on objects that
  leave or are held.
- **Occlusion inventory (object tracks; hand-track episodes counted apart, 86 in every arm).**
  (a) / (b) / (c) / (d): **284 / 317 / 247 / 218 episodes** on 194 / 215 / 180 / 173 tracks; lost at
  the 30-frame timeout 187 / 208 / 178 / 171, re-acquired 97 / 109 / 69 / 47 (the re-acquired ones
  last 9-11 frames median, 24-26 p90). Inferred state: `held` (inside a hand box in >= 2 views)
  **142 / 166 / 102 / 89**, of which inside a hand box in every projecting view 16 / 19 / 6 / 5;
  `contained` **125 / 135 / 138 / 144**, by container **micro-tube rack 89 / 100 / 109 / 118**,
  centrifuge 16 / 13 / 12 / 9, magnetic rack 15 / 13 / 12 / 10, vortex 4 / 4 / 1 / 2, 50 ml rack
  1 / 5 / 2 / 5; identical-instance classes or group slots (group-track candidates) **145 / 156 /
  171 / 161**, of which ending in an ambiguity or a `possibly_same_as` successor 74 / 80 / 88 /
  80; association misses (detector still within the gate in >= 2 views) 21 / 16 / 21 / 13;
  unexplained 66 / 68 / 42 / 32.
  By class: micro tubes 123-136 episodes in every arm (the racks' tubes coasting when a hand
  passes over the rack: `contained` in the micro-tube rack is the dominant state), the blue
  pipette 74 / 116 / 30 / 24, the 8-channel pipette 27 / 17 / 26 / 12, blue tips 16 (they open no
  slot and are tracked from detector rows only), the red pipette 14 / 10 / 10 / 10, the
  centrifuge 8 / 7 / 0 / 0.
  Reading for `p3-tracker-ext`: **`contained` would serve 125-144 episodes per arm, four fifths
  of them tubes in the micro-tube rack** (the rack is one static volume; the plan's group track
  for the rack's tubes and the `contained` state are the same fix from two sides), the
  centrifuge's own containment 12-16; **`held` 89-166 by the loose test but only 5-19 by the
  strict one**, and the in-hand pipette's real problem is not the hand box but the stationary
  prior at a 27-30 px gate (it fragments into 22-80 ids before any occlusion); **group tracks**
  145-171 episodes, half of them ending ambiguous, which is the per-tube identity the plan already
  rules out. The 3 zero-length episodes per arm (a track re-acquired on the frame its support
  dropped) are coasting events without a coasting row and are not counted.
- **Deviations.** (1) The plate slot is plan-driven (above). (2) The adapter speckle rule
  (above). (3) The hand-off gate came out at 27 px, half the preflight's, for the reason above;
  it was used as computed. (4) (d) ran on the identity clause of the pre-registered rule, not the
  IoU clause, and came out negative. (5) (e)
  DAM4SAM was not run (optional; the GPU went to (d)). (6) The tracker also tracks the hands;
  they are in the observation files as the rig treats them (probes) and are excluded from the
  inventory's object counts and the `objects_only` identity line, not from the tracker's totals.
  (7) The `sanity` subcommand reads every mask of a run before keeping the first N frames, so a
  full-run check takes the adapter's 3-5 minutes; the 300-frame smoke runs are quicker.
- **Tests and lint.** Default tier **945 passed / 14 skipped** (934 before this lane; +10 arms
  tests, +1 observation test); `uv run ruff check src tests scripts` and `ruff format --check` clean on the
  files touched. No `gpu`-marked test added (the arms' GPU numbers are recorded here and in the
  run README).
- **For the confidence / events / viewer phase.** Every arm directory has the same shape:
  `observations.jsonl` (`FineBioObservation` rows in raw pixels and raw frames: every DINO box at
  >= 0.3 plus, in (b)-(d), the SAM3 rows with mask bbox, area centroid, area, `sam3_object_score`
  and the detector box when one matches, `provenance.detector_box_iou`, `object_id` `sam3-NN`
  for the mask file), `tracks/tracks.jsonl` (`Track3D`), `tracks/events.jsonl` (`TrackEvent`;
  `handoff_reseed` / `detector_reseed` carry `view` and `box_xyxy_px`), `tracks/residuals.jsonl`,
  `tracks/identity_metrics.json`, `measures.json`, `occlusion_inventory.jsonl` (schema in the
  README). The worker runs' `masks/<frame:06d>_<slot:02d>.png` are in analysis frames (raw - 600)
  under `b-box-decode/<view>/`, `c-video-memory/<view>/`, `d-video-memory/<view>/`. The (d)
  schedules with the corrections the tracker emitted are `d-reseed-schedules/<view>.json`. Open,
  read off these runs for `p5-confidence`: `sam3_object_score` means different things per arm;
  the per-slot `frac >= 0.5` and the area jumps flag the drifted slots in (c) without a label; the
  tracker's `slot_disagreement_views` fired 28-64 times.

### Sep 25: anchor frames and workspace for gate 2 (p6-anchors, prepared)

- **What this is.** The `p6-anchors` todo of the FineBio 3D-tracking plan
  ([`docs/plan-2026-09-25-finebio-3d-tracking.md`](plan-2026-09-25-finebio-3d-tracking.md)),
  prepared, not labelled: the anchor frames were chosen from arm (b), the candidate masks the
  human will choose among were decoded, and the scoreboard exists and runs on the empty record.
  No cell is labelled; `runs/finebio-anchors-P03_03_01-20260925/decisions.json` does not exist.
  **Claim boundary:** when labels land they are one person's choice among SAM3 image-decoder
  masks (or a hidden / box-level mark) on 18 frames of two views plus one six-view frame of one
  trial; they rank arms (a)-(d) against each other and are not ground truth, not a dataset, and
  support no accuracy claim; the detector that proposed every box was trained on this lab's
  objects and cameras. **GPU:** one MuggledSAM decode of 33 s (40 image encodes, 1011 decodes,
  2.57 GiB peak) run **beside the trial-2 lane's arm (b) fpv worker** (pid 1292239, 3.3 GiB;
  4.5 of 16.3 GiB in use before the start): the guard refused first ("another Battle GPU worker
  is running"), the card had 11.7 GiB free against a 2.8 GiB expected peak, so the PID was named
  with `--allow-gpu-neighbour` rather than waiting an hour for that queue; the trial-2 worker's
  wall-clock timing may show a 33 s contention window at about 06:20 UTC-4, nothing was killed.
  Nothing under `runs/` or `data/` is committed (FineBio licence); the anchor config carries frame
  numbers and slot labels only. Commit `e2aae0b` (tool, tests, config; its `pyproject` line went
  in with the confidence lane's `4ecf332`, which staged the whole file) and this entry's commit.
- **The tool (`src/battle/finebio_anchors.py`, console script `battle-finebio-anchors`).**
  `select` reads arm (b)'s `observations.jsonl` (SAM3 rows, pre-filtered on the raw text so the
  877k-row file reads in a second), the seeds' slot lists and the clip config, and writes the
  typed anchor config; `workspace` builds the candidate decode requests, runs the decode under
  the MuggledSAM interpreter (`decode-worker`, the seed tool's spawn pattern and guard, encoder
  1280, one image encode per frame-view), assembles the cells, renders the sheets and writes
  `workspace.json`, `decisions.template.json`, `README.md`; `score` scores arm directories
  against a decisions file (missing or partial is fine); `export` writes the committed
  human-record skeleton. The Assembly101 anchor machinery (`battle-anchor-export`,
  `battle-anchor-iou`, the calibration web workspace) was **not** generalised: the web
  workspace builds its manifest from an Assembly101 clip config and a four-part manual-seed
  target policy (`muggled_calibration.build_manifest`, `load_manual_seed_target_config`),
  and re-pointing it at FineBio proxies, per-view slot lists and the raw/proxy frame mapping was
  judged more than the two hours allowed; gate 2 follows the gate-1 pattern instead (static
  sheets plus a decisions JSON), which is also what the human used for gate 1. Scoring
  definitions were carried over where they apply (labeled / hidden / unlabeled cells, IoU and
  area, hidden false positives, missing run masks, the built-in check that an arm scores 1.0
  on its own accepted masks).
- **Fixed view: T4.** Rule: among the fixed views whose plate slot has a mask on >= 90% of the
  window, the one with the most individually tracked (non-group) slots, ties by pooled det-box
  IoU. Arm (b) per view: T1 7 non-group + 4 group slots (plate 100%, det-box IoU 0.918), T2 10 +
  1 (97.9%, 0.918), T3 10 + 1 (100%, 0.918), **T4 11 + 0 (97.1%, plate 0.906, pooled 0.926)**,
  T5 9 + 2 (100%, 0.931). Group slots (a rack of tubes as one union box) have no single detector
  box and are outside the disagreement pool and the arms' IoU pool, which is why T5, the
  top-down camera with two group slots, ranks below T4 despite the slightly higher pooled IoU.
  T4's eleven slots are the four landmarks (centrifuge, vortex, PCR machine, plate), trash can,
  yellow / red / blue pipette, `8_tube_stripes#0`, `50ml_tube#0` (600-1921) and `50ml_tube#1`
  (2564-3059), the objects that go to the centrifuge among them. Table on the config
  (`fixed_view_candidates`).
- **Frame rule and the frames** (`configs/qa/finebio_P03_03_01_review_anchors.json`; raw frame,
  proxy = raw - 600). Per frame the disagreement score is the minimum of
  `provenance.detector_box_iou` over the non-group SAM3 rows of the fpv and T4 in arm (b)
  (14-19 slot rows pooled per frame; distribution p10 / p50 / p90 = 0.37 / 0.70 / 0.86; the
  argmin slot is the blue pipette on 1449 of 3600 frames, the fpv yellow pipette on 534).
  Eligible: a SAM3 row in both views and a valid fpv pose (13 frames excluded). Chosen lowest
  first, each >= 90 frames (3 s) from every frame already chosen including 916, the centrifuge
  cycles [1176, 1228) and [3224, 3311) padded by 90 frames ([1086, 1318), [3134, 3401)) served
  first until three frames lie inside them; then 5 random frames from
  `numpy.default_rng(20260925)` uniform over the eligible frames with the same spacing (draws
  8, 9, 18, 19, 20 accepted).
  **`916`** (proxy 316) six views, the frame the FineBio authors annotated in every camera.
  **Disagreement (12):** in the neighbourhoods `1120` (T4 centrifuge 0.192, the lid opening
  before the first spin), `1294` (T4 yellow pipette 0.205), `3372` (T4 vortex 0.216); window-wide
  `1521` (fpv blue pipette 0.170), `1739` (fpv 8-channel pipette 0.128), `1966` (fpv red pipette
  0.215), `2380` (fpv blue pipette 0.133), `2748` (T4 blue pipette 0.206), `3054` (fpv blue
  pipette 0.140), `3481` (fpv blue pipette 0.223), `3571` (T4 blue pipette 0.234), `4032` (T4 blue
  pipette 0.055, the lowest in the window). **Random (5):** `637`, `1635`, `2132`, `2240`,
  `4122`. 18 frames, 40 frame-views, **440 cells** (11 slots per view), of which **337 have a
  detector box in arm (b)** and hence candidates, and 103 have none (the slot's object had not
  arrived, had left, or was out of frame: `50ml_tube#1` 17, `15ml_tube#1` 15, `50ml_tube#0` 15,
  `15ml_tube#0` 14, `8_tube_stripes#0` 9, ...; those cells take only `hidden` or `none_fits`).
  The config also records the slot lists per view (label, class, role, rule, start frame), the
  reason and the arm-(b) minimum per frame, every rule parameter, and the SHA-256 of the seeds
  file and of arm (b)'s observations, so `select` re-run on the retained run reproduces it
  (the `real_data` test does exactly that).
- **Workspace (`runs/finebio-anchors-P03_03_01-20260925/`, gitignored).** Per cell with a box:
  **c0** = arm (b)'s mask for that frame and slot (`b-box-decode/<view>/.../masks/<proxy
  frame>_<slot>.png`, copied after the arms' speckle rule so the anchor and the scored masks are
  measured alike), **c1** = the 0.15-margin box, **c2** = the tight box plus one positive point
  (arm (b)'s mask centroid when it lies inside the box, else the box centre; the centroid was
  used on every cell with a mask), **c3** = the tight box's second-ranked decoder output (the
  same decode as c0 with the other granularity). 1348 candidates; mask-bbox IoU vs the reference
  box, median per kind: c0 0.927, c1 0.779, c2 0.924, c3 0.925. **556 of the 1011 alternates
  duplicate an earlier candidate** (IoU >= 0.97: c2 256 of 337, c3 231, c1 69), marked
  `duplicate_of` and greyed on the sheet, so a row typically offers two or three distinct masks
  (1 / 2 / 3 / 4 distinct on 67 / 144 / 67 / 59 cells); the margin box is the alternate that
  differs most often. Sheets `sheets/f<raw>_<view>.jpg` (40, 1280 px wide, one row per slot,
  columns = context crop with the reference box, c0..c3 with `dec`, `bbox IoU`, area) and
  `sheets/f<raw>_<view>_overview.jpg` (the frame with every slot's box and number, for
  identity); 40 MB of sheets, 11 MB of candidate PNGs. `decisions.template.json`: one entry per
  cell with `decision: null` (a candidate index, `"box"`, `"hidden"`, `"none_fits"` or null) and
  `instance_identity` pre-filled for the bench's singletons (centrifuge, vortex, PCR machine,
  plate, trash can; not for pipettes, whose colour the detector confuses, nor for tubes). The
  human's steps, the viewing command (a read-only static server on the Tailscale address, not
  started) and the scoring command are in the gate-2 section of
  [`docs/labeling-sessions-2026-09-25-finebio.md`](labeling-sessions-2026-09-25-finebio.md).
- **Scoreboard (`score --workspace ... --record decisions.json --arms a=...,b=...,c=...,d=...`).**
  Per arm and labelled cell: the arm's row is its SAM3 row with the cell's slot label (the
  worker label), else the same-class detector row overlapping the anchor's reference at IoU >=
  0.1 (the boxes-only arm, or an arm without a mask there); **mask IoU** between the arm's mask
  (read through the speckle rule) and the accepted candidate; **box IoU** between the arm's
  mask bbox or detector box and the candidate's bbox, or the reference box for `box` accepts;
  `run_mask_missing` when no row exists; a mask on a `hidden` cell is a `hidden_false_positive`
  with its area; `none_fits` and unlabelled cells are counted, not scored; means per class, view,
  origin (six-view / disagreement / random) and inside the centrifuge neighbourhoods.
  **Identity:** on every labelled cell with an `instance_identity`, the track id behind the
  arm's row is read from `tracks/tracks.jsonl` (`support_slots[view] == the row's slot`), and
  IDF1 = 2 IDTP / (named cells + cells with a track) with IDTP the maximum one-to-one matching
  between names and track ids (`multiview_tracks.hungarian`), on frame 916 across the six
  cameras and pooled over every labelled frame (so a name kept across frames also measures
  persistence), with the names split across several ids, the ids covering several names and
  the named cells without a track. The table's first line says how many of the 440 cells are
  labelled. **Run on the empty record today**: 0 / 440 labelled, every arm 0 cells scored, 18 s
  for four arms. **Synthetic check, not labels:** a throw-away record (100 cells: c0 on frame
  916, c1 on the pipettes there, `hidden` on the 13 box-less cells of 916, c0 / `box` on frames
  1120 and 4032, names on 65 cells) scored the four arms in 20 s: (b) 1.000 on every c0 accept
  (the built-in check) and 0.83-0.95 where the margin alternate was chosen; (a) 10 `missing`
  cells (the box stream came from tracklets at score >= 0.2, arm (a)'s rows are the detector at
  >= 0.3); (c) 3 hidden false positives (31k px: video memory answers on frames the detector
  had no box) and the drift the arms scoreboard already showed (micro tube 0.21, 8-channel
  pipette 0.43); IDF1 0.87-0.91 on 916 for all four arms. The record and its scoreboard were
  written under `/tmp` and are not in the workspace.
- **Export.** `export --workspace ... --record decisions.json --output
  docs/qa/finebio-P03_03_01-review-anchors.human-record.json` writes the committed skeleton:
  per cell the state, candidate index and kind, SHA-256 of the accepted mask PNG, identity;
  counts; author and reviewed_at from the decisions file; claim boundary and licence; no pixel
  leaves `runs/`. Not written today (nothing to export).
- **Deviations and choices on the record.** (1) Static sheets + decisions JSON instead of the
  calibration web workspace (above). (2) The cycle "neighbourhood" is the closed-lid interval
  padded by 90 frames on each side, one spacing unit; the loading and unloading happen there.
  (3) Group slots are outside the disagreement pool, as in the arms' measures. (4) Frames with
  an invalid fpv pose are not eligible (13 of 3600). (5) Two decisions beyond the plan's
  accept / hidden: `box` (the reference box is the object, no mask fits; scored by box IoU) and
  `none_fits` (excluded, counted). (6) Identity pre-filled for five singleton classes only.
  (7) The spacing applies against 916 and the random frames too, so no random or disagreement
  frame sits within 3 s of another anchor. (8) The task brief estimated ~90 frame-views; the
  plan's counts give 40 (18 frames x 2 views + 916 x 4 more), 440 cells.
- **Tests and lint.** `tests/test_finebio_anchors.py`: 13 default-tier (disagreement pooling;
  the selection with spacing, cycle-first and reproducible seeded randoms; the fixed-view rule;
  the committed config against the plan's counts, spacing, neighbourhoods and views; the
  workspace through the CLI without a GPU on two synthetic proxies and an arm (b) with masks,
  including c0 copied byte-equal, the template's identity pre-fill and the margin box; request
  building with the box-centre fallback; every `score_cell` branch; row matching by label then
  overlap; IDF1 on perfect / split / merge / missing; the scoreboard on an empty record via the
  CLI; a partial record over synthetic arms a / b / c with masks, box, hidden and identity, and
  the export skeleton; record validation; the worker command and environment) + 1 `real_data`
  (re-running `select` on the retained arm (b) reproduces the committed frames). Default tier
  **964 passed / 14 skipped** (945 at the arms entry; the confidence and events lanes added the
  rest); `ruff check` / `format --check` clean on the files touched here (three E501 lines in
  `finebio_events.py` belong to the events lane).
- **Next (the human's, optional, ~1.5 h).** Per the brief: frame 916 in six views first (the
  cross-camera identity), then the three centrifuge-neighbourhood frames, then the rest; copy
  `decisions.template.json` to `decisions.json`, fill `decision` and `instance_identity`, run
  `score`, then `export`. The arms are already ranked on the label-free measures; the anchors
  add the human's reading of the same frames and never block the pipeline.
