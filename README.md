# Battle
A 3D object tracker for a wet-lab bench made with SAM3, six cameras and no training

This is a method lab that went from SAM3 smoke tests on a toy car assembly (Assembly101) to a
six-camera 3D object tracker on a wet-lab bench (FineBio) during its first 22 days. The current
work focuses on pipette masks and initialization. I trained nothing and annotated no ground
truth, so every number on this page says what it was measured against. This page is the
five-minute version. The day by day is in [`docs/story.md`](docs/story.md), the
numbers are in [`docs/results.md`](docs/results.md), the commands are in
[`docs/pipeline.md`](docs/pipeline.md) and the lab's shorthand is translated in the
[glossary](docs/writing-style.md#glossary).

![One tube, one identity, through a closed centrifuge lid](media/story/2026-09-25-finebio-3d.gif)

The six cameras on the bench and the tracker's top-down view of them. The highlighted micro tube
keeps the same identity while the centrifuge lid is closed over it.

## Current checkpoint

**A good box, two positive points and one targeted negative gave the strongest masks in the
latest pipette study.** Put the positives inside the visible body and exposed shaft. Put the
negative inside a glove, rack or neighbouring object that the mask wrongly includes. Start with
fewer clicks and stop when the mask is good.

The study compared 19 prompt recipes on seven views of one blue pipette. Each recipe was decoded
independently, with all points supplied together and no previous-mask feedback. All 532 candidate
masks were inspected against the original images.

| Recipe, with a box | Mostly correct masks chosen by SAM3 / seven views | Measured against |
|---|---:|---|
| One body positive | 3 | One AI reviewer's visual grade of at least four out of five |
| Body and shaft positives, one targeted negative | 6 | Same visual rubric |
| Five positives | 2 | Same visual rubric |
| Eight positives | 2 | Same visual rubric |

SAM3 sometimes ranked a glove-contaminated mask above a cleaner candidate. More clicks did not
consistently help. These are exploratory visual judgments on correlated views, not ground-truth
IoU or tracking accuracy. Rack leakage and support-tube contamination remain unresolved.

The earlier initialization review sampled raw frames 0 through 600 every 100 in all six cameras.
After five passes, 26 of 42 masks were approximate usable drafts; 16 still need review. That
review uses a different rubric from the point study. Production tracking still starts at raw
frame 600, and the study masks have not been adopted.

Open the [interactive point comparison](runs/finebio-pipette-improvement-20260930/point-prompt-study/review.html)
on this machine. The [saved study report](docs/qa/pipette-checkpoint-20261001/point-prompt-study/README.md)
and [checkpoint notes](docs/qa/pipette-checkpoint-20261001/README.md) preserve the findings,
label selections, dependencies and steps to resume. The next experiment is validation on unseen
views, followed by sequential corrective clicks with previous-mask feedback.

Evidence: `runs/finebio-pipette-improvement-20260930/point-prompt-study/` and
`every100-review/`; [checkpoint](docs/qa/pipette-checkpoint-20261001/README.md), saved Oct 1 2026
in local commit `dbbad8a`. A 1.25 GB local archive preserves the reviews, masks, GIFs and Rerun
files. All 3,365 artifact files were verified against their hashes. The archive is on this
machine, and the checkpoint commit has not been pushed.

## What it does

Every static object on the bench gets one 3D identity from six cameras and a detector's boxes. This
worked in both rooms I tried. The pipeline has six parts.

- Camera solve: printed markers on the bench rank the dataset's shipped camera poses. Any pose more
  than 10 px off is re-solved from the markers to within 2 px. No camera is faked or dropped.
- Masks from boxes: the FineBio detector's box is the only prompt SAM3 gets, and a fresh mask is
  decoded from it on every frame in every camera. SAM3's video memory lost that comparison so it
  stays around as a second arm (arm is the lab's word for one candidate method).
- 3D tracks with persistent ids: any object two cameras can see is triangulated into one world
  frame and tracked by predict, project and gate. The gates are formulas on each trial's own data.
- Containers, held objects, groups: four opt-in extensions keep a tube's identity inside a closed
  centrifuge, follow an object in a hand and hold a rack of identical tubes as one group. They cut
  spurious track births by 40% and identity ambiguities by two-thirds.
- Confidence and events: every track row gets a label-free confidence rank or an abstention.
  `contained`, `held` and `proximity` episodes are just geometry on the tracks against the rig's
  volumes.
- Rerun review: one recording per trial with World, Cameras and Evidence presets, a storyboard on
  the timeline, seven cross-checks and a negative control.

The five numbers that matter, and what each one was measured against:

| Number | Value | Measured against | Where |
|---|---|---|---|
| Mask agreement, per-frame box decode, trial 1 | 0.928 median IoU, 99.3% of masks at 0.5 or above | the detector's own box on the same frame, model against model | `runs/finebio-arms-P03_03_01-filtered-20260927/scoreboard/` |
| Masks against my choice, trial 1 | 0.984 mean IoU, 98.8% at 0.5 or above | 325 cells on 18 frames where I picked the right SAM3 mask | `runs/finebio-anchors-P03_03_01-20260925/scoreboard/` |
| Static bench objects | one 3D identity each, every arm, both rooms | the tracker's own ids over the 120 s window | `<arm>/tracks-ext/identity_metrics.json` |
| Trial 2, room 2, nothing tuned | 0.919 median IoU, 98.5% at 0.5 or above | the detector's box, as in trial 1 | `runs/finebio-arms-P20_03_01-20260925/scoreboard/` |
| GPU time | about 2.3 h per trial for the two mask arms, 8.8 h for the phase | wall clock on one RTX 5070 Ti | `runs/finebio-arms-<trial>-20260925/README.md` |

## Day by day

Each line is the failure that caused the next change and links to that day in
[`docs/story.md`](docs/story.md). This table covers the first 22 days. Later work is summarized
in the checkpoint above. The only success clip is the one at the top. At gate 1 I accepted or rejected each seed box. At gate 2
I picked the right mask on a sample of frames.

| Day | What went wrong, and what it forced | Clip |
|---|---|---|
| [1](docs/story.md#sep-8-text-prompts-fail-on-the-head-camera) | Text prompts never found the toy wheel and only found a hand on the head camera, so I stopped prompting with words | <a href="docs/story.md#sep-8-text-prompts-fail-on-the-head-camera"><img src="media/story/2026-09-08-text-prompts-fail.gif" width="140" alt="Text prompts fail on both cameras"></a> |
| [2](docs/story.md#sep-9-box-seeds-replace-text-prompts) | Boxes drawn on frame 0 tracked four targets, then the first frame turned out to be a poor seed for a part that turns | <a href="docs/story.md#sep-9-box-seeds-replace-text-prompts"><img src="media/story/2026-09-09-box-seed-fix.gif" width="140" alt="Box seeds give both hands and both toy parts"></a> |
| [3 to 5](docs/story.md#sep-10-to-12-no-runs) | No runs | |
| [6](docs/story.md#sep-13-masks-leak-between-keyframes) | Masks leaked between keyframes. Corrections at three frames reset them, and every confidence had been reading 1.0 | <a href="docs/story.md#sep-13-masks-leak-between-keyframes"><img src="media/story/2026-09-13-keyframe-corrections.gif" width="140" alt="Keyframe corrections reset the leaking masks"></a> |
| [7](docs/story.md#sep-14-no-text-prompt-finds-the-black-base) | Five text prompts for the black base failed, so I picked its mask out of the decoder's candidates | [still](docs/story.md#sep-14-no-text-prompt-finds-the-black-base) |
| [8](docs/story.md#sep-15-stable-slots-unstable-identities) | Slots kept their ids while the masks inside them swapped parts, so the window moved to the reassembly | <a href="docs/story.md#sep-15-stable-slots-unstable-identities"><img src="media/story/2026-09-15-exploratory-identity.gif" width="140" alt="Chassis and cabin masks on the same part"></a> |
| [9](docs/story.md#sep-16-the-exploratory-queue-drifts-off-the-parts) | Two of the three challenger trackers drifted off the parts that SAM3 held on the same 20 s | <a href="docs/story.md#sep-16-the-exploratory-queue-drifts-off-the-parts"><img src="media/story/2026-09-16-exploratory-arms.gif" width="140" alt="Four trackers on the same 20 s"></a> |
| [10](docs/story.md#sep-17-the-first-human-review-finds-the-swap) | The first close review found the chassis and interior swapping as a hand sweeps past. The automatic correction missed it | <a href="docs/story.md#sep-17-the-first-human-review-finds-the-swap"><img src="media/story/2026-09-17-human-flags-swap.gif" width="140" alt="Interior leaks into the chassis and the labels swap"></a> |
| [11](docs/story.md#sep-18-eight-cameras-disagree) | Eight cameras flagged the chassis failure by disagreeing about it, and fixed nothing | <a href="docs/story.md#sep-18-eight-cameras-disagree"><img src="media/story/2026-09-18-eight-views-disagree.gif" width="140" alt="Eight cameras disagree about the chassis"></a> |
| [12](docs/story.md#sep-19-anchors-become-the-yardstick) | My 52 anchor masks replaced self-consistency and showed the swap window sitting at 0.0 to 0.6 IoU | [still](docs/story.md#sep-19-anchors-become-the-yardstick) |
| [13](docs/story.md#sep-20-other-cameras-detect-the-failure-and-cannot-fix-it) | Corrections from the other cameras got to the floor and never to the level of mine. A detector, not yet a corrector | <a href="docs/story.md#sep-20-other-cameras-detect-the-failure-and-cannot-fix-it"><img src="media/story/2026-09-20-interior-stays-human.gif" width="140" alt="Interior stays human"></a> |
| [14](docs/story.md#sep-21-finebio-arrives-and-text-fails-on-the-plate) | On the bench, text never found the transparent plate. The shipped detector boxed it on every frame | <a href="docs/story.md#sep-21-finebio-arrives-and-text-fails-on-the-plate"><img src="media/story/2026-09-21-text-fails-on-plate.gif" width="140" alt="Text prompts miss the transparent plate"></a> |
| [15 and 16](docs/story.md#sep-22-and-23-cleanup-then-nothing) | Cleanup, then nothing | |
| [17](docs/story.md#sep-24-camera-6-is-94-px-off-its-shipped-pose) | The top-down camera's shipped pose was 94 px off the markers. The preflight re-solved it and rewrote the plan | [still](docs/story.md#sep-24-camera-6-is-94-px-off-its-shipped-pose) |
| [18](docs/story.md#sep-25-the-3d-tracker-runs-in-both-rooms) | The tracker ran in both rooms. Video memory drifted and the pipette in the hand fragmented into 59 identities | <a href="docs/story.md#sep-25-the-3d-tracker-runs-in-both-rooms"><img src="media/story/2026-09-25-pipette-fragments.gif" width="140" alt="The pipette in the hand fragments"></a> |
| [19](docs/story.md#sep-26-gate-1-rejects-six-seeds) | Gate 1 rejected six seeds (two objects under one box, a glove, a table edge) and no verdict moved | [still](docs/story.md#sep-26-gate-1-rejects-six-seeds) |
| [20](docs/story.md#sep-27-gate-2-rejects-30-masks-and-the-redo-holds) | Gate 2 rejected 30 of 337 masks, almost all pipettes in a hand. The label-free gap shrank from 20 percentage points to 5 | [still](docs/story.md#sep-27-gate-2-rejects-30-masks-and-the-redo-holds) |
| [21 to 22](docs/story.md#sep-28-to-29-pipettes-as-3d-lines-and-where-it-stopped) | Pipettes tracked as 3D lines fit the shaft but cut the held pipette's ids 2.8x, not 5x, and named the wrong end where the hand covered it. Not adopted | <a href="docs/story.md#sep-28-to-29-pipettes-as-3d-lines-and-where-it-stopped"><img src="media/story/2026-09-28-pipette-lines.gif" width="140" alt="The line fits the shaft and its tip flips while the hand covers it"></a> |
| [22](docs/story.md#sep-29-reprojection-prompts) | The orientation vote names the cone. Reprojection masks still pick gloves and other pipettes, and the second pass is not adopted | [readout](docs/results.md#reprojection-prompts) |

## Recordings

Three recordings are worth opening. Each one has a World, a Cameras and an Evidence preset next to
it. They live under `runs/`, which is not committed, so they have to be rebuilt first with the
commands in [`docs/pipeline.md`](docs/pipeline.md#finebio-one-command-per-stage). Then run one of
these.

```bash
uv run rerun runs/finebio-review-P03_03_01-filtered-20260927/review.rrd runs/finebio-review-P03_03_01-filtered-20260927/world.rbl   # trial 1 on human-filtered seeds, with the extensions (recommended)
uv run rerun runs/finebio-review-P03_03_01-20260925/review.rrd runs/finebio-review-P03_03_01-20260925/world.rbl                     # trial 1, core tracker, the recording the guide describes
uv run rerun runs/finebio-review-P20_03_01-20260925-ext/review.rrd runs/finebio-review-P20_03_01-20260925-ext/world.rbl             # trial 2, with the extensions
```

The 20-minute route through both trials is in the
[review guide](docs/review-guide-2026-09-25-finebio-3d.md#what-to-look-at-first-20-minutes-both-trials).
It covers the negative control, the centrifuge story three ways, the transparent plate, the object
in the hand, a confidence drop that is a real failure, the decision and what did not transfer.

The earlier pipette initialization review has a separate Rerun recording. It shows sparse samples
at raw frames 0 through 600 and the selected masks before and after refinement:

```bash
uv run rerun runs/finebio-pipette-improvement-20260930/every100-review/refinement.rrd runs/finebio-pipette-improvement-20260930/every100-review/labels.rbl
```

The newer point study is in the interactive HTML comparison above. Both reviews are local outputs
under `runs/`; the checkpoint notes describe archive restoration.

## Notes

The pipeline numbers compare one model with another, or a model with my choice among its masks.
The point-study counts are one AI reviewer's visual judgments. None of these is accuracy against
ground truth. The long version of the pipeline claim boundaries is in
[`docs/results.md`](docs/results.md#claim-boundaries).

- The FineBio detector was trained on FineBio's own objects and on frames from these same cameras.
  Any IoU against its box is two models agreeing with each other, and how well it seeds here is an
  upper bound for a new bench.
- The anchors are 337 cells on 18 frames of one trial, my pick among SAM3's decoder masks. They
  rank the arms against each other and are not a dataset. The 103 cells with no
  detector box were left unlabeled.
- Three things are overfit to trial 1: the rig's witness class list (it put room 2's hand-off gate
  at its 80 px cap), the slot cap (it handed a rack the plate's slot) and the viewer's negative
  control (drawn for camera 6 only).
- The pipette in the hand fragments in both rooms (59 identities for blue and 48 for yellow, with
  the extensions on). I read it as an observation problem, since a long object seen from one or two
  cameras has no single 3D point at its box center. Tracking it as a 3D line cut that to 18 and 17
  and failed its own rule, so the observation is only part of it.
- Proximity events are zero by protocol, because no pipette comes near the plate in either window.
  The mechanism has only been tested on synthetic tracks.
- Trial 2 has no human gate. Its 66 seed tiles are undecided, no anchor set was chosen and its
  numbers rest on the detector's own seeds.

License: FineBio is gated, non-commercial research data and Assembly101 is CC BY-NC 4.0. Nothing
under `data/` or `runs/` is committed. The only dataset pixels in Git are the story media under
`media/story/`, and the repo is private. [`docs/LICENSES.md`](docs/LICENSES.md#committed-media-sep-27)
has the terms they carry and says they must be removed before any redistribution.

## Open problems

- The pipette in the hand. I tried tracking it as a 3D line fitted to the mask axes, with a
  length prior and a hold on the hand. The ids fell 2.8x (51 to 18), not the 5x I pre-registered,
  and the tip landed 3 cm from my clicks against a 2 cm rule, so I did not adopt it. The line
  fits the shaft, and the ids die where the hand covers it. The next lever is in the tracker:
  carry a held pipette on the hand and refuse a new id until the hand lets go. Named, not built.
- Room 2 has no human gate. Gate 2 was only run on trial 1, so room 2 has no number anchored on
  my choices. The web workspace for it exists and the sitting would take about 1.5 h.
- Production tracking parameters have not been swept. Frames before a re-seed, the memory write
  threshold, the hysteresis widths, the window length and the container heights remain at their
  first values. The point-prompt comparison above tests mask initialization only.
- Cost per trial. About 2.3 h of GPU for the two mask arms on one RTX 5070 Ti, 25 minutes for the
  detector pass, 8 minutes of CPU for the recording and about 2 h of my time for the two gates.

Turning this into a product would start with the three things the lab took from the dataset: a
detector trained on the target bench's objects, printed markers to solve the cameras and a list
of containers with their heights for each bench. The camera solve, the rig check, the tracker and
the review recording all moved to a second room with the trial id as the only change, and the
three overfit items above are the first patches. The two review gates would have to become
routine instead of a sitting.

## Structure

`src/battle/` is one package of about 120 modules. The console scripts are in `pyproject.toml` and
every one of them answers `uv run battle-<name> --help` on the CPU. By role:

- FineBio pipeline: `finebio_cameras*.py` (camera solve), `finebio_preprocessing.py` (proxies and
  clip configs), `finebio_detect.py` (the shipped detector), `finebio_rig.py` (gates as
  formulas), `detector_seed.py` (slots, seeds, gate 1 sheets), `finebio_arms.py` (arms and
  scoreboard), `multiview_tracks.py` (the 3D tracker and its four extensions),
  `finebio_confidence.py`, `finebio_events.py`, `finebio_viewer.py`, `finebio_anchors*.py`
  (gate 2), `finebio_slice.py`, `finebio_frames.py`, `finebio_observations.py`, `multiview_schemas.py`.
- SAM3 workers: `muggled_worker.py` (runs in the MuggledSAM interpreter), `muggled_arms.py`
  (box decode and video memory drivers), `muggled_smoke.py`, `muggled_calibration*.py` (the
  browser seed workspace), `muggled_agent_correction.py`, `worker_common.py`, `video_driver.py`,
  `mask_ops.py`, `mask_cache.py`.
- Assembly101 phase: `assembly101_*.py` (fetch, clocks, cameras, dataset poses),
  `multiview_geometry.py`, `multiview_consensus.py`, `multiview_visual_hull.py`,
  `multiview_reprompt.py`, `multiview_seed_transfer.py`, `review_anchors.py`,
  `review_metrics*.py`, `detector_scorecard*.py`, `exemplar_*.py`, `seed_search.py`,
  `human_seeds.py`, `ensemble_reference*.py`, `interaction_review*.py`, `four_part_*.py`,
  `policy_ablation.py`, `correction_acceptance_search.py`.
- Other methods, smoke tier: `mediapipe_hands.py`, `wilor_*.py`, `boxmot_*.py`, `drop_dtw_*.py`,
  `grounding_dino_sam2_video*.py`, `samurai_video*.py`, `dam4sam_*.py`, `athena_*.py`,
  `kineo_*.py`, `egoexo_correspondence.py`, `sam3_appearance.py`.
- Shared: `schemas.py` (the versioned contracts), `observations.py`, `exporter.py`,
  `rerun_logging.py`, `review_presets.py`, `contact_sheet.py`, `gpu_guard.py`,
  `overnight_queue.py`, `digest_cache.py`, `media_probe.py`, `story_media.py` (renders
  `media/story/`).
- `tests/`: the default tier needs nothing outside the repo and runs in about 25 s. The
  `real_data`, `gpu` and `slow` markers gate the rest.
- `configs/`: `finebio/` (cameras, trials), `clips/`, `rerun/` (the presets), `qa/`, `story/`,
  and the Assembly101 seed and correction policies.
- `scripts/`: the detector installer, the GPU queue, the run pruner, the Tailscale serve script
  and `archive/`.
- `docs/`: the index is [`docs/README.md`](docs/README.md). Everything historical is kept verbatim
  under [`docs/archive/`](docs/archive/README.md), including the
  [lab notebook this page replaced](docs/archive/README-lab-notebook-2026-09-27.md) and the
  [method ledger](docs/archive/method-ledger.md).

## Setup

CPython 3.12 through `uv` (`uv sync --python 3.12`) and one RTX 5070 Ti are required. The FineBio
detector gets its own CUDA venv from `scripts/install_finebio_detector.sh --cuda` and SAM3 runs in
the separately managed MuggledSAM interpreter. Everything else is one command per stage and can be
found in [`docs/pipeline.md`](docs/pipeline.md#setup).
