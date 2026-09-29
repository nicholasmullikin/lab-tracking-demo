# The story in 20 days

This page tells the lab's 20 days one day at a time, and each section opens with that day's
verdict. The clips are struggle-first: each shows the failure that drove the next change, and
the one success clip is the tube that keeps its identity through a closed centrifuge lid.
Every section ends with the number that changed and where the evidence sits; the lab's
shorthand is in the [glossary](writing-style.md#glossary) and the numbers in full are in
[`results.md`](results.md).

## Sep 8: text prompts fail on the head camera

**Text prompts could not name the toy parts, and on the monochrome head camera they found only
a hand.** Three prompts (`hand`, `yellow toy body`, `toy wheel`) ran through SAM3 on 300 frames
of each camera. The static camera started two of the three concepts and never the wheel. The
head camera started `hand` alone, with a track that wandered. By midnight the plan had changed:
draw boxes, inspect what the decoder returns for them, stop prompting with words.

![Text prompts fail: 'toy wheel' never appears and the ego view finds only the hand](../media/story/2026-09-08-text-prompts-fail.gif)

The number that changed: concepts started by text, 2 of 3 on the static camera and 1 of 3 on
the head camera.

Evidence: `runs/muggledsam-sam3-smoke-static-c10379-20260909t025854z/` and
`runs/muggledsam-sam3-smoke-ego-hmc21110305-20260909t025910z/`; archive entries
[core-method smoke](archive/method-ledger.md#sep-8-late-evening-muggledsam-sam3-core-method-smoke-g2)
and [ego monochrome diagnostic](archive/method-ledger.md#sep-8-late-evening-muggledsam-sam3-ego-monochrome-diagnostic).

## Sep 9: box seeds replace text prompts

**Boxes drawn on frame 0 gave the head camera both hands and both toy parts, and then the first
frame proved a poor seed.** A browser workspace replaced an OpenCV prompt tool within an hour,
and a second head camera (e4) was picked because it started every concept. Four seeded targets
covered 98 to 100% of the 5,400 frames. Watching the result beside the static camera showed the
tracked "toy wheel" was the wrong part, so the vocabulary changed. A seed drawn on one frame
does not describe a part that turns, which is what the next run addressed.

![Text prompts gave one hand; human box seeds on frame 0 give both hands and both toy parts](../media/story/2026-09-09-box-seed-fix.gif)

The number that changed: targets tracked on the head camera, 1 to 4, at 98 to 100% frame
coverage over 180 s.

Evidence: `runs/muggledsam-sam3-full-ego-manual-seed-multiplexed-ego-hmc21179183-20260910t024052z/`;
archive entries [four-target baseline](archive/method-ledger.md#sep-9-evening-four-target-180-second-ego-baseline)
and [the head-camera screen](archive/run-report-assembly101-ego-viewpoint-screen.md).

## Sep 10 to 12: no runs

- Sep 10: the labelling workspace's two-step flow became one "Finalize tracking plan" action. No run.
- Sep 11: no work.
- Sep 12: no work.

## Sep 13: masks leak between keyframes

**Seeded masks on the head camera leaked from one part to the next between keyframes, and
corrections at three later frames reset them.** With corrections at frames 75, 150 and 240,
three of four targets were present on every frame of the 300-frame clip, the best head-camera
result so far. The same evening found that the worker had thrown away SAM3's own score and
clamped its presence logit, so every observation read confidence 1.0. Recording both signals
split the failures in two: one loss the model knew about, and five where it was confidently
wrong. A resolution sweep fixed the encoder side at 720 px, and the repository got its first
commits after five days of work.

![Ego masks leak between keyframes; human corrections at 75, 150 and 240 reset them](../media/story/2026-09-13-keyframe-corrections.gif)

The number that changed: the weakest target's IoU rose 44% from 504 to 720 px encoder side,
and 1,193 observations stopped reading confidence 1.0.

Evidence: `runs/muggledsam-sam3-smoke-multi-keyframe-corrections-ego-hmc21179183-20260913t221457z/`;
archive entries [multi-keyframe corrections](archive/method-ledger.md#sep-13-evening-muggledsam-sam3-e4-multi-keyframe-human-corrections)
and [tracker diagnostics](archive/method-ledger.md#sep-13-evening-tracker-score-and-iou-diagnostics).

## Sep 14: no text prompt finds the black base

**Five text prompts for the black toy base all failed, one of them on an unrelated black
fixture, so the base was seeded from a mask I picked among the decoder's
candidates.** The static camera's hybrid run kept text prompts for both hands and the yellow
top and used that one reviewed mask for the base, over all 5,400 frames. Reviewing the toy piece
by piece then showed the composite names were wrong for the task. The vocabulary narrowed to
four physical parts: chassis, interior, rear body and cabin.

![No text prompt finds the black base; I pick its frame-0 mask from decoder candidates](../media/story/2026-09-14-black-base-candidates.png)

The number that changed: text prompts tried for the black base, 5, accepted 0; the target list
went from 2 composite names to 4 physical parts.

Evidence: `runs/muggledsam-sam3-static-black-toy-top-base-calibration-20260914t022125z/` and
`runs/muggledsam-sam3-g3-full-hybrid-static-static-c10379-20260915t005919z/`; archive entry
[aligned static hybrid](archive/method-ledger.md#sep-14-aligned-static-hybrid-candidate).

## Sep 15: stable slots, unstable identities

**Four tracked slots kept their ids for 5,901 frames while the masks inside them swapped
parts.** In the exploratory full run the chassis and cabin masks sat on the same object at
about 0.9 IoU within the first minutes, so a stable id said nothing about a stable identity.
The window moved to the point where all four parts start separated and are then assembled,
92.7 s of video, with a fresh seed and corrections at three later frames.

![Stable slots, unstable identities: chassis and cabin masks land on the same part](../media/story/2026-09-15-exploratory-identity.gif)

The number that changed: frames where two parts' masks overlapped above 0.5 IoU, 111 in the
exploratory run to 1 in the focused run.

Evidence: `runs/muggledsam-sam3-four-part-static-full-exploratory-static-c10379-20260916t012945z/`
and `runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260916t023700z/`;
archive entry [four-part static reassembly](archive/method-ledger.md#sep-15-four-part-static-reassembly-experiment).

## Sep 16: the exploratory queue drifts off the parts

**Eight more methods ran in one unattended pass, and on the same 20 s two of the three
challenger trackers drifted off the parts SAM3 held.** Grounded-SAM-2 with an open vocabulary
and SAMURAI with the reviewed seeds both wandered; DAM4SAM held and then slipped near frame 330.
WiLoR, BoxMOT, CLIP with Drop-DTW, ATHENA and Kineo reached smoke tier: ATHENA stalled on
missing intrinsics and Kineo produced a body only. MediaPipe hands became the second core
method, and one review recording gathered everything.

![Grounded-SAM-2 and SAMURAI drift off the parts; SAM3 with reviewed seeds holds](../media/story/2026-09-16-exploratory-arms.gif)

The number that changed: methods attempted, 1 to 10 in one day; on the shared 20 s, 1 of 3
challengers stayed on all four parts.

Evidence: `runs/grounding-dino-sam2-open-vocabulary-four-part-20s-20260916t1004z/`,
`runs/samurai-four-part-reviewed-seed-20s-20260916t1007z/` and
`runs/dam4sam-four-part-reviewed-seed-20s-20260916t1009z/`; archive entries
[exploratory queue](archive/method-ledger.md#sep-16-exploratory-queue-autonomous-pass) and
[four-part comparison](archive/method-ledger.md#sep-16-focused-static-four-part-segmentation-comparison).

## Sep 17: the first human review finds the swap

**The first close review of the minute named the failures the metrics had not: the interior
leaks into the chassis as the part turns, and the two labels swap as a hand sweeps past.** An
automatic correction at frame 1172 landed inside the swap and did not undo it; a correction I
picked at 1235 did. A correction at the leak onset (frame 1020) did not hold either.
The first minute became the comparison unit, DAM4SAM ran over it, and a per-part reference took
DAM4SAM's chassis inside the failing interval. Label-free review metrics ranked 198 episodes for
the next review.

![Interior leaks into the chassis from 1020 and the labels swap; a human correction at 1235 undoes it](../media/story/2026-09-17-human-flags-swap.gif)

![The v4 review's pinned moments: swap, leaks and hand drop-outs I was asked to judge](../media/story/2026-09-17-v4-review-sheet.png)

The number that changed: the comparison unit shrank from 92.7 s to the first 60 s, and the
interior was marked hidden over frames 1024 to 1172.

Evidence: `runs/interaction-review-first-minute-v4/` and
`runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t001210z/`;
archive entries [v4 human review](archive/method-ledger.md#sep-17-first-minute-v4-human-review-and-follow-up-rebuild)
and [leak onset](archive/method-ledger.md#sep-1718-leak-onset-correction-attempt-20-s-v5-hand-layer-dam4sam-first-minute).

## Sep 18: eight cameras disagree

**Eight static cameras on one rig flagged the chassis failure by disagreeing about it, and
fixed nothing.** Overnight, the seven other static views were fetched, their clocks and cameras
fitted, and SAM3 run on each from seeds transferred by geometry. A cross-view consensus and a
carved visual hull both threw C10379's chassis out over frames 1620 to 1700, and both flagged
the swap window the review had named. ATHENA triangulated hands against the dataset's, Kineo ran
on all eight cameras, and LM-EEC mapped parts into the head camera. Every number from that
night is disagreement between estimates.

![Eight cameras disagree: C10379's chassis is thrown out of the consensus over frames 1620-1700](../media/story/2026-09-18-eight-views-disagree.gif)

The number that changed: cameras on one timeline, 2 to 12, with 25 queued GPU jobs and 25
successes.

Evidence: `runs/multiview-static-comparison-first-minute/` and
`runs/muggledsam-sam3-four-part-multiview-first-minute-static-*/`; archive entries
[cross-view SAM3](archive/method-ledger.md#sep-18-track-2-of-the-overnight-multicam-pass-cross-view-sam3-with-a-geometric-combiner)
and [visual hulls](archive/method-ledger.md#sep-18-track-5-of-the-overnight-multicam-pass-per-part-visual-hulls-from-eight-silhouettes).

## Sep 19: anchors become the yardstick

**My 52 anchor masks replaced self-consistency as the yardstick, and the first thing they
showed was the swap window.** Chassis and interior fell to 0.0 to 0.6 IoU on frames
1050 to 1150 for the reference and the best arm alike. The label-free proxies had also been
wrong about resolution: 1280 px beat 720 px on the anchors and 1920 px saturated. Appending
corrections to SAM3's prompt memory was the one memory change that helped. DAM4SAM under the
same schedule tied SAM3 at 1280 px (0.715 against 0.724). Nineteen arms went into one table.

![Against my anchors, chassis and interior fall to 0.0-0.6 IoU on frames 1050-1150](../media/story/2026-09-19-anchors-swap-window.png)

The number that changed: the best arm scored 0.743 mean IoU on the 52 anchor cells against the
old reference's 0.668.

Evidence: `runs/anchor-scoreboard-20260919/`; archive entries
[review anchors](archive/method-ledger.md#sep-18-human-review-anchors-labeling-run-prepared-not-labelled-sep-18-labelled-sep-19),
[memory arms](archive/method-ledger.md#sep-19-sam3-correction-memory-arms-on-c10379-at-1280-plan-step-1b)
and [anchor scoreboard](archive/method-ledger.md#sep-19-anchor-scoreboard-over-every-first-minute-arm-plan-step-3-scoring-half).

## Sep 20: other cameras detect the failure and cannot fix it

**Corrections proposed by the other cameras reached the seed-only floor and never the level
of my corrections, so multicam is a detector, not yet a corrector.** SAM3's own object score
ranked its failures (AUROC 0.91 to 0.96) but no threshold transferred. Seeds for the rear body
and cabin transferred from other cameras; the chassis and interior did not. On the interior
window the consensus corrections scored 0.29 and 0.32 IoU against the anchors, where mine
scored 0.64 and 0.59. A second recording ran with zero human input and stayed
unscored, and the review surface became one recording with three presets.

![Interior stays human: consensus corrections from the other cameras reach 0.29-0.32 IoU, mine 0.59-0.64](../media/story/2026-09-20-interior-stays-human.gif)

![The v6 review recording under its segmentation preset: reference, provenance and six candidate arms](../media/story/2026-09-20-v6-review.png)

The number that changed: corrections from the other cameras reached 0.659 mean IoU against
0.743 with my corrections, and 0.288 against 0.585 on the interior.

Evidence: `runs/multiview-reprompt-20260920/` and `runs/interaction-review-first-minute-v6/`;
archive entries [the consensus re-prompt loop run](archive/method-ledger.md#sep-20-b4-arms-on-c10379-the-consensus-re-prompt-loop-run-multicam-plan-headline-commits-78f2eca-21da378-cf8007e)
and [v6 review surface](archive/method-ledger.md#sep-20-v6-review-surface-one-recording-with-three-blueprint-presets-track-d-of-the-multicam-plan).

## Sep 21: FineBio arrives and text fails on the plate

**On the wet-lab bench SAM3's text prompts never found the transparent 6-well plate or the tube
rack, and FineBio's shipped detector boxed both on every frame.** One 20 s head-camera clip and
five prompts: three concepts started at frame 0, the plate and the racks found nothing, and the
pipette was lost at frame 106. The detector, run on the CPU beside it, put `cell_culture_plate`
on all 600 frames and six rack classes on the racks. The box became the way in. The same night
the Assembly101 labelling sessions were acted on and a distractor guard added.

![SAM3's text prompt never finds the transparent 6-well plate and loses the pipette at frame 106](../media/story/2026-09-21-text-fails-on-plate.gif)

![The shipped FineBio detector boxes the plate and tubes the text prompt could not](../media/story/2026-09-21-detector-boxes.png)

The number that changed: frames with the plate found, 0 of 600 by text prompt and 600 of 600 by
the detector's box.

Evidence: `runs/finebio-sam3-smoke-20260921/` and `runs/finebio-dino-20260921/`; archive
entries [FineBio first look](archive/method-ledger.md#sep-21-finebio-first-look-sam3-zero-shot-text-prompts-on-one-first-person-clip)
and [FineBio shipped detector](archive/method-ledger.md#sep-21-finebio-shipped-detector-first-run-mmdetection-dino-and-deformable-detr-on-the-cpu-beside-the-sam3-smoke).

## Sep 22 and 23: cleanup, then nothing

- Sep 22: SAM3's own appearance signals were scored as failure detectors, and three deduplication passes removed 1,675 lines with equivalence proven on three artifacts. No new result.
- Sep 23: no work.

## Sep 24: camera 6 is 94 px off its shipped pose

**The preflight found the top-down camera's shipped pose 94 px off the bench markers, and
re-solving it from the markers brought it to 0.7 px.** The Assembly101 phase closed first,
tagged, with 3.67 GB of unreferenced runs deleted. Then every assumption behind the FineBio plan
was checked before anything ran: the camera mapping, synchronisation within one frame, static
objects triangulated to half their heights, and the transparent plate masked in all six views
from the detector's box. The in-hand object was the weakest detection in every fixed view. The
plan was rewritten on what the preflight found, the trial windows chosen, and the detector built
for the GPU.

![Camera 6's shipped pose (green) sits 94 px off the markers (red); marker PnP re-solves it to 0.7 px](../media/story/2026-09-24-camera6-pose.png)

The number that changed: camera 6's marker residual, 94 px shipped to 0.7 px re-solved.

Evidence: `runs/preflight-finebio-20260924/`; archive entries
[the preflight](archive/preflight-2026-09-24-finebio.md),
[Assembly101 phase closed](archive/method-ledger.md#sep-24-assembly101-phase-closed) and
[plan approved, trial windows chosen](archive/method-ledger.md#sep-24-night-plan-approved-trial-windows-chosen-p0-trials-p-docs).

## Sep 25: the 3D tracker runs in both rooms

**A six-camera 3D tracker ran end to end on trial 1 and then, with nothing tuned, on trial 2;
per-frame box decode beat SAM3's video memory, and the pipette in the hand fragmented.**
Cameras, rig, proxies, seeds, detector runs on 86,400 frames, four arms and the tracker went
from fixtures to a review recording in one day. A fresh mask decoded from the detector's box on
every frame agreed with the box on 99.1% of masks; video memory, seeded once, on 79.3%, and
re-seeding it made it worse. Four extensions (motion model, containers, groups, held objects)
cut spurious track births by 40% (308 to 186) and identity ambiguities by two thirds (158 to
54). A micro tube kept one identity through both closed-lid spins. The blue pipette in the hand
split into 59 identities in two minutes.

![One tube, one identity, through a closed centrifuge lid](../media/story/2026-09-25-finebio-3d.gif)

![Video memory drifts off the micro tube; per-frame box decode holds](../media/story/2026-09-25-video-memory-drifts.gif)

![The pipette in the hand splits into new identities every few seconds (59 ids in two minutes)](../media/story/2026-09-25-pipette-fragments.gif)

Trial 2 (room 2, recording `P20_03_01`) went through the same pipeline with the trial id as the
only change. Four of its cameras were 15 to 32 px off their shipped poses and were re-solved to
0.7 to 2.1 px. Per-frame box decode agreed with the detector on 98.5% of masks, video memory on
65.3%. The static objects were one identity each, two tubes kept their ids through three spins,
and the yellow pipette in the hand fragmented into 48. Three things were named as overfit to
room 1: the rig's witness class list, which put the hand-off gate at its 80 px cap; the slot
cap, which gave a rack the plate's slot; and the viewer's negative control drawn for camera 6
only.

![The Sep 25 review recording under its World preset: six cameras, containers and arm (b) tracks](../media/story/2026-09-25-finebio-world-view.png)

The number that changed: masks agreeing with the detector's box at 0.5 IoU or above, 99.1%
per-frame decode against 79.3% video memory in room 1, and 98.5% against 65.3% in room 2.

Evidence: `runs/finebio-arms-P03_03_01-20260925/`, `runs/finebio-arms-P20_03_01-20260925/` and
`runs/finebio-review-P03_03_01-20260925/`; archive entries
[tracking arms on trial 1](archive/method-ledger.md#sep-25-tracking-arms-on-trial-1-p4-arms-p1-rig-run),
[tracker extensions](archive/method-ledger.md#sep-25-tracker-extensions-on-the-inventorys-evidence-p3-tracker-ext)
and [second trial with zero tuning](archive/method-ledger.md#sep-25-second-trial-p20_03_01-with-zero-tuning-p6-trial2).

## Sep 26: gate 1 rejects six seeds

**I accepted 52 of the 60 detector seeds, rejected 6 and left 2 undecided, and the arithmetic
said the verdicts would not move.** Three rejects were two objects under one box, one
had a glove inside the mask, one caught the table edge, and one was hidden by a hand on its seed
frame. The six slots were 9% of the SAM3 rows; removing them would lift per-frame box decode
from 99.1% to about 99.4% and video memory from 79.3% to about 81.4%. The same day the gate 2
workspace became web pages with one keypress per row, because a 7,866-line decisions file was
not going to be edited by hand.

![Gate 1 rejects six seeds: two objects under one box, a glove inside the mask, a table edge](../media/story/2026-09-26-gate1-rejected-seeds.png)

The number that changed: seed slots kept on trial 1, 66 to 60 with the plate, and no verdict
moved.

Evidence: `runs/finebio-seeds-P03_03_01-20260925/` and
[`qa/finebio-P03_03_01-seed-decisions.human-record.json`](qa/finebio-P03_03_01-seed-decisions.human-record.json);
archive entries [gate 1 held](archive/method-ledger.md#sep-26-gate-1-held-seed-decisions-on-trial-1-p2-gate1)
and [gate 2 web workspace](archive/method-ledger.md#sep-26-gate-2-web-workspace-battle-finebio-anchors-web).

## Sep 27: gate 2 rejects 30 masks and the redo holds

**On 337 anchor cells I took per-frame box decode's own mask 307 times and rejected it on 30,
almost all pipettes in a hand; the ranking held with a smaller gap.** Against the
accepted masks, per-frame box decode scored 0.984 mean IoU and 98.8% at 0.5 or above, video
memory 0.897 and 93.5%. The 20-point label-free gap became 5 points, because the 18 frames
sample video memory's dead slots instead of counting every frame of them. Video memory won
identity through the centrifuge lid, one id where the per-frame decode had two. Trial 1 was then
redone on the seeds gate 1 kept: per-frame decode 0.928 and 99.3%, video memory 0.915 and
80.4%, its drift moved between slots rather than removed. Finally, 270 GB of re-seed checkpoints
were deleted.

![Gate 2 rejects arm b's mask on 30 of 337 cells: pipettes in the hand, a hidden object, a wider fit](../media/story/2026-09-27-gate2-rejected-masks.png)

The number that changed: the gap between per-frame box decode and video memory on masks at 0.5
IoU or above, 20 points against the detector's box to 5 points against my choice.

Evidence: `runs/finebio-anchors-P03_03_01-20260925/scoreboard/` and
`runs/finebio-arms-P03_03_01-filtered-20260927/`; archive entries
[gate 2 held](archive/method-ledger.md#sep-27-gate-2-held-the-anchor-scoreboard-on-trial-1-p6-anchors)
and [the redo on the filtered seeds](archive/method-ledger.md#sep-27-trial-1-downstream-redone-on-the-human-filtered-seeds-post-gate).
