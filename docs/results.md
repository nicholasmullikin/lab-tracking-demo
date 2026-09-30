# Results

This page holds the numbers the lab produced, what each was measured against and where it sits on
disk. The FineBio phase comes first because it is the current result, then the Assembly101 phase,
then the goals scorecard in the founder's words and the claim boundaries. The lab's shorthand
(arm (b), slot, gate 1, LOO) is mapped to plain words in the [glossary](writing-style.md#glossary).

## FineBio: one 3D identity per object, in two rooms

**Six cameras on a wet-lab bench become one world frame, and every static object on it gets one
3D identity, in both rooms.** The FineBio detector's box is SAM3's only prompt. A fresh mask decoded
from that box on every frame agrees with the detector on 99.3% of trial 1's masks (median IoU 0.928)
and 98.5% of trial 2's (0.919). SAM3's video memory, seeded once, agrees on 80.4% and 65.3%, so
the memory-free decode is the mask source. Four tracker extensions (motion model, containers, groups,
held objects) cut spurious track births by 40% and identity ambiguities by two-thirds on trial 1,
and by a similar amount in room 2 with nothing tuned. A tube keeps its identity through a closed
centrifuge lid. The one failure is the pipette in the hand, which splits into 50 to 60 identities
over two minutes in both rooms. Tracking it as a 3D line cut that to 18 and 17 and failed its
own rule, so the line tracker is not adopted (its section below). Every IoU here is agreement
between SAM3 and a detector trained on this bench, not accuracy; the boundaries are in the last
section.

Evidence: `runs/finebio-arms-P03_03_01-filtered-20260927/scoreboard/scoreboard.md`,
`runs/finebio-arms-P20_03_01-20260925/scoreboard/scoreboard.md`, archive entries "p-docs" (the
close-out) and "post-gate" (the redo on human-filtered seeds) in
[`archive/method-ledger.md`](archive/method-ledger.md).

## The two trials, same build, nothing tuned

**Room 2 reproduced every room-1 verdict with the trial id as the only change.** Both columns are
the runs on the detector's own seeds, so the rooms sit on one footing; trial 1's redo on
human-filtered seeds follows in the next section. Arms are named in plain words: per-frame box
decode is arm (b), video memory is arm (c), boxes only is arm (a). "Core" is the tracker without
the four extensions.

| Measure | Trial 1 (room 1) | Trial 2 (room 2) | Measured against | Where |
|---|---|---|---|---|
| Cameras re-solved from markers | one (T5), 0.72 px | four (T1 to T4), 0.7 to 2.1 px | ArUco marker corners on three frames per camera; the shipped poses read 93.7 px and 14.7 to 32.2 px | `runs/finebio-cameras-<trial>-20260925/mapping.md` |
| Association gate, from the formula | 30.1 px | 27.5 px | three times the median leave-one-camera-out residual of the static objects (10.0 and 9.2 px) | `runs/finebio-rig-<trial>-600-4200/rig.json` |
| Hand-off gate into the head camera | 27.0 px | 80.0 px, the cap | the p90 held-out residual of the moving witness objects; the cap is overfit item 1 below | same |
| Seeds the rule accepted | 66 of 66 | 66 of 66, plate unseen in T2 | SAM3 mask box against detector box at 0.6 or above | `runs/finebio-seeds-<trial>-20260925/with-plate/seeds.md` |
| Mask agreement, per-frame box decode | 0.926 median, 99.1% at 0.5 or above, 148,362 masks | 0.919, 98.5%, 141,159 masks | mask box against the best same-class detector box of the frame, model against model | `runs/finebio-arms-<trial>-20260925/scoreboard/scoreboard.md` |
| Mask agreement, video memory | 0.914, 79.3% | 0.883, 65.3% | same | same |
| Mask agreement, video memory plus re-seeds | 0.911, 75.7%, worse than either | not run, negative in room 1 | same | same, `decision_c_vs_b.json` |
| Track births, per-frame box decode, core to extensions | 308 to 186, down 40% | 321 to 202, down 37% | the tracker's own count over the 120 s window | `<arm>/tracks/identity_metrics.json`, `<arm>/tracks-ext/identity_metrics.json` |
| Track births, boxes only | 285 to 191 | 290 to 166 | same | same |
| Track births, video memory | 270 to 147 | 208 to 152 | same | same |
| Identity ambiguities, per-frame box decode | 158 to 54, down two-thirds | 97 to 35 | same | same |
| Identity ambiguities, boxes only | 126 to 33 | 70 to 8 | same | same |
| Identity ambiguities, video memory | 133 to 22 | 25 to 7 | same | same |
| Static bench objects (plate, centrifuge, vortex, PCR machine, trash can) | one id each, every arm | one id each, every arm | the tracker's ids over the window | same |
| Pipette in the hand, ids with the extensions | blue: 59 per-frame decode, 41 boxes only, 20 video memory | yellow: 48, 24, 21 | same | same |
| Tube kept as one id through the closed centrifuge | one tube, both closures | two tubes, three spins | events geometry on 3D tracks against the rig's container volume | `<arm>/events-ext/events_summary.json` |
| GPU time for the trial's arms | about 4.0 h, 1.8 h of it the re-seed arm | about 2.3 h | wall clock on one RTX 5070 Ti | `runs/finebio-arms-<trial>-20260925/README.md` |

The whole phase cost about 7.6 h of GPU through the close and 1.3 h more for the redo below, about
8.8 h in all. One trial's two mask arms cost about 2.3 h.

Evidence: the paths in the table; archive entries "p6-trial2" (the second trial) and "p-docs" (the
close-out).

## After the two human gates

**Both gates were held on trial 1, and neither moved a verdict.** At gate 1 I accepted 52 seed
tiles, rejected 6 and left 2 undecided out of 60; the rejects were pipettes under the wrong color
word and a tube group under one mask. At gate 2 I labeled 337 of the 440 anchor cells, every cell
that had a detector box. Trial 1 was then redone downstream of both gates.
The per-frame decode is a row filter of its own masks, so it changed only by arithmetic. Video
memory was re-run per camera on the filtered seeds, 1.26 h of GPU.

| Measure, trial 1 | Detector's seeds | Human-filtered seeds | Measured against | Where |
|---|---|---|---|---|
| Mask agreement, per-frame box decode | 0.926 median, 99.1% | 0.928, 99.3% | mask box against detector box, model against model | `runs/finebio-arms-P03_03_01-filtered-20260927/scoreboard/scoreboard.md` |
| Mask agreement, video memory | 0.914, 79.3% | 0.915, 80.4% | same | same |
| Dead video-memory slots (median IoU under 0.35) | 9 of 58 | 10 of 53 | same | `c-video-memory-arm/measures.md` |
| Track births with extensions, per-frame decode | 186 | 183 | the tracker's own count | `b-box-decode-arm/tracks-ext/identity_metrics.json` |
| Identity ambiguities with extensions, per-frame decode | 54 | 35 | same | same |
| Blue pipette ids with extensions, per-frame decode | 59 | 51 | same | same |
| Blue pipette ids with extensions, video memory | 20 | 30 | same | `c-video-memory-arm/tracks-ext/identity_metrics.json` |

One finding came out of the redo. Removing a seed from a multiplexed video-memory run changes the
other slots of that camera, because they share one memory bank. Video memory's drift moved between
the head camera's tube and pipette slots rather than going away.

Evidence: `runs/finebio-arms-P03_03_01-filtered-20260927/README.md`,
`qa/finebio-P03_03_01-seed-decisions.human-record.json`, archive entries "p2-gate1" (gate 1 held)
and "post-gate" (the redo).

## What the masks I chose say

**On human-chosen masks the ranking holds: per-frame box decode first, video memory second,
re-seeding third, with a gap of 5 percentage points instead of 20.** The 18 anchor frames sample
video memory's dead slots instead of counting every frame of them, which is where the 20 points
went. Candidate 0 on every cell was the per-frame decode's own mask, and I accepted it on 307 of
325 mask cells, so that arm scores 1.0 there by construction. Masks beat the detector's boxes on
the static bench objects and lose on the tubes and the held pipettes.

| Arm | Mask IoU against accepted mask, mean | Masks at 0.5 or above | Pooled IDF1 on six named objects | Measured against | Where |
|---|---|---|---|---|---|
| Per-frame box decode | 0.984 | 98.8% | 0.921 | 325 cells, my choice among SAM3 decoder masks on 18 frames of trial 1 | `runs/finebio-anchors-P03_03_01-20260925/scoreboard/anchor_scoreboard.md` |
| Video memory | 0.897 | 93.5% | 0.962 | same | same |
| Video memory plus re-seeds | 0.871 | 91.7% | 0.950 | same | same |
| Boxes only (no masks) | box IoU 0.903 mean, 0.929 median | not measured | 0.956 | the detector box against the accepted mask's box, 307 cells | same |
| Per-frame box decode, human seeds | 0.988 | 99.0% | 0.919 | the 312 cells not on a rejected seed | `.../scoreboard-filtered/anchor_scoreboard.md` |
| Video memory, human seeds | 0.903 | 93.9% | 0.965 | the 311 such cells | same |

The one identity disagreement the record can see is the centrifuge: one id in video memory, two in
the per-frame decode across the lid cycles. Video memory's label-free win, identity through
appearance change, holds on a human name. The six named objects are all static but one, so the
pipette's fragmentation has no human-anchored reading.

Evidence: `qa/finebio-P03_03_01-review-anchors.human-record.json` (the record without pixels),
archive entry "p6-anchors" (gate 2 held).

## Pipettes as 3D lines

**Tracking the pipettes as 3D line segments fitted to the mask axes cut the held pipette's ids
2.8x, not the 5x I pre-registered, and I did not adopt it.** Each elongated mask contributes a
plane through its camera and each compact mask a ray. The line is the null space of the stacked
constraints, completed to a length prior, and the tracker associates on line distance. The
extension sits behind `--line-classes`, so the point tracker's outputs are byte for byte
unchanged. Four attempts ran on both trials with the same flags and nothing tuned between rooms.
The first wrote impossible segments (one row in ten longer than two pipettes). The second fixed
the geometry defects behind them. The third added the detector's tip boxes, a two-state length
and the mask's end widths. The fourth read the tip side from the mask's thin tail and the tip
state from the 3D length. The third attempt (v3) is the arm the tables read; the fourth cost
room 2 ten ids.

| Clause of the rule | Read-out | Numbers (v3) | Measured against |
|---|---|---|---|
| Ids per held pipette fall at least 5x | **No.** | 51 to 18 in room 1 (2.8x), 48 to 17 in room 2 (2.8x) | the tracker's own ids against the point tracker on the same rows |
| Held-out line residual beats the box centre's | **No, a draw.** | 8.36 against 8.84 px in room 1, 13.4 against 12.9 in room 2 | the held-out camera's own mask axis on frames with three axis views |
| Tip error median under 2 cm | **No.** | 3.03 cm on 48 anchors, the wrong end named on 14 of 47 | my clicks at the cone end on 30 frames of trial 1, triangulated through the rig |
| Flat-rest geometry within the rig's static noise | Yes on v3, no on v4 | the 8-channel pipette at 86.5 deg against the stand's 85.8 in room 1, 71.9 against 70.5 in room 2 | the stand report on the same rows |

The rule also said that an id gain without the tip clause is luck, and the anchors read it that
way. Over the 30 click frames the blue pipette spans ten v3 ids, the same ten as the point
tracker, and the one long v3 track is the resting red pipette absorbing the blue and yellow
anchors the detector's blue class put on it.

| Held pipette | Point tracker | Lines v1 | v2 | v3 | v4 | Measured against |
|---|---|---|---|---|---|---|
| Room 1, blue | 51 | 34 | 17 | 18 | 18 | the tracker's own ids over the 120 s window |
| Room 2, yellow | 48 | 34 | 20 | 17 | 27 | same |

| Tracks | Median cm | p90 cm | Across the axis cm | Rest (n) | Held (n) | Around the tip events (n) | Wrong end | Measured against |
|---|---|---|---|---|---|---|---|---|
| Lines v3 | 3.03 | 24.8 | 1.98 | 2.35 (19) | 13.0 (13) | 2.54 (15) | 14 of 47 | the 48 anchors of the second click sitting |
| Lines v4 | 3.47 | 23.1 | 2.04 | 2.30 (20) | 5.80 (13) | 2.62 (14) | 8 of 47 | same |
| Lines v2 | 8.76 | 24.9 | 1.88 | 13.1 (20) | 8.76 (13) | 12.8 (14) | 21 of 47 | same |
| Point tracker, box centre | 13.9 | 17.0 | not a tip | 14.6 (19) | 13.1 (9) | 14.0 (13) | not a tip | same, 41 anchors matched |

The p90 is a pipette length on every line arm, and the across-axis error is about 2 cm on all
three, so the axis is right and the end choice is what fails. The held anchors are where v3
loses (13.0 cm on 13), and the resting red pipette is where the line is best (0.9 cm across the
axis on 15).

**The id count did not move because a line needs two cameras to see the same shaft on the same
frame, and the hand takes that away.** The held blue pipette has two axis views on 69% of room
1's frames and three on 19%. On the rest the hand covers part of the shaft, the mask breaks into
a blob or a stub, and the tracker has a ray or nothing. Those are the frames where the first
attempt invented geometry and the later ones coast, and 30 frames of coasting end an id. The 18
ids that remain are born where the pipette comes back into two views after a gap the coast did
not cover, or turns while one camera sees it and returns outside the gate. Room 2 has better
coverage (two axis views on 91% of frames) and its ids fell less, so coverage is not the whole
story there: the yellow pipette's ids are born where the hand covers the shaft and where the
class rule splits a track whose masks changed colour. The tip boxes, the end widths and the tail
rule all name the tip end better and touch none of this.

What stands from the phase:

- Every SAM3 observation carries a mask axis, its two endpoints, an elongation and a width:
  165,551 rows in room 1 and 162,676 in room 2 re-measured, no old field changed.
- The line geometry module. On the resting red pipette the line sits 0.9 cm across the axis from
  my clicks (15 anchors). The negative controls break the held-out residual: 3–5x on the real
  rows with camera 6's shipped pose, 35–131x on synthetic segments on the real rig.
- The pipettes rest flat on the bench at 86–87 deg from the bench normal, not upright in a
  stand, so the stand check became a flat-rest check.
- The disposable tip is inside SAM3's mask when one is on. The blue pipette's butt-to-end length
  has two modes, 22.3 cm bare and 29.5 cm with a tip, which is a P1000 tip. The yellow and red
  pipettes rest bare the whole window, at 21.7 and 24.5 cm.
- The colour is the plunger button at the top of the pipette, not a ring at the tip. The colour
  vote disagrees with the detector on 9 of 39 coloured tracks in room 1 and reads yellow on
  almost every disagreement in room 2, a sampler problem there rather than a tracking one.
- The detector's `*_tip` classes fire on 1% of pipette rows, and SAM3 prompted with an empty box
  returns a blob the size of the box, so neither is a tip finder.
- The tip-click tool and two human records. The first sitting accepted a suggested marker and
  put 70 of 138 clicks on the plunger end; the second, with no marker, put every one of its 133
  clicks at the cone end and gave 48 anchors.
- The fourth attempt's thin-tail rule for the tip side, checked by eye in room 1, did not travel:
  room 2's held ids went from 17 to 27.

The next lever is a tracker change, not another observation: carry a held pipette on the
triangulated hand and refuse a new id until the hand lets go. A separate plan is being written.

Evidence: `runs/finebio-lines-P03_03_01-20260928/README.md` (sections "The pre-registered rule",
"Third attempt" and "Fourth attempt"), `runs/finebio-lines-P20_03_01-20260928/README.md`,
`runs/finebio-tipseg-P03_03_01-20260929/README.md`,
`runs/finebio-tips-P03_03_01-20260929/scoreboard/tip_scoreboard.md`,
`qa/finebio-P03_03_01-tip-clicks.human-record.json` and
`qa/finebio-P03_03_01-tip-clicks-2.human-record.json`. The archive closed before this phase, so
the run READMEs are its record.

## Orientation vote

**The orientation vote is adopted under the Part 1 rule.** Against the clean second sitting, median
tip error is 2.38 cm and 1 of 47 anchors names the wrong end. Rest error falls to 1.26 cm and held
error to 4.27 cm. The rule asks for at most four wrong ends, a median under 2.5 cm and no rest
regression.

| Tracks | All median cm | Rest median cm | Held median cm | Wrong end | Measured against |
|---|---|---|---|---|---|
| lines-v3 | 3.03 | 2.35 | 13.04 | 14 of 47 | frozen click anchors |
| lines-v4 | 3.47 | 2.30 | 5.80 | 8 of 47 | frozen click anchors |
| lines-v5a | 2.38 | 1.26 | 4.27 | 1 of 47 | frozen click anchors |
| lines-v5a-inverted | 3.20 | 2.63 | 4.51 | 9 of 47 | frozen click anchors |

I ran both rooms with the fourth attempt’s flags plus `--line-orientation vote`. Each cue
contributes through its strongest camera, weighted by visible length. Colour samples are computed
once. The offline pass writes one orientation over each episode, using the peak absolute sum of its
evidence. It changes endpoint labels, but does not repair an online hold on the wrong end.

The inverted-colour control raises wrong-end anchors from 1 to 9. Cue agreement is against the final
episode decision, not an independent accuracy measure. The hand cue disagrees on 1,574 frames in
room 1.

Evidence: `runs/finebio-lines-P03_03_01-20260928/orientation_readout.json`, both trials’
`scoreboard-v5a/` and `runs/finebio-tips-P03_03_01-20260929/scoreboard-v5a/`. The run READMEs record
this phase.

## Assembly101 in numbers

**SAM3 at 1280 px with an appended prompt memory is the best arm on the toy car, DAM4SAM ties it,
and other cameras detect its failures without fixing them.** The yardstick is 52 anchor cells on 13
frames of the static camera C10379. Each cell is my choice of a SAM3 decoder mask for one visible
part: 51 masks and one hidden mark.

| Arm or question | Value | Measured against | Where |
|---|---|---|---|
| `pm-append`: SAM3 1280 px, corrections appended to memory | 0.743 mean IoU | the 52 anchor cells | `runs/anchor-scoreboard-20260919/anchor_iou.md` |
| SAM3 1280 px, corrections replace memory | 0.724 | same | same |
| DAM4SAM large, the same schedule | 0.715, a tie within the 0.01 rule | same | same |
| The old 720 px reference | 0.668, so 1280 px is the run condition | same | same |
| Corrections proposed by the other cameras alone | 0.659 against 0.743 with human corrections | same | `runs/multiview-reprompt-20260921/anchor_iou_arms.md` |
| The interior, the part no other camera tracks | 0.288 from consensus against 0.585 human | same | same |
| The head camera e4 | 0.558 with 3-part seeds, 0.589 with human-accepted 4-part seeds | 42 scored cells of its 26-frame anchors | `runs/multiview-reprompt-20260921/anchor_iou_e4.md` |
| ATHENA hand triangulation, eight cameras | wrist 24 mm median from the dataset's | the dataset's own hand tracker, itself an estimate | `runs/athena-hands-first-minute-mediapipe/manifest.json` |
| Kineo, known cameras | wrists 17 mm median from the dataset's | same | `runs/kineo-multiview-known-first-minute-20260918/manifest.json` |
| Kineo, self-calibrated cameras | 7.3 degrees and 115 mm median from the dataset's poses | the dataset's extrinsics after similarity alignment | `runs/kineo-multiview-selfcal-first-minute-20260918/manifest.json` |
| LM-EEC, static to head camera | chassis 0.41, interior 0.27, rear body and cabin 0.00 median IoU | the head camera's SAM3 mask at 12 keyframes | `runs/egoexo-correspondence-first-minute-20260918/manifest.json` |

The verdict of the multi-camera work is that **multicam is a detector, not yet a corrector**. SAM3's
own object score ranks its failures (AUROC 0.91 to 0.96 across three arms) but no threshold
transfers. Seeds for the rear body and cabin transfer from other cameras; the chassis and interior
do not.

Evidence: `runs/interaction-review-first-minute-v6/interaction_review_combined.rrd` with its three
presets, `runs/detector-scorecard-20260920/*/scorecard.md`, archive entry "Assembly101 phase
closed".

## Goals scorecard

The goals as the founder set them, first for Assembly101 and then for FineBio ("SAM3.1 + multi cam
+ detector, segmentation from multiple angles to keep track, compelling demo"), each with its final
verdict. Test counts are at the close of the phase.

| Goal | Verdict | Measured against | Evidence |
|---|---|---|---|
| Typed manifests, fixture tests, inference-free Rerun exporter | **Done.** 995 tests in the default tier, 72 more behind `real_data` and `gpu`; ruff clean | the repository's own suite | `src/battle/schemas.py`, `src/battle/exporter.py` |
| Pin one Assembly101 segment with source, analysis, annotation and pose clocks | **Done, twice.** Recording nusar-9033 at 215 to 395 s, and recording nusar-9061 fetched and calibrated for the generalization test | the dataset's own clocks | `configs/clips/*.json` |
| SAM3 over the full 180 s static view | **Done.** The comparison unit became the first minute at 1280 px, 0.743 mean IoU | the 52 anchor cells | `runs/anchor-scoreboard-20260919/anchor_iou.md` |
| SAM3 over the full 180 s head-camera view | **Done, only with human-seeded masks.** 0.558 to 0.589 on the retained minute | 42 anchor cells on e4 | `runs/multiview-reprompt-20260921/anchor_iou_e4.md` |
| Both views on one synchronized Rerun timeline | **Done.** The v6 recording with three presets, the 9-pose-frame static lag applied | the dataset's pose clock | `runs/interaction-review-first-minute-v6/` |
| Five pre-accuracy measures recorded per run | **Done.** | every run's own manifest | `worker_result.json`, `manifest.json` in each run |
| MediaPipe Hands static-view baseline | **Done.** One 60 s run, then every static view, then triangulated | the dataset's hand tracker | `runs/mediapipe-hands-*-60s-20260918/` |
| A second method in the viewer | **Done.** MediaPipe, WiLoR, the dataset's hands and ATHENA under the `hands.rbl` preset | the same recording | `runs/interaction-review-first-minute-v6/hands.rbl` |
| Fixed two-timestamp human QA per method | **Closed without dispositions.** Records prepared, never marked; superseded by the anchors and `battle-anchor-iou` | 52, 64 and 75 anchor cells on three cameras | `qa/`, `runs/anchor-scoreboard-*/` |
| The exploratory queue (WiLoR, BoxMOT, CLIP + Drop-DTW, Grounded-SAM-2, SAMURAI, DAM4SAM, ATHENA, Kineo) | **All eight attempted at smoke tier.** DAM4SAM then given a fair run, 0.715, tied with SAM3; SAMURAI's schedule unsupported upstream | the 52 anchor cells | `runs/dam4sam-arms-20260919/` |
| Multi-view geometry from the dataset's extrinsics | **Done, as context.** Twelve cameras fitted; the rig reproduces the shipped 2D to 0.0023 px RMS or less on the eight static views and 0.27 to 5.1 px on the head cameras | the dataset's shipped 2D points | `runs/assembly101-multiview-rig-check/report.json` |
| Cross-view SAM3 consensus and visual hull | **Done and answered: a detector, not yet a corrector.** 0.659 against 0.743 | the 52 anchor cells | `runs/multiview-reprompt-20260921/anchor_iou_arms.md` |
| Multi-view hand triangulation (ATHENA) | **Done.** Wrist 24 mm median from the dataset; WiLoR's 3D hopping is its per-frame depth, and a triangulated wrist removes it | the dataset's hand tracker | `runs/athena-hands-first-minute-*/` |
| Kineo multi-camera | **Done, both arms.** Known cameras 17 mm; self-calibrated 7.3 degrees and 115 mm off | the dataset's poses and extrinsics | `runs/kineo-multiview-*-first-minute-20260918/` |
| Ego-exo correspondence (LM-EEC) | **Done at 12 keyframes.** Finds the chassis and interior about half the time, never the rear body or cabin | the head camera's SAM3 mask | `runs/egoexo-correspondence-first-minute-20260918/` |
| No training, no annotation project, no accuracy claims | **Held.** Anchors are my choice among decoder masks and rank arms; no measure against ground truth anywhere | the plan's rule | every scoreboard's footer |
| Four physical components through reassembly | **First minute closed at 0.743.** The interior stays human on every camera (0.585 human, 0.29 to 0.36 automatic); the late reassembly never got a verdict | the 52 anchor cells | `runs/anchor-scoreboard-20260919/` |
| Git history from the start | **Missed, then repaired.** First commit after five days of work; the phase closes at tag `assembly101-lab-close` | the git log | `git tag -n9 assembly101-lab-close` |
| FineBio: SAM3.1 plus the shipped detector, the box as SAM3's only prompt | **Done.** Per-frame box decode 0.928 / 99.3% and 0.919 / 98.5%; human-anchored 0.984 mean / 98.8% on trial 1 | detector box, then 325 anchor cells | `runs/finebio-arms-*/scoreboard/`, `runs/finebio-anchors-P03_03_01-20260925/scoreboard/` |
| FineBio: multi-camera, one world frame, segmentation from multiple angles to keep track | **Done with its limits named.** Static objects one id each in both rooms; births down 40%, ambiguities down two-thirds; the held pipette fragments | the tracker's own counts | `<arm>/tracks-ext/identity_metrics.json` |
| FineBio: compelling demo | **Done as a review surface, not judged by a viewer.** Four recordings with World, Cameras and Evidence presets, a storyboard and a negative control, all `rerun rrd verify` clean | the Rerun CLI | `runs/finebio-review-*/review_index.json` |
| FineBio: generalizes to a second room with zero tuning | **Done.** The same verdicts in room 2; three things named as overfit, one stop rule fired | the same measures | `runs/finebio-arms-P20_03_01-20260925/README.md` |
| FineBio: the two soft human gates | **Both held on trial 1.** 52 / 6 / 2 seeds; 337 of 440 anchor cells; trial 1 redone on the filtered seeds; trial 2 has no gate | my decisions at both gates | `qa/finebio-P03_03_01-*.human-record.json` |
| Audio | **Deferred by plan, never revisited.** | nothing | the Assembly101 close |

Evidence: the "Goals scorecard" table in [`archive/method-ledger.md`](archive/method-ledger.md),
whose rows these rewrite.

## Claim boundaries

**Every number above is a comparison between models, or between a model and my choice among that
model's masks. None is accuracy against ground truth.** The boundaries, once:

- The FineBio detector was trained on FineBio's own objects and on frames from these cameras. Every
  IoU against the detector box is agreement between two models. Its seeding quality is an upper
  bound for a new lab's bench.
- The anchors are 337 cells on 18 frames of one trial, my choice among SAM3 decoder masks. They
  rank arms against each other and are not a dataset. I named no tube or held pipette, so per-tube
  identity and the pipette's fragmentation have no human-anchored reading. The 103 cells without a
  detector box were left unlabeled, so false positives on detector-less frames are unmeasured.
- Trial 2 has no human gate: its 66 seed tiles are undecided and no anchor set was chosen. Its
  numbers rest on the detector's own seeds.
- Identity measures against the SAM3 per-camera slots are a proxy. Confidence ranks rows within one
  arm and is not a probability. Events are geometry on model output.
- The pipette in the hand fragments in both rooms (blue 59, yellow 48 ids with the extensions). I
  read it as an observation problem: a long object seen from one or two cameras has no single 3D
  point in its box center. Tracking it as a 3D line cut the ids 2.8x and failed its rule, so the
  observation was only part of it. The hold on the hand in the tracker is the next lever, named,
  not built.
- Proximity events are zero by protocol: no pipette comes near the plate in either window. The
  mechanism is tested on synthetic tracks only.
- Parameters the plan left open were never swept: frames before a re-seed, the memory-write
  threshold, the hysteresis widths, the window length, the container heights.
- FineBio's annotation archives were treated as unavailable. The Assembly101 anchors have the same
  status as FineBio's: my choice, review evidence, not ground truth. The ensemble-v2 row
  in the Assembly101 scoreboard copies masks from the two arms that scored best on those anchors,
  so it is not an out-of-sample number.

Three things are named as overfit to trial 1, in full:

1. **The rig's moving-object witness class list.** The hand-off gate is the p90 held-out residual of
   the tracked moving objects, and the class list was set to the plate and the blue pipette on the
   preflight, where the blue pipette was in use. In room 2 the blue pipette rests on the bench as a
   weak, flickering detection, its residuals put the gate at the 80 px cap instead of 27 px, and the
   head camera's residuals rose from 9 to 15 px. The formula, floor and cap transfer; the class list
   does not.
2. **The trial-1 slot cap.** `--slot-cap 11` was chosen to make room for the plate in every camera.
   In room 2 the detector never sees the plate in T2, so the 11th slot went to the next-ranked
   container, a `magnetic_rack`, which trial 1's T2 did not have.
3. **The viewer's camera-6-only negative control.** The drawn control (a second frustum and a second
   marker set in T5) is logged only when T5 was re-solved, and its text names camera 6. In room 2
   the re-solved cameras are T1 to T4, so the control exists there as numbers only. The fix is to
   draw it for every re-solved camera; recorded, not made.

License: FineBio is non-commercial research data and Assembly101 is CC BY-NC 4.0. Nothing under
`data/` or `runs/` is committed. The one set of dataset pixels in Git is the story media under
`media/story/`; [`LICENSES.md`](LICENSES.md) states the terms they carry.

Evidence: the "Outcome" section of
[`archive/plan-2026-09-25-finebio-3d-tracking.md`](archive/plan-2026-09-25-finebio-3d-tracking.md)
and the FineBio section of
[`archive/README-lab-notebook-2026-09-27.md`](archive/README-lab-notebook-2026-09-27.md).
