# FineBio trials, Sep 25, 2026: the two windows the 3D tracking plan runs on (`p0-trials`)

The plan ([`archive/plan-2026-09-25-finebio-3d-tracking.md`](../archive/plan-2026-09-25-finebio-3d-tracking.md),
todo `p0-trials`) asked for a ~120 s window of `P03_03_01` that contains the six-view annotated
frame 916 and at least two centrifuge cycles, chosen from a top-down contact sheet with the
centrifuge lid state read per frame, and for `P20_03_01` (room 2, same protocol) as the second
trial. This file is the evidence and the decision; the machine-readable result is
[`configs/finebio/trials.json`](../../configs/finebio/trials.json). Everything below was run on
the night of Sep 24 on the CPU (the FineBio DINO detector at ~2.2 s/frame, ffmpeg/ffprobe, cv2,
PIL); no GPU, no viewer. Images (contact sheets, 1:1 centrifuge crops, every 30th frame of the
T5 videos) are under `runs/preflight-finebio-20260924/trials/<trial>/` (gitignored; FineBio
licence, non-commercial research, nothing redistributed). Only numbers are in this file.

Claim boundary: the lid state is a pixel heuristic (teal fraction of a fixed rotor region of
the top-down view), checked by eye on 36 frames of `P03_03_01` and 16 of `P20_03_01`, all
agreeing; "hands" and "moved" counts are FineBio DINO detections at 1 fps, a model's output,
not annotation; the frame-difference energy is a proxy for bench activity, not a measure of
anything semantic. The frame numbers are exact (every frame of T5 was read for the lid series),
the seconds are frame / (30000/1001).

## Videos, frame counts, poses

Counted with `ffprobe -count_frames` (decode of every frame), not the container's `nb_frames`:

| trial | fpv (1920x1440) | T1 | T2 | T3 | T4 | T5 (each 1920x1080) | fps | fpv pose `rets` length | valid |
|---|---|---|---|---|---|---|---|---|---|
| `P03_03_01` | 8492 | 8492 | 8492 | 8492 | 8492 | 8492 | 30000/1001 | 8492 | 8407 (99.0%) |
| `P20_03_01` | 6045 | 6045 | 6045 | 6045 | 6045 | 6045 | 30000/1001 | 6045 | 5716 (94.6%) |
| `P03_01_01` (smoke, from the preflight) | 5032 | 5032 | 5032 | 5032 | 5032 | 5032 | 30000/1001 | 5032 | 4890 (97.2%) |

Six views with one frame count per trial, and a pose file of that length, so one window applies
to every view (the preflight put the six clocks at 0 +/- 1 frame). The shipped detection
images give the frames the authors annotated: `P03_03_01` has **916 in all six cameras**
(`P03_03_01_T1..T5_000916.jpg` and `P03_03_01_000916.jpg`) plus fpv-only 210, 1544, 3970,
7197, 8206; `P20_03_01` has **1442 in all six** plus fpv-only 695, 1112, 2368, 2769, 4196,
4749.

## How the lid state was read

- **Where the centrifuge is.** One FineBio DINO pass (`scripts/finebio_preflight_detect.py
  --views T5 --spaced 0:N:30`, class `centrifuge`, score 0.84-0.87 on every sampled frame) on
  every 30th frame of each T5 video. `P03_03_01`: box `(1110, 271, 1294, 461)` while the lid is
  closed and `(1128, 162, 1341, 462)` while it is open (the flipped lid extends the box upward
  by 110 px; the base does not move). `P20_03_01`: closed `(765, 129, 930, 332)`, open
  `(728, 0, 952, 332)` (the open lid touches the top edge of the frame). The preflight's
  P03_01_01 box (~590-770 x 140-330) is a different bench layout; the protocol-03 bench puts the
  centrifuge right of centre.
- **The statistic.** A fixed rotor region inside the base (`P03_03_01`: `(1135, 300, 1270, 445)`;
  `P20_03_01`: `(800, 180, 900, 285)`, raw T5 pixels), and per frame the fraction of its
  pixels whose OpenCV HSV hue is in 75..100 with saturation > 70 and value > 50 (the teal of the
  dome). Open lid: the white/grey rotor with its dark tube holes fills the region, fraction
  ~0.02. Closed lid: the dome covers it, fraction ~0.85-0.97; with a gloved hand over the
  centre of the dome (every `P03_03_01` spin) 0.36-0.62.
- **The threshold, 0.24**, is the midpoint of the widest empty gap in the 1 fps sample of
  `P03_03_01` (nothing between 0.123 and 0.357). Full-rate histogram of `P03_03_01` (8492
  frames): 7029 in [0, 0.05), 14 in [0.05, 0.10), 62 spread over [0.10, 0.35) (the transition
  frames: the lid mid-swing), 158 in [0.35, 0.40), 142 in [0.40, 0.65), 2 in [0.65, 0.75),
  0 in [0.75, 0.80), 78 in [0.80, 0.85), 1007 in [0.85, 0.90). `P20_03_01` (6045 frames):
  5867 in [0, 0.05), 37 over [0.05, 0.35), 20 over [0.35, 0.85), 121 in [0.85, 1.0].
  Sensitivity: thresholds 0.15, 0.20, 0.24, 0.30, 0.35 all give the same 10 state runs on
  `P03_03_01` (closed-frame total 1439 -> 1387, so each of the eight transitions moves by a
  few frames) and the same 9 runs on `P20_03_01` (158 -> 141). The series is not
  threshold-sensitive between the modes.
- **By eye.** 1:1 crops of the centrifuge at every 10 s over 0-150 s of `P03_03_01` (16
  frames) and at the frames on either side of each transition (22 more; 36 distinct): every
  frame the statistic calls closed shows the teal dome down, every open frame shows the rotor,
  and the 0.36-0.62 frames are the dome with a gloved hand on it. `P20_03_01`: 16 frames around
  the four transitions plus 1440 and 6030, the same; frame 5190 (0.25) is the lid halfway
  down. The stable-state values also show the region is placed correctly: closed without a
  hand 0.84-0.87 (P03), closed with the hands at the rim of the dome 0.90-0.96 (P20).

## Trial 1: `P03_03_01` (protocol 03, room 1, day 221013)

**Top-down contact sheet** (57 frames, every 150th raw frame = 5.0 s; `T5_contact_5s.jpg`).
The black bench from above; the participant at the bottom edge with the head-mounted GoPro
visible; three ArUco markers (one top centre beside the vortex mixer, two at the bottom edge).
Left to right along the far edge: a pink tip rack and an 8-tube-strip rack, the 6-well cell
culture plate, the blue vortex mixer, the teal/white mini centrifuge right of centre, a cream
50 ml tube stand and a yellow tip rack; nearer the participant a printed protocol sheet, a
yellow notes sheet, a white micro-tube rack, a pen, the blue, red and 8-channel pipettes and a
blue tip rack at the right. What changes over 283 s: for the first 37 s the centrifuge lid is
closed and the hands work at the lower half of the bench (pipette in hand at 20 s, a hand on
the vortex at 30 s); the lid opens at 36.7 s, tubes go in, a hand presses the lid for the
first spin at 39-41 s; from 41 s to 108 s the lid stays open while micro tubes move between the
rack, the vortex and the centrifuge and the pipettes are picked up and put down (the busiest
stretch of the trial for tube traffic, 50-90 s); spins at 108-110 s, 168-171 s and 249-252 s
each preceded by loading and followed by unloading; the lid is open at the end.

**Lid state, every frame** (closed = teal fraction >= 0.24):

| state | frames (end exclusive) | seconds | length |
|---|---|---|---|
| closed (initial) | [0, 1099) | 0.00-36.67 | 1099 |
| open | [1099, 1176) | 36.67-39.24 | 77 (tubes loaded) |
| **closed, spin 1** | **[1176, 1228)** | 39.24-40.97 | 52 |
| open | [1228, 3224) | 40.97-107.57 | 1996 |
| **closed, spin 2** | **[3224, 3311)** | 107.57-110.48 | 87 |
| open | [3311, 5020) | 110.48-167.50 | 1709 |
| closed, spin 3 | [5020, 5111) | 167.50-170.54 | 91 |
| open | [5111, 7458) | 170.54-248.85 | 2347 |
| closed, spin 4 | [7458, 7545) | 248.85-251.75 | 87 |
| open | [7545, 8492) | 251.75-283.35 | 947 |

Four cycles (open -> tubes in -> closed -> open) in the trial, each spin 1.7-3.0 s with a
hand on the lid. The preflight's "at least six" came from a 12-frame contact sheet and counted
loosely; four is the number.

**Activity per 10 s** (hands = mean number of DINO `left_hand`/`right_hand` boxes at score
>= 0.3 per sampled frame, 1 fps; motion = mean absolute grey difference between consecutive
frames on a 192x108 downscale of the lower 75% of T5, every frame; moved = number of
(class, instance) pairs among the movable classes whose box centre at second t is > 40 px from
every same-class box at t-1, summed over the bin; lid = the 1 fps state, `o` open, `C` closed):

| start frame | s | hands | motion | moved | lid per second |
|---|---|---|---|---|---|
| 0 | 0 | 1.8 | 0.85 | 13 | `CCCCCCCCCC` |
| 300 | 10 | 2.0 | 0.31 | 8 | `CCCCCCCCCC` |
| 600 | 20 | 2.1 | 0.33 | 2 | `CCCCCCCCCC` |
| 900 | 30 | 2.1 | 1.00 | 11 | `CCCCCCCooo` |
| 1200 | 40 | 2.0 | 0.66 | 13 | `Cooooooooo` |
| 1500 | 50 | 2.0 | 0.76 | 19 | `oooooooooo` |
| 1800 | 60 | 1.9 | 0.92 | 19 | `oooooooooo` |
| 2100 | 70 | 1.9 | 0.85 | 15 | `oooooooooo` |
| 2400 | 80 | 2.0 | 0.50 | 19 | `oooooooooo` |
| 2700 | 90 | 1.7 | 0.32 | 6 | `oooooooooo` |
| 3000 | 100 | 2.1 | 0.71 | 9 | `ooooooooCC` |
| 3300 | 110 | 2.0 | 0.69 | 11 | `Cooooooooo` |
| 3600 | 120 | 1.9 | 0.46 | 6 | `oooooooooo` |
| 3900 | 130 | 1.9 | 0.95 | 15 | `oooooooooo` |
| 4200 | 140 | 2.1 | 0.28 | 7 | `oooooooooo` |
| 4500 | 150 | 2.4 | 0.24 | 14 | `oooooooooo` |
| 4800 | 160 | 2.0 | 0.70 | 8 | `ooooooooCC` |
| 5100 | 170 | 2.0 | 0.61 | 9 | `Cooooooooo` |
| 5400 | 180 | 1.8 | 0.41 | 6 | `oooooooooo` |
| 5700 | 190 | 2.0 | 0.81 | 15 | `oooooooooo` |
| 6000 | 200 | 1.8 | 0.82 | 14 | `oooooooooo` |
| 6300 | 210 | 2.0 | 0.54 | 11 | `oooooooooo` |
| 6600 | 220 | 1.6 | 0.27 | 6 | `oooooooooo` |
| 6900 | 230 | 1.6 | 0.48 | 6 | `oooooooooo` |
| 7200 | 240 | 2.2 | 0.43 | 9 | `oooooooooC` |
| 7500 | 250 | 2.2 | 0.58 | 7 | `CCoooooooo` |
| 7800 | 260 | 1.9 | 0.37 | 10 | `oooooooooo` |
| 8100 | 270 | 2.0 | 0.37 | 12 | `oooooooooo` |
| 8400 | 280 | 2.0 | 1.34 | 9 | `oooo` |

Both hands are on the bench in almost every sampled frame of the trial (1.6-2.4 boxes), so
the hand count does not separate windows; motion and moved do. Which classes moved, 40-110 s:
`micro_tube` 4, 10, 12, 4, 12, 4, 3 per 10 s bin (tubes between rack, vortex and
centrifuge), `8_channel_pipette` 4, 2, 1, 3, 0, 0, 1, `blue_pipette`, `red_pipette`,
`yellow_pipette` 1-2 each per bin, `blue_tip_rack` 1, 3, 2, 1 in the 40, 60, 70 and 80 s bins
(the relocation the preflight saw), `micro_tube_rack` 3 at 50 s, `50ml_tube` 1-2. 20-30 s is the quietest bin
of the whole trial (motion 0.33, 2 moved instances, a pipette).

**Candidate windows** (3600 frames each; `full cycles` = closed runs with open frames on both
sides inside the window; `annotated` = shipped annotated frames inside; hands / motion / moved
as above but over the window; fpv pose = valid fraction over the window):

| window (raw frames) | seconds | 916 offset | closed runs inside | full cycles | annotated | hands | motion | moved | fpv pose |
|---|---|---|---|---|---|---|---|---|---|
| [0, 3600) | 0.0-120.1 | 916 (30.6 s) | [1176, 1228), [3224, 3311) | 2 | 210, 916, 1544 | 1.97 | 0.658 | 145 | 99.6% |
| [300, 3900) | 10.0-130.1 | 616 (20.6 s) | same | 2 | 916, 1544 | 1.98 | 0.626 | 138 | 99.6% |
| **[600, 4200)** | **20.0-140.1** | **316 (10.5 s)** | same | **2** | **916, 1544, 3970** | 1.97 | **0.679** | 145 | 99.6% |
| [750, 4350) | 25.0-145.1 | 166 (5.5 s) | same | 2 | 916, 1544, 3970 | 1.97 | 0.679 | 150 | 99.6% |
| [900, 4500) | 30.0-150.2 | 16 (0.5 s) | same | 2 | 916, 1544, 3970 | 1.97 | 0.675 | 150 | 99.6% |
| [0, 4200) (140 s) | 0.0-140.1 | 916 | same | 2 | 210, 916, 1544, 3970 | 1.96 | 0.664 | 166 | 99.7% |

No 120 s window that contains 916 can hold spin 3 (a window ending at 5111 starts at 1511, past
916), so two cycles is the maximum and every candidate has both.

**Decision: `P03_03_01` window = raw frames [600, 4200)**, 20.02-140.14 s, 3600 frames.

- Frame 916 sits 316 frames (10.5 s) after the start, so the tracker has 10 s of history when
  the six-view anchor frame arrives; the hands are on the bench and the lid is still closed
  there, and the first lid opening (1099), loading (1099-1176), spin 1 (1176-1228) and
  unloading follow 6-10 s later.
- Both cycles fall inside with their loading and unloading (spin 2 at 3224-3311 leaves 29.7 s
  of unloading and tube traffic before the window ends; the 130-140 s bin is one of the busiest,
  motion 0.95).
- Three of the six shipped annotated frames are inside (916 six-view, 1544 and 3970 fpv).
- It has the highest mean bench motion of the candidates and the fpv pose is valid on 3587 of
  3600 frames (99.6%; longest invalid run 5 frames at 2038-2043).
- Rejected: [0, 3600) starts at the trial start and spends 10-30 s in the quietest stretch of
  the trial, leaves only 9 s after spin 2 and loses frame 3970; [900, 4500) puts 916 sixteen
  frames after the start, where nothing has been born yet; [750, 4350) is the same window
  shifted by 5 s with 916 at 5.5 s, no gain; [0, 4200) is 140 s, over the budget for five arms
  x six views, for the quiet 20 s it adds.

## Trial 2: `P20_03_01` (protocol 03, room 2, day unknown until the camera solve)

Checks: fpv and T1..T5 present (`finebio_videos_fpv_test/finebio_videos/P20_03_01.mp4`,
`finebio_videos_tpv_test/finebio_videos/P20_03_01_T{1..5}.mp4`), 6045 decoded frames in every
one, 1920x1440 / 1920x1080 at 30000/1001; pose file `P20_03_01.npz` with `rets (6045,)`,
`rots (6045, 3, 1)`, `trans (6045, 3, 1)`, 5716 valid (94.6%). `P20_03_01` passes, so the
protocol-05 fallbacks (`P13_05_01`, `P28_05_01`) were not evaluated.

**Top-down contact sheet** (21 frames every 300th raw frame = 10.0 s; `T5_contact_10s.jpg`).
The room-2 top-down camera sits higher: the whole bench fits with margin, four ArUco markers
(two at the far edge, two near the participant), the participant's shoulders at the bottom.
Far edge left to right: a spray bottle and a green tissue box, a dark 8-tube-strip rack above
the 6-well plate, the teal/white mini centrifuge (left of centre, its open lid reaching the top
of the frame), the blue vortex mixer, a cream 50 ml tube stand; nearer the participant: a red
tip rack, the printed protocol, a white micro-tube rack, a yellow tip box, a blue Finntip box
and three pink Rainin tip racks at the right, the blue, red, yellow and 8-channel pipettes. The lid is open in every
10 s tile except 30 s (both hands pressing the closed dome). Over 201 s the hands are on the
bench in every tile; pipetting over the plate and the rack, tubes carried to the centrifuge,
the 8-channel pipette in hand at 48 s (frame 1440).

**Lid state, every frame** (same statistic, region `(800, 180, 900, 285)`, threshold 0.24):

| state | frames | seconds | length |
|---|---|---|---|
| open (initial) | [0, 895) | 0.00-29.86 | 895 |
| **closed, spin 1** | **[895, 937)** | 29.86-31.26 | 42 |
| open | [937, 2500) | 31.26-83.42 | 1563 |
| **closed, spin 2** | **[2500, 2539)** | 83.42-84.72 | 39 |
| open | [2539, 3798) | 84.72-126.73 | 1259 |
| **closed, spin 3** | **[3798, 3834)** | 126.73-127.93 | 36 |
| open | [3834, 5190) | 127.93-173.17 | 1356 |
| closed, spin 4 | [5190, 5224) | 173.17-174.31 | 34 |
| open | [5224, 6045) | 174.31-201.70 | 821 |

Four spins of 1.1-1.4 s each, both hands on the lid; the lid is open from the first frame.

**fpv pose and the centrifuge.** The 329 invalid pose frames fall in 11 runs; the five of 10
frames or more are [892, 909), [915, 992), [2523, 2589), [3812, 3883), [5209, 5280): one per
spin, each starting between 3 frames before and 23 frames after the lid closes and ending 49-56
frames after it re-opens. When the participant leans over the centrifuge the head camera loses
the bench markers, so in this trial the fpv has no pose for the 2-3 s around every spin and the
fixed cameras carry the containment episodes alone. `P03_03_01` has no such coupling (its two
long invalid runs, [4851, 4907) and [8287, 8300), are outside the window).

**Activity per 10 s** (as for trial 1):

| start frame | s | hands | motion | moved | lid per second |
|---|---|---|---|---|---|
| 0 | 0 | 1.8 | 1.06 | 14 | `oooooooooo` |
| 300 | 10 | 2.0 | 0.44 | 11 | `oooooooooo` |
| 600 | 20 | 2.0 | 0.90 | 16 | `oooooooooo` |
| 900 | 30 | 2.1 | 1.28 | 20 | `CCoooooooo` |
| 1200 | 40 | 2.1 | 1.02 | 16 | `oooooooooo` |
| 1500 | 50 | 2.0 | 1.06 | 16 | `oooooooooo` |
| 1800 | 60 | 2.2 | 1.07 | 21 | `oooooooooo` |
| 2100 | 70 | 1.9 | 0.86 | 24 | `oooooooooo` |
| 2400 | 80 | 2.1 | 1.16 | 16 | `ooooCooooo` |
| 2700 | 90 | 1.7 | 0.81 | 14 | `oooooooooo` |
| 3000 | 100 | 2.0 | 1.12 | 28 | `oooooooooo` |
| 3300 | 110 | 2.0 | 0.47 | 5 | `oooooooooo` |
| 3600 | 120 | 2.1 | 1.29 | 17 | `oooooooCoo` |
| 3900 | 130 | 1.7 | 0.88 | 15 | `oooooooooo` |
| 4200 | 140 | 1.7 | 0.86 | 13 | `oooooooooo` |
| 4500 | 150 | 2.6 | 0.90 | 12 | `oooooooooo` |
| 4800 | 160 | 2.0 | 0.92 | 16 | `oooooooooo` |
| 5100 | 170 | 2.2 | 1.08 | 22 | `oooCCooooo` |
| 5400 | 180 | 1.7 | 0.75 | 11 | `oooooooooo` |
| 5700 | 190 | 1.7 | 0.77 | 11 | `oooooooooo` |
| 6000 | 200 | 2.0 | 0.56 | 1 | `oo` |

Room 2 is busier throughout (motion 0.8-1.3 in most bins against 0.3-1.0 in room 1; the
camera is higher, so the same motion covers fewer pixels, which makes the comparison
conservative). `micro_tube` moves 7-15 instances per bin around every spin (30, 60-80, 100,
120, 170 s).

**Candidate windows** (1442 offset instead of 916):

| window | seconds | 1442 offset | closed runs inside | full cycles | annotated | hands | motion | moved | fpv pose |
|---|---|---|---|---|---|---|---|---|---|
| [0, 3600) | 0.0-120.1 | 1442 (48.1 s) | [895, 937), [2500, 2539) | 2 | 695, 1112, 1442, 2368, 2769 | 1.99 | 0.937 | 201 | 95.3% |
| [300, 3900) | 10.0-130.1 | 1142 (38.1 s) | + [3798, 3834) | 3 | same | 2.02 | 0.957 | 204 | 93.1% |
| **[600, 4200)** | **20.0-140.1** | **842 (28.1 s)** | **[895, 937), [2500, 2539), [3798, 3834)** | **3** | **695, 1112, 1442, 2368, 2769, 4196** | 1.99 | **0.994** | **208** | 93.1% |
| [900, 4500) | 30.0-150.2 | 542 (18.1 s) | [2500, 2539), [3798, 3834) | 2 | 1112, 1442, 2368, 2769, 4196 | 1.97 | 0.990 | 205 | 93.3% |
| [1200, 4800) | 40.0-160.2 | 242 (8.1 s) | same two | 2 | 1442, 2368, 2769, 4196, 4749 | 2.01 | 0.959 | 197 | 95.7% |

**Proposal: `P20_03_01` window = raw frames [600, 4200)**, 20.02-140.14 s, the same offset
and length as trial 1 (no per-trial tuning of the window either). Three full cycles, frame
1442 at 28.1 s, six of the seven shipped annotated frames inside, the highest bench motion and
moved count of the candidates. Its fpv pose validity, 93.1% (248 invalid frames, longest run
77 at [915, 992)), is lower than [0, 3600)'s 95.3% for the reason above: every cycle costs 2-3 s
of fpv pose, and this window has three of them. That is the trade the second trial should
make: the fpv drop-out during containment is the situation the tracker is meant to survive.
Rejected: [0, 3600) (one fewer cycle, frame 4196 lost, starts at the trial start); [1200, 4800)
(two cycles, 1442 at 8 s, frame 695 and 1112 lost).

## What `P03_01_01` remains

The smoke and the preflight reference: raw frames [1798, 1858) (60 consecutive frames from
60 s), six views, detections and SAM3 masks on disk under `runs/preflight-finebio-20260924/`,
the committed observation fixtures under `tests/fixtures/finebio_preflight/`, and the
regression target for `battle-finebio-cameras` and `-rig`. Protocol 01 moves one plate and one
pipette and opens the centrifuge once near the end, so it is not a tracking trial; its
per-view frame count is 5032 in all six videos, pose 4890/5032 valid.

## For lane B (`p1-configs`)

- Cut proxies on exactly `[600, 4200)` for both `P03_03_01` and `P20_03_01` with
  `finebio_frames.proxy_ffmpeg_args(raw, out, 600, 3600)` (native resolution and rate, no `fps=`
  filter); proxy frame k == raw frame 600 + k, and the shipped fpv pose row for proxy frame k is
  `rets[600 + k]`. The six-view annotated frames are proxy frame 316 (`P03_03_01`) and 842
  (`P20_03_01`).
- Cycles in proxy frames: `P03_03_01` closed [576, 628) and [2624, 2711); `P20_03_01` closed
  [295, 337), [1900, 1939), [3198, 3234). The centrifuge T5 boxes and rotor regions in
  `trials.json` are raw T5 pixels and apply unchanged to a native-resolution proxy.
- `P20_03_01` needs its camera solve (`p0-cameras`) before the rig check; the fpv pose gaps
  around the spins are real drop-outs of the shipped pose, not outliers to gate.

## Files

- [`configs/finebio/trials.json`](../../configs/finebio/trials.json): the three trials with
  role, room, day, per-view frame counts, pose file and validity, window, annotated frames in
  the window, centrifuge boxes, rotor region, lid intervals and cycles, notes.
- `runs/preflight-finebio-20260924/trials/<trial>/` (gitignored): `frames/f%06d.jpg` (every
  30th raw T5 frame), `motion.csv` (per-frame energy), `lid_fullrate.csv` (per-frame teal
  fraction and mean saturation of the rotor region), `lid_state.csv` (1 fps with hands and
  motion), `detections_T5_every30/T5.jsonl` (DINO boxes at 1 fps, score >= 0.05),
  `T5_contact_5s.jpg` / `T5_contact_10s.jpg`, `T5_centrifuge_*_1to1.jpg` (the crops looked at),
  `T5_window_keyframes.jpg`; `detect_every30.log`. Analysis scripts were one-offs under `/tmp`
  (a sequential cv2 pass for motion and frames, the HSV statistic, PIL contact sheets); nothing
  new under `scripts/`.
