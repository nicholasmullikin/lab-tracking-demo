# Review guide, Sep 25, 2026: the FineBio 3D object-tracking recordings (trial 1 `P03_03_01`, trial 2 `P20_03_01`)

One Rerun recording per trial window (raw frames [600, 4200), 20.02-140.14 s, six views)
with three layout presets, built by `battle-finebio-viewer` from the tracking arms, the
confidence and events passes, the rig check, the seeds and the lane-B proxies
(plan `docs/plan-2026-09-25-finebio-3d-tracking.md`, todos `p5-confidence`, `p5-events`,
`p5-viewer`, `p6-trial2`; ledger entries "Sep 25: confidence, events and the review recording"
and "Sep 25: FineBio 3D tracking phase, close-out"). The first part of this guide describes the
trial-1 recording on the core tracker; the trial-2 recording (room 2, built on the tracker
extensions with nothing tuned) and the two-trial scoreboard are in
[Trial 2](#trial-2-p20_03_01-room-2-the-same-build-on-the-second-trial), and a 20-minute route
through both is in [What to look at first](#what-to-look-at-first-20-minutes-both-trials).
Nothing in the recordings is committed (FineBio licence: they hold the proxy videos and SAM3
masks); the presets under `configs/rerun/finebio_{world,cameras,evidence}.rbl` hold entity paths
only.

**Claim boundary, read first.** Everything drawn is model output. The FineBio DINO detector was
trained on FineBio's own bench and on frames from these cameras, so its boxes, and every SAM3
mask prompted from them, are agreement between two models, not accuracy. "Confidence" ranks a
track's rows within one arm by five label-free signals; it is not a probability of being right.
Events are geometry on 3D tracks against volumes derived from the rig. The Sep 25 recordings
predate both gates; the Sep 27 one is built on gate 1's human-filtered seeds but draws no anchor
either; the human-anchored numbers that exist since Sep 27 (trial 1 only, 337 cells) are under
[Key numbers](#key-numbers) and rank the arms without making any of this accuracy. The
human-facing frame is the six-view annotated frame 916, which the storyboard lands on.

## Open

**Recommended for trial 1 since Sep 27: the post-gate recording**
`runs/finebio-review-P03_03_01-filtered-20260927/review.rrd`, built on the arms re-run on the
human-filtered seeds of gate 1 (60 slots of 66; the six rejected seed slots are absent from the
tiles) with the tracker extensions (`tracks-ext/`, arms a / b / c), scored against the human
anchors of gate 2. Its numbers are at least as good as the Sep 25 recordings' on the default arm
(b): det-box IoU 0.928 / 99.3% >= 0.5 (Sep 25 0.926 / 99.1%), ambiguities 35 (54), the blue
pipette 51 ids (59), the human-anchored mask IoU identical on every shared anchor cell (0.988
mean / 99.0% >= 0.5), two of the four hidden false positives gone; arm (c) 0.915 / 80.4% (0.914 /
79.3%) with its drift moved between slots rather than removed (ledger entry "Sep 27: trial-1
downstream redone on the human-filtered seeds (post-gate)"; run README
`runs/finebio-arms-P03_03_01-filtered-20260927/README.md`). 867.5 MB, 389 entity paths, 37,984
mask cut-outs, `rerun rrd verify` clean, presets validated; the same three presets sit beside it
(`--preset-dir ""` was used, so the committed `configs/rerun/finebio_*.rbl` copies are still the
Sep 25 core recording's; they fit this one too, being entity paths on the same application id).

```bash
# Post-gate recording (trial 1, human-filtered seeds, tracks-ext; recommended)
uv run rerun runs/finebio-review-P03_03_01-filtered-20260927/review.rrd runs/finebio-review-P03_03_01-filtered-20260927/world.rbl
uv run rerun runs/finebio-review-P03_03_01-filtered-20260927/review.rrd runs/finebio-review-P03_03_01-filtered-20260927/cameras.rbl
uv run rerun runs/finebio-review-P03_03_01-filtered-20260927/review.rrd runs/finebio-review-P03_03_01-filtered-20260927/evidence.rbl
```

What differs from the Sep 25 recordings below: 60 mask entities per mask arm instead of 66
(T1 8, T3 9, fpv 10 slots), `world/<view>/seeds` from `with-plate/filtered/seeds.json`, the
storyboard's tube is `micro_tube-031` (renumbered; the same object, `contained` from raw 887
through both closures with the same id), and its confidence-drop item is arm (c)'s
`50ml_tube-064` at raw 2037 (T3 `50ml_tube#1`, IoU 0.89 -> 0.00). Everything else in this guide
(presets, cross-checks, negative control, timeline) reads the same. The Sep 25 recordings stay
as built and are what the sections below describe:

```bash
# World preset (default): the 3D rig with tracks, volumes, events strip, confidence, storyboard
uv run rerun runs/finebio-review-P03_03_01-20260925/review.rrd runs/finebio-review-P03_03_01-20260925/world.rbl
# Cameras preset: six tiles with the proxy video, detector boxes, SAM3 masks, track ids
uv run rerun runs/finebio-review-P03_03_01-20260925/review.rrd runs/finebio-review-P03_03_01-20260925/cameras.rbl
# Evidence preset: the seven cross-checks, the negative control, scorecards, inventories
uv run rerun runs/finebio-review-P03_03_01-20260925/review.rrd runs/finebio-review-P03_03_01-20260925/evidence.rbl
```

The presets are also under `configs/rerun/finebio_<name>.rbl` (identical copies; they are bound
to the recording's application id `finebio-review-P03_03_01`). Switching presets inside the
viewer: open the recording once, then drag the other `.rbl` files in, or use the blueprint panel.
The timeline is `frame` (raw frame index; the video tiles follow it) with `source_time` beside
it. Raw frame = proxy frame + 600 = worker analysis frame + 600.

Rebuild (about 8 min for the recording, 15 s / 7 s per arm for confidence / events; CPU). The
recording described here is on the core tracker's `tracks/`; the three tools default to
`tracks-ext/` when it exists, so name the tracks directory:

```bash
R=runs/finebio-arms-P03_03_01-20260925
for arm in a-boxes-only b-box-decode-arm c-video-memory-arm d-video-memory-arm; do
  uv run battle-finebio-confidence --arm-dir $R/$arm --detections runs/finebio-detect-P03_03_01-600-4200-20260925/dino --tracks-dir tracks --output $R/$arm/confidence
  uv run battle-finebio-events --arm-dir $R/$arm --config configs/clips/finebio_P03_03_01_600-4200.json --rig runs/finebio-rig-P03_03_01-600-4200/rig.json --tracks-dir tracks --output $R/$arm/events
done
uv run battle-finebio-viewer --clip-config configs/clips/finebio_P03_03_01_600-4200.json \
  --arm-dirs a=$R/a-boxes-only,b=$R/b-box-decode-arm,c=$R/c-video-memory-arm,d=$R/d-video-memory-arm \
  --rig runs/finebio-rig-P03_03_01-600-4200/rig.json --seeds runs/finebio-seeds-P03_03_01-20260925/with-plate \
  --tracks-dir tracks --output runs/finebio-review-P03_03_01-20260925
```

On the tracker extensions' output (`<arm>/tracks-ext/`, same file layout, `held` / `contained`
states with `container_id` / `held_by` / `group_size`; arms a/b/c today), the same three
commands with `--tracks-dir tracks-ext`, the confidence and events outputs to
`$R/$arm/confidence-ext` and `$R/$arm/events-ext` (the viewer reads `confidence<tag>/` and
`events<tag>/` beside `tracks<tag>/`), and the viewer to
`runs/finebio-review-P03_03_01-20260925-ext` with `--preset-dir ""` (the committed presets are
the core recording's; they fit both, being entity paths). The states are entity paths
(`world/tracks/<arm>/contained`, `.../held`), so they render, with the container or hand in the
label, without a code change; see the section on that recording below.

## What each preset shows

**World** (`world.rbl`). Left: the rig in board centimetres, z into the bench (`world`, view
coordinates `RIGHT_HAND_Z_DOWN`): the bench outline, the board origin, the day's three ArUco
markers, the five fixed frusta (`world/T1..T5`, T5 from the marker PnP), the head camera's frustum
moving per frame with its trail (`world/fpv`, `world/fpv_trail`), the rig's static objects with
their heights (`world/static_objects`) and per-frame hand triangulations (`world/left_hand`,
`world/right_hand`), the container volumes as boxes (`world/containers/<name>`: centrifuge, PCR
machine, vortex, trash can, three racks; footprints from the rig's static points and the
detector's box widths, per-class heights) and the centrifuge lid state as a red solid slab when
closed / green wireframe when open (`world/containers/centrifuge_lid`), and the 3D tracks of
every arm under `world/tracks/<arm>/<state>` (points coloured by class, radius growing with the
uncertainty, label `id [state] in <container> by <hand> !` where `!` marks an abstaining row;
non-observed states are drawn in the class colour dimmed) with the default arm's 30-frame
trails (`world/tracks/b/trails`). Arm (b) is visible by default; arms (a), (c), (d) are logged
and hidden (toggle them in the entity tree). Right: the events strip as time series
(`events/<arm>/contained|held|proximity`, counts of active episodes; `events/lid_closed` 0/1;
arm (b) shown, others hidden), the confidence series of arm (b) per class plus the abstain
fraction and the storyboard tracks (`confidence/b/...`), and tabs with the storyboard's current
item (`storyboard/current`), the active relations at the current frame (`events/b/active`), the
storyboard guide, the storyboard marks and the event log.

**Cameras** (`cameras.rbl`). Six tiles (T1..T5 and the fpv), each the proxy video
(`world/<view>/video`, an `AssetVideo` + one `VideoFrameReference` per frame at the asset's own
frame timestamps, so proxy frame k is shown at raw frame 600 + k), the DINO boxes at score >= 0.3
(`world/<view>/detector`, `class score`), the SAM3 masks of arm (b) as RGBA cut-outs per slot
(`world/<view>/masks/b/<slot>`, slot labels with `#` written as `-`, e.g. `cell_culture_plate-0`;
arm (c) under `masks/c/` hidden by default; arm (d) only on the storyboard frames), the tracks
of arms (b) and (c) projected into the view with their ids and abstain flags
(`world/<view>/tracks/<arm>`), and the accepted seed boxes at their seed frames
(`world/<view>/seeds`). Masks are logged every 6th frame for (b) and every 30th for (c), plus
every storyboard frame (and 15 frames either side of the confidence-drop frame for all three
mask arms), to keep the recording near a gigabyte: step the timeline in sixes (proxy frames
0, 6, 12, ...) to see a mask on every step. Below the tiles: the confidence and events series.

**Evidence** (`evidence.rbl`). The rig view without tracks or videos (frusta including the
negative control `world/T5_shipped`, static objects, hands, markers, fpv), the T5 tile with the
markers projected through the marker-PnP pose (green, `world/T5/markers_projected`) and through
the shipped pose (red, `world/T5/markers_projected_shipped`) beside the static objects, the hands
and the fpv centre, and tabs of documents: the negative control's numbers, the clock scan, the
gates, the arms scoreboard, per arm the measures, occlusion inventory, identity metrics,
confidence and events reports, the seeds report, the rig report, the claim boundary. Below: the
leave-one-view-out residual per class and view (`checks/loo/<class>/<view>`), the plate's fixed
-> fpv hand-off residual with its inside-the-box flag (`checks/handoff/...`), the cross-view
residual per arm and view (`checks/residuals/<arm>/<view>`, median over live tracks) and the
coasting-track counts (`checks/occlusion/<arm>/coasting_tracks`).

### The seven cross-checks, as entities on every build

| # | check | entities | where the numbers are |
|---|---|---|---|
| 1 | marker reprojection per view | `world/<view>/markers_projected` (fixed views static, fpv per frame) | `checks/gates`, `checks/rig` (T1-T4 shipped 6.3 / 6.1 / 6.7 / 4.8 px, T5 marker PnP 0.72 px) |
| 2 | fpv camera centre in the fixed views | `world/<view>/fpv_camera_centre` per frame | lands on the head-mounted GoPro in T1 / T4 |
| 3 | static-object triangulation | `world/static_objects`, `world/<view>/static_reprojected` | `checks/rig`: static LOO median 10.0 px over 42 cells, heights half the physical ones (centrifuge 8.4 cm with the lid open) |
| 4 | hands as probes | `world/left_hand`, `world/right_hand`, `world/<view>/<hand>_triangulated` | left 2546/3600 frames at 10.9 px median, right 1904 at 51.7 px (raised) |
| 5 | clock scan | `checks/clock_scan` | offsets 0 in every view, none significant |
| 6 | LOO on the moving objects | `checks/loo/<class>/<view>` per frame (plate, blue pipette, hands) | `checks/gates`: per-view p90 T1 14.1, T2 11.9, T3 4.5, T4 11.7, T5 8.1, fpv 27.0 |
| 7 | fixed -> fpv hand-off | `checks/handoff/cell_culture_plate`, `_inside` | 8.6 px median / 27.0 p90, inside the fpv box on 96.5% of 3556 frames |

**Negative control** (`checks/negative_control`, `world/T5_shipped`,
`world/T5/markers_projected_shipped`): camera 6's shipped pose beside the marker-PnP pose in use:
markers 93.7 vs 0.72 px, static LOO median 97.7 vs 4.8 px (every object 91-112 px under the
shipped pose), left hand 54.4 vs 9.5 px, centres 6.4 cm apart. The red marker outlines in the T5
tile sit a hand's width off the printed markers; the green ones sit on them.

## Storyboard (marked on the `frame` timeline: `storyboard/marks`, `storyboard/index`, `storyboard/current`; also `storyboard.md` beside the recording)

| # | story | raw | proxy | s | what to see |
|---|---|---|---|---|---|
| 1 | a tube goes into the centrifuge | 1099 | 499 | 36.7 | `micro_tube-059` (arm b) enters the centrifuge volume; `contained` starts (`events/b/contained` rises, the World tab lists it) |
| 2 | the lid closes | 1176 | 576 | 39.2 | `world/containers/centrifuge_lid` turns red; the tube is still `contained` |
| 3 | inside while the lid is closed | 1189 | 589 | 39.7 | every tile is empty for the tube (no detector box, no mask); the 3D point sits inside the centrifuge box, dimmed, `[coasting]` |
| 4 | the lid opens | 1228 | 628 | 41.0 | the lid slab turns green; `micro_tube-059` was lost at 1203, 28 frames into the 52-frame closure: the core tracker's 30-frame coast timeout is shorter than the spin |
| 5 | a successor id in the same place | 1230 | 630 | 41.0 | `micro_tube-073` is born where -059 was lost; the events cross-table lists it as the successor; identity across the closure is not claimed |
| 6 | the fpv still sees the plate | 2862 | 2262 | 95.5 | the plate (`cell_culture_plate-009`, one id over the whole window in every arm) projects inside the fpv tile; hand-off residual ~9 px |
| 7 | the fpv pans off the plate | 2923 | 2323 | 97.5 | raw [2863, 2983): the head camera looks away, the plate projects outside its image; the fixed cameras hold the id, the fpv tile shows no plate |
| 8 | and back | 2983 | 2383 | 99.5 | the plate is inside the fpv image again with the same id |
| 9 | the transparent plate masked in six views | 916 | 316 | 30.6 | the six-view annotated frame: `world/<view>/masks/b/cell_culture_plate-0` in all six tiles (the plate has six masks on 3402 of 3600 frames in arm b); the yellow pipette rests beside it in T3-T5 |
| 10 | a confidence drop that is a real failure | 3031 | 2431 | 101.1 | arm (c): `8_channel_pipette-153` falls from confidence 0.64 to 0.08 over the 15 frames either side; its T5 slot `8_channel_pipette#0` mask-vs-box IoU falls from 0.97 to 0.32 as the pipette is picked up: the video-memory mask stays on the rack while the detector box follows the pipette (`world/T5/masks/c/8_channel_pipette-0` against `world/T5/detector`); the track then loses its support and coasts. Every arm's masks are logged on frames 3016-3046 for this |

The picker searched every arm for the largest confidence drop that a slot's IoU confirms; the
runner-up is arm (d)'s `yellow_pipette-128` at raw 3854 (confidence 0.48 -> 0.22, T3 slot
IoU 0.93 -> 0.00, a `track_reproject` re-seed follows at 3869). The centrifuge story repeats on
the second cycle ([3224, 3311), 87 frames): `micro_tube-078` (arm b), contained since 1272, is
lost at 3250 and `micro_tube-230` / `-231` are born at 3312 / 3313. In arm (d) one tube
(`micro_tube-050`) keeps its id through the first closure because the video-memory slot kept
emitting a mask on the closed centrifuge; that is the memory holding an appearance, not a
sighting, and the same track is lost in the second closure.

### The same recording on the tracker extensions (`tracks-ext/`, arms a/b/c)

`runs/finebio-review-P03_03_01-20260925-ext/` was built with `--tracks-dir tracks-ext` on the
`p3-tracker-ext` output (confidence and events re-run into `<arm>/confidence-ext/` and
`<arm>/events-ext/`; 887.6 MB, 402 entities, 41,835 masks, presets validated). There the
storyboard's centrifuge story is the one the plan wrote: `micro_tube-032` (arm b) is
`contained` in the centrifuge from raw 887 to the window end, through both closures
(`world/tracks/b/contained`, label `micro_tube-032 [contained] in centrifuge`), "the lid opens"
at 3311 and "the same id back" at 3313 with no successor; `contained` episodes fall from 145 to
45 in arm (b) because the rack's tubes are one group track, and `held` rows render under
`world/tracks/b/held`. Confidence on the extended tracks: (a) 132,518 rows abstain 1.000; (b)
134,007 rows, median 0.520, abstain 0.668 (0.058 on the 47,271 rows with all five signals); (c)
123,474 rows, 0.485, 0.638 (0.009 on 45,058). Its confidence-drop item is arm (c)'s
`50ml_tube-060` at raw 1919 (T4 `50ml_tube#0` IoU 0.90 -> 0.00).

## Key numbers

Arms (`checks/scoreboard`; det-box IoU is the mask's bbox against the best same-class DINO box,
model vs model): (b) per-frame decode 0.926 median / 99.1% >= 0.5; (c) video memory 0.914 /
79.3%; (d) (c) + re-seeds 0.911 / 75.7%; (b) is the mask source of the floor and the default
arm here. Identity: tracks born 285 / 308 / 270 / 264 (a / b / c / d), the plate and the
machines one id each in every arm, the centrifuge one id in (c)/(d) and 8-9 in (a)/(b), the
in-hand blue pipette 47 / 80 / 30 / 22 ids.

**Human-anchored numbers (gate 2, held Sep 27 on trial 1; not in the recordings).** One person
chose, on 337 of the 440 anchor cells (18 frames: 916 in six views, 12 disagreement and 5 random
frames on the fpv and T4; the 103 cells without a detector box left unlabelled), which SAM3
image-decoder mask is the object, or that the box is not: arm (b)'s own mask on 307, another
candidate on 18, the box rejected on 12 (nine of them the blue-pipette slot sitting on the red
pipette). Against the accepted masks: **(b) 1.000 median / 0.984 mean / 98.8% >= 0.5 IoU** (1.0
wherever its own mask was accepted, the scorer's built-in check), **(c) 0.965 / 0.897 / 93.5%**,
**(d) 0.961 / 0.871 / 91.7%**; the boxes-only arm (a) 0.929 median box IoU against (b) 0.975 and
(c) 0.901. The same order as the label-free scoreboard above, with a 5-point gap on the fraction
>= 0.5 instead of 20: the 18 frames sample the dead memory slots (13 (c) cells at 0.000, the fpv
8-channel pipette and 15 ml tube and the T4 blue pipette) rather than counting every frame of
them. Masks pay over the detector's boxes on the static bench objects, the red pipette and the
tube strip, tie on the yellow pipette, and lose to the boxes on the tubes and the held pipettes.
Identity on the six names given (five static singletons and `red_pipette` on the T2
`blue_pipette#0` cell): pooled IDF1 (c) 0.962 > (a) 0.956 > (d) 0.950 > (b) 0.921, the whole
difference being the centrifuge (one id in (c)/(d), two in (a)/(b) across the lid cycles, the
storyboard's item 1 read on a human name), (b) 1.000 on the six-view frame; `tracks-ext/` gives
the same identity numbers because no tube or held pipette was named. The record without pixels is
[`docs/qa/finebio-P03_03_01-review-anchors.human-record.json`](qa/finebio-P03_03_01-review-anchors.human-record.json);
the tables are in `runs/finebio-anchors-P03_03_01-20260925/scoreboard/README.md` and the ledger
entry "Sep 27: gate 2 held, the anchor scoreboard on trial 1 (p6-anchors)". These numbers say
which arm's masks a human chose and how far the others are from that choice on 18 frames of one
trial; they rank arms and are not accuracy.

**After the gates (Sep 27): the arms on the human-filtered seeds.** Trial 1 was redone downstream
of both gates into `runs/finebio-arms-P03_03_01-filtered-20260927/` ((b) as a row filter of the
Sep 25 observations to the 60 kept slots, exact because the per-frame decode is per-slot
independent; (c) re-run per view on the filtered schedules, 1.26 h of GPU; (a) unchanged and
linked; (d) not re-run: negative on Sep 25, and on the filtered seeds the pre-registered rule
would not have started it). Label-free: (b) 0.928 median / 99.3% >= 0.5 (0.926 / 99.1%), (c)
0.915 / 80.4% (0.914 / 79.3%); the three views that lost a seed answer differently on their
remaining video-memory slots (T2 / T4 / T5 reproduce the Sep 25 masks object for object), so
(c)'s drift moved between the fpv tube and pipette slots (dead slots 9 of 58 -> 10 of 53) rather
than going away. Human-anchored, on the 312 / 311 anchor cells that are not on a rejected slot:
(b) 0.988 mean / 99.0% >= 0.5, identical cell by cell to Sep 25; (c) 0.903 / 93.9% (Sep 25 on
the same cells 0.903 / 93.9%; 14 cells moved by more than 0.02, 8 down and 6 up); counting the 13
rejected mask cells as 0 against the filtered arms, (b) 0.949 / 95.1% and (c) 0.867 / 90.1%.
Identity on the human names: (b) pooled IDF1 0.921 -> 0.919 (the T2 `red_pipette` cell lost its
track with T1's and T3's blue slots), (c) 0.962 -> 0.965; the ranking (b) > (c) on masks, (c) >
(b) on identity through the centrifuge, is unchanged. Tracker identity with the extensions: (b)
born 186 -> 183, ambiguities 54 -> 35, blue pipette 59 -> 51 ids, 8-channel 8 -> 11; (c) 147 ->
155, 22 -> 14, blue pipette 20 -> 30. Full tables in the run README and the ledger entry "Sep 27:
trial-1 downstream redone on the human-filtered seeds (post-gate)"; the anchor scoreboard on the
filtered arms is `runs/finebio-anchors-P03_03_01-20260925/scoreboard-filtered/`. Trial 2 was not
redone (the gates were trial-1 sittings; the plan forbids per-trial tuning).

Confidence (`checks/confidence/<arm>`; five signals ranked within the arm: detector-vs-mask bbox
IoU, cross-view residual, SAM3 object score, detector score, support; abstain when support < 2,
any signal undefined, or bottom decile):

| arm | rows | confidence median (p10 / p90) | abstain, all rows | abstain, rows with all five signals | DDETR disagreement (of which already abstaining) |
|---|---|---|---|---|---|
| (a) boxes only | 153,944 | 0.535 (0.17 / 0.72) | 1.000 (no mask signal exists) | - | 1.7% (100%) |
| (b) per-frame decode | 158,290 | 0.521 (0.19 / 0.72) | 0.723 | 0.070 on 47,196 | 5.4% (42%) |
| (c) video memory | 145,966 | 0.504 (0.21 / 0.70) | 0.698 | 0.038 on 45,867 | 6.2% (83%) |
| (d) + re-seeds | 142,974 | 0.501 (0.22 / 0.71) | 0.716 | 0.024 on 41,549 | 6.2% (81%) |

The abstain fraction is high by construction: tracks supported only by detector rows (racks,
individual micro tubes, hands, pens) carry no mask signal and abstain under the stated rule; on
the rows that carry every signal (the seeded slots' tracks) it is 2-7%. The DINO-vs-DDETR
agreement column is near-redundant: agreement rises from 0.73-0.87 in the lowest confidence
quartile to 0.96-0.97 in the highest, and most disagreeing rows already abstain.

Events (`checks/events/<arm>`; model output; hysteresis enter <= 0 cm / exit >= 3 cm / dwell 5
frames; `held` within 12 cm of a hand track once the object has moved >= 2 cm; `contained` for
tubes, strips and the plate against the container volumes; `proximity` = pipette tip in the plate
volume):

| arm | contained episodes (rack / magnetic rack / centrifuge / vortex) | held (blue pipette / micro tube / 50 ml tube) | proximity | cycle [1176,1228): entered before / ended while closed / same id through / successor after | cycle [3224,3311) |
|---|---|---|---|---|---|
| (a) | 140 (80 / 45 / 10 / 5) | 53 (25 / 19 / 9) | 0 | 3 / 2 / 0 / 3 | 2 / 2 / 0 / 2 |
| (b) | 145 (81 / 50 / 9 / 5) | 64 (34 / 17 / 11) | 0 | 3 / 1 / 0 / 3 | 2 / 2 / 0 / 2 |
| (c) | 158 (90 / 60 / 7 / 1) | 56 (23 / 20 / 11) | 0 | 2 / 1 / 0 / 0 | 2 / 2 / 0 / 2 |
| (d) | 192 (113 / 72 / 6 / 1) | 33 (13 / 8 / 10) | 0 | 3 / 1 / 1 / 0 | 2 / 2 / 0 / 1 |

No `proximity` episode exists in this window: no pipette comes within 30 cm of the plate (the
plate is a bench object in protocol 03; `--proximity-targets` accepts other volumes). The lid is
closed on 638 of the 3600 frames; contained track-frames while closed 11-12k per arm.

## Trial 2 (`P20_03_01`, room 2): the same build on the second trial

`runs/finebio-review-P20_03_01-20260925-ext/review.rrd` is the trial-2 recording (`p6-trial2`;
ledger entries "Sep 25: second trial P20_03_01 with zero tuning" and "Sep 25: FineBio 3D
tracking phase, close-out"): raw frames [600, 4200) of `P20_03_01` (room 2, protocol 03,
six-view annotated frame 1442 = proxy 842), six views, built on the tracker extensions'
`tracks-ext/` of arms (a), (b), (c) with the three commands below and **no parameter changed
against trial 1** (arm (d) was negative on trial 1 and did not run here). Room 2's own camera
solve (day 221124; cameras 1-4 re-solved by marker PnP at 0.7-2.1 px because no shipped pose of
the day fits them; camera 6 kept shipped at 7.5 px) and its own rig gates (association 27.5 px;
hand-off 80 px, the cap). **817.2 MB, 399 entity paths, 37,058 mask cut-outs, `rerun rrd
verify` clean ("1 file verified without error"), 356 s to build**; its three presets resolve
every query against its own entity tree (`world.rbl` 8 views / 47 queries, `cameras.rbl` 8 /
41, `evidence.rbl` 28 / 65) and live beside the recording only: the committed
`configs/rerun/finebio_*.rbl` are the trial-1 presets, bound to that recording's application id
(`finebio-review-P03_03_01`), so trial 2 was built with `--preset-dir ""` and its presets are
bound to `finebio-review-P20_03_01`.

### Open

```bash
E=runs/finebio-review-P20_03_01-20260925-ext
uv run rerun $E/review.rrd $E/world.rbl      # World: rig, arm (b) tracks incl. contained / held, volumes, events, confidence, storyboard
uv run rerun $E/review.rrd $E/cameras.rbl    # Cameras: six tiles, DINO boxes, masks of (b), track ids, seeds
uv run rerun $E/review.rrd $E/evidence.rbl   # Evidence: cross-checks, the negative control on T1-T4 (numbers), reports
```

Raw frame = proxy frame + 600, as on trial 1. Rebuild (CPU: 12-16 s per arm for confidence,
5-8 s for events, about 6 min for the recording); the lid intervals and the three centrifuge
cycles in the window ([895, 937), [2500, 2539), [3798, 3834)) are read by both tools from
`configs/finebio/trials.json` through the clip config's trial id, nothing is passed for them:

```bash
R=runs/finebio-arms-P20_03_01-20260925
for arm in a-boxes-only b-box-decode-arm c-video-memory-arm; do
  uv run battle-finebio-confidence --arm-dir $R/$arm --detections runs/finebio-detect-P20_03_01-600-4200-20260925/dino --tracks-dir tracks-ext --output $R/$arm/confidence-ext
  uv run battle-finebio-events --arm-dir $R/$arm --config configs/clips/finebio_P20_03_01_600-4200.json --rig runs/finebio-rig-P20_03_01-600-4200/rig.json --tracks-dir tracks-ext --output $R/$arm/events-ext
done
uv run battle-finebio-viewer --clip-config configs/clips/finebio_P20_03_01_600-4200.json \
  --arm-dirs a=$R/a-boxes-only,b=$R/b-box-decode-arm,c=$R/c-video-memory-arm \
  --rig runs/finebio-rig-P20_03_01-600-4200/rig.json --seeds runs/finebio-seeds-P20_03_01-20260925/with-plate \
  --tracks-dir tracks-ext --preset-dir "" --output runs/finebio-review-P20_03_01-20260925-ext
```

### What differs from the trial-1 recording

- **The negative control is a document here, not a drawing.** In room 2 the cameras the solve
  replaced are T1-T4, and `checks/negative_control` carries their numbers (marker RMS 0.97 /
  1.80 / 0.71 / 2.09 px under the marker-PnP pose vs 29.4 / 16.2 / 14.7 / 32.2 shipped; static
  LOO median 7.9 / 11.8 / 8.0 / 7.0 vs 23.7 / 29.9 / 19.0 / 31.6 px; centres 2.8-4.8 cm apart):
  the checks catch the shipped error on four cameras where trial 1 had one. But the viewer draws
  the shipped frustum and the red marker set only for T5 (`world/T5_shipped`,
  `world/T5/markers_projected_shipped`, logged when T5 is `marker_pnp`), and T5 is the one
  camera that kept its shipped pose in room 2, so neither entity exists in this recording and
  the document's opening sentence still speaks of camera 6. **Named as a trial-1 constant in the
  review surface**: the drawn control should iterate over every `marker_pnp` view; recorded,
  not patched, in the close-out entry.
- **The head camera's frustum has gaps.** The fpv pose is invalid on 248 frames in six runs
  around the three spins ([892, 908] + [915, 991], [2503, 2511] + [2523, 2588], [3800, 3806] +
  [3812, 3882]); `world/fpv`, its trail and markers, `world/<view>/fpv_camera_centre` and the
  `checks/handoff/*` series are empty there, and the fpv tile shows the video without a
  projected track. The fixed cameras carry the spins (the centrifuge volume was built from
  T2/T3/T4/T5).
- **No plate mask in the T2 tile, by design.** DINO never sees the plate in T2 (recorded as
  `detector_unseeded`; Deformable DETR's box was not substituted), so there is no
  `world/T2/masks/b/cell_culture_plate-0` entity; the plate is one id on 3600/3600 frames from
  T1 / T3 / T5 / fpv and T4 from raw 1250. T2's eleventh slot is the `magnetic_rack` instead
  (the trial-1 slot cap's side effect).
- **Storyboard: six items, not ten.** The picker found no fpv look-away (the plate projects
  inside the head camera on 100% of its 3352 valid-pose frames, against 96.5% in room 1) and no
  frame with the plate masked in all six views (impossible without T2), so the three fpv items
  and the plate item are absent by the picker's own rules; the centrifuge story is the plan's
  as written, and the confidence drop is arm (c)'s again:

  | # | story | raw | proxy | s | what to see |
  |---|---|---|---|---|---|
  | 1 | a tube in the centrifuge | 600 | 0 | 20.0 | `micro_tube-013` (arm b) is inside the centrifuge volume from the window start; `contained` starts |
  | 2 | the lid closes | 895 | 295 | 29.9 | `world/containers/centrifuge_lid` turns red (cycle [895, 937), 42 frames); the tube stays `contained` |
  | 3 | inside while the lid is closed | 915 | 315 | 30.5 | every tile empty for the tube; the point sits inside the box under `world/tracks/b/contained` |
  | 4 | the lid opens | 937 | 337 | 31.3 | the slab turns green; `micro_tube-013` has kept its id through the closure |
  | 5 | the same id back | 939 | 339 | 31.3 | `micro_tube-013` observed again inside the centrifuge; the same id also holds through the spins at [2500, 2539) and [3798, 3834), as does `micro_tube-033` (contained from 838) |
  | 6 | a confidence drop that is a real failure | 1188 | 588 | 39.6 | arm (c): `50ml_tube-008` falls from confidence 0.58 to 0.30 over the 15 frames either side; its T4 slot `50ml_tube#0` mask-vs-box IoU falls from 0.92 to 0.00 as the video-memory mask leaves the detector's tube (`world/T4/masks/c/50ml_tube-0` against `world/T4/detector`) |

- **Eight container volumes** (`world/containers/*`: centrifuge, PCR machine, vortex, trash
  can, micro-tube rack, tube-strip rack and, new against room 1's seven, the magnetic rack; no
  15 ml rack on this bench), from the rig's static points and the arm's own box widths;
  centrifuge centre (-1.2, -22.4) cm, half-extent 9.7 cm.
- **The fpv residual series sit higher.** `checks/residuals/<arm>/fpv` reads 15-16 px median
  against 9 in room 1: the fpv gate is the hand-off gate, which came out at the 80 px cap
  because the rig's moving-object witness list names the `blue_pipette` (in use in room 1,
  resting in room 2), the one overfit the trial-2 entry names. The fixed views' residuals are
  4.5-7.7 px in both rooms.

### Two-trial scoreboard

Same measures, same reference (the best same-class DINO box; model vs model), same code and
parameters; trial 2 differs only in the trial id. Core = `tracks/`, ext = `tracks-ext/`; (a) /
(b) / (c) throughout, trial 1's (d) in brackets where it exists.

| measure | trial 1 `P03_03_01` (room 1) | trial 2 `P20_03_01` (room 2) | where |
|---|---|---|---|
| cameras in use | T1-T4 shipped (4.8-6.7 px), T5 marker PnP (0.72 px; shipped 93.7) | T1-T4 marker PnP (0.7-2.1 px; shipped 14.7-32.2), T5 shipped (7.47 px) | `configs/finebio/cameras/<trial>_600-4200.json` |
| static LOO median -> association gate | 10.0 px -> 30.1 px | 9.2 px -> 27.5 px | `runs/finebio-rig-<trial>-600-4200/rig.json` |
| hand-off gate | 27.0 px (the fpv sets it) | **80.0 px, the cap** (the resting blue pipette sets it) | same |
| fpv pose valid / plate fixed -> fpv hand-off | 3587 / 3600; 8.6 px, inside 96.5% | 3352 / 3600; 4.4 px, inside 100% | same |
| seeds accepted / plate slots | 66 / 66; six views | 66 / 66; five views (T2 `detector_unseeded`) | `runs/finebio-seeds-<trial>-20260925/with-plate/seeds.md` |
| (b) det-box IoU median / >= 0.5 | **0.926 / 99.1%** | **0.919 / 98.5%** | `runs/finebio-arms-<trial>-20260925/scoreboard/scoreboard.md` |
| (c) det-box IoU median / >= 0.5 | 0.914 / 79.3% [(d) 0.911 / 75.7%] | 0.883 / 65.3% [(d) not run] | same |
| (c) - (b) on the median; slots with median IoU < 0.35 | -0.012; 9 of 58 | -0.036; 24 of 61 | same; `c-video-memory-arm/measures.md` |
| tracks born, core -> ext | 285 / 308 / 270 -> 191 / 186 / 147 | 290 / 321 / 208 -> 166 / 202 / 152 | `<arm>/tracks[-ext]/identity_metrics.json` |
| ambiguities, core -> ext | 126 / 158 / 133 -> 33 / 54 / 22 | 70 / 97 / 25 -> 8 / 35 / 7 | same |
| pipette in use, ids, core -> ext | blue 47 / 80 / 30 -> 41 / 59 / 20 | yellow 32 / 58 / 25 -> 24 / 48 / 21 | same |
| micro tubes, ids, core -> ext | 96 / 100 / 101 -> 32 / 26 / 17 | 100 / 100 / 41 -> 18 / 17 / 9 | same |
| centrifuge ids (core) | 8 / 9 / 1 | 7 / 2 / 1 | same |
| plate, machines, trash can, tip racks in use | one id each, every arm | one id each, every arm (yellow tip rack 2: relocated) | same |
| confidence (ext) median; abstain all rows / rows with all five signals | (a) 0.531 / 1.000; (b) 0.520 / 0.668 / 0.058 on 47,271; (c) 0.485 / 0.638 / 0.009 on 45,058 | (a) 0.516 / 1.000; (b) 0.484 / 0.623 / 0.013 on 45,791; (c) 0.505 / 0.640 / 0.015 on 41,264 | `<arm>/confidence-ext/confidence_summary.json` |
| DDETR disagreement (ext, b) / agreement by confidence quartile | 6.3% / 0.75 -> 0.97 | 3.7% / 0.74 -> 0.96 | same |
| events (ext): contained / held / proximity | 43 / 45 / 36; 55 / 61 / 58; 0 | 21 / 20 / 11; 60 / 76 / 25; 0 | `<arm>/events-ext/events_summary.json` |
| centrifuge cycles in the window; lid closed frames | 2 ([1176, 1228), [3224, 3311)); 638 | 3 ([895, 937), [2500, 2539), [3798, 3834)); 117 | `configs/finebio/trials.json` |
| (b) tubes `contained` in the centrifuge with the same id through every closure | 1 (`micro_tube-032`, from 887) | 2 (`micro_tube-013` from 600, `-033` from 838; `-037` through the last two) | `<arm>/events-ext/events_summary.json` -> `centrifuge_cycles_vs_contained` |
| recording (ext): size / entities / masks / storyboard items | 887.6 MB / 402 / 41,835 / 10 | 817.2 MB / 399 / 37,058 / 6 | `runs/finebio-review-<trial>-20260925-ext/review_index.json` |
| worker cost | (b) 150 ms per prompted frame, 2.2 GiB; (c) 195-217 ms/step, 3.6-4.2 GiB | (b) 150 ms, 2.2 GiB; (c) 202-213 ms/step, 3.6-4.1 GiB | `runs/finebio-arms-<trial>-20260925/README.md` |
| GPU | about 4.0 h (with (d) 1.8 h) | about 2.3 h | same |

Every (a) row abstains on both trials because a boxes-only arm carries no mask signal; its
median is the rank combination of the three signals it does have and compares to nothing.

## What to look at first (20 minutes, both trials)

Times are rough; every step names the preset and the entities. Both recordings step on the
`frame` timeline (raw frame index); `storyboard/marks` jumps to the storyboard frames.

1. **The checks can fail (3 min).** Trial 1, Evidence preset: the T5 tile with the green
   (`world/T5/markers_projected`) and red (`_shipped`) marker outlines a hand's width apart, the
   two T5 frusta in the rig view, `checks/negative_control` (93.7 vs 0.72 px on the markers,
   97.7 vs 4.8 px static LOO). Then trial 2's `checks/negative_control`: the same table on four
   cameras (T1-T4), numbers only.
2. **The storyboard's centrifuge story, three ways (5 min).** Trial 1 core recording
   (`runs/finebio-review-P03_03_01-20260925/`), World preset, items 1-5: the tube is lost 28
   frames into a 52-frame closure and a successor id is born (the 30-frame coast timeout).
   Trial 1 ext recording (`-ext/`), items 1-5: `micro_tube-032 [contained] in centrifuge`
   through both closures, the same id back at 3313. Trial 2 ext, items 1-5: `micro_tube-013`
   through three spins; watch `events/lid_closed` against `events/b/contained` in the strip.
3. **The transparent plate (3 min).** Cameras preset, trial 1 at raw 916 (the six-view
   annotated frame): `world/<view>/masks/b/cell_culture_plate-0` in all six tiles, the yellow
   pipette beside it in T3-T5. Trial 2 at raw 1442 (proxy 842): the plate in five tiles and an
   empty T2 tile, by design.
4. **The object in the hand, where everything is weakest (3 min).** Trial 1, Cameras at raw
   4032 (T4 blue pipette, the anchors' lowest mask-vs-box IoU 0.06) and 1521 / 2380 / 3054 (fpv):
   the mask on the glove, the ids under `world/<view>/tracks/b` changing. Trial 2, Cameras
   over raw 3950-4087: eight `yellow_pipette-*` ids are born in those 137 frames (48 over the
   window on `tracks-ext/`), the densest turnover of the in-hand object;
   `confidence/b/yellow_pipette` in the series below the tiles (median 0.23, abstain 45%).
5. **A confidence drop that is a real failure (2 min).** Trial 1 item 10 (raw 3031): arm (c)'s
   `world/T5/masks/c/8_channel_pipette-0` staying on the rack while `world/T5/detector` follows
   the pipette; `confidence/c/tracks/8_channel_pipette-153` 0.64 -> 0.08. Trial 2 item 6 (raw
   1188): `world/T4/masks/c/50ml_tube-0` vs the detector, `confidence/c/tracks/50ml_tube-008`
   0.58 -> 0.30. Toggle `world/<view>/masks/b/*` on the same frames: the per-frame decode has
   not drifted.
6. **The decision and its evidence (2 min).** Evidence preset, `checks/scoreboard` on either
   trial: (b) 0.926 / 0.919 median and 99.1 / 98.5% >= 0.5 against (c) 0.914 / 0.883 and 79.3 /
   65.3%; `checks/measures` for arm (c), the per-slot table with the 0.000 rows.
7. **What did not transfer (2 min).** Evidence preset, `checks/gates` on both trials (hand-off
   27.0 vs 80.0 px) and `checks/residuals/b/fpv` (9 vs 15 px median): the witness class list
   inside the hand-off formula. And in trial 2's World preset, the fpv frustum vanishing for
   2.6-3.1 s at every spin (`world/fpv` empty on the 248 invalid-pose frames).

## Files

`runs/finebio-review-P03_03_01-20260925/`: `review.rrd` (915.6 MB: the six proxies about
490 MB, 44,448 mask cut-outs about 400 MB, the rest tracks, boxes and series; 508 entity paths;
`rerun rrd verify` clean; built in 503 s), `world.rbl`, `cameras.rbl`,
`evidence.rbl`, `presets_check.json` (every view's queries resolved against the entity tree),
`review_index.json` (inputs, strides, entity counts, storyboard, containers, timings),
`entity_paths.txt`, `storyboard.md` / `.json`, `rrd_verify.txt`, `rrd_stats.txt`. Per arm:
`<arm>/confidence/{confidence.jsonl,confidence_summary.json,confidence.md}` and
`<arm>/events/{events.jsonl,episodes.jsonl,events_strip.jsonl,events_summary.json,events.md}`.

`runs/finebio-review-P03_03_01-20260925-ext/` (trial 1 on `tracks-ext/`, 887.6 MB) and
`runs/finebio-review-P20_03_01-20260925-ext/` (trial 2 on `tracks-ext/`, 817.2 MB, 89,285
chunks, 698,620 rows, built in 356 s): the same file set, presets bound to their own
recording; per arm `<arm>/confidence-ext/` and `<arm>/events-ext/` under
`runs/finebio-arms-<trial>-20260925/`. Build logs beside each directory
(`runs/finebio-review-<trial>-20260925[-ext].build.log`). All gitignored.

`runs/finebio-review-P03_03_01-filtered-20260927/` (trial 1 on the human-filtered seeds and
`tracks-ext/`, arms a / b / c; the recommended trial-1 recording since Sep 27): the same file
set, presets beside the recording, `review_index.json` naming the filtered arms root
`runs/finebio-arms-P03_03_01-filtered-20260927/` (per arm `<arm>/confidence-ext/`,
`<arm>/events-ext/`), build log `runs/finebio-review-P03_03_01-filtered-20260927.build.log`.
Gitignored.
