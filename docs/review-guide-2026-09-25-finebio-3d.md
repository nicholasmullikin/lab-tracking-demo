# Review guide, Sep 25, 2026: the FineBio 3D object-tracking recording (trial 1, `P03_03_01`)

One Rerun recording of the trial-1 window (raw frames [600, 4200), 20.02-140.14 s, six views)
with three layout presets, built by `battle-finebio-viewer` from the tracking arms, the
confidence and events passes, the rig check, the seeds and the lane-B proxies
(plan `docs/plan-2026-09-25-finebio-3d-tracking.md`, todos `p5-confidence`, `p5-events`,
`p5-viewer`; ledger entry "Sep 25: confidence, events and the review recording"). Nothing in
the recording is committed (FineBio licence: it holds the proxy videos and SAM3 masks); the
presets under `configs/rerun/finebio_{world,cameras,evidence}.rbl` hold entity paths only.

**Claim boundary, read first.** Everything drawn is model output. The FineBio DINO detector was
trained on FineBio's own bench and on frames from these cameras, so its boxes, and every SAM3
mask prompted from them, are agreement between two models, not accuracy. "Confidence" ranks a
track's rows within one arm by five label-free signals; it is not a probability of being right.
Events are geometry on 3D tracks against volumes derived from the rig. No human anchor exists yet
(`p6-anchors`); the human-facing frame is the six-view annotated frame 916, which the storyboard
lands on.

## Open

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

## Files

`runs/finebio-review-P03_03_01-20260925/`: `review.rrd` (915.6 MB: the six proxies about
490 MB, 44,448 mask cut-outs about 400 MB, the rest tracks, boxes and series; 508 entity paths;
`rerun rrd verify` clean; built in 503 s), `world.rbl`, `cameras.rbl`,
`evidence.rbl`, `presets_check.json` (every view's queries resolved against the entity tree),
`review_index.json` (inputs, strides, entity counts, storyboard, containers, timings),
`entity_paths.txt`, `storyboard.md` / `.json`, `rrd_verify.txt`, `rrd_stats.txt`. Per arm:
`<arm>/confidence/{confidence.jsonl,confidence_summary.json,confidence.md}` and
`<arm>/events/{events.jsonl,episodes.jsonl,events_strip.jsonl,events_summary.json,events.md}`.
