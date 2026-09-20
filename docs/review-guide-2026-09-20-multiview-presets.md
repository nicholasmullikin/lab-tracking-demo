# Review guide: the v6 package, one recording and three presets (Sep 20)

Written for the human who asked three things of the v5 package: which analysis is actually
multi-camera and why the recording did not show it; how to compare SAM3, DAM4SAM and the human
label on the same frame; and whether hands and segmentation could be separate layouts that switch
every panel at once. v6 answers each with logged data, not prose: nine camera tiles, the
candidate arms beside the reference, the human anchors drawn on their frames, and three
blueprint presets over one recording. Everything is CPU work over retained runs; no model ran, the
GPU was not touched, and the agent did not open a viewer (see "How the presets were validated").
Paths are for this machine (`runs/` is gitignored).

Standing rule: the human anchors are 13 frames on one view (C10379); they rank arms against each
other and support no accuracy claim. Every other view's seeds were chosen by an agent from
geometry and no human has reviewed a mask on them. Consensus, hull and detector series measure
disagreement between estimates, never accuracy. Ensemble v2 remains a candidate reference; adopting
it is your disposition.

## Open it

One recording, three layouts. Pass the preset after the recording:

```bash
cd /home/nick/src/battle
uv run rerun runs/interaction-review-first-minute-v6/interaction_review_combined.rrd runs/interaction-review-first-minute-v6/segmentation.rbl
uv run rerun runs/interaction-review-first-minute-v6/interaction_review_combined.rrd runs/interaction-review-first-minute-v6/hands.rbl
uv run rerun runs/interaction-review-first-minute-v6/interaction_review_combined.rrd runs/interaction-review-first-minute-v6/multiview.rbl
```

To switch presets without reloading the 406 MB recording, drag the other `.rbl` onto the open
viewer window (or File > Open); a blueprint file for the same application id
(`battle-interaction-review-v6`) replaces the active layout and keeps the data. The combined
recording carries **no** embedded layout on purpose, so a preset is the only blueprint in play;
opened alone it falls back to Rerun's automatic layout, which is not useful here.

Over the tailnet: `scripts/serve_review_over_tailscale.sh runs/interaction-review-first-minute-v6/interaction_review_combined.rrd`
on this machine, then `rerun --connect rerun+http://100.64.0.7:9876/proxy` on the client
(native viewer, Rerun 0.37.1, `ffmpeg` >= 5.1 for the H.264). The presets are files on this
machine: copy the three `.rbl` to the client and drag them onto the connected viewer.

Files (`runs/interaction-review-first-minute-v6/`):

| file | size | what |
| --- | --- | --- |
| `interaction_review_combined.rrd` | 406.4 MB | the one recording: the v5-style package + the nine-view comparison in one store (`application_id battle-interaction-review-v6`, `recording_id interaction_review_first_minute_v6`) |
| `interaction_review_first_minute_v4.rrd` | 154.5 MB | the v5-style package alone (builder output, fingerprinted by `interaction_review_index.json`; carries the builder's default layout) |
| `runs/multiview-static-comparison-first-minute-r1280-pm-append/multiview_static_comparison.rrd` | 253.9 MB | the nine-view comparison alone (no embedded layout; same ids, so `multiview.rbl` applies to it too) |
| `segmentation.rbl`, `hands.rbl`, `multiview.rbl` | 131 / 94 / 132 KB | the presets |
| `presets_check.json` | | the validation report (every view's contents resolved against the recording's 384 entity paths) |
| `interaction_review_index.json`, `review_guide.md`, `pinned_moments_contact_sheet.png` | | the usual package index, guide and sheet |

Build: `battle-build-multiview-static-comparison` 14.5 s, `battle-build-interaction-review-v4`
47.4 s (validate 2.8, geometry 13.1, export 31.5), merge and presets 7 s; 69 s in total, from the
mask caches (`battle-cache-masks` was run once for the three arms that had none: 38-67 s each).
The exact commands are in the README ("First-minute v4 review", v6 paragraph).

## Which analyses are multi-camera, plainly

- **SAM3 runs on nine cameras**: C10379 and the seven other statics (C10095, C10115, C10118,
  C10119, C10390, C10395, C10404) plus the ego camera e4 (HMC_21179183), every one at 1280 px
  with `--prompt-memory-semantics append`, first minute, one tracker per view. v5 showed only
  C10379 because the nine-view recording lived in another file (Sep 18, 720 px, eight views, no
  interior, an older consensus). v6 logs all nine under
  `world/assembly101_multiview_first_minute/views/<VIEW>/` in the same recording.
- **Only C10379 has human input**: the four frame-0 seeds, the four corrections (327, 900, 1172,
  1235) and the 52 anchors. The other eight views were **seeded by the agent** from geometry
  (Sep 18 table-plane transfer; the Sep 20 seed search re-seeded rear_body and cabin but those
  runs are the `-seeded` roots, not the canonical ones shown here) and **no human has reviewed
  any mask on them**. Their tiles say so in the name: `(agent seeds, unreviewed)`.
- **No view but C10379 tracks the interior.** The consensus and hull for the interior are
  therefore empty (`multiview_consensus_frames_interior: 0`), and every interior series that
  depends on other cameras is undefined by construction.
- **Cross-view consensus** (`diagnostics/multiview/<part>/{c10379_error_px, views_used}` in the
  review root; per-view error under the multiview root) triangulates the nine masks' centroids
  and reports how far each view's mask sits from the majority: C10379 is contradicted on 115
  frames with `pm-append` as reference (chassis `[296,313)` `[475,494)` `[508,515)` `[1049,1057)`
  `[1058,1085)`, rear_body `[1662,1667)` `[1762,1794)`), 105 episodes over all views.
- **Visual hull** (`world_mm_3d/hull/<part>`, 1 fps voxels; `views/<VIEW>/hull_projection/<part>`)
  is the intersection of the nine silhouette cones: a lower bound on the part, over-carved
  wherever a mask misses a part's occluded half.
- **Single-camera, C10379 only**: the reference masks and every candidate arm, WiLoR (raw and
  stabilized), MediaPipe, BoxMOT, Kineo, the contact geometry, the detector confidence series.
  **Multi-camera hands**: the dataset's own 60 fps hand tracker (world mm, projected into C10379
  through an estimated camera) and the two ATHENA triangulations (eight-view MediaPipe,
  three-view WiLoR) in the world-mm 3D view.

## The three presets

### `segmentation.rbl` (15 views): SAM3 vs DAM4SAM vs the human label on one frame

Top row: the **reference** (ensemble v2: `pm-append` primary, DAM4SAM-large fallback inside the
label-free intervals; magenta boxes mark the 48 fallback frames; white-on-colour outlines are the
human anchor masks on the 13 anchor frames) beside the series column. Second row, **six
same-frame tiles**, one per candidate arm, each the video plus that arm's four part masks:

| tile (`comparison/segmentation/<arm>`) | run | anchor IoU all |
| --- | --- | --- |
| `pm-append` | SAM3 1280 px, append prompt bank, four human corrections (the v2 primary) | 0.743 |
| `dam4sam-large-1024-sched-60s` | DAM4SAM large, shared predictor, same schedule (the v2 fallback) | 0.715 |
| `off-r1280-sched` | SAM3 1280 px, replace-semantics bank (the 1280 baseline) | 0.724 |
| `old-reference-off-r720-sched` | SAM3 720 px, the run `configs/ensemble_reference/first_minute_v1.json` calls primary (`...20260918t001210z`) | 0.668 |
| `ensemble-v1` | the v4 reference (720 px primary, DAM4SAM tiny fallback, hidden interior) | 0.681 |
| `human_anchors` | the human's accepted decoder masks, present on the 13 anchor frames only | (the yardstick) |

The anchor scores are the Sep 19 scoreboard's; the arm choice for v2 saw those anchors, so the
first two rows are selection-biased. The tiles are locked to the same frame and the same visual
bounds, so a frame-step compares all six at once. Their mask areas are the panel
`diagnostics/segmentation/<arm>/<part>` (area in px; a collapse or a growth is a step there).

Series column, top to bottom:

- **Reference provenance** (`diagnostics/reference_provenance/<part>`): `1 sam3 primary`,
  `2 dam4sam fallback`. There is **no code 3**: ensemble v2 labels no hidden interval, and the v5
  legend that still listed `3 hidden` was wrong. The frame-1700 screwdriver is not a provenance
  state; it is the rear_body `[1660,1800)` `not_contact_eligible: distractor_confusion` interval,
  named in the legend and visible as `diagnostics/segmentation_contact_eligible/rear_body`
  dropping to 0 (the mask is still drawn so you can see the failure).
- **Consensus contradiction** (`diagnostics/multiview/<part>/c10379_error_px`, `views_used`):
  C10379's centroid error against the other cameras in raw px, and how many views formed the
  consensus.
- **Detector confidence** (`diagnostics/confidence/<part>/confidence`, crosses =
  `.../abstain`), see the next section.
- **Anchor marks** (`metadata/anchors/anchor_frame` diamonds, `failed_cells` crosses).
- **Candidate arm areas**.

Bottom row: the **anchor log** (one line per anchor frame naming each part's state and whether
`pm-append` fails it; click a line to jump), the per-frame document (provenance and eligibility
per part, the active fine-grained GT, the consensus lines), and the package guide.

### `hands.rbl` (10 views): every hand source, no masks

Top row, four same-frame tiles: stabilized WiLoR (the primary hand layer), raw WiLoR, MediaPipe,
and the Assembly101 dataset hands projected into C10379 (+9 pose-frame static offset). Bottom
row: the world-mm 3D view (dataset hands in yellow/mint, ATHENA eight-view MediaPipe in
orange/blue, ATHENA three-view WiLoR in magenta/green, the C10379 frustum; the consensus
centroids and hull voxels are excluded from this view), the WiLoR camera-relative non-metric 3D,
and the series: MediaPipe-vs-WiLoR disagreement with the stabilized layer's state counts
(`low_confidence_continuation` / `fallback` / `missing`), dataset hand confidence and
wrist-to-WiLoR distance, ATHENA wrist disagreement in mm and contributing views, the detector
confidence (context only), and the hand-to-part contact distances. Nothing segmentation-related
is drawn on the tiles.

### `multiview.rbl` (18 views): the nine cameras

Top: a 3 x 3 grid of camera tiles, each with its SAM3 part masks (every second frame,
`--mask-every 2`), the consensus point reprojected into that view with its pixel error as the
label, the hull projection (toggle it in the tree), and:

- on **C10379**: the human anchor masks as outlines at the 13 anchor frames
  (`views/C10379/human_anchor_outlines/<part>`);
- on the **eight other views**: the seed-search proposals as outlined candidates at their frames
  (`views/<VIEW>/proposals/<part>`: chassis and rear_body at frame 0, interior at frame 427-430;
  white = candidate 00, orange = 01, cyan = 02, violet = 03; the label reads
  `proposal: accept/reject pending` and lists the strategies). Deciding them is session (c) of
  [`labeling-sessions-2026-09-20.md`](labeling-sessions-2026-09-20.md).

Bottom: the world-mm 3D view (consensus centroids, dataset hands, eight static frusta, hull voxels
at 1 fps), tabs with the C10379 contradiction series, one per-part tab of every view's error, and
the anchor marks, and two documents: the episode list of this consensus build and the same list as
the review package logged it.

## How to read the confidence and abstain series

`runs/detector-scorecard-20260920/pm-append/confidence.jsonl` is logged as
`diagnostics/confidence/<part>/confidence`: per frame and part, `1 - suspicion`, where suspicion
is the mean normalized rank of the three detectors that scored best against the 52 anchors on
`pm-append` (SAM3's own object score, seed-area drift, area jump; in-sample AUROC 0.941). A cross
on `.../abstain` marks a frame whose confidence is at or below the in-sample threshold that
recalls >= 0.8 of the anchor failures (confidence <= 0.187): 1,022 of 7,200 frame x part rows
(chassis 277, interior 484, rear_body 261, cabin 0). Read it as a **ranker of suspicious frames,
not a gate**: held out one frame at a time the same threshold recalls 0.44 of the failures at
precision 0.36 (F1-max: 0.43 / 0.33), because on 13 frames the threshold sits at the lowest
training positive and does not transfer. The series is frame-noisy (no smoothing). Where it is
low and the other cameras agree with C10379 (consensus error small), the detectors are reacting
to something the centroid test cannot see; where both fire, look at the tile.

Anchor marks: `metadata/anchors/anchor_frame` is 1 at each of the 13 anchor frames (300, 370,
400, 600, 650, 700, 900, 1050, 1100, 1150, 1200, 1500, 1700) and cleared on the next frame;
`metadata/anchors/failed_cells` is how many of the four cells `pm-append` fails there (IoU < 0.5,
or, on the hidden cell, more than 300 px): 600 chassis; 650 chassis; 700 interior; 1050 chassis
and interior; 1100 interior; 1150 interior; 1500 rear_body; 1700 rear_body (1,265 px on the
screwdriver); 9 cells in all, 0 in `[279,408)` and at 900. The anchor log names them. Human
anchor outlines exist only on the anchor frame itself; step to the frame (the diamonds on the
time panel, or the log) rather than expecting them to persist.

## Frames to scrub

- **0** (`multiview.rbl`): the nine seeds. C10379's four are human; the other eight views'
  chassis/rear_body/cabin are agent transfers, and the proposal outlines on those tiles are the
  seed search's alternatives. No view but C10379 has an interior.
- **300 / 370 / 400** (`segmentation.rbl`): the window you reported (279-408). Human outlines on
  every arm; no anchor cell fails on `pm-append` here, and the other cameras do not contradict
  C10379. Whatever you saw is not a centroid displacement.
- **480-515** (`segmentation.rbl`): label-free chassis fallback (magenta) in the reference; the
  DAM4SAM tile shows the source mask, the `pm-append` tile the SAM3 chassis that shrank; the
  consensus series shows why the interval exists; the confidence series flags 480 and 500 (two
  of the detector-selected extra frames sent to the C10119 session).
- **600 / 650 / 700** (`segmentation.rbl`): the rotation window. Chassis fails at 600 and 650 on
  `pm-append` (0.25 / 0.20), interior at 700 (0.41); compare the six tiles against the outline.
  `multiview.rbl`: with `pm-append` the other cameras no longer contradict C10379 here.
- **1048-1085** (`segmentation.rbl`): the swap. 1049-1056 the SAM3 chassis collapses and the
  substitution is declined; 1057-1084 the magenta DAM4SAM chassis takes over and overlaps the
  SAM3 interior (both tiles show it). Anchor 1050: chassis and interior fail; 1100 / 1150 interior
  fails but the chassis is back.
- **1500** (`segmentation.rbl`): rear_body 0.00 in every SAM3 arm (label on the table piece);
  DAM4SAM tile for comparison.
- **1660-1800** (`segmentation.rbl` and `multiview.rbl`): the screwdriver. Every arm's rear_body
  sits on it at 1700 (hidden cell, 1,265 px); the reference marks the interval
  `distractor_confusion`; on the other tiles the agent-seeded rear_body slots are on the same
  yellow piece from about 1660 (the consensus contradicted C10379 on `[1762,1794)` in this build;
  the seeded rerun removes that contradiction because all views then agree on the wrong piece).
- **427-430** (`multiview.rbl`): the interior rest-frame proposals on the eight other views.

## How the presets were validated (no viewer was opened)

`rerun --headless` does not exist in 0.37.1 and the agent did not open a window. Instead:

1. each `.rbl` was written with `rerun.blueprint.Blueprint.save(application_id, path)` and read
   back through `rerun.experimental.RrdReader` (one blueprint store, application id equal to the
   recording's);
2. `rerun rrd verify` passes on the combined recording and on each preset;
3. `battle-review-presets` walks every view of each preset (its `space_origin` and
   `ViewContents` queries), expands `$origin`, and requires every inclusion to match at least one
   of the recording's 384 entity paths (`/**` = the path or any descendant, otherwise exact;
   exclusions `- ...` are not required to match). `presets_check.json`: segmentation 15 views /
   25 queries, hands 10 / 21, multiview 18 / 47, all matched. `tests/test_review_presets.py`
   covers the resolver and the writer on a synthetic recording.

What that does not prove: that the layout is pleasant, that the video decodes on your client,
or that a 3D view's camera starts where you want it. If a tile is black, check the entity toggle
in the blueprint tree first; every entity a preset names exists in the file.

## Caveats, again

- 13 anchor frames on one view; the candidate arms' anchor numbers are the Sep 19 scoreboard's
  and ensemble v2's arm choice saw those anchors.
- The nine-view recording draws masks on every second frame (`--mask-every 2`); the consensus
  markers, anchor outlines and proposals are on every frame they exist.
- The seed-search proposals and the agent seeds on the other views are agent output; the
  `proposal: accept/reject pending` label is a request, not a status.
- 406 MB in one file; the Tailscale server script's 4 GiB buffer is enough, a client below
  8 GB of RAM may prefer the two separate files (`interaction_review_first_minute_v4.rrd` with
  `segmentation.rbl` / `hands.rbl`; the multiview recording with `multiview.rbl`).
