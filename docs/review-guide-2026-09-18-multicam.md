# Morning review guide: the Sep 18 overnight multi-camera pass

Written for the human who left the plan running overnight. Everything below is pulled from
the `### Sep 18:` sections of [`method-ledger.md`](method-ledger.md) and from the manifests
on disk; every path was checked to exist when this guide was committed. `runs/` is
gitignored, so the paths are for this machine.

Standing rule for every recording and every number here: the dataset's poses and
extrinsics are external context, the intrinsics are fitted estimates, the seeds on the new
views were chosen by an agent, and every millimetre or IoU is cross-source disagreement
between estimates. Nothing is accuracy. The GPU is idle and stays idle; nothing in this
guide runs a model.

## (a) What ran overnight

Queue: 9 queue starts, 25 jobs, 25 `succeeded`, none failed, timed out or skipped, no
`NVRM`/`Xid` line in the kernel journal (`runs/overnight-multicam-20260918/queue.log`;
per-job output under `runs/overnight-multicam-20260918/logs/`). GPU wall is the queue's
`job_end.duration_s` summed per track; CPU-only tracks never touched the card.

| Track | What | Run directories | GPU wall (queue) | Peak VRAM | Commit |
| --- | --- | --- | --- | --- | --- |
| 0 | 7 static views fetched for the 92.7 s window, 720p proxies, per-view clock offsets, 12 camera fits | `runs/assembly101-clock-offsets/`, `configs/assembly101/clock_rules.json`, `configs/assembly101/<view>_camera_estimate.json` (12), `configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_all_static_g2.json` | 0 (CPU; ~5 min network + encode, 429 s offset scans, 4.5 s fits) | n/a | `0fdfeb7` |
| queue | serial GPU runner with watchdog | `runs/overnight-multicam-20260918/` | 0 | n/a | `994fbd5` |
| 1 | `CameraRig` over 12 views; two-view recordings rebuilt with the 0.150 s static lag | `runs/assembly101-multiview-rig-check/`, `runs/four-part-focused-first-minute-comparison-offset-v2/`, `runs/ego-static-synchronized-comparison-offset-v2/` | 0 (CPU; 3.3 s rig check, 58 s + 20 s rebuilds) | n/a | `a2e4157` |
| 3a | MediaPipe on the 7 new static views with a dataset-derived crop, plus the ego mono stress arm | `runs/mediapipe-hands-<view>-60s-20260918/` for c10095, c10115, c10118, c10119, c10390, c10395, c10404, hmc21110305 | 0 (CPU; ~35 s per view) | n/a | `a7d9807` |
| 3b | ATHENA triangulation of the 8-view MediaPipe hands vs the dataset 3D | `runs/athena-hands-first-minute-mediapipe/` (+ `-ego/`, `-rigdlt/` variants) | 0 (CPU; 6.7 s) | n/a | `5c0d87e` |
| 4 prep | Kineo trims, YAMLs, dataset-camera PKLs, pixi validation | `runs/kineo-multiview-selfcal-first-minute-20260918/`, `runs/kineo-multiview-known-first-minute-20260918/` | 0 | n/a | `3a81384` |
| 2 | triangulated frame-0 seeds on 7 static views, 7 SAM3 first-minute runs, cross-view consensus, 8-view comparison recording, v4 layer | `runs/multiview-seed-transfer-20260918/<view>/`, `runs/muggledsam-sam3-four-part-multiview-first-minute-static-<view>-20260918t0*/` (7, timestamps `055420z` to `061020z`), `runs/multiview-part-consensus-first-minute/`, `runs/multiview-static-comparison-first-minute/` | 65.3 s (7 seed decodes, 8.9-10.1 s each) + 1,119.5 s (7 SAM3 runs, 156.3-163.8 s each; worker 117-120 s) | 2.10 GB per SAM3 run | `577df77` |
| 5 | per-part visual hulls from 8 silhouettes, hull-vs-mask disagreement | `runs/multiview-visual-hull-first-minute/` | 0 (CPU; 184 s) | n/a | `cd6d857` |
| 6 | e4 seeded and run as a moving camera; e4 variants of consensus and hull; e1/e2 visibility audit | `runs/muggledsam-sam3-four-part-multiview-first-minute-ego-hmc21179183-20260918t061608z/`, `runs/multiview-part-consensus-first-minute-with-e4/`, `runs/multiview-visual-hull-first-minute-with-e4/`, `runs/multiview-ego-visibility-audit/`, `runs/multiview-seed-transfer-20260918/HMC_21179183/`, `runs/multiview-seed-transfer-20260918/HMC_21176875/` | 9.0 s (seed decode) + 173.3 s (SAM3; worker 132 s) | 2.10 GB | `c7fe790` |
| 7 prep | LM-EEC cloned, pinned, installed in its own venv; pairs and frames prepared | `runs/egoexo-correspondence-first-minute-20260918/`, `/home/nick/src/LM-EEC` @ `b37e50e5`, `scripts/install_lm_eec.sh`, `scripts/lm_eec_driver.py` | 0 | n/a | `4e6f0e2` |
| 3 WiLoR arm | WiLoR on C10379, C10395, C10115; ATHENA on the three | `runs/wilor-hands-c10379-60s-20260918/`, `runs/wilor-hands-c10395-60s-20260918/`, `runs/wilor-hands-c10115-60s-20260918/`, `runs/athena-hands-first-minute-wilor/` | 340.1 s (134.9 + 77.3 + 118.9 WiLoR, 6.9 ATHENA, 2.1 review) | 2.63 GiB per WiLoR run | `5e11133` |
| 4 run | Kineo known-camera and self-calibration arms, evaluated and rendered | `runs/kineo-multiview-known-first-minute-20260918/`, `runs/kineo-multiview-selfcal-first-minute-20260918/` | 502.4 s (222.2 known + 280.2 selfcal; 8.4 min of the 2 h box) | 5.34 GiB known / 5.48 GiB selfcal (torch reserved; ONNX Runtime excluded) | `5e11133` |
| 7 run | LM-EEC exo->ego for 4 parts at 12 keyframes, evaluated and rendered | `runs/egoexo-correspondence-first-minute-20260918/` | 6.3 s | 752 MiB | `5e11133` |

Total GPU wall across the night: 2,216 s (36.9 min). Idle GPU memory before every
job 1,180-1,313 MiB across all 25 checks. Tests at close-out: 419 default-tier, 37 `real_data`, ruff clean.
`scripts/prune_runs.py` now lists 74 unreferenced run directories (1.93 GB), up from 72;
nothing was deleted.

## (b) Which Rerun recordings to open, in what order, and what to look for

Each command opens one recording in the native viewer; open them one at a time. Frame
numbers below are analysis frames of the focused first minute (30 fps, C10379 clock,
`pose_frame = 17649 + 2p`) unless stated otherwise.

### 1. The rebuilt v4 package (start here)

```bash
rerun runs/interaction-review-first-minute-v4/interaction_review_first_minute_v4.rrd
```

78.4 MB. Same package you reviewed on Sep 17 plus three overnight layers, all in the
existing views; no blueprint change:

- `assembly101_multiview` (Track 2): consensus part centroids in the
  `contexts/assembly101_world_mm_3d` view next to the dataset hands and the C10379
  frustum; a new time panel `diagnostics/multiview/<part>/{views_used, c10379_error_px}`;
  the episode document as a static tab; per-frame consensus lines and active episodes in
  the navigation document. Look at `c10379_error_px` for chassis around 585-702 and
  1089-1172: those are the frames where C10379's mask sits more than 40 raw px from a
  consensus formed by at least two other static cameras. Then look at 279-408, which you
  reported and which the other cameras do *not* contradict.
- `athena_hands` (Track 3b): the eight-view MediaPipe triangulation (orange left / blue
  right, thin smoothed skeletons) under `contexts/assembly101_world_mm_3d/athena_hands`,
  with `diagnostics/assembly101/athena_hands` series for contributing views, wrist
  disagreement in mm and per-view reprojection RMS.
- `athena_hands_wilor` (GPU results): the three-view WiLoR triangulation beside it in
  magenta / green, with the `wilor_wrist_camera_frame_C10379` series. Compare the
  triangulated wrist's frame-to-frame motion with the existing stabilized WiLoR 3D layer:
  the hopping you reported is WiLoR's per-frame depth, and the triangulated wrist does not
  have it.
- The visual hull is *not* in this package (it lives in recording 2); the consensus
  centroids are.

### 2. The eight-view static comparison

```bash
rerun runs/multiview-static-comparison-first-minute/multiview_static_comparison.rrd
```

248.7 MB (masks on every second frame, `--mask-every 2`). Eight 2D views, one per static
proxy, each with its SAM3 part masks as RGBA cut-outs and the consensus point reprojected
as a marker labelled with that view's pixel error; a world-mm 3D view with the consensus
centroids, dataset hands, all eight frusta and the Track 5 hull voxels at 1 fps; hull
projections as a toggleable overlay per view; a per-part time panel of every view's error;
the episode document.

What to check, in this order:

1. Frame 0 on all eight views: these are the agent-authored seeds. C10379 is the only view
   a human has seen. The interior has no mask on any view but C10379 (never seeded).
2. Chassis `[585,702)` (the four contradicted runs `[585,604)`, `[643,658)`, `[665,689)`,
   `[697,702)` sit inside your reported 573-722 window): does C10379's chassis mask sit
   where the seven other cameras put the chassis?
3. Chassis `[1089,1172)` (`[1089,1163)` and `[1167,1172)`, ending at the agent correction
   frame 1172): the chassis/interior swap you reported at 1020-1172. The consensus starts
   contradicting at 1089, not 1020; the hull's C10379 IoU is 0.00 over `[1020,1172)`.
4. Rear body `[1677,1800)` (`[1677,1751)`, `[1759,1781)`, `[1786,1800)`): the retained
   late-degradation note (label stays on the table piece). Note that adding e4 shrinks this
   contradiction to three 5-6 frame runs, so it is the fragile one.
5. C10395: the other grazing camera, agreement 0.73 / 0.81 / 0.73 and 30 of the 83
   episodes. Decide for yourself whether its masks are worse or whether the low viewpoint
   is simply hard for a centroid test.
6. Toggle the hull projections: every SAM3 mask is larger than the over-carved hull (the
   "mask larger than hull" fraction is 0.26-0.99 in every view), so read the hull as a
   lower bound on the part, not as a reference mask.

### 3. Kineo self-calibration vs the dataset cameras

```bash
rerun runs/kineo-multiview-selfcal-first-minute-20260918/comparison.rrd
```

6.3 MB. Dataset frusta as in the other recordings, Kineo's similarity-aligned frusta with
magenta labels, Kineo's 55-joint body in green, dataset hands, wrist-disagreement series.
Look for: the eight magenta frusta sit inside the dataset's ring (Kineo puts the cameras
~37 % closer to the subject, 598 vs 944 mm per unit, with 4-11 deg of compensating
rotation; medians 7.26 deg / 114.7 mm after alignment; camera-centre RMS 148.7 mm). The
body itself is plausible; its wrists under the *camera* alignment are 83 / 44 mm from the
dataset's, and would be 17 mm under a body alignment (diagnostic only, not in the
manifest). The known-camera arm has no comparison recording (cameras are the dataset's by
construction, wrists 16.9 / 17.4 mm median); its native Kineo output is
`runs/kineo-multiview-known-first-minute-20260918/kineo_outputs/assembly101_known_first_minute.rrd`
(121.7 MB) if you want the body alone. The self-calibration arm's native recording is
934 MB and not needed for this decision.

### 4. LM-EEC ego-exo correspondence, per keyframe

```bash
rerun runs/egoexo-correspondence-first-minute-20260918/correspondence.rrd
```

4.7 MB. Twelve keyframes (C10379 frames 0, 150, ..., 1650; ego HMC_21110305 frame
`p + 4`). Two 2D views per keyframe: the exo frame with the human/agent query cut-outs, and
the ego frame with LM-EEC's prediction in magenta, the hull projection in white and the
ego SAM3 mask in the part colours; an IoU time panel; the manifest.

Look for: chassis and interior predictions land on the part in roughly half the keyframes
(chassis vs ego SAM3 per keyframe 0.35, 0.73, 0.41, 0.26, 0.01, 0.40, 0.70, 0.69, 0.96,
0.57, 0.40, 0.33; interior 0.00, 0.12, 0.51, 0.69, 0.93, 0.33, 0.02, 0.00, 0.27, 0.39,
0.23, no pair at 1050). Rear body has two hits (0.73 at 150, 0.74 at 1350) and cabin peaks
at 0.34 at frame 0; elsewhere the small dark parts get 1,200-84,000 px predictions that
are not the part. The model's own IoU prediction separates the cases (0.70 chassis /
interior, 0.03-0.05 rear_body / cabin). The one question to answer while looking: do the
magenta exo->ego masks look like *ego-view* shapes, or like exo-view shapes pasted into
the ego frame? That decides the checkpoint-direction item in (c).

### 5. The offset-corrected two-view recordings

```bash
rerun runs/four-part-focused-first-minute-comparison-offset-v2/four_part_focused_first_minute_ego_static_comparison.rrd
rerun runs/ego-static-synchronized-comparison-offset-v2/ego_manual_seed_vs_static_g3_synchronized_comparison.rrd
```

84.5 MB (focused first minute, C10379 vs HMC_21110305) and 122.0 MB (canonical 180 s,
C10379 vs HMC_21179183). Both apply the measured 0.150 s static lag: scrub the
`analysis_time` timeline for exact alignment; on the integer `analysis_frame` timeline the
static entries are shifted by round(4.5) = 4 frames, so static frame 0 appears at
analysis_frame 4 and the static view is empty for the first 0.150 s. The zero-offset
originals (`runs/four-part-focused-first-minute-comparison/` and the 131.5 MB recording
inside the ego run directory) are kept as superseded. Look for whether the hand that
crosses both views now arrives at the same instant on `analysis_time`.

### Optional

- `rerun runs/athena-hands-first-minute-mediapipe/hands.rrd` (7.8 MB) and
  `rerun runs/athena-hands-first-minute-wilor/hands.rrd` (7.1 MB): the standalone
  triangulation recordings (dataset hands yellow/mint, triangulated orange/blue, eight
  frusta, time series). The v4 package carries the same entities.

## (c) Decisions only you can make

Generated manifests under `runs/` are fingerprinted by the recordings and indexes that
consume them (for example the v4 index carries the SHA-256 of the consensus manifest), so
do not edit them. Record each decision in a new tracked human-feedback record,
`docs/qa/interaction-review-first-minute-v4-multicam.human-feedback.json` (does not exist
yet; it is the one file this guide names that you create), using the
`HumanFeedbackReviewRecord` schema in `src/battle/schemas.py` (`manifest_kind:
human_feedback_review_record`, `author_type: human`, `provenance_tag:
human_feedback_report`, one `feedback` item per decision with your words in `verbatim`,
`reviewed_recording_uri` pointing at the recording you looked at). The existing
`docs/qa/interaction-review-first-minute-v4.human-feedback.json` is the template. The
agent then carries each decision into the typed places named below and cites the record.

1. **The 13 consensus-proposed `not_contact_eligible` intervals for C10379.** Source:
   `runs/multiview-part-consensus-first-minute/manifest.json`, field
   `proposed_validity_intervals[0..12]`, each with `target`, `start_frame`,
   `end_frame_exclusive`, `proposed_state: not_contact_eligible`, `trigger:
   multiview_disagreement`, `applied: false`. The list: chassis `[585,598)`, `[599,604)`,
   `[643,651)`, `[652,658)`, `[665,689)`, `[697,702)`, `[1089,1163)`, `[1167,1172)`,
   `[1464,1471)`; rear_body `[1525,1530)`, `[1677,1751)`, `[1759,1781)`, `[1786,1800)`.
   Per interval: accept, reject, or narrow. Note that `[1089,1172)` overlaps the existing
   `not_contact_eligible` interval `[1020,1172)` already in the v4 package and the
   ensemble reference, so accepting it changes nothing there; the chassis 585-702 and
   rear_body 1677-1800 proposals would be new. The 37 hull proposals in
   `runs/multiview-visual-hull-first-minute/manifest.json` (`proposed_validity_intervals`)
   are the noisier signal and include windows where the hull itself is tiny; read them
   next to `part_summaries` / `voxel_count` rather than accepting them wholesale.
2. **The agent-authored frame-0 seeds on the seven static views and e4.** Source:
   `runs/multiview-seed-transfer-20260918/<view>/seed_manifest.json` for C10095, C10115,
   C10118, C10119, C10390, C10395, C10404 and HMC_21179183; per part `parts[*].status`
   (`accepted` / blocked), `parts[*].accepted.mask.uri` (the PNG that seeded the run),
   `parts[*].accepted.acceptance_basis` (`plane_warp_iou` or the centroid-ray rule),
   `parts[*].accepted.backprojection_iou`, `centroid_ray_distance_mm`,
   `area_ratio_vs_expected`, and the rejected alternatives under `parts[*].candidates`;
   hands under `hands[*]`. `selected_by` is `agent` and stays `agent`: no tool flips it,
   and a human seed on these views would have to come through the calibration workspace.
   The decision is whether the consensus and hull built on these seeds are admissible as
   review evidence; if any view's seed is wrong, its SAM3 run, its consensus vote and its
   silhouette in the hull are wrong with it.
3. **Whether the consensus / hull disagreement triggers enter `contact_eligible`.** Today
   the eligibility intervals are fixed in code: `_sam3_validity_intervals` in
   `src/battle/interaction_review_v4.py` (frames 1020-1172 `not_contact_eligible` by
   `human_feedback_report`, 1200-1800 by `agent_authored_visual_review`) and the ensemble
   policy in `src/battle/ensemble_reference.py` (`runs/ensemble-reference-first-minute-v1/`).
   Adopting a proposal means adding a `SegmentationValidityInterval` row with
   `provenance: human_feedback_report` and your rationale; the alternative is to leave the
   `multiview_disagreement` and `hull_disagreement` episodes as review triggers beside the
   reference, which is where they are now.
4. **LM-EEC checkpoint direction.** Source:
   `runs/egoexo-correspondence-first-minute-20260918/pairs.json`,
   `directions[0].checkpoint` (currently
   `/home/nick/src/LM-EEC/checkpoints/LM-EEC-checkpoint/ExoEgo_checkpoint.pt`, SHA-256
   `b3130bcb…83dcd9c`), used for exo->ego on its file name alone; both released
   checkpoints carry identical training metadata and no direction label. If the magenta
   masks in recording 4 look like exo shapes, the decision is to swap to
   `EgoExo_checkpoint.pt` (`a79234ab…af259ad`) and re-queue
   `runs/overnight-multicam-20260918/jobs_t7_egoexo.json` (6.3 s of GPU, your call to
   spend it). `independent` mode (state reset per keyframe) was also not run.
5. **Whether to seed e1 (HMC_21176875) mid-minute.** Source:
   `runs/multiview-seed-transfer-20260918/HMC_21176875/seed_manifest.json`
   (`run_decision: skip`, `run_decision_reason` starting `seed_transfer_failed: no part
   projects inside the view at frame 0`; the cabin lands at proxy row 745 of 720) and
   `runs/multiview-ego-visibility-audit/report.json` (e1 sees the cabin in 88 % and the
   chassis in 61 % of sampled frames, hands 27 % / 23 %). The clip config
   `configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_ego_e1_g2.json`
   exists. A run needs a profile that starts at the first frame the cabin is in view, which
   is not built; the alternative is to record e1 as `not_run` with this reason, as e2
   already is (best part 45 %).

Still open from before, unchanged tonight: the fixed two-timestamp QA dispositions under
`docs/qa/*.human-qa.json`; hidden-interior semantics for `[1024,1172)`; the late
reassembly beyond the first minute; the G5 / CC BY-NC question, which now covers seven
more views; and the 74 unreferenced run directories that `scripts/prune_runs.py` lists.

## (d) Known caveats and non-claims, verbatim from the ledger

Track 0:

> Everything in this track is external review context: the dataset's poses and extrinsics
> are the dataset's, the fitted intrinsics are estimates of the dataset's own internal
> projection (the archive ships none), the measured clock offsets describe when each
> camera's video started relative to the pose clock. None of it is ground truth for any
> method compared here and no accuracy number rests on it; CC BY-NC 4.0 attribution
> applies.

> The offsets are per-camera constants for this recording; nothing was checked about drift
> within the 92.7 s window beyond the three-chunk agreement.

Track 1:

> The rig is geometry on dataset context (shipped extrinsics, per-frame ego poses) and on
> fitted estimates (intrinsics). Its residuals against the dataset's own 2D and 3D
> landmarks show that it reproduces the dataset's internal projection; they are not
> accuracy claims about the dataset, the cameras or any method. Triangulated points are
> cross-view agreement.

> The offset is assumed constant over the recording; it was measured on the focused window
> only.

Track 3b:

> The triangulated hands are a DLT on MediaPipe detections through fitted intrinsics and
> the dataset's shipped extrinsics; the dataset `landmarks3D` are the output of the
> dataset's own multi-view tracker with a fixed-scale hand model. Their millimetre
> difference is cross-source disagreement between two estimates, never accuracy of either.
> Hand-side correspondence borrows the dataset wrist projected into each view (method
> handedness is unreliable); positions do not. Intrinsics are estimates.

> "Fixable" therefore means replacing WiLoR's translation with a triangulated wrist and
> keeping WiLoR's articulation, which is what the queued three-view WiLoR arm will test; it
> is not a claim that either estimate is right.

Track 2:

> Every seed on a view other than C10379 was chosen by an agent from geometry
> (`selected_by: agent`, provenance `geometric_seed_transfer`); no human reviewed a mask on
> the seven new static views or on e4. The consensus, agreement scores and disagreement
> episodes below measure how far runs of one tracker disagree with each other across
> calibrated views; they are not accuracy. The human-corrected C10379 masks stay the
> reference and were not modified; the `not_contact_eligible` intervals emitted here are
> *proposals* for human review.

> The human-reported 279-408 window is *not* contradicted: the other views agree with
> C10379 there, so whatever the human saw in that window is not a centroid displacement the
> static cameras can resolve.

Track 5:

> A visual hull is the intersection of the views' silhouette cones. It is an upper bound on
> the object's volume only when every silhouette is a superset of the object; with SAM3
> masks that miss a part's occluded half, the intersection is *over-carved* and the hull
> shrinks or collapses. Hull counts, centroids, hull-vs-mask IoU and the
> `hull_disagreement` episodes are therefore geometry-only comparison evidence and never a
> reconstruction or accuracy claim. Nothing is substituted into any reference.

> the rear_body and cabin lists are far longer than Track 2's and include windows where the
> hull itself is tiny, so the hull proposals are the noisier of the two signals and should
> be read next to `voxel_count`.

Track 6:

> e4's seeds are agent-authored geometric transfers like the static ones; its per-frame
> pose is the dataset's; its intrinsics are the 0.31 px rational estimate; e1/e2 intrinsics
> are 4-5 px estimates fitted on few hand observations ("projection-only cameras").
> Visibility is geometric (inside the sensor, in front of the camera) and ignores
> occlusion. Nothing here is accuracy.

> the gate passes on 1800/1800 frames, which says the pose data are self-consistent, not
> that the video is (the gate has no video-side observation; the 30 px triangulation filter
> is what actually drops e4 when its mask disagrees).

Track 4 and the Kineo results:

> The self-calibration arm compares Kineo's estimated extrinsics with the dataset's shipped
> extrinsics after a similarity alignment of camera centres: a calibration comparison
> against dataset context, not pose accuracy, and the only place in the pass where the
> dataset extrinsics act as a reference for a method. Wrist numbers in both arms compare
> Kineo's SMPL-X body wrists with the dataset hand-tracker wrists: cross-source
> disagreement between two estimates. Intrinsics are fitted estimates. Ego cameras are
> excluded (Kineo assumes static cameras).

> Reading: Kineo's body 3D is consistent with the dataset up to a similarity; its camera
> placement is not the dataset's under any similarity.

Track 7 and the LM-EEC results:

> Every number this track will produce is an IoU between two estimates of the same
> ego-view region: the LM-EEC prediction from a human C10379 mask, the ego SAM3 mask of the
> Sep 16 run (human frame-0 seed, one tracker), and Track 5's visual hull (agent-seeded
> static masks, fitted intrinsics, dataset extrinsics) projected into the ego camera.
> Cross-source disagreement, not accuracy; no ego ground truth exists for the parts. The
> ego video is monochrome and LM-EEC was trained on colour Aria frames; both views are
> squashed to 480x480 inside the model.

> both carry identical training metadata (epoch 60, 11,280 steps) and no direction label,
> so `ExoEgo` is used for exo->ego on its file name alone (recorded as a claim boundary).

> the hull projection disagrees with the ego SAM3 masks as much as the model does (SAM3 vs
> hull median 0.006 for the chassis), so on this monochrome ego view there is no reference
> the other two agree with.

WiLoR arm:

> Reading: the hopping is WiLoR's per-frame depth; replacing its translation with the
> triangulated wrist cuts the depth step 3.5x (left) / 6x (right) and lands the wrist
> series on the dataset tracker's own steadiness, while WiLoR's 2D and articulation are
> untouched. That is a statement about steadiness and cross-source agreement, not about
> which wrist is right. The substitution itself is not built; the arm only measures it.

Licences, from the GPU results section:

> Licences: Assembly101 CC BY-NC 4.0; WiLoR checkpoints CC-BY-NC-ND (MANO and Ultralytics
> separate); Kineo research/evaluation only, checkout dirty (fingerprinted in
> `prepare.json`); LM-EEC weights are research artefacts of a NeurIPS 2025 paper with no
> licence file, SAM 2 Apache-2.0; ATHENA MIT.
