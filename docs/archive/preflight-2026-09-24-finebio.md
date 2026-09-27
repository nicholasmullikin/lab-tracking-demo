# FineBio preflight, Sep 24, 2026: the plan's assumptions checked before anything runs

Run on the evening of Sep 24 against the plan in
`/home/nick/.cursor/plans/finebio_3d_object_tracking_demo_5b2e9c17.plan.md` (and its repo copy
[`plan-2026-09-24-finebio-detector-seeded-lab.md`](plan-2026-09-24-finebio-detector-seeded-lab.md)),
after the user asked for every assumption to be challenged first: whether SAM3.1 holds up with
this many objects, whether attached/occluded objects and head motion are tractable, whether the
input data is good enough, whether the arm set is right, and how we would know the multicam rig
is working. Every number below is from a check that ran tonight; nothing is inferred from
Assembly101. CPU except the SAM3 checks (about 5 min GPU in total, peak 2.7 GiB, no viewer
opened). Outputs under `runs/preflight-finebio-20260924/` (gitignored; FineBio licence,
non-commercial research, nothing redistributed).

Claim boundary: the FineBio DINO detector is the reference for every SAM3 number here and it
was trained on FineBio's own objects and on frames from these same cameras; "IoU vs detector
box" is agreement between two models, not accuracy. No human looked at a mask pixel by pixel.

## What ran

| step | command | time |
|---|---|---|
| detector pass, 6 views x 78 raw frames (60 consecutive at 60 s + every 30th over 60-80 s) | `CUDA_VISIBLE_DEVICES="" /home/nick/src/finebio-detector/.venv/bin/python scripts/finebio_preflight_detect.py --trial P03_01_01 --consecutive 1798:60 --spaced 1798:2398:30 --output runs/preflight-finebio-20260924/detections` | 914 s CPU |
| camera mapping and day | `uv run python scripts/finebio_preflight.py --trial P03_01_01 mapping --seconds 30,60,90` (and `--trial P03_03_01 --output .../P03_03_01 mapping --seconds 30,120,240`) | 3 s each |
| fpv pose vs markers | `uv run python scripts/finebio_preflight.py --trial P03_01_01 fpv-pose --day 221013 --step 25` (and P03_03_01, `--step 50`) | 35 s |
| rig checks | `uv run python scripts/finebio_preflight.py --trial P03_01_01 rig --detections runs/preflight-finebio-20260924/detections` | 2 s |
| SAM3 box-prompt decode + 300-frame tracks | `CUDA_VISIBLE_DEVICES=0 /home/nick/.pyenv/versions/muggled_sam/bin/python scripts/finebio_preflight_sam3.py --trial P03_01_01 --frame 1798 --detections .../detections --output .../sam3_<views> --decode-views ... --track-views ...` (four invocations: fpv+T1+T2 / T3+T2 / T4 / T5) | 72-86 s each on the GPU |
| Rerun recording | `uv run python scripts/finebio_preflight.py --trial P03_01_01 rerun --detections .../detections` -> `runs/preflight-finebio-20260924/preflight.rrd` (16 MB) | 4 s |
| clock by motion energy (Sep 24, earlier) | 128x72 grey frame-difference energy per view, cross-correlated against T5 over 60-80 s | 3 s |

Also read-only: one frame per view at 60 s through the detector (Sep 24 earlier, `/tmp`), and
top-down contact sheets across the full trial for `P03_01_01`, `P03_03_01`, `P03_05_01`.

## A. Cameras, day, first-person pose

**Mapping** (`mapping/mapping.md`, overlays `mapping/T*_markers.jpg`; ArUco `DICT_6X6_50`,
three frames per view, residual = median corner RMS of the day's `marker_points` projected
through the shipped extrinsics against the detected corners):

| view | camera id | day | shipped residual | best other camera | PnP centre vs shipped |
|---|---|---|---|---|---|
| T1 | 1 | 221013 | 6.3 px | cam6 69.5 px | 0.3 cm |
| T2 | 2 | 221013 | 6.9 px | cam4 96.8 px | 0.6 cm |
| T3 | 3 | 221013 (221109 within 1.6 px; cam 3 did not move between days) | 5.1 px | cam6 85.9 px | 0.9 cm |
| T4 | 4 | any room-1 day within 2.6 px (cam 4 did not move) | 2.2-4.8 px | cam6 105 px | 1.0 cm |
| T5 | 6 | 221013 | **93.7 px on every day** | cam4 98.6 px | **6.4 cm** |

- `T1..T5` are cameras `1,2,3,4,6` in order; the ids form a permutation. Same result on
  `P03_03_01`.
- P03 was recorded on **221013**: T1/T2 discriminate it (next day 10.3 / 105 px); 221013 is
  also the only day with three markers, and the third marker of the video matches it.
- **The top-down camera (6) does not fit its shipped extrinsics on this trial**: a uniform
  ~90 px image shift, PnP from the three detected markers puts the camera 6.4 cm from the
  shipped position with 0.7 px residual. Camera 6 also moves 10-15 cm between the room-1 days
  in the shipped files, so it was re-mounted between sessions. The rig below uses the
  marker-PnP pose for T5 (`provenance: marker_pnp`) and the shipped poses for T1-T4.
- Intrinsics: the shipped videos are already downscaled (fixed 1920x1080, fpv 1920x1440) from
  the calibration resolutions; a uniform 0.5 / 0.48 rescale of `intrinsic_matrix` with the
  shipped distortion reproduces the markers to 2-7 px, so no crop was applied by the authors.

**First-person pose** (`fpv_pose/fpv_pose.md`): the shipped per-frame pose projects the
markers onto the detected ArUco corners with corner RMS **median 0.9 px, p90 2.0 px** on
196/196 checked frames of `P03_01_01` (3 frames over 20 px; the pose is a marker PnP and
those are its outliers), and 1.0 / 1.9 px on 169 frames of `P03_03_01`. Validity: 4890/5032
(97.2%) and 8407/8492 (99.0%). The "jitter" reported earlier (p99 2.5 cm, max 10.7 cm) is
real head motion plus roughly 1.5% outlier frames; a per-frame marker-residual or velocity gate
removes the outliers. Projecting the fpv camera centre into T1/T4 lands on the head-mounted
GoPro (`rig/T1_fpv_centre_f1798.jpg`, `rig/T4_fpv_centre_f1798.jpg`).

## B. Rig, clocks, hand-off

**Static objects** (`rig/rig.md`; five-view triangulation of median detector box centres over
the 21 spaced frames, raw 1920 px):

| object | height of box centre above bench | leave-one-view-out residual per view |
|---|---|---|
| pcr_machine / 8_tube_stripes_rack / micro_tube_rack / magnetic_rack | 0.2-0.9 cm | 1-13 px |
| vortex_mixer | 1.8 cm | 1-11 px |
| centrifuge | 4.1 cm | 2-19 px |
| tip racks | 3.3-5.9 cm | 1-32 px (T2 up to 61 on a 3-view object) |
| trash_can | 9.7 cm | 5-31 px |

Heights are half the objects' physical heights, so the units are centimetres and the board
plane is the bench. Over 49 (object, view) cells the LOO residual is **median 10.4 px, p90
25.3 px** at 1920; box-centre parallax (the box centre of a 3D object is not one 3D point)
accounts for most of it. Rerun: `world/static_objects`, `world/<view>/static_reprojected`.

**Hands as probes**: `left_hand` triangulates from >= 3 fixed views on 61/61 consecutive
frames, residual median 7.7 px, 3.3 cm above the bench (resting on the plate). `right_hand`
(raised, holding the pipette) has >= 3 fixed views on only **14/61** frames, residual 14 px,
26.6 cm above the bench.

**Clock**: motion-energy cross-correlation puts T1-T4 at -1..0 frames relative to T5 (r
0.91-0.95), fpv at 0 (r 0.57). The detector-box scan (-15..+15) agrees where the object moved
(the right hand: T3 +1, T4 +1, T5 -1, with residual doubling at +/-3) and is flat, hence
uninformative, for the plate and left hand, which barely moved in the 2 s window. Verdict:
synchronised to +/-1 frame; there is no Assembly101-style per-view offset.

**Leave-one-view-out on moving objects** (centre residual of the 4-view triangulation
reprojected into the held-out view, 61 consecutive frames): plate median 17-44 px, p90 30-46,
**inside the held-out box on 98-100%** of frames in every view; held pipette median 6-24 px,
inside 92-97% (T4 has it on 3 frames only); left hand 2-16 px, 98-100%.

**Fixed -> fpv hand-off**: the plate triangulated from the fixed views and projected through
the shipped fpv pose lands **inside the fpv detector box on 45/45 frames**, median 37 px from
its centre (p90 52 px at 1920x1440); the left hand 32 px median.

## C. SAM3 with detector boxes as the only prompt

**Decode** (`sam3_*/decode_table.md`; one raw frame per view at 60 s; encoder max side 1280,
square sizing; IoU between the mask's bounding box and the prompt box; box size in px):

| class | fpv | T1 | T2 | T3 | T4 | T5 |
|---|---|---|---|---|---|---|
| cell_culture_plate (transparent) | 0.94 (359x252) | 0.77 (83x48) | 0.84 (177x83) | 0.91 (180x146) | 0.91 (128x150) | 0.90 (175x93) |
| blue_pipette (in hand) | 0.95 | 0.90 | 0.88 | 0.90 | not detected | 0.78 |
| centrifuge | 0.99 | 0.96 | 0.94 | 0.95 | 0.97 | 0.98 |
| 50ml_tube | 0.96 | 0.88 | 0.94 | 0.84 | 0.91 | 0.68 (90x67) |
| micro_tube_rack / blue_tip_rack / 8_tube_stripes_rack / vortex_mixer | 0.87-0.94 | 0.92-0.97 | 0.86-0.94 | 0.89-0.94 | 0.91-0.94 | 0.91-0.96 |
| micro_tube (one) | 0.65 (33x36) | 0.58 (17x15) | 0.84 (28x20) | 0.79 (19x20) | 0.80 (21x23) | 0.58 (22x23) |

- The transparent plate is a **recognition** problem for SAM3, not a delineation one: given
  the detector's box it is segmented in every view, including the 83x48 px T1 case. The
  PVS prompt search / rim points / exemplar arm planned for it are unnecessary.
- Encoder side 1920 vs 1280 on the fixed views: bounding-box IoU changes by mean -0.002 over
  43 cells (min -0.18, max +0.15). No gain, as on Assembly101.
- Objects under ~30 px (individual micro tubes) are marginal (0.58-0.84) at either side.

**Tracks** (SAM3.1 multiplex, prompt memory 1, frame memory 4, no corrections, 1280; 299
steps from 60 s; IoU vs the same-class detector box on the 67 frames where the detector ran):

| view | plate | held pipette | centrifuge | 50ml tube | ms/step |
|---|---|---|---|---|---|
| fpv | 0/299 lost, IoU 0.95 | 9 lost (1905-1913, frame-edge exit), 0.93 | 0 lost, 0.98 | 0 lost, 0.92 | 220 |
| T2 | 0 lost, 0.85 | 0 lost, 0.83 | 0 lost, 0.95 | **14 lost (1912-1925, occluded by the arm), re-acquired on the same tube**, 0.92 | 219 |
| T4 | 0 lost, 0.90 | not seeded | 0 lost, 0.96 | 0 lost, 0.91 | 220 |
| T5 | 0 lost, 0.87 | 0 lost, 0.71 | 0 lost, 0.98 | 0 lost, 0.70 | 214 |

Four slots at 1280 cost 214-220 ms/step regardless of view; peak VRAM 2.6-2.7 GiB.

## D. Trial choice and object traffic

Top-down contact sheets across the whole trial (`/tmp`, 12 frames each): `P03_01_01`
(protocol 01, 168 s) has one plate and one pipette moving and the centrifuge opening once near
the end. `P03_03_01` (protocol 03, 283 s, same day and rig, fpv pose 99.0% valid, six-view
frame count 8492 in every video) opens the centrifuge at least six times, moves tubes in and
out, and relocates a tip rack; `P03_05_01` (protocol 05, 351 s) also has traffic. The shipped
detection images for `P03_03_01` include a **six-view annotated frame at 916** (all of T1-T5
and the fpv) plus five more fpv frames; `P20_03_01` (room 2, protocol 03) has the same
structure with its six-view frame at 1442.

## Verdicts on the five questions

1. **Does SAM3.1 hold up with this many objects?** Compute yes (image encoder dominates; 4
   slots 220 ms/step, ~2.6 GiB). Box-seeded slots on the plate, pipette, centrifuge and a tube
   held for 300 frames in four views with one occlusion re-acquired correctly. Not tested and
   still the real risk: identity among identical instances (8-13 micro tubes at 17-28 px,
   two identical white pipettes); the preflight shows those are below the size SAM3 or the
   detector resolve in the fixed views, so per-tube identity inside racks is out of scope by
   evidence, not by choice.
2. **Attached / occluded / head motion.** Containment and hand occlusion: tractable (T2 tube
   lost 14 frames behind the arm, re-acquired). Attachment: no observable for tips (never a
   stable detection) or the plate lid (same box as the plate). Head motion: the fpv pose is
   accurate (0.9 px) on 97-99% of frames, the fixed-to-fpv hand-off landed inside the fpv box
   on 45/45 frames, and SAM3 lost the fpv pipette only on frame-edge exits.
3. **Input data good enough?** Yes with corrections: cameras synchronised to +/-1 frame;
   cameras 1-4 calibrated to 2-7 px; **camera 6 must be re-solved from the markers** (6.4 cm
   off); intrinsics rescaled 0.5 / 0.48; undistort before triangulating; rig residual ~10 px
   median / 25 px p90 at 1920 sets the gates; objects under ~30 px are not trackable from the
   fixed cameras.
4. **Arms.** The plate-specific arms (PVS search, exemplar) and the 1920 encoder can go; the
   arm the plan lacks is per-frame box-prompted decode with no video memory, which the decode
   table says works at the frame level and which cannot drift.
5. **Multicam cross-checks.** Seven now exist as code and as Rerun entities in
   `preflight.rrd`: marker reprojection per view, fpv camera centre in the fixed views,
   static-object triangulation with LOO, hands as probes, clock scan, LOO for moving objects,
   fixed-to-fpv hand-off residual. The in-hand object is the weakest detection in every fixed
   view (0.44-0.57 vs 0.8-0.9 for bench objects), so seeding by score persistence selects the
   wrong objects.

## Proposed plan modifications

1. `p0-mapping`: done here; record `T1..T5 = 1,2,3,4,6`, day 221013 for P03, marker-PnP for
   camera 6 (`provenance: marker_pnp`, 0.7 px) as the rule for every trial: fit every camera
   to the markers and take the shipped pose only where it agrees within 10 px.
2. `p0-presence`: drop. Box prompt from the detector seeds the plate in every view.
3. `p0-trials`: trial 1 = `P03_03_01` (protocol 03), window around the six-view annotated frame
   916 and the centrifuge cycles; trial 2 = `P20_03_01` (room 2, same protocol, six-view
   frame 1442). `P03_01_01` stays as the smoke.
4. `p1-configs`: fixed-camera proxies at 1920x1080 (no downscale; SAM3 encodes at 1280 either
   way), fpv 1920x1440; undistorted normalised coordinates for all triangulation; intrinsics
   rescale recorded.
5. `p1-rig`: done here; gates from the numbers: association 30 px, LOO/hand-off 50-60 px at
   1920, birth allowed from 2 fixed views plus a valid fpv pose (the raised in-hand object has
   >= 3 fixed views on 14/61 frames).
6. `p2-seeds`: seed rule = objects that move (box displacement over a window) or sit inside a
   hand box, at score >= 0.3 with persistence, plus a named static set; not "score >= 0.5
   persistence", which picks the bench and misses the hand.
7. `p2-plate`: reduce to a monitor: bounding-box IoU vs the detector, not fill ratio (the
   plate's fill is 0.50-0.54 when correct).
8. `p3-tracker`: identical-instance groups (tubes in racks) are one track per rack until a
   tube leaves; no per-tube identity claimed.
9. `p4-arms`: (a) boxes-only, (b) per-frame box-prompted decode, no memory, (c) SAM3 video
   memory seeded once, (d) (c) + detector re-seed + hand-off; DAM4SAM fpv last. Drop 1920.
10. `p5-events`: containment (tube in centrifuge, lid state) and hand-held; drop tip and lid
    attachment.
11. `p5-viewer`: the seven cross-checks above are standing entities in the review recording,
    logged on every build.

## Files

- `scripts/finebio_preflight_detect.py` (detector venv), `scripts/finebio_preflight.py`
  (`mapping`, `fpv-pose`, `rig`, `rerun`), `scripts/finebio_preflight_sam3.py` (MuggledSAM
  interpreter).
- `runs/preflight-finebio-20260924/`: `detections/*.jsonl` (6 views x 78 frames),
  `mapping/` (+ `P03_03_01/mapping/`), `fpv_pose/` (+ `P03_03_01/fpv_pose/`), `rig/`,
  `sam3_fpv/`, `sam3_T2/`, `sam3_T4/`, `sam3_T5/` (decode tables, track series, overlays),
  `preflight.rrd` (view with `uv run rerun runs/preflight-finebio-20260924/preflight.rrd`;
  left pane the rig in cm with z down, right pane the six cameras with markers, detector
  boxes, triangulated points and the fpv camera centre, ten frames on the `frame` timeline).
