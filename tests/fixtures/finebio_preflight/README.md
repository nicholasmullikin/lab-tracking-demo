# FineBio preflight fixtures (P03_01_01, Sep 24, 2026)

Numeric derived data from the FineBio preflight of Sep 24
([`docs/preflight-2026-09-24-finebio.md`](../../../docs/preflight-2026-09-24-finebio.md)),
converted into the `p0-contracts` schemas so the tracker lane can develop against six real
views without `data/`, `runs/` or a GPU. Loader: `tests/finebio_fixtures.py`
(`load_preflight_fixtures()`); tests: `tests/test_finebio_fixtures.py`.

## Licence

FineBio is licensed for non-commercial research. This directory holds **numbers only**:
detector boxes and scores, SAM3 mask bounding boxes, areas and scores, camera intrinsics,
extrinsics and per-frame poses, and residual statistics. No video frame, video, mask image,
contact sheet or Rerun recording is committed here or anywhere else in the repository.

## Files

| file | bytes | content |
|---|---|---|
| `observations.jsonl` | ~4.4 MB | 20,209 `FineBioObservation` rows (compact JSONL: `schema_version` and empty `provenance` omitted, restored on load) |
| `cameras.json` | 5.2 kB | the `FineBioCameraConfig` of P03_01_01, byte-identical to `configs/finebio/cameras/P03_01_01.json` |
| `fpv_poses.json` | ~40 kB | the shipped per-frame fpv pose (`rvec`, `tvec`, validity) for the 310 raw frames that carry an observation |
| `rig_reference.json` | ~20 kB | the preflight numbers a regression test reproduces (below) |

## Provenance and commands

Detector pass (FineBio DINO, the detector venv, CPU, 914 s):

    CUDA_VISIBLE_DEVICES="" /home/nick/src/finebio-detector/.venv/bin/python \
        scripts/finebio_preflight_detect.py --trial P03_01_01 \
        --consecutive 1798:60 --spaced 1798:2398:30 \
        --output runs/preflight-finebio-20260924/detections

Camera mapping, rig checks and the fpv pose check (battle interpreter, CPU):

    uv run python scripts/finebio_preflight.py --trial P03_01_01 mapping --seconds 30,60,90
    uv run python scripts/finebio_preflight.py --trial P03_01_01 fpv-pose --day 221013 --step 25
    uv run python scripts/finebio_preflight.py --trial P03_01_01 rig \
        --detections runs/preflight-finebio-20260924/detections

SAM3 box-prompt decode and 300-frame video-memory tracks (MuggledSAM interpreter, GPU):

    CUDA_VISIBLE_DEVICES=0 /home/nick/.pyenv/versions/muggled_sam/bin/python \
        scripts/finebio_preflight_sam3.py --trial P03_01_01 --frame 1798 \
        --detections runs/preflight-finebio-20260924/detections \
        --output runs/preflight-finebio-20260924/sam3_<views> --decode-views ... --track-views ...

Conversion into this directory:

    uv run python scripts/finebio_preflight_fixtures.py \
        --preflight runs/preflight-finebio-20260924 --trial P03_01_01 \
        --output tests/fixtures/finebio_preflight

## Observation rows

All pixels are raw-video pixels (fixed views 1920x1080, fpv 1920x1440) and every
`frame_index` is a raw frame index of the shipped mp4s (the six videos of a trial are
synchronised to +/-1 frame and have the same frame count). Boxes are rounded to 0.1 px and
scores to 4 decimals.

- `source: detector` (15,695 rows): every FineBio DINO detection with score >= 0.3 on the 78
  preflight frames (1798..1858 consecutive, then 1888..2368 every 30th) in all six views.
  `slot` is `<class>#<k>` with k the same-class score rank in that frame (0 = top); it is
  **not** an identity across frames. `pose_valid` is the shipped fpv pose flag for fpv rows
  and always true for fixed views.
- `source: sam3_decode` (52 rows): the SAM3 image-decoder mask from the top detector box of
  each class at frame 1798, encoder side 1280, in all six views (`box_xyxy_px` = the prompt,
  `mask_bbox_px`, `mask_area_px`, provenance `decoder_iou_pred` and
  `mask_bbox_iou_vs_prompt`).
- `source: sam3_video` (4,462 rows): the SAM3.1 multiplex video-memory tracks seeded at frame
  1798 and run to 2097 (299 steps) in fpv, T2, T4 and T5 on cell_culture_plate, blue_pipette
  (not seeded in T4), centrifuge and 50ml_tube. `sam3_object_score` is the tracker's object
  score, `mask_area_px` is `area_frac x image area`, `detector_score` and `box_xyxy_px` are
  present on the 67 frames where the detector also ran (matched by class and score), and
  provenance carries `detector_box_iou`. Frames where the slot reported no mask are **absent**
  rows (T2 50ml_tube 1912-1925 behind the arm, fpv blue_pipette 1905-1913 frame-edge exit);
  the omitted frames are listed in `rig_reference.json` under `observations.sam3.track`.

**Centroid approximation.** The preflight kept only each mask's bounding box, so
`mask_centroid_px` on SAM3 rows is the **centre of the mask bounding box**, not the mask's
area centroid. Real runs write the area centroid; the difference matters for elongated or
partially occluded masks (the held pipette), and a tracker test that is sensitive to it should
say so.

## `rig_reference.json`

From `mapping/mapping.json`, `rig/rig.json` and `fpv_pose/fpv_pose.json`: per fixed view the
camera id, day, shipped median corner RMS, PnP RMS and PnP-vs-shipped centimetres; the day
decision; the eleven static objects' five-view points with per-view all-view and
leave-one-view-out residuals and the LOO summary (median 10.4 px, p90 25.3 px); hands as probes
(left 61/61 frames, right 14/61); the clock scan's best offsets; LOO on the moving objects
(median, p90, inside-box fraction per view); the fixed-to-fpv plate hand-off (45 frames, median
37 px, p90 52 px, inside the fpv box on 100%); the fpv pose check (196 frames, corner RMS
median 0.93 px). `tests/test_finebio_fixtures.py` triangulates the static objects and the plate
hand-off from these rows and the camera config and matches the reference within 0.05 cm /
0.5 px.
