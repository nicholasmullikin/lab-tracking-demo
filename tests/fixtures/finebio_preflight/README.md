# FineBio preflight fixtures (P03_01_01, Sep 24 2026)

This directory holds numeric derived data from the FineBio preflight of Sep 24
([`docs/archive/preflight-2026-09-24-finebio.md`](../../../docs/archive/preflight-2026-09-24-finebio.md)).
I converted them into the `p0-contracts` schemas so that the tracker lane can develop against six
real views without `data/`, `runs/` or a GPU. The loader is `tests/finebio_fixtures.py`
(`load_preflight_fixtures()`) and the tests are `tests/test_finebio_fixtures.py`.

## License

**Numbers only, no pixels.** FineBio is licensed for non-commercial research. This directory
holds detector boxes and scores, SAM3 mask bounding boxes, areas and scores, camera intrinsics,
extrinsics and per-frame poses, and residual statistics. I committed no video frame, video, mask
image, contact sheet or Rerun recording here. The only FineBio pixels in the repository are the
story media under `media/story/`, covered in `docs/LICENSES.md` under "Committed media (Sep 27)".

## Files

| file | bytes | content |
|---|---|---|
| `observations.jsonl` | about 4.4 MB | 20,209 `FineBioObservation` rows in compact JSONL: `schema_version` and an empty `provenance` are left out and the loader restores them |
| `cameras.json` | 8.1 kB | the `FineBioCameraConfig` of P03_01_01, byte-identical to `configs/finebio/cameras/P03_01_01.json`. `battle-finebio-cameras` rewrote that file on Sep 25: T4's shipped residual is now the chosen day's 4.78 px, not the best day's 2.23 px, and the poses are unchanged |
| `fpv_poses.json` | about 40 kB | the shipped per-frame fpv (head camera) pose, `rvec`, `tvec` and validity, for the 310 raw frames that carry an observation |
| `rig_reference.json` | about 20 kB | the preflight numbers a regression test reproduces, described below |

## Provenance and commands

The detector pass ran FineBio DINO from the detector's own virtual environment on the CPU in
914 s:

    CUDA_VISIBLE_DEVICES="" /home/nick/src/finebio-detector/.venv/bin/python \
        scripts/finebio_preflight_detect.py --trial P03_01_01 \
        --consecutive 1798:60 --spaced 1798:2398:30 \
        --output runs/preflight-finebio-20260924/detections

Camera mapping, the rig checks and the fpv pose check ran in the battle interpreter on the CPU:

    uv run python scripts/finebio_preflight.py --trial P03_01_01 mapping --seconds 30,60,90
    uv run python scripts/finebio_preflight.py --trial P03_01_01 fpv-pose --day 221013 --step 25
    uv run python scripts/finebio_preflight.py --trial P03_01_01 rig \
        --detections runs/preflight-finebio-20260924/detections

The SAM3 box-prompt decode and the 300-frame video-memory tracks ran in the MuggledSAM
interpreter on the GPU:

    CUDA_VISIBLE_DEVICES=0 /home/nick/.pyenv/versions/muggled_sam/bin/python \
        scripts/finebio_preflight_sam3.py --trial P03_01_01 --frame 1798 \
        --detections runs/preflight-finebio-20260924/detections \
        --output runs/preflight-finebio-20260924/sam3_<views> --decode-views ... --track-views ...

The conversion into this directory:

    uv run python scripts/finebio_preflight_fixtures.py \
        --preflight runs/preflight-finebio-20260924 --trial P03_01_01 \
        --output tests/fixtures/finebio_preflight

## Observation rows

All pixels are raw-video pixels (fixed views 1920 × 1080, fpv 1920 × 1440) and every
`frame_index` is a raw frame index of the shipped mp4s. The six videos of a trial are
synchronized to within one frame and have the same frame count. I rounded boxes to 0.1 px and
scores to four decimals.

- `source: detector` (15,695 rows): every FineBio DINO detection with a score of at least 0.3
  on the 78 preflight frames (1798–1858 consecutive, then every 30th from 1888 to 2368) in all
  six views. `slot` is `<class>#<k>`, with k the same-class score rank in that frame and 0 the
  top. It is not an identity across frames. `pose_valid` is the shipped fpv pose flag for fpv
  rows and always true for fixed views.
- `source: sam3_decode` (52 rows): the SAM3 image-decoder mask from the top detector box of
  each class at frame 1798, encoder side 1280, in all six views. `box_xyxy_px` is the prompt,
  and the row carries `mask_bbox_px`, `mask_area_px` and the provenance fields
  `decoder_iou_pred` and `mask_bbox_iou_vs_prompt`.
- `source: sam3_video` (4,462 rows): the SAM3.1 multiplex video-memory tracks seeded at frame
  1798 and run to 2097 (299 steps) in fpv, T2, T4 and T5 on cell_culture_plate, blue_pipette
  (not seeded in T4), centrifuge and 50ml_tube. `sam3_object_score` is the tracker's object
  score and `mask_area_px` is `area_frac x image area`. `detector_score` and `box_xyxy_px` are
  present on the 67 frames where the detector also ran, matched by class and score, and
  provenance carries `detector_box_iou`. Frames where the slot reported no mask are absent
  rows, not empty ones: T2 50ml_tube 1912–1925 behind the arm, fpv blue_pipette 1905–1913 at a
  frame-edge exit. `rig_reference.json` lists the omitted frames under
  `observations.sam3.track`.

The centroid is an approximation. The preflight kept only each mask's bounding box, so
`mask_centroid_px` on SAM3 rows is the center of the mask bounding box, not the mask's area
centroid. Real runs write the area centroid. The difference matters for elongated or partly
occluded masks, such as the held pipette, and a tracker test that is sensitive to it should say
so.

## `rig_reference.json`

The file collects the numbers from `mapping/mapping.json`, `rig/rig.json` and
`fpv_pose/fpv_pose.json`. Per fixed view it holds the camera id, the day, the shipped median
corner RMS (root mean square), the PnP (perspective-n-point) RMS and the PnP-against-shipped
distance in centimeters. It holds the day decision and the 11 static objects' five-view points,
with per-view all-view and LOO (leave-one-view-out) residuals and the LOO summary at a median of
10.4 px and a p90 (90th percentile) of 25.3 px. It holds hands as probes (left in 61 of 61
frames, right in 14 of 61), the clock scan's best offsets and LOO on the moving objects (median,
p90 and inside-box fraction per view). It holds the fixed-to-fpv plate hand-off (45 frames,
median 37 px, p90 52 px, inside the fpv box on 100%) and the fpv pose check (196 frames, corner
RMS median 0.93 px). `tests/test_finebio_fixtures.py` triangulates the static objects and the
plate hand-off from these rows and the camera config, and matches the reference within 0.05 cm
and 0.5 px.
