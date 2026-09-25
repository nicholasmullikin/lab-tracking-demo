# Source and provenance ledger

This ledger distinguishes source/provenance approval from a legal determination of
permitted use. Raw inputs and generated experiment outputs remain outside Git.

## Required record for each future asset

- Asset ID and clip ID
- Creator or dataset owner
- Canonical source URL
- Access date and access method
- License or dataset-agreement version
- Attribution/citation text
- Intended use and display scope
- Redistribution, derivative-work, and noncommercial constraints
- Local storage class (outside Git) and checksum
- Reviewer, review date, and approval status

## Approved local source: Assembly101 G2 clip

- Status: `approved` for the user-frozen G1/G2 local smoke inputs documented in
  `configs/clips/assembly101_nusar_9033_g2.json`.
- Clip ID: `assembly101-nusar-9033-g2`; dataset: `cvml-nus/assembly101`; pinned
  revision: `bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967`.
- Approved assets: static `C10379_rgb` and ego `HMC_21110305_mono10bit`, plus the
  timestamp-exact, audio-free 30 FPS proxies for source interval 215.000–395.000 s.
- Stated license: `CC BY-NC 4.0`; local acquisition facts are in
  `data/raw/assembly101/nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532/acquisition_report.md`.
- Smoke display/processing scope: first 10.0 seconds (proxy frames 0–299) of each
  approved proxy only. No annotations or poses are used.
- Ego diagnostic derivative scope: two additional first-10-second-only, ignored local
  run artifacts used the same approved ego proxy. One deterministically transformed only
  its decoded grayscale representation (p1–p99 normalization plus CLAHE, then
  three-channel replication); the other used original decoded BGR with a human-recorded
  frame-0 box. Neither condition downloaded an asset, weight, annotation, or additional
  video, and neither changes the preserved three-text-concept zero-shot baseline.
- Unresolved use decision: the G1/G2 approval does not determine whether job-seeking,
  private demos, or public displays satisfy CC BY-NC 4.0 or dataset-specific terms.
  Obtain a rights-holder or institutional determination before sharing beyond the
  approved local procedure.
- Repository policy: raw recordings, proxies, annotations, masks, model checkpoints,
  and generated RRDs stay in ignored storage and are not redistributed.

## Model source: MuggledSAM / SAM3

- MuggledSAM source path: `/home/nick/src/muggled_sam`; source license: Apache-2.0.
- Adapter basis: `simple_examples/video_segmentation_multiplexed.py`; Battle does not
  copy that source file.
- SAM3.1 checkpoint status: already available locally for the smoke run; MuggledSAM
  documents separate agreement requirements for SAM3 weights. Battle did not download,
  modify, bundle, or redistribute that checkpoint. Acquisition authorization and
  sharing rights were not independently verified.

## Model source: MediaPipe Hand Landmarker

- MediaPipe package/source: `mediapipe` 1.0.1 /
  `https://github.com/google-ai-edge/mediapipe`; source repository declares Apache-2.0.
- Task model: official Hand Landmarker float16 bundle, version `1`, retrieved Sep 15,
  2026 from
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`.
- Local ignored path: `models/mediapipe/hand_landmarker_float16_v1.task`; SHA-256
  `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.
- The bundle packages palm detection and 21-point hand landmark models. Battle does not
  copy vendor source or commit/redistribute the model.
- The official model page documents behavior and training-data summary, but the model
  bundle's license was not independently confirmed during this run. Treat redistribution
  as unapproved until that review is complete.

## Model source: WiLoR

- Source path: `/home/nick/src/WiLoR`; git revision `fcb911312a38fa8badd30d9656a167485d61b8f9`.
- Audit status: the checkout has uncommitted tracked changes (`demo.py`, `requirements.txt`,
  and `wilor/models/backbones/__init__.py`). The revision alone does not reproduce the
  Sep 16 evidence run.
- Checkpoints (ignored, outside Git): `pretrained_models/wilor_final.ckpt` SHA-256
  `3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2`;
  `pretrained_models/detector.pt` SHA-256
  `5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033`.
- Stated license: CC-BY-NC-ND for WiLoR weights; MANO and Ultralytics carry separate
  terms. Battle does not copy vendor source; the wilor pyenv runs `wilor_worker.py`.

## Model source: BoxMOT + YOLOv8n detector

- BoxMOT: pip package 25.0.0 / `https://github.com/mikel-brostrom/boxmot`.
- Installed package metadata declares AGPL-3.0. Ultralytics 8.1.34 metadata also declares
  AGPL-3.0; this is a license caveat, not a permission determination.
- Independent detector: Ultralytics YOLOv8n (`yolov8n.pt`), local ignored path
  `models/yolo/yolov8n.pt`, SHA-256
  `31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95`.
- ReID weights downloaded by BoxMOT to the wilor environment on first use
  (`osnet_x0_25_msmt17.pt`); not committed.

## Model source: OpenCLIP + Drop-DTW

- OpenCLIP: `open-clip-torch` 3.3.0, model `ViT-B-32` / `openai` weights (downloaded on
  first run into the wilor environment cache).
- Pinned rerun: Hugging Face `timm/vit_base_patch32_clip_224.openai` snapshot
  `a6f597a30f7b82c51704746581f9a4e41421e878`,
  `open_clip_model.safetensors` SHA-256
  `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`.
  The installed OpenCLIP package metadata declares MIT. `battle-drop-dtw-align` requires
  that local checkpoint and runs with `HF_HUB_OFFLINE=1`.
- Drop-DTW: `/home/nick/src/Drop-DTW` @ `32ce9c82c6a0d717a94f4139b1902ad146923444`.
- Assembly101 coarse transcript: local file under approved raw tree; weak supervision only.

## Approved local source: Assembly101 ego viewpoint screen

- Status: `approved` only for the bounded e1/e2/e4 viewpoint-screen inputs in
  `configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json`; this is separate
  from and does not alter the original static/e3 G2 pair.
- Dataset/revision: `cvml-nus/assembly101` at
  `bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967`; selected source interval
  215.000–395.000 seconds / raw `[12900,23700)` at 60 FPS / proxy `[0,5400)` at 30 FPS.
- Assets: only `HMC_21176875_mono10bit`, `HMC_21176623_mono10bit`, and
  `HMC_21179183_mono10bit` were selectively downloaded. Their acquisition facts and
  checksums are recorded in the separate ignored local report
  `data/raw/assembly101/nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532/
  ego_viewpoint_screen_acquisition_report.md`.
- Processing/display scope: audio-free 954×720 CFR 30-FPS proxies; a sequential
  first-300-frame/10.0-second SAM3 smoke per new view; original decoded BGR input and
  the fixed three text concepts only. The preserved e3 original zero-shot smoke is
  review-only comparison evidence and was not altered or rerun.
- Unresolved use decision: as for the original G2 pair, G1/G2 operational approval
  does not decide whether job-seeking, private demos, or public display satisfy CC
  BY-NC 4.0 or dataset-specific terms.

## Approved local source: Assembly101 poses, extrinsics and fine-grained annotations

- Status: acquired Sep 17 at the user's explicit request for this recording only; this
  crosses the earlier "no annotations / no poses" gate. Same dataset/revision
  (`cvml-nus/assembly101` @ `bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967`), CC BY-NC 4.0.
- Assets: the ten `AssemblyPoses.zip` members for the recording, range-extracted from the
  72 GB archive (never downloaded whole; per-member CRC-32 verified), and this recording's
  fine-grained rows streamed out of the split CSVs. Acquisition facts, checksums, and
  format findings are in the ignored
  `data/raw/assembly101/nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532/
  selective_poses_annotations_acquisition_report.md` and `acquisition_manifest.json`.
- Derived, checked in: `configs/assembly101/c10379_camera_estimate.json`, a Brown camera
  model for C10379 fitted to the dataset's own 2D/3D landmark pairs through its shipped
  camera-to-world pose (fit RMS 5e-5 px, 4,388 points). It reproduces the dataset's
  internal projection and is labelled `estimated_from_dataset_landmark_projection`; the
  archive ships no intrinsics.
- Derived, ignored: `runs/assembly101-reference-first-minute-v1/` from
  `battle-build-assembly101-reference` (first-minute window, static clock offset +9 pose
  frames). Use: external review context in the v4 interaction review. Not ground truth for
  any method; no accuracy claim rests on it. TSM/DINOv2 features were inspected and not
  downloaded (bulk LMDB archives; DINOv2 only exists for view C10119).

## Approved local source: Assembly101 all static views, focused window (G1, Sep 18)

- Status: `approved` (overnight multicam plan, user-approved) for the focused window only:
  source interval 294.000–386.700 s / raw 60 fps frames `[17640, 23202)` of the seven static
  views not previously on disk. Same dataset/revision (`cvml-nus/assembly101` @
  `bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967`), CC BY-NC 4.0, same unresolved use decision
  as the original G2 pair.
- Access method: `battle-fetch-assembly101-view`, Sep 18 2026. `hf_hub_url` +
  `get_hf_file_metadata` resolve the pinned file to a signed CDN URL; ffmpeg reads it through
  a local counting proxy that forwards HTTP Range requests (`-ss 294 -t 92.7`, one decode,
  two encodes). The full files were never downloaded. Per view: two requests
  (`bytes=0-`, 3,145,728 B for the moov atom; one open-ended range for the mdat window).
- Assets (HF path `recordings/<recording>/<view>_rgb.mp4`; the HF LFS etag is the file's
  SHA-256; "bytes" is what the proxy delivered to ffmpeg):

  | view | HF size (B) | full-file SHA-256 (etag) | mdat range start | bytes transferred | raw60 trim SHA-256 | 720p proxy SHA-256 |
  |---|---:|---|---:|---:|---|---|
  | C10095 | 2,462,520,301 | `ab9bea9b24d95db6364f7addac1839bfa4ecc48c9f7b10e4eef913e48b7eb052` | 779,920,264 | 256,901,120 | `c01909c59a14acd2efec3b7e41329517be8d79b68a114839bb4940d1e99fe9d6` | `a791ef14094f565d0de7322b5dd4cd721ec6f60bb8b2099fb8d1119da6f262bb` |
  | C10115 | 2,631,101,730 | see manifest | 820,987,888 | 277,872,640 | `fe26926894e8db2ee159c562b0cbe4ec6a2326e813a96cb91d64e222fcc79191` | see clip config |
  | C10118 | 4,000,615,566 | see manifest | 1,246,098,998 | 428,867,584 | `8e80c266d833822758a6598a5c0047b15ec3a6244cae1a214dd5e22945d8c4b9` | see clip config |
  | C10119 | 2,133,892,339 | see manifest | 663,479,796 | 228,589,568 | `cc6182ea38048d114a48e727c85a94cc1d5b93e157f04e9d0715df4563898765` | see clip config |
  | C10390 | 1,876,452,183 | see manifest | 591,470,811 | 198,180,864 | `77075620643eb122239cc8d60ee118359b85cc6f3287643ec88c8f629419ad47` | see clip config |
  | C10395 | 2,613,875,251 | see manifest | 817,254,675 | 286,261,248 | `1b16c70c8acd475400aad9263e403c73cad8880b592edd1f9412f64bf089935c` | see clip config |
  | C10404 | 1,628,339,186 | see manifest | 507,404,173 | 176,160,768 | `fd8e19e61645bcd625fb7f96c8038ca441b10beb31e2d652dd86acf2699445a6` | see clip config |

  Total 1,852,833,792 B. Every full-file SHA-256 (etag) and proxy SHA-256 is in the tracked
  `configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_all_static_g2.json`
  (`raw_source.checksum_sha256` / `checksum_sha256`), and all measured facts (sizes, frame
  counts, ranges, ffmpeg version) in the ignored
  `data/raw/assembly101/<recording>/static_views_focused_acquisition_report.md` +
  `static_views_focused_acquisition_manifest.json`.
- Local storage class: ignored `data/derived/assembly101/<recording>/` (`*_raw60.mp4` trims,
  1920x1080 60 fps, libx264 crf 18; `*_1280x720_30fps.mp4` proxies). The trims of C10379 and
  the four HMC cameras, plus 954x720 proxies at 294.000 s for HMC_21176623/21176875/21179183,
  were made from the recordings already on disk with the same recipe.
- Derived, checked in: `configs/assembly101/<view>_camera_estimate.json` for all eight static
  and four ego views (estimates of the dataset's internal projection, provenance
  `estimated_from_dataset_landmark_projection`) and `configs/assembly101/clock_rules.json`
  (per-view video-vs-pose offsets measured by `battle-assembly101-clock-offset`). Review
  context only; no accuracy claim rests on any of it.

## Approved local source: Assembly101 recording 2 (`nusar_9061`), 80 s window (Track C1, Sep 20)

- Status: `approved` under the Sep 20 multicam plan (Track C, "a second Assembly101 recording
  of the same toy with a different subject"), window only. Recording
  `nusar-2021_action_both_9061-c02a_9061_user_id_2021-02-09_141537` (toy `c02a`, subject 9061;
  recording 1 is subject 9033). Same dataset/revision (`cvml-nus/assembly101` @
  `bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967`), CC BY-NC 4.0, same unresolved use decision as
  the original G2 pair. Choice and rejected candidates with reasons:
  `configs/assembly101/nusar_9061/recording_selection.json`; registry entry:
  `configs/assembly101/recordings.json`.
- Window: source 374.000-454.000 s (raw 60 fps frames `[22440, 27240)`, 4,800 trim / 2,400
  proxy frames); the 60 s analysis span is 384.000-444.000 s = proxy frames `[300, 2100)`,
  10 s of margin each side. Coarse actions inside the span (dataset labels, 30 fps): attach
  interior, screw chassis, attach body, attempt to attach bumper, screw chassis, attach bumper,
  attempt to attach cabin, unscrew chassis, detach interior, attach interior, attach cabin.
- Access method: `battle-fetch-assembly101-view --recording nusar_9061` (Sep 20 2026), the
  Track 0 recipe unchanged (signed CDN URL, local counting proxy, HTTP Range, `-ss 374 -t 80`,
  one decode, two encodes, ffmpeg 8.1.2). Two requests per view (`bytes=0-` for the moov atom,
  one open-ended range for the mdat window). The full files were never downloaded.
- Video assets (HF path `recordings/<recording>/<view>_rgb.mp4` or `_mono10bit.mp4`; the HF
  LFS etag is the file's SHA-256 and is recorded as the raw-source checksum; "bytes" is what
  the proxy delivered to ffmpeg):

  | view | HF size (B) | full-file SHA-256 (etag) | mdat range start | bytes transferred |
  |---|---:|---|---:|---:|
  | C10095 | 1,561,006,255 | `7bbbf85830e89de06296708bb1bf8c34cb5369b1e255e1cb28055f77cc0d0f5c` | 1,036,853,453 | 244,318,208 |
  | C10115 | 1,610,542,189 | `b901175c00bdc00779616a7f2773dc91bb4697a2641568f0872256992aa531fb` | 1,065,821,495 | 258,998,272 |
  | C10118 | 2,316,443,553 | `83da90db44b642c0d6384b61afb0de22eb65a9181d809f50a1ecc2d7a816e62e` | 1,563,533,624 | 332,398,592 |
  | C10119 | 1,218,389,649 | `2f8848cd1099750f8edebf55cbc66859eb0114191f7e95b640ff6ff21de3cc64` | 803,186,811 | 204,472,320 |
  | C10379 | 1,539,394,683 | `d80d4136022bf188d69db7211da88dde316d7fd632fca4a108b50beaef2f7148` | 1,026,288,297 | 246,415,360 |
  | C10390 | 950,213,939 | `5c0c158ce5ce29add2f95d81627576b0e0cbd7d511fbabadabc4dae2dc84c48a` | 634,300,474 | 155,189,248 |
  | C10395 | 1,453,831,016 | `185cae75c0c866f79a5c01bda5c841af940a47e2bede859e701aceca214924dd` | 950,471,736 | 235,929,600 |
  | C10404 | 993,837,625 | `ca3a869454ee49725c0062b02ac1c7e2e6d911adab9508d029d4a44e99a9ecdf` | 660,522,739 | 156,237,824 |
  | HMC_21110305 (e3) | 86,372,095 | `c76029bcf45c5bc9ba6f97b032189eb6a81bb8a1deeb9d77c5dd400fbd448301` | 56,827,782 | 16,777,216 |
  | HMC_21179183 (e4) | 99,072,741 | `8dbd3891590993757f2bd05a6f87b0091bd9c9de4e50bf29e9714351cfbce0e4` | 65,384,074 | 19,922,944 |

  Video total 1,870,659,584 B (14-20 % of each file). The two ego views are the e3/e4 camera
  ids Track 6 found useful on recording 1; recording 2 uses the same headset serials, so the
  choice transferred by camera id (the visibility audit needs hulls that do not exist yet).
  HMC_21176623 and HMC_21176875 exist on the Hub and were not fetched. Trim and proxy
  SHA-256s, frame counts and ranges are in the tracked clip configs
  (`configs/clips/assembly101_nusar_9061_four_part_reassembly_focused_all_static_g2.json`,
  `..._ego_hmc_21110305_g2.json`, `..._ego_hmc_21179183_g2.json`) and in the ignored
  `data/raw/assembly101/<recording>/static_views_focused_acquisition_{report.md,manifest.json}`.
- Poses and annotations (`battle-fetch-assembly101-poses --recording nusar_9061`): the ten
  `AssemblyPoses.zip` members for the recording, range-extracted from the 72 GB archive
  (281,494,021 compressed bytes; per-member CRC-32 verified; `members_manifest.json`); the
  fine-grained split CSVs streamed once (train/validation/test, 183 MB; full-file SHA-256 equal
  to the LFS etag for all three; 3,516 rows for this recording, all in `train`, kept as
  `train__<recording>.csv`); the coarse labels file and four small lookup tables (git blob ids
  verified). Total 465,555,825 B. Derived, ignored:
  `data/derived/assembly101/<recording>/assembly101_landmarks2D_60fps_frames_22440_27240.npz`
  (12 views, no missing pose frame). Record:
  `data/raw/assembly101/<recording>/poses_annotations_acquisition.json`.
- Local storage class: ignored `data/derived/assembly101/<recording>/` (`*_raw60.mp4` trims,
  1.43 GB; `*_1280x720_30fps.mp4` / `*_954x720_30fps.mp4` proxies, 0.40 GB; the contact sheet
  `contact_sheet_374.000-454.000.png`); ignored `data/raw/assembly101/<recording>/` (0.83 GB).
- Derived, checked in: `configs/assembly101/nusar_9061/<view>_camera_estimate.json` for the
  ten fetched views (estimates of the dataset's internal projection, provenance
  `estimated_from_dataset_landmark_projection`) and `configs/assembly101/clock_rules_nusar_9061.json`
  (per-view video-vs-pose offsets measured by `battle-assembly101-clock-offset`). Review
  context only; no accuracy claim rests on any of it.

## External source: Grounded-SAM-2

- Checkout: `/home/nick/src/Grounded-SAM-2` @ `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28`
  (`IDEA-Research/Grounded-SAM-2`, Apache-2.0).
- Environment: pyenv `grounded_sam2` (torch 2.11.0+cu128).
- Weights (ignored): `sam2.1_hiera_tiny.pt` SHA-256
  `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`; SAM2 config
  `configs/sam2.1/sam2.1_hiera_t.yaml` from the checkout.
- Detector: Hugging Face `IDEA-Research/grounding-dino-tiny` revision
  `a2bb814dd30d776dcf7e30523b00659f4f141c71` (pinned in Battle worker); prompt `hand.`.
- Battle adapter: `battle-grounding-dino-sam2-video` (`transformers_grounding_dino_plus_sam2_video_smoke`).
  Preserved 300-frame run:
  `runs/transformers_grounding_dino_plus_sam2_video_smoke-10s-20260916t0518z/`.
- Legacy one-frame external JSON (un-pinned HF revision, no video propagation) remains under
  `runs/grounded-sam2-static-frame0-smoke-20260916t0450z/` and is imported by
  `battle-import-external-smoke grounded-sam2`.
- Install note: local Grounding DINO CUDA extension build failed (CUDA 13.2 vs torch 12.8);
  smoke uses HF detector path only. This is not the vendor CUDA Grounded-SAM-2 extension.

## External source: SAMURAI

- Checkout: `/home/nick/src/samurai` @ `76ba195984892b0d1e3db5d9c90bb62175680a`
  (Apache-2.0). External checkout fingerprint: clean except untracked
  `sam2.1_hiera_tiny.pt` symlink at repo root (not used; worker reads
  `sam2/checkpoints/sam2.1_hiera_tiny.pt`).
- Environment: pyenv `samurai` (torch 2.11+cu128); SAM2.1 tiny checkpoint SHA-256
  `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.
- Battle adapter: `battle-samurai-video` (`samurai_sam2_video_smoke`). Worker config
  `configs/samurai/sam2.1_hiera_t.yaml` sets `samurai_mode: true`.
- Preserved 10 s run:
  `runs/samurai_sam2_video_smoke-10s-20260916t052328z/`.

## External source: DAM4SAM

- Checkout: `/home/nick/src/DAM4SAM` @ `9c954504b39ebca4c412f207be0787c26bfac85a`.
  External checkout fingerprint: clean except untracked symlink
  `sam2.1_hiera_tiny.pt` → `checkpoints/sam2.1_hiera_tiny.pt` (pre-existing local aid).
- Environment: pyenv `samurai` (torch 2.11+cu128) because official torch 2.1+cu121 env
  is incompatible with sm_120 GPU; `vot-toolkit==0.7.1` installed for `vot.region` imports.
- Battle adapter: `battle-dam4sam-video` (`dam4sam_video_smoke`). Uses
  `DAM4SAMTracker('sam21pp-T')` headless bbox initialization, not interactive
  `run_bbox_example.py` or the VOT wrapper.
- Preserved 10 s run:
  `runs/dam4sam_video_smoke-10s-20260916t052430z/`.
- Legacy 602-frame external mask import remains under
  `runs/dam4sam-static-20s-smoke-20260916t0520z/` (1-indexed filenames; superseded by
  source-aligned 0-indexed integrated runs).

## External source: ATHENA (blocked for Assembly101 real data)

- Checkout: `/home/nick/src/athena` @ `e85bd49444253aed9532439ace8ede146d1b6470` (MIT).
- Range probe evidence: `data/logs/athena_hf_calibration_probe.json` and selective extract under
  `data/raw/assembly101/athena_calibration_extract/` from official `cvml-nus/assembly101`
  `AssemblyPoses.zip` @ `bfc15ea5…` (72 GB; central directory inspected, not whole-archive download).
- Archive members for the approved recording include extrinsics/positions/timestamps but no
  intrinsics path; ATHENA fixture smoke only: `uv run battle-athena-fixture-smoke`.

### Sep 18: ATHENA installed and used for real triangulation

- Same checkout and revision (`e85bd49444253aed9532439ace8ede146d1b6470`, clean tree), installed
  editable into `/home/nick/src/athena/.venv` (py3.12; `uv venv --python 3.12 .venv && uv pip
  install --python .venv/bin/python -e .`): mediapipe 0.10.21, numpy 1.26.4, opencv-python
  4.11.0, scipy 1.17.1, athena 1.0. The battle env keeps mediapipe 1.0.1; the two never share an
  interpreter.
- Used functions: `athena.triangulaterefine._triangulate_with_filtering` and `_smooth3d`, called
  by `scripts/athena_triangulate_worker.py` under that interpreter from `battle-athena-hands`.
  The intrinsics blocker is gone in practice: intrinsics are the fitted estimates in
  `configs/assembly101/*_camera_estimate.json` (Track 0), not a dataset member.

## External source: Kineo (`kineo_nlf_only_partial`)

- Checkout: `/home/nick/src/kineo` @ `03b36e31…` (dirty working tree during smokes).
- Checked-in Battle config: `configs/kineo_nlf_headless_only.yaml`.
- Wrapper: `uv run battle-kineo-nlf --seconds 20`.
- Fused-crop wrapper: `CUDA_VISIBLE_DEVICES=0 uv run battle-kineo-fusion`. It consumes only the
  approved static-view, source/time-aligned native Kineo and BoxMOT runs; its per-frame source,
  gate measurements, and bounded residual inference are recorded in `box_fusion.json`.
- Native PKLs under ignored `runs/kineo/infer_nlf_headless_only/`; normalized runs under
  `runs/kineo-nlf-headless-*` and `runs/kineo-nlf-fused-*`. Both remain NLF-only partial evidence,
  not a full Kineo integration.
- **Sep 18, Track 4 (`battle-kineo-multiview`).** Same checkout, pinned
  `03b36e31c79bd40bc8bb1ce4c9c08907140952dd`, working tree dirty: `M kineo/pipeline/stages/rerun_export.py`,
  `M kineo/pipeline/stages/sfm_camera_extrinsics_initialization.py` (two-line guard on graph
  edges without point pairs), `M pixi.lock`, `M pyproject.toml`, `?? kineo/demo/rerun_export.py`;
  dirty-tree fingerprint (SHA-256 of `git status --short` + `git diff --stat`)
  `64a2ff02a2b83df4a231dcff29bf571208128c1e6e9d280a8d6b8aa6bb2878eb`, recorded per run in
  `prepare.json`. Nothing in the checkout was edited for this track; the generated YAMLs live in
  the run directories, the runner is `scripts/kineo_multiview_runner.py` (battle-owned, executed
  with `pixi run python` from the Kineo checkout). Models: `checkpoints/nlf_l_multi_0.3.2.torchscript`
  (SHA-256 `52bee28edb6ea9148691331df87cfc238d7e3d9134dc60104a5aaed282a9ddad`), MoGe
  `Ruicheng/moge-2-vitl` (self-calibration arm only), rtmlib YOLOX-tiny (openmmlab URL),
  `body_models/smplx/SMPLX_NEUTRAL.npz` (self-calibration arm only). Inputs are the eight
  static G1 proxies trimmed to 1,800 frames (fingerprints in `prepare.json`).

## External source: LM-EEC (ego-exo correspondence, Sep 18, Track 7)

- Repository: `https://github.com/juneyeeHu/LM-EEC` ("Robust Ego-Exo Correspondence with
  Long-Term Memory", Hu et al., NeurIPS 2025, arXiv 2510.11417). Checkout
  `/home/nick/src/LM-EEC` pinned at `b37e50e50fd03ae8625e6100da37bad3dfeb6aa4` (clean tree;
  nothing edited). Code is Meta's SAM 2 (Apache-2.0 headers) plus the authors' additions; the
  repository ships no LICENSE file of its own.
- Environment: own venv (`.venv`, Python 3.10, torch 2.7.1+cu128, torchvision 0.22.1, editable
  install with `SAM2_BUILD_CUDA=0`, plus `timm`, `matplotlib`, `scikit-learn`, `networkx`,
  `natsort`, `pycocotools`, `opencv-python-headless`), created by `scripts/install_lm_eec.sh`.
  The battle env is untouched.
- Weights (retrieved Sep 18, 2026 with `gdown` from the authors' Google Drive folder
  `1tc5HNWl0j7BcJE4uX0Bzb6PiYdlIWvXx`, linked from the README; not on Hugging Face):
  `checkpoints/LM-EEC-checkpoint/ExoEgo_checkpoint.pt` SHA-256
  `b3130bcbb8c907bf86a0c3afa321aa9d31c74ae64b842c71e1503ffba83dcd9c` (1,003,932,430 B) and
  `EgoExo_checkpoint.pt` SHA-256
  `a79234ab9ac20ca7a6d493ee8b65dbf970e6b5d0ace52321c500e850daf259ad` (1,003,932,238 B). Both are
  full fine-tuned SAM 2.1 base-plus models (627 tensors, epoch 60, 11,280 steps) trained on the
  Ego-Exo4D correspondence split; the direction each serves is not documented and is inferred
  from the file name (`ExoEgo` -> exo query, ego prediction). The SAM 2.1 base-plus checkpoint
  under `checkpoint/sam2.1_hiera_base_plus.pt` is a symlink to the Grounded-SAM-2 copy and is
  not read at inference.
- Battle-owned glue: `src/battle/egoexo_correspondence.py`, `scripts/lm_eec_driver.py` (run
  under the LM-EEC interpreter), `scripts/install_lm_eec.sh`. Inputs: the C10379 and
  HMC_21110305 G1 proxies (twelve keyframes each), the ensemble reference masks, the Sep 16 ego
  SAM3 masks, the Track 5 hull voxels (fingerprints in `pairs.json`).
- Fallback not exercised: `/home/nick/src/ObjectRelator` (cloned earlier, not installed) and
  the `wangzeze/ObjectRelator-Exo2Ego-Small` checkpoint were not needed because LM-EEC installed
  and constructed on CPU within the 90 min box.

## Approved local source: FineBio (Sep 21, 2026)

- Status: `access granted`; local handling approved for the non-commercial research
  smoke recorded in `docs/method-ledger.md` ("Sep 21: FineBio first look"). No
  determination has been made about demos, job-seeking or any display beyond the local
  procedure; obtain one before sharing anything derived from these videos.
- Dataset: FineBio, "FineBio: A Fine-Grained Video Dataset of Biological Experiments with
  Hierarchical Annotation", Takuma Yagi, Misaki Ohashi, Yifei Huang, Ryosuke Furuta, Shungo
  Adachi, Toutai Mitsuyama, Yoichi Sato; International Journal of Computer Vision 133,
  7352-7367 (2025), `https://doi.org/10.1007/s11263-025-02523-2`; arXiv 2402.00293. Data
  release and licence form via the AIST repository `https://github.com/aistairc/FineBio`
  (the code there is MIT; the videos, metadata and annotations are not).
- Access: by signed FineBio licence agreement submitted through the form linked from that
  repository; the dataset link and credentials were sent to the user by email after approval.
  Terms as stated there: non-commercial research/development use only; citation of the IJCV
  paper required. The signed agreement and the download date are the user's records; the
  repository holds no copy of either.
- Downloaded and extracted under `data/raw/finebio/` (gitignored), with the `7z` extraction
  logs beside each directory: `finebio_videos_fpv_test` (35 first-person MP4s at 1920x1440,
  5.8 GB), `finebio_videos_fpv_all_w640` (226 first-person MP4s downscaled to 640 px wide,
  6.3 GB), `finebio_videos_tpv_test` (12 GB) and `finebio_videos_tpv_valid` (14 GB)
  third-person views, and `finebio_object_detection_images` (249 MB). No annotation files
  were read by the Sep 21 smoke.
- Clip handled so far: `finebio_videos_fpv_test/finebio_videos/P03_01_01.mp4` (SHA-256
  `cfa11f06333cb00aaca4b348aaa9f405fb633741bd913939012fe5a2f2f1e896`), source interval
  60.000-80.000 s, as the 600-frame 1280x960 30 fps proxy
  `data/derived/finebio/P03_01_01/P03_01_01_060.000-080.000_1280x960_30fps.mp4` (SHA-256
  `0691edeb48312d563c3cd55498c749219b6bf0fb74f3257cc8a1736a6811fddf`). The proxy, masks,
  contact sheet and RRD under `runs/finebio-sam3-smoke-20260921/` are derivatives under the
  same terms and stay outside Git.
- Repository policy: no FineBio frame, video, mask or checksum-bearing manifest of a frame
  is committed; `configs/` holds no FineBio clip config because the Assembly101 G2 manifest
  schema does not describe this source (see the ledger entry).
- Second derivative on the same clip (Sep 21, later): `runs/finebio-dino-20260921/`
  (detections, contact sheets, RRD carrying the video and the SAM3 smoke's masks) from the
  FineBio shipped detector below; same terms, outside Git.

## External source: MMDetection and the FineBio shipped detector weights (Sep 21, 2026)

- Code: MMDetection, `https://github.com/open-mmlab/mmdetection`, Apache-2.0. Checkout
  `/home/nick/src/finebio-detector/mmdetection` pinned at tag `v3.3.0`
  (`44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`, the last release accepting mmcv < 2.2); the
  only change to the tree is the two FineBio config files copied into `configs/dino/` and
  `configs/deformable_detr/` as the authors' README instructs. Companion packages in the same
  venv: mmcv 2.1.0 (OpenMMLab prebuilt CPU wheel for torch 2.1, Apache-2.0), mmengine 0.10.7
  (Apache-2.0), torch 2.1.2+cpu / torchvision 0.16.2+cpu (BSD-3), all in
  `/home/nick/src/finebio-detector/.venv` (Python 3.10), created by
  `scripts/install_finebio_detector.sh`. The battle env is untouched.
- FineBio detector configs and code: `github.com/aistairc/FineBio/object_detection/` at commit
  `cb8d16ef13c7c9901418c13bcbaa50a3bdf3a2c3` (the repository's code is MIT):
  `dino-4scale_r50_8xb2-12e_finebio.py` SHA-256
  `6ac73d40c540ca856c9ec5f2083f8e6e4d1f6a95f92fcd7605371fe2ab7e4a1e` and
  `deformable-detr-refine-twostage_r50_16xb2-50e_finebio.py` SHA-256
  `227ea4291bca14f317828cf7c2e7eed444d95602e6b5ab2ffb70bd6ef8bc91d5`; both set
  `bbox_head.num_classes=35` over the MMDetection COCO base configs and list the 35 classes.
  Their two custom metric files (`AP_manipulated`, `AP_affected`) are not installed; they are
  evaluation-only.
- Weights (the authors' released checkpoints, retrieved Sep 21, 2026 with `gdown` from the
  Google Drive links in the FineBio README's 2025-12-15 update; the
  `finebio.s3.abci.ai/ckpts/` host named in the object_detection README no longer resolves):
  `checkpoints/dino.pth` SHA-256
  `e63995318ac28e230105f61f3e1db6c5de40748cb73850576f7c75cfc8029d94` (579,232,009 B; DINO
  4-scale R50, epoch 12, 47.7 M parameters, full training checkpoint with optimizer state,
  `dataset_meta` carries the 35 classes; the authors report AP 53.3 / AP50 77.4 on their test
  split) and `checkpoints/deformable-detr.pth` SHA-256
  `35982a45a17b4c7abf09f894feee8c43fd20413d105d9b984c1ff45be85bd6c3` (515,194,905 B;
  two-stage Deformable DETR with refinement, R50, epoch 50, 41.2 M parameters; AP 56.1 /
  78.5). Both are fine-tuned from the OpenMMLab COCO checkpoints named in the configs. The
  weights are research artefacts trained on FineBio annotations and are treated under the
  FineBio licence (non-commercial research); nothing is redistributed.
- Not on this machine: `finebio_coco_annotations.zip`, so no AP can be computed here; the
  Sep 21 run is qualitative.
- Battle-owned glue: `scripts/finebio_dino_detect.py` (detect phase under the detector
  interpreter, export under Battle), `scripts/install_finebio_detector.sh`.

## Approved local source: FineBio camera poses and shipped checkpoints (Sep 24, 2026)

- Status: `access granted` under the same signed FineBio licence agreement as the videos
  above (non-commercial research/development, citation of Yagi et al., IJCV 2025); local
  handling approved for the 3D object tracking plan
  ([`docs/plan-2026-09-25-finebio-3d-tracking.md`](plan-2026-09-25-finebio-3d-tracking.md)). No
  determination has been made about any display beyond the local procedure. The dataset
  README (`github.com/aistairc/FineBio`, read Sep 24) notes the licence agreement was updated
  on 2026-09-10; which version the user signed is the user's record, not the repository's.
- Retrieved Sep 24, 2026 (files placed 21:04-21:05 local) from the dataset release's `misc/`
  and `ckpts/` directories, the same gated download as the videos; the server mtimes
  (2024-06-27) are preserved on the files. Stored under `data/raw/finebio/` (gitignored).
- **Camera poses**: `misc/finebio_camera_poses.zip`, 64,013,894 B, SHA-256
  `ee8ee467804d84ff8584565155a81b95322f66d3559014c7548752597a217c3a`, extracted beside it to
  `misc/finebio_camera_poses/` (291 files, 77,073,685 B): `intrinsic_parameters/` with two
  GoPro 9 calibrations by `cv2.findChessboardCorners`, `gopro9_5_wide_4k_43_0.50.npz` (first
  person, 4000x3000 wide, SHA-256
  `58b5e95f3920afd0d5c9acfa5c581ed28554849eef9fc5d122cd8f62ed8d76c8`) and
  `gopro9_6_linear_4k_169_0.50.npz` (third person, 3840x2160, SHA-256
  `fe5080041180278008d82f79407046a9525a4b51fe7c5a795e70a04a3e42187d`);
  `third_person_camera_poses/<yymmdd>/`
  for the ten recording days `221013, 221021, 221109, 221110, 221117, 221118, 221124, 221125,
  221207, 221208`, each with `extrinsics/{1,2,3,4,6}_board.npz` (rotation and translation
  vectors from `cv2.calibrateCamera` on a checkerboard at the table centre, origin at its
  top-left corner) and `params/marker_points.npy` (AR marker positions by PnP from those
  extrinsics); `first_person_camera_poses/` with 226 per-trial `.npz` files (`rets`, `rots`,
  `trans`, one row per video frame, obtained by the authors from the markers); the authors'
  `README.txt` (1,714 B, SHA-256 `6c20376f…36789`), `vis_extrinsic_parameters.py` (6,614 B,
  `145e3116…6cf23`) and `vis_first_person_camera_poses.py` (4,519 B, `de2a4b92…31e3c`). The
  intrinsics are for the calibration resolutions and are rescaled by 0.5 (fixed) / 0.48 (fpv)
  for the shipped videos, as the README instructs and the preflight verified.
- How Battle uses them: the shipped fixed-camera extrinsics are kept only where they fit the
  bench markers within 10 px and replaced by a marker PnP where they do not (camera 6 on every
  day so far), the shipped per-frame fpv pose is used with an outlier gate, and the resulting
  per-trial camera config is committed as numbers only
  (`configs/finebio/cameras/<trial>.json`, six poses and two intrinsics fitted or copied from
  the dataset's calibration); that config is attributed to the dataset and is not a
  redistribution of the archive. `configs/finebio/trials.json` records frame counts, pose
  validity fractions and centrifuge lid intervals derived from the videos: numbers only.
- **Checkpoints** (`ckpts/`, the seven files the release ships; SHA-256, bytes, what each is
  per the dataset README and the benchmark READMEs):
  `dino.pth` `e63995318ac28e230105f61f3e1db6c5de40748cb73850576f7c75cfc8029d94`, 579,232,009 B
  (object detection, MMDetection DINO 4-scale R50, identical to the Sep 21 `gdown` copy in the
  detector venv); `deformable-detr.pth`
  `35982a45a17b4c7abf09f894feee8c43fd20413d105d9b984c1ff45be85bd6c3`, 515,194,905 B (object
  detection, two-stage Deformable DETR, identical to the Sep 21 copy); `dino_checkpoint_e30.pth`
  `70558986bc02324f95c6c3c383c085ba6ba7dc54dfe7224dfc88b64da474497d`, 561,278,348 B (the
  IDEA-Research DINO codebase object detector, 30 epochs, used frozen inside the
  manipulated/affected object detection benchmark); `handobj_checkpoint_e5.pth`
  `e092e4ec76bef02e5c7b160e3d442f6846d5c6d2299231bd6792bbc8794b4af2`, 188,348,734 B (the
  authors' manipulated/affected object detector: Shan et al., "Understanding Human Hands in
  Contact at Internet Scale", CVPR 2020, re-implemented on IDEA DINO with hand-state,
  manipulated-object and affected-object heads, 5 epochs); `actionformer.pth.tar`
  `20498a7a50ca028439c8320af73d60525097e38d5ea1e3555b4df3a1ed397fb4`, 540,945,256 B (atomic
  operation detection, ActionFormer on I3D features); `asformer.model`
  `85ae91fe30581c629f7a047a0536301476c4f7d9eb475a800afd466c2c62a237`, 7,513,633 B (step
  segmentation, ASFormer on I3D); `mstcn.model`
  `54332a08f8be319ee81f0e2e98fc04c69dc467468d58a37501499f1e6e27e6df`, 5,725,097 B (step
  segmentation, MS-TCN++ on I3D). Only `dino.pth` and `deformable-detr.pth` are used (through
  the detector venv, see above); the other five are inventoried, not loaded, and the atomic
  operation and step segmentation models are out of the plan's scope. All seven are the
  authors' research artefacts trained on FineBio annotations, carry no licence file of their
  own and are treated under the FineBio agreement; the FineBio README asks that the underlying
  methods (DINO, Deformable DETR, Hand Object Detector, ActionFormer, ASFormer, MS-TCN++, I3D,
  RAFT) be cited if the baselines are used.
- Repository policy, restated: nothing under `data/` is committed (no pose file, marker file,
  checkpoint, frame, video, mask or `.rrd`); derived numbers (camera configs, trial windows,
  lid intervals, observation fixtures) are.

## Candidate source: creator-uploaded video

- Status: `not approved`.
- Constraint: verify the upload-level license, creator identity, attribution, and
  redistribution rights. A platform's default license is not evidence of permission
  for this lab.
