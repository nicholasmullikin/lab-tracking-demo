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
- Checkpoints (ignored, outside Git): `pretrained_models/wilor_final.ckpt` SHA-256
  `3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2`;
  `pretrained_models/detector.pt` SHA-256
  `5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033`.
- Stated license: CC-BY-NC-ND for WiLoR weights; MANO and Ultralytics carry separate
  terms. Battle does not copy vendor source; the wilor pyenv runs `wilor_worker.py`.

## Model source: BoxMOT + YOLOv8n detector

- BoxMOT: pip package 25.0.0 / `https://github.com/mikel-brostrom/boxmot`.
- Independent detector: Ultralytics YOLOv8n (`yolov8n.pt`), local ignored path
  `models/yolo/yolov8n.pt`, SHA-256
  `31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95`.
- ReID weights downloaded by BoxMOT to the wilor environment on first use
  (`osnet_x0_25_msmt17.pt`); not committed.

## Model source: OpenCLIP + Drop-DTW

- OpenCLIP: `open-clip-torch` 3.3.0, model `ViT-B-32` / `openai` weights (downloaded on
  first run into the wilor environment cache).
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

## External source: Grounded-SAM-2

- Checkout: `/home/nick/src/Grounded-SAM-2` @ `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28`
  (`IDEA-Research/Grounded-SAM-2`, Apache-2.0).
- Environment: pyenv `grounded_sam2` (torch 2.11.0+cu128).
- Weights (ignored): `sam2.1_hiera_tiny.pt` SHA-256
  `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`; HF
  `IDEA-Research/grounding-dino-tiny` at runtime.
- Install note: local Grounding DINO CUDA extension build failed (CUDA 13.2 vs torch 12.8);
  smoke used HF detector path only.

## External source: SAMURAI

- Checkout: `/home/nick/src/samurai` @ `76ba195984892b0d1e3db5d9c90bb62175680a`
  (Apache-2.0).
- Environment: pyenv `samurai`; SAM2.1 tiny checkpoint from Meta public URL.
- Battle does not copy vendor source; headless `scripts/demo.py` smoke only.

## External source: DAM4SAM

- Checkout: `/home/nick/src/DAM4SAM` @ `9c954504b39ebca4c412f207be0787c26bfac85a`.
- Environment: executed from pyenv `samurai` (torch 2.11) after official torch 2.1 env
  failed on sm_120 GPU; `vot-toolkit==0.7.1`.
- Headless bbox-init smoke; interactive `run_bbox_example.py` not used.

## External source: ATHENA (blocked)

- Checkout: `/home/nick/src/athena` @ `e85bd49444253aed9532439ace8ede146d1b6470` (MIT).
- Assembly101 HF probe: see `data/logs/athena_hf_calibration_probe.log`.
- Unofficial extrinsics sample downloaded from `pablovela5620/assembly101-720p` for
  inspection only; not approved upstream calibration.

## External source: Kineo

- Checkout: `/home/nick/src/kineo` (pixi env); headless NLF-only smoke config stored under
  ignored `data/logs/kineo_nlf_headless_only.yaml`.
- Outputs under ignored `runs/kineo/infer_nlf_headless_only/`.

## Candidate source: FineBio

- Status: `pending access and license review`; no application or download is performed
  by session one.
- Constraint: the plan describes noncommercial and citation requirements. Record the
  signed agreement/version and permitted display scope before handling a clip.

## Candidate source: creator-uploaded video

- Status: `not approved`.
- Constraint: verify the upload-level license, creator identity, attribution, and
  redistribution rights. A platform's default license is not evidence of permission
  for this lab.
