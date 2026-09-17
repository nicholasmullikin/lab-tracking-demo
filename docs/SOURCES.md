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
