# License and handling policy

This repository contains original code, configuration, and synthetic test coordinates.
It includes no third-party recordings, annotations, model weights, or copied
third-party source code. Since Sep 27 it also holds rendered frames from two licensed
datasets under `media/story/`; see "Committed media (Sep 27)" below. The ignored local
`data/` directory contains approved G2 inputs outside Git.

## Noncommercial and provenance constraints

- An NC license can restrict job-seeking, demos, or other activity that confers a
  commercial advantage. This lab does not presume a private presentation is permitted.
- Dataset terms can add conditions beyond the headline license. Keep the current terms,
  access agreement, required citation, and display restrictions with each source entry.
- Do not redistribute raw videos, annotations, derived mask payloads, checkpoints, or
  gated assets through Git, Rerun recordings, screenshots, or release archives unless
  the applicable rights explicitly allow it.
- Rerun outputs from this session reference mask artifact URIs only; they never embed
  full mask payloads. Future exports must preserve this rule unless approved terms say
  otherwise.

## Current core-method inputs

- **Assembly101 clip `assembly101-nusar-9033-g2`:** the approved configuration records
  `CC BY-NC 4.0` and the local provenance ledger. G1 and G2 are user-approved for the
  exact raw recordings and their 180-second, audio-free proxies. This is provenance
  approval for the local smoke procedure, not a legal conclusion that a job-seeking,
  private demo, or public display is noncommercial. Raw video, proxy video, derived
  masks, annotations, and RRD files remain ignored and must not be redistributed. The
  one exception is the story media under `media/story/` (see "Committed media (Sep 27)").
- **Assembly101 poses and fine-grained annotations (Sep 17):** the same `CC BY-NC 4.0`
  terms and citation (Sener et al., CVPR 2022) cover the selectively acquired hand poses,
  extrinsics, and this recording's fine-grained rows, and the derived reference window under
  `runs/`. The checked-in camera estimate `configs/assembly101/c10379_camera_estimate.json`
  holds six numbers fitted from dataset landmarks plus the dataset's shipped 4x4 extrinsics;
  it is attributed to the dataset and is not redistribution of the dataset assets.
- **MuggledSAM source:** `/home/nick/src/muggled_sam` declares Apache-2.0. Battle uses
  its public interface and does not copy its code. Apache-2.0 covers that source, not
  the SAM3 checkpoint.
- **SAM3/SAM3.1 checkpoint:** MuggledSAM's README states that SAM3 weights require an
  agreement before downloading. The smoke run used an already-local checkpoint; Battle
  did not download, bundle, or redistribute it. Its checkpoint license/access status
  must be reviewed separately before a successful model run can be shared.
- **MediaPipe:** the MediaPipe source repository declares Apache-2.0. The separately
  downloaded official Hand Landmarker float16 v1 task bundle remains ignored and is not
  redistributed; its model-specific license was not independently confirmed, so no
  redistribution permission is claimed.
- **WiLoR:** checkpoints are CC-BY-NC-ND; MANO and Ultralytics dependencies carry
  separate licenses. Camera-relative 3D outputs are non-metric model estimates. The
  evidence checkout was dirty, so its base revision is not a reproducible source snapshot.
- **BoxMOT / YOLOv8n:** installed BoxMOT 25.0.0 and Ultralytics 8.1.34 package metadata
  declares AGPL-3.0. Association is conditional on independent per-frame COCO-person
  detections and is not comparable to SAM3 masks. The detector file is hashed, but its
  upstream acquisition/model-license decision remains unrecorded.
- **OpenCLIP + Drop-DTW:** the pinned `open-clip-torch` 3.3.0 package declares MIT; the
  OpenCLIP checkpoint is content-addressed in `docs/SOURCES.md`. Alignment uses Assembly101
  coarse labels as declared weak supervision; intervals and cost are not accuracy claims.
- **Grounded-SAM-2 / SAM2 / Grounding DINO:** Apache-2.0 source; Meta SAM2.1 tiny checkpoint
  and IDEA Grounding DINO Tiny (HF revision `a2bb814…`) used by
  `transformers_grounding_dino_plus_sam2_video_smoke`. Local Grounding DINO CUDA extension
  build failed; Battle uses the HF fallback detector, not the vendor CUDA extension.
- **SAMURAI / DAM4SAM:** Apache-2.0 and project licenses respectively; SAM2.1 tiny checkpoints
  from Meta public URLs (SHA-256 `7402e0…be69`). Battle adapters `samurai_sam2_video_smoke` and
  `dam4sam_video_smoke` preserve native SAMURAI/DAM4SAM semantics on a shared deterministic hand
  bbox seed; they do not establish accuracy or distractor-scene proof.
- **ATHENA:** MIT source checkout; Assembly101 real triangulation blocked (no intrinsics in
  official `AssemblyPoses.zip` member inventory). Fixture smoke only. Official sources place
  extrinsics/positions in `AssemblyPoses.zip` but do not establish that intrinsics are absent.
  *Sep 18 update:* ATHENA (MIT, Neural Control & Computation Lab, 2025) is now installed in its
  own virtualenv and its DLT/filter/smoothing functions run on Assembly101 detections via
  `battle-athena-hands`; MIT permits this use with attribution retained in the checkout's
  `LICENSE`. Its pinned dependency MediaPipe 0.10.21 (Apache-2.0) lives only in that venv. The
  triangulated outputs derive from Assembly101 video and poses and inherit the dataset's
  CC BY-NC 4.0 terms; the queued WiLoR arm additionally inherits WiLoR's CC-BY-NC-ND weights
  caveat.
- **Kineo:** research/evaluation terms per upstream README; its checkout was dirty during the
  NLF-only partial smoke. The readable PKLs are person-centric outputs without SfM, metric
  world scale, BVH, or hand-part claims. The Sep 18 multi-view arms (`battle-kineo-multiview`)
  stay under the same research/evaluation terms; their inputs and every comparison number
  derive from Assembly101 video, extrinsics and poses and inherit CC BY-NC 4.0.
- **LM-EEC (Track 7):** the repository ships no LICENSE file; its code is SAM 2 (Apache-2.0
  headers) plus the authors' NeurIPS 2025 additions, and the two released checkpoints are
  research artefacts fine-tuned from Meta's SAM 2.1 base-plus (Apache-2.0) on Ego-Exo4D, whose
  own licence (Ego-Exo4D license agreement, non-commercial research) travels with the weights.
  Treat both checkpoints as research/non-commercial only; nothing is redistributed. Every
  prediction is made on Assembly101 frames and masks and inherits CC BY-NC 4.0. The
  ObjectRelator fallback (`wangzeze/ObjectRelator-Exo2Ego-Small`, LLaVA/PSALM stack, likewise
  non-commercial) was not installed or run.
- **FineBio (Sep 21):** videos and images obtained under the FineBio licence agreement
  (Yagi et al., IJCV 2025; access by signed agreement via `github.com/aistairc/FineBio`,
  credentials by email), which limits use to non-commercial research/development and
  requires citation; the dataset is gated, so treat redistribution as forbidden. Everything
  under `data/raw/finebio/`, the `data/derived/finebio/` proxy and the
  `runs/finebio-sam3-smoke-20260921/` outputs (masks, contact sheet, RRD) are covered by
  those terms and stay outside Git. The SAM3 zero-shot smoke on one clip makes no accuracy
  claim; its outputs are not evidence for any FineBio-related result and are not to be shown
  outside the local procedure without a rights determination.
- **MMDetection and the FineBio shipped detector (Sep 21, later):** MMDetection v3.3.0, mmcv
  2.1.0 and mmengine 0.10.7 are Apache-2.0 and live in their own venv
  (`/home/nick/src/finebio-detector`); Battle uses their public inference API and copies no
  code. The FineBio object_detection configs come from a repository whose code is MIT. The two
  released checkpoints (`dino.pth`, `deformable-detr.pth`, checksums in `docs/SOURCES.md`)
  are the authors' research artefacts fine-tuned from OpenMMLab COCO checkpoints on the
  FineBio annotations; they carry no licence file of their own, so they are treated under the
  FineBio agreement (non-commercial research, citation of Yagi et al., IJCV 2025) and are not
  redistributed. Every detection in `runs/finebio-dino-20260921/` is made on FineBio frames
  and inherits those terms; the run is unscored (no annotations here) and makes no accuracy
  claim for the detector or for SAM3.
- **FineBio camera poses and shipped checkpoints (Sep 24):** the calibration archive
  (`misc/finebio_camera_poses/`: two GoPro intrinsic files, ten recording days of fixed-camera
  extrinsics and marker points, 226 per-trial first-person pose files, the authors' README and
  two visualisation scripts) and the seven checkpoints under `ckpts/` (`dino.pth`,
  `dino_checkpoint_e30.pth`, `deformable-detr.pth`, `handobj_checkpoint_e5.pth`,
  `actionformer.pth.tar`, `asformer.model`, `mstcn.model`; SHA-256 of each in
  `docs/SOURCES.md`) came through the same gated FineBio download and are covered by the same
  FineBio licence agreement (Yagi et al., IJCV 2025; non-commercial research/development,
  citation required; the authors note the agreement text was updated 2026-09-10). They carry
  no licence of their own. The `handobj` model is the authors' re-implementation of the Shan et
  al. (CVPR 2020) hand-object detector on the IDEA DINO codebase; the FineBio README asks that
  the underlying methods be cited if the baselines are used, and only the two object detectors
  are used here. Everything under `data/raw/finebio/misc/` and `data/raw/finebio/ckpts/` stays
  outside Git, as does every video and `.rrd` derived from the videos and every frame, mask
  and contact sheet under `runs/` (the Sep 24 trial-selection images under
  `runs/preflight-finebio-20260924/trials/` included). Until Sep 27, what the repository
  committed from these sources was numbers: per-trial camera configs fitted to or copied from
  the dataset's calibration, trial windows, centrifuge lid intervals, pose validity fractions
  and the preflight observation fixtures, attributed to the dataset. Since Sep 27 it also
  commits the rendered story media described in the next section. No sharing determination
  beyond the local procedure is made.

## Committed media (Sep 27)

`media/story/` holds GIFs and stills rendered on Sep 27 by `battle-story-media` from the
proxies, masks and tracks this lab produced, with `media/story/manifest.json` naming the
source of every file. They are the first pixels from either dataset committed to Git.
The Assembly101 frames are covered by `CC BY-NC 4.0` (Sener et al., CVPR 2022) and are
attributed to the dataset. The FineBio frames come from a gated dataset whose licence
agreement limits use to non-commercial research and requires citation of Yagi et al.,
IJCV 2025. This repository is a private GitHub repository (`origin` is
`github.com/nicholasmullikin/lab-tracking-demo`), which is the only reason these files are
in it. Before the repository is made public, or a clone, an archive or any of these files is
otherwise redistributed, every file under `media/story/` must be removed from the tree and
from history, or a rights determination obtained for each dataset. Committing them makes no
sharing determination and does not change the terms above.

## Dependency and future-model review

Python packages are pinned in `pyproject.toml` and resolved in `uv.lock`. Review their
current license notices before redistribution. Future adapters must separately document
the code license, checkpoint license, access requirements, and attribution for every
model; an open-source repository does not automatically grant rights to its weights.

## Decision record template

For each future source or model, record:

1. identifier and version;
2. authoritative terms URL and retrieval date;
3. intended use, audience, and duration;
4. attribution or citation required;
5. redistribution decision;
6. reviewer and approval status.
