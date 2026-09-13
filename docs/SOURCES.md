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
