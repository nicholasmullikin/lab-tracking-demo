# Methods inventory, Sep 2026 (was `configs/methods.yaml`)

Hand-maintained snapshot from Sep 8-16, never loaded by code; moved out of `configs/` on Sep 24.
[`SOURCES.md`](SOURCES.md) and the ledger's goals table are the maintained records.

```yaml
schema_version: "1.0"
hardware_assumption:
  gpu: "NVIDIA RTX 5070 Ti"
  vram_gb: 16
  measured: true

clocks:
  source_fps: 60
  analysis_fps: 30
  annotation_fps: 30
  pose_fps: 60

methods:
  muggledsam_sam3:
    state: succeeded
    environment: "~/.pyenv/versions/muggled_sam"
    adapter: battle.muggled_smoke
    external_source: /home/nick/src/muggled_sam
  mediapipe_hands:
    state: succeeded
    environment: battle uv / CPython 3.12
    adapter: battle.mediapipe_hands
  wilor_hands:
    state: external_partial
    environment: "~/.pyenv/versions/wilor"
    adapter: battle.wilor_hands
    external_source: /home/nick/src/WiLoR
    license_caveat: CC-BY-NC-ND; non-metric camera-relative 3D only; source checkout was dirty
  boxmot:
    state: succeeded
    environment: "~/.pyenv/versions/wilor"
    adapter: battle.boxmot_track
    external_source: https://github.com/mikel-brostrom/boxmot
    detector_source: ultralytics YOLOv8n per-frame COCO detections; not MuggledSAM IDs
  drop_dtw:
    state: integrated_smoke
    environment: "~/.pyenv/versions/wilor"
    adapter: battle.drop_dtw_align
    external_source: https://github.com/SamsungLabs/Drop-DTW
    external_revision: 32ce9c82c6a0d717a94f4139b1902ad146923444
    supervision: assembly101_gt_transcript_weak_supervision; OpenCLIP ViT-B-32/openai checkpoint pinned
  fine_substep_crop_clip:
    state: integrated_smoke
    environment: "~/.pyenv/versions/wilor"
    adapter: battle.fine_substep_align
    external_source: https://github.com/SamsungLabs/Drop-DTW
    external_revision: 32ce9c82c6a0d717a94f4139b1902ad146923444
    supervision: agent_authored_visual_review labels only; WiLoR-primary crops; baseline SAM3 part boxes
    sample_fps: 3
    claim_boundary: not benchmark accuracy against Assembly101 GT
  grounded_sam2:
    state: smoke_only
    adapter: battle.grounding_dino_sam2_video
    environment: "~/.pyenv/versions/grounded_sam2"
    external_source: /home/nick/src/Grounded-SAM-2
    external_revision: b7a9c29f196edff0eb54dbe14588d7ae5e3dde28
    smoke_scope: bounded HF Grounding-DINO-tiny frame-0 detect + SAM2.1 tiny video propagation
    install_caveat: local Grounding DINO CUDA extension failed (CUDA 13.2 vs torch 12.8); HF fallback only
  samurai:
    state: integrated_smoke
    environment: "~/.pyenv/versions/samurai"
    adapter: battle.samurai_video
    external_source: /home/nick/src/samurai
    external_revision: 76ba195984892b0d1e3db5d9c90bb62175680a
    smoke_scope: bounded SAMURAI samurai_mode SAM2.1 tiny video propagation with frame-0 hand bbox seed
  dam4sam:
    state: integrated_smoke
    environment: "~/.pyenv/versions/samurai"
    adapter: battle.dam4sam_video
    external_source: /home/nick/src/DAM4SAM
    external_revision: 9c954504b39ebca4c412f207be0787c26bfac85a
    smoke_scope: bounded DAM4SAMTracker sam21pp-T headless bbox init with native DRM path
    install_caveat: official torch 2.1.0+cu121 env incompatible with sm_120 GPU; pyenv samurai torch 2.11+cu128 used
  athena:
    state: blocked
    external_source: /home/nick/src/athena
    external_revision: e85bd49444253aed9532439ace8ede146d1b6470
    blocker: AssemblyPoses.zip exposes extrinsics/positions but no intrinsics member for ATHENA calibration
    evidence_log: docs/athena_hf_calibration_probe.json
    fixture_command: battle-athena-fixture-smoke
    comparison_policy: metadata_only; synthetic fixture must not share the Assembly101 RGB timeline
  kineo:
    state: external_partial
    environment: pixi-managed /home/nick/src/kineo
    adapter: battle.kineo_nlf
    smoke_scope: kineo_nlf_only_partial; 600-frame bounded export (456 PKL bbox/body-joint rows on latest rerun; no SfM/BVH)
    install_caveat: full offline config requires multi-view SfM or interactive SAM2 UI
    license_caveat: calibration-free person-centric; not hand-part specialized
  exploratory_comparison:
    state: succeeded
    adapter: battle.exploratory_comparison
    artifact_uri: runs/exploratory-first-20s-comparison/exploratory_first_20s_comparison.rrd
    index_uri: runs/exploratory-first-20s-comparison/exploratory_comparison_index.json
    scope: source-aligned static RGB frames 0-599; inference-free composition only
  four_part_segmentation_comparison:
    state: succeeded
    adapter: battle.four_part_segmentation + battle.four_part_comparison
    contract_uri: configs/four_part_segmentation_comparison.json
    artifact_uri: runs/four-part-segmentation-comparison/four_part_segmentation_comparison.rrd
    index_uri: runs/four-part-segmentation-comparison/four_part_segmentation_comparison_index.json
    scope: focused static RGB frames 0-599 / source 294.0-314.0; masks only
    initialization: Grounding-DINO arm uses independent prompts; SAM2 control, SAMURAI, and DAM4SAM use identical reviewed frame-zero masks
    claim_boundary: output coverage and model detections are not segmentation accuracy; human semantic QA remains pending
  deferred:
    - LM-EEC
    - ObjectRelator
    - Qwen video VLMs
    - supervised temporal-action models
    - long-video VLMs
```
