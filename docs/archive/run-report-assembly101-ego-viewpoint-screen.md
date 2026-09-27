# Assembly101 bounded ego viewpoint-screen run report

## Scope

- Purpose: compare e1/e2/e4 against the preserved e3 original zero-shot smoke before
  asking for any method change or longer ego run.
- Dataset/revision: `cvml-nus/assembly101` at
  `bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967`.
- Input interval: raw 60-FPS `[12900,23700)` / 215.000–395.000 seconds; proxy 30-FPS
  `[0,5400)` / 180 seconds.
- Screen condition: exactly 300 proxy frames / 10.0 seconds, sequentially per new view;
  original decoded BGR images; text prompts `hand`, `yellow toy body`, `toy wheel`;
  one continuous tracker stream with one prompt-memory and four frame-memory entries.
- Exclusions: no contrast normalization, manual box seed, annotation/pose use, model or
  checkpoint download, 60-second ego run, or 180-second ego run.

## Inputs and validation

The three new raw recordings, their byte sizes/SHA-256 values, and their 636×480,
60/1-FPS, 925.683333-second, 55,541-frame `h264/yuv420p` ffprobe facts are recorded in
`data/raw/assembly101/nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532/
ego_viewpoint_screen_acquisition_report.md`.

`scripts/create_assembly101_ego_viewpoint_screen_g2_proxies.sh` made and revalidated the
three separate 954×720 `h264/yuv420p` proxies: CFR 30/1 FPS, 180.000000 seconds, 5,400
frames, no audio. The separate G1/G2 manifest is
`configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json`; it explicitly
labels these additions as not the original static/e3 pair.

## Run records

- e1 / `ego-hmc21176875`:
  `runs/muggledsam-sam3-smoke-ego-hmc21176875-20260909t033053z/`.
  Succeeded, 300/300 frames / 10.0 seconds, 14.574 s elapsed, 4.327 s TTFU,
  1,965,729,280-byte peak allocated VRAM, 45 external masks. Initial result: `hand`
  (`sam3-00`) only. Hand emitted 270/300 frames (90.0%) with gaps `[164,182)` and
  `[228,240)`; no post-initialization ID introduced.
- e2 / `ego-hmc21176623`:
  `runs/muggledsam-sam3-smoke-ego-hmc21176623-20260909t033109z/`.
  Succeeded, 300/300 frames / 10.0 seconds, 14.641 s elapsed, 4.327 s TTFU,
  1,965,729,280-byte peak allocated VRAM, 49 external masks. Initial result: `toy
  wheel` (`sam3-02`) only; `hand` did not initialize or emit. Wheel emitted 294/300
  frames with gap `[36,42)`; no post-initialization ID introduced.
- e3 / `ego-hmc21110305` preserved baseline:
  `runs/muggledsam-sam3-smoke-ego-hmc21110305-20260909t025910z/`.
  Read-only comparison evidence: 300/300 frames / 10.0 seconds, 14.056 s elapsed,
  4.588 s TTFU, 1,965,065,728-byte peak allocated VRAM, 50 external masks. `hand`
  (`sam3-00`) emitted 300/300 frames, but its previously documented visual quality
  remains poor; the other two concepts did not initialize.
- e4 / `ego-hmc21179183`:
  `runs/muggledsam-sam3-smoke-ego-hmc21179183-20260909t033125z/`.
  Succeeded, 300/300 frames / 10.0 seconds, 15.261 s elapsed, 4.319 s TTFU,
  1,965,320,704-byte peak allocated VRAM, 149 external masks. Initial results:
  `hand` (`sam3-00`), `yellow toy body` (`sam3-01`), and `toy wheel` (`sam3-02`).
  Hand and wheel each emitted 300/300 frames; yellow toy body emitted 294/300 with
  gap `[36,42)`; no post-initialization ID introduced.

Each new run has schema-valid `manifest.json`, normalized `observations.jsonl`,
`worker_result.json`, runtime settings, worker logs, external masks, bounded input video,
and `smoke.rrd`. The consolidated machine-readable evidence is
`runs/muggledsam-sam3-ego-viewpoint-screen-20260909t033053z/viewpoint_screen_report.json`.

## G3 review and bounded judgement

`runs/muggledsam-sam3-ego-viewpoint-screen-20260909t033053z/g3_review/
ego_viewpoint_screen_4view_contact_sheet.png` is a fixed four-view sheet at 0.000,
5.000, and 9.967 seconds. It overlays recorded model outputs only.

By visual hand association at those three samples—not ground truth—the tentative order is
e4, e1, e3, e2. E4's `hand` output visibly remains co-located with the working hand;
e1 initializes on a hand but has visible loss/drift; e3 remains the known poor baseline;
e2 has no `hand` output. This is not a detection, segmentation, association, or
cross-view accuracy result.

## Required human decision

Approve or reject one 60-second e4-only candidate under the unchanged condition and a
defined human review protocol. If rejected, choose either a labelled evaluation protocol
or to stop ego continuation. No aggregate ego claim or other longer ego run is authorized.

## Executed e4-only 60-second candidate

The human-approved e4 candidate completed as
`runs/muggledsam-sam3-g4-e4-candidate-ego-hmc21179183-20260909t033519z/`. It used the
existing e4 proxy only, original decoded BGR, the unchanged `hand`, `yellow toy body`,
and `toy wheel` text prompts, and the same continuous one-prompt/four-frame memory
policy. It processed exactly frames `[0,1800)` / 60.0 seconds; the remaining 120 seconds
were not run.

- State: succeeded; normalized run manifest and observations validate against schema
  1.0. Source, proxy, and config fingerprints are in `e4_candidate_report.json`.
- Runtime: 69.792 s elapsed; 4.297 s time to first usable output; 1,965,320,704-byte
  peak allocated VRAM; 887 external masks at no more than 5 FPS.
- Initialization: all three fixed concepts produced `sam3-00`/hand,
  `sam3-01`/yellow toy body, and `sam3-02`/toy wheel on frame 0.
- Continuity: hand emitted 1,791/1,800 frames with gaps `[1345,1353)` and
  `[1365,1366)`; yellow toy body 1,768/1,800 with five gaps; toy wheel 1,742/1,800 with
  ten gaps. No concept introduced a post-initialization ID; intentional resets are false.
- Export: `g4_e4_60_second_candidate.rrd` succeeded through the central exporter,
  logging the bounded input once, boxes at every analysis frame, and external mask
  references only.
- QA: `g3_review/ego-hmc21179183_g4_e4_candidate_qa.png` uses frames 0, 900, and 1799
  (0.000, 30.000, 59.967 seconds). It has three, two, and zero mask overlays at those
  samples due to the 5-FPS mask cadence; boxes remain rendered at all three frames.

This remains an e4-only monochrome, reviewable candidate rather than an accuracy or
general ego-performance result. G4 now requires one explicit choice: **accept** this
view-scoped baseline, **reject** it and stop/define labelled evaluation, or **run the
remaining 120 seconds** under the unchanged condition.
