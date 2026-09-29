# Archive

This directory holds every historical document of the lab, moved here verbatim on Sep 27 2026.
Each entry below names the file's era, what it was for and which live page replaced it. Read the live
pages first (`README.md`, `docs/story.md`, `docs/results.md`, `docs/pipeline.md`) and come here for the
full record behind a number.

The era tags are **Assembly101** (the toy-car assembly phase, Sep 8 to 24), **FineBio** (the wet-lab
bench phase, Sep 21 to 27) and **cross-cutting** (records that span both).

## Files

| File | Era | What it was | Superseded by |
|---|---|---|---|
| `method-ledger.md` | cross-cutting | The append-only lab record: the original ask, the goals scorecard, a dated timeline and one detailed record per run with its claim boundary. Closed with a banner on Sep 27. | `docs/story.md` and `docs/results.md` |
| `README-lab-notebook-2026-09-27.md` | cross-cutting | The root README as it stood on Sep 27: a short status block followed by a dated how-to for every arm ever run. | the root `README.md` and `docs/pipeline.md` |
| `methods-inventory-2026-09.md` | cross-cutting | A hand-kept snapshot of every candidate method and its environment, once `configs/methods.yaml`, never read by code. | `docs/SOURCES.md` and `docs/results.md` |
| `plan-2026-09-08-assembly-rerun-lab.md` | Assembly101 | The operating plan for the Assembly101 comparison lab: scope, the five human gates, the method matrix and the pre-accuracy measures. | `plan-2026-09-24-finebio-detector-seeded-lab.md` here; its scorecard now sits in `docs/results.md` |
| `run-report-assembly101-ego-viewpoint-screen.md` | Assembly101 | The Sep 9 screen of the four head cameras that picked e4 for every later ego run. | the Sep 9 entry of `docs/story.md` |
| `vigra-build.md` | Assembly101 | How VIGRA 1.12.4 was built into the project environment for the calibration browser's display-only edge and corner aids. Still the only build recipe; `src/battle/calibration_view_filters.py` points here. | nothing; the tool it serves belongs to the Assembly101 phase |
| `review-guide-2026-09-18-multicam.md` | Assembly101 | The guide to the Sep 18 overnight pass: eight static cameras, consensus and visual hull, on the v4 recording. | `review-guide-2026-09-20-multiview-presets.md` here, then `docs/pipeline.md` |
| `review-guide-2026-09-20-ensemble-v2.md` | Assembly101 | The guide to the ensemble reference v2 candidate on the v5 recording. | `review-guide-2026-09-20-multiview-presets.md` here; the candidate's disposition is in the ledger's Sep 24 close |
| `review-guide-2026-09-20-multiview-presets.md` | Assembly101 | The guide to the v6 package: one recording, three presets (segmentation, hands, multiview), nine camera tiles and the anchors drawn on their frames. | the Assembly101 review section of `docs/pipeline.md` |
| `labeling-sessions-2026-09-20.md` | Assembly101 | The briefs for the five anchor-labeling sessions on the Assembly101 cameras and the outcome of each. | the scoreboards under `qa/` here and `docs/results.md` |
| `athena_hf_calibration_probe.json` | Assembly101 | The Sep 16 HTTP Range probe of Assembly101's `AssemblyPoses.zip` on Hugging Face, written by `battle-athena-calibration-probe`: the member inventory and the finding that no intrinsics ship in it. | the ATHENA entry of `docs/SOURCES.md` |
| `frame-rate-comparison-2026-09-13.json` | Assembly101 | The Sep 13 comparison of 30 fps against 60 fps SAM3 arms on the head camera, keyed on source seconds, written by `scripts/archive/compare_frame_rate_arms.py`. | the Sep 13 entry of `docs/story.md` |
| `qa/anchor-scoreboard-c10119-20260920.md` | Assembly101 | Arm ranking against the anchors on the top-down camera C10119, 26 frames, first labeling. | the Assembly101 scoreboard in `docs/results.md` |
| `qa/anchor-scoreboard-c10119-20260921.md` | Assembly101 | The same camera after the four-part rerun and the guarded re-prompt. | the Assembly101 scoreboard in `docs/results.md` |
| `qa/anchor-scoreboard-c10379-arms-20260921.md` | Assembly101 | Arm ranking on the static camera C10379, 13 frames: `pm-append` 0.743 and the DAM4SAM tie. | the Assembly101 scoreboard in `docs/results.md` |
| `qa/anchor-scoreboard-e4-20260920.md` | Assembly101 | Arm ranking on the head camera e4, 26 frames, first labeling. | the Assembly101 scoreboard in `docs/results.md` |
| `qa/anchor-scoreboard-e4-20260921.md` | Assembly101 | The same camera after the four-part rerun. | the Assembly101 scoreboard in `docs/results.md` |
| `qa/runs-archive-list-2026-09-24.md` | Assembly101 | The Sep 24 listing of `runs/` at the Assembly101 close: every run's size, which were archived and which were deleted. | nothing; it is the record of that deletion. `docs/pipeline.md` describes pruning |
| `plan-2026-09-24-finebio-detector-seeded-lab.md` | FineBio | The first FineBio plan, detector-seeded and objects only, written before the preflight. | `plan-2026-09-25-finebio-3d-tracking.md` here, one day later |
| `preflight-2026-09-24-finebio.md` | FineBio | The check of the plan's assumptions before anything ran: camera mapping, the transparent plate, rig accuracy and the choice of trials. | its numbers live on in `tests/fixtures/finebio_preflight/` and `configs/finebio/`; the summary is in `docs/results.md` |
| `plan-2026-09-25-finebio-3d-tracking.md` | FineBio | The FineBio 3D object tracking plan with its todo table and per-phase status, complete. | `docs/results.md` for the numbers and `docs/pipeline.md` for the commands |
| `labeling-sessions-2026-09-25-finebio.md` | FineBio | The briefs for gate 1 (seeds) and gate 2 (anchors) on trial 1, both held. | the records under `docs/qa/` and their notes in `docs/qa/README.md` |

Still live beside this directory: `docs/review-guide-2026-09-25-finebio-3d.md` (the 20-minute viewer route),
`docs/LICENSES.md`, `docs/SOURCES.md` and the gate records under `docs/qa/`.

## Reading the archive

The prose is as written. The one edit is the three-line banner at the top of `method-ledger.md`.

Paths written as `docs/<file>` inside these pages name the location before the move. The file now sits in this
directory under the same name.

Relative links inside `method-ledger.md` were written from `docs/` and were not touched. Its links into
`qa/*.json`, `qa/finebio-trials-2026-09-25.md`, `SOURCES.md`, `review-guide-2026-09-25-finebio-3d.md` and
`../scripts/archive/README.md` are therefore one directory short; add one `../`. The other archived pages had
their links to live files fixed.

`configs/finebio/trials.json` still names `docs/plan-2026-09-25-finebio-3d-tracking.md` and
`docs/preflight-2026-09-24-finebio.md` in its provenance strings. The three committed clip configs under
`configs/clips/finebio_*.json` pin that file's SHA-256 (`trials_source.sha256`), so editing it would break the
provenance chain. Read those two paths as pointing here.
