# Documentation index

Five live pages tell the story of this lab for a reader with five minutes; everything historical
sits verbatim under `archive/`. Start with the root README, then the day-by-day story, then the
numbers. The glossary that maps the lab's shorthand to plain words is in `writing-style.md`.

## Live pages

| Page | What it tells you |
|---|---|
| [`../README.md`](../README.md) | What the lab built in 22 days, the five numbers that matter, what is real and what is not, and how to set up |
| [`story.md`](story.md) | One section per active day, each with a GIF, a headline and the number that changed |
| [`results.md`](results.md) | The goals scorecard in the founder's words, the two-trial FineBio table, the tip-vote, reprojection and stricter association readouts, the numbers anchored on the masks I chose, the Assembly101 headline, the claim boundaries |
| [`pipeline.md`](pipeline.md) | Setup, the FineBio pipeline one command per stage, the two gates, the recordings, the Assembly101 review package, tests, the GPU queue, pruning |
| [`review-guide-2026-09-25-finebio-3d.md`](review-guide-2026-09-25-finebio-3d.md) | The 20-minute route through the FineBio recordings: presets, storyboard, cross-checks, the numbers |
| [`writing-style.md`](writing-style.md) | The ten rules the live pages follow, the house style taken from The Economist Style Guide, the [glossary](writing-style.md#glossary) and one before-and-after |
| [`LICENSES.md`](LICENSES.md) | The license and handling policy for every input, and the terms the committed story media carry |
| [`SOURCES.md`](SOURCES.md) | The source and provenance record for every dataset, model and external repository used |
| [`qa/README.md`](qa/README.md) | The review records: gate decisions, review anchors and QA files, tracked as numbers and hashes without pixels |

## Checkpoints

- [Pipette tracking, 2026-10-01](qa/pipette-checkpoint-20261001/README.md): annotation state, point-prompt study, saved artifacts and resumption instructions.

## Archive

Every historical document was moved to [`archive/`](archive/README.md) on one day, prose untouched,
with an index that names each file's era and what replaced it. The eras are **Assembly101** (the
toy-car phase), **FineBio** (the wet-lab phase) and **cross-cutting** (records that span both).

| File | Era | What it was |
|---|---|---|
| [`archive/method-ledger.md`](archive/method-ledger.md) | cross-cutting | The append-only lab record: the original ask, the goals scorecard, a dated timeline and one detailed record per run |
| [`archive/README-lab-notebook-2026-09-27.md`](archive/README-lab-notebook-2026-09-27.md) | cross-cutting | The root README as it stood before the cleanup: a status block and a dated how-to for every arm ever run |
| [`archive/methods-inventory-2026-09.md`](archive/methods-inventory-2026-09.md) | cross-cutting | A hand-kept snapshot of every candidate method and its environment |
| [`archive/plan-2026-09-08-assembly-rerun-lab.md`](archive/plan-2026-09-08-assembly-rerun-lab.md) | Assembly101 | The operating plan: scope, the five human gates, the method matrix, the pre-accuracy measures |
| [`archive/run-report-assembly101-ego-viewpoint-screen.md`](archive/run-report-assembly101-ego-viewpoint-screen.md) | Assembly101 | The screen of the four head cameras that picked e4 |
| [`archive/vigra-build.md`](archive/vigra-build.md) | Assembly101 | How VIGRA was built for the calibration browser's display-only edge and corner aids |
| [`archive/review-guide-2026-09-18-multicam.md`](archive/review-guide-2026-09-18-multicam.md) | Assembly101 | The guide to the overnight eight-camera pass: consensus and visual hull on the v4 recording |
| [`archive/review-guide-2026-09-20-ensemble-v2.md`](archive/review-guide-2026-09-20-ensemble-v2.md) | Assembly101 | The guide to the ensemble reference v2 candidate on the v5 recording |
| [`archive/review-guide-2026-09-20-multiview-presets.md`](archive/review-guide-2026-09-20-multiview-presets.md) | Assembly101 | The guide to the v6 package: one recording, three presets, nine camera tiles |
| [`archive/labeling-sessions-2026-09-20.md`](archive/labeling-sessions-2026-09-20.md) | Assembly101 | The briefs for the five anchor-labeling sessions on the Assembly101 cameras |
| [`archive/athena_hf_calibration_probe.json`](archive/athena_hf_calibration_probe.json) | Assembly101 | The HTTP range probe of Assembly101's pose archive: no intrinsics ship in it |
| [`archive/frame-rate-comparison-2026-09-13.json`](archive/frame-rate-comparison-2026-09-13.json) | Assembly101 | The comparison of 30 fps against 60 fps SAM3 arms on the head camera |
| [`archive/qa/anchor-scoreboard-c10119-20260920.md`](archive/qa/anchor-scoreboard-c10119-20260920.md) | Assembly101 | Arm ranking on the top-down camera C10119, first labeling |
| [`archive/qa/anchor-scoreboard-c10119-20260921.md`](archive/qa/anchor-scoreboard-c10119-20260921.md) | Assembly101 | The same camera after the four-part rerun and the guarded re-prompt |
| [`archive/qa/anchor-scoreboard-c10379-arms-20260921.md`](archive/qa/anchor-scoreboard-c10379-arms-20260921.md) | Assembly101 | Arm ranking on the static camera C10379: `pm-append` 0.743 and the DAM4SAM tie |
| [`archive/qa/anchor-scoreboard-e4-20260920.md`](archive/qa/anchor-scoreboard-e4-20260920.md) | Assembly101 | Arm ranking on the head camera e4, first labeling |
| [`archive/qa/anchor-scoreboard-e4-20260921.md`](archive/qa/anchor-scoreboard-e4-20260921.md) | Assembly101 | The same camera after the four-part rerun |
| [`archive/qa/runs-archive-list-2026-09-24.md`](archive/qa/runs-archive-list-2026-09-24.md) | Assembly101 | The listing of `runs/` at the phase close: sizes, what was archived, what was deleted |
| [`archive/plan-2026-09-24-finebio-detector-seeded-lab.md`](archive/plan-2026-09-24-finebio-detector-seeded-lab.md) | FineBio | The first FineBio plan, detector-seeded and objects only, written before the preflight |
| [`archive/preflight-2026-09-24-finebio.md`](archive/preflight-2026-09-24-finebio.md) | FineBio | The check of the plan's assumptions before anything ran: cameras, the transparent plate, rig accuracy, trials |
| [`archive/plan-2026-09-25-finebio-3d-tracking.md`](archive/plan-2026-09-25-finebio-3d-tracking.md) | FineBio | The FineBio 3D object tracking plan with its todo table and Outcome section |
| [`archive/labeling-sessions-2026-09-25-finebio.md`](archive/labeling-sessions-2026-09-25-finebio.md) | FineBio | The briefs for gate 1 (seeds) and gate 2 (anchors) on trial 1, both held |
