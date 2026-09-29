# Archived scripts (moved Sep 24 2026)

One-off helpers from the Sep 9–16 phases. I moved them here at the Assembly101 close so that
`scripts/` holds only what I still run. Each still runs from this directory with
`uv run python scripts/archive/<name>.py`. `docs/archive/method-ledger.md` cites the run roots
they read or wrote, and those roots stay on disk: none is in the Sep 24 archive list,
`docs/archive/qa/runs-archive-list-2026-09-24.md`, and I deleted none in the Sep 24 cleanup pass.

- `compare_frame_rate_arms.py`: the Sep 13 evening comparison of the 30 fps and 60 fps arms,
  keyed on source seconds. It read the three
  `runs/muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260913t22{3959,4508,4556}z`
  arms and wrote `runs/frame_rate_comparison.json`, which I copied to
  `docs/archive/frame-rate-comparison-2026-09-13.json`.
- `report_e4_candidate.py`: continuity measures (`e4_candidate_report.json`) for the Sep 9
  e4-only 60-second candidate, `runs/muggledsam-sam3-g4-e4-candidate-ego-hmc21179183-20260909t033519z`.
- `report_ego_viewpoint_screen.py`: `viewpoint_screen_report.json` for the Sep 9 four-view
  ego-viewpoint screen, `runs/muggledsam-sam3-ego-viewpoint-screen-20260909t033053z`.
- `render_overnight_v3_audits.py`: the raw, stabilized and fallback hand sheets and the mask and
  contact audit sheets under `runs/interaction-review-overnight-v3/audits`, from the WiLoR
  stabilized-v2-r3 and audited-source-state runs and the Sep 16 four-part focused C10379 run.
- `profile_view_route.py`: timing of the calibration browser's display-only view-aid route
  (`/api/view-filters/render`). It serves a throwaway workspace from a temporary directory and
  reads no run. The numbers it produced are the timing table in the lab notebook's rapid browser
  workspace section, `docs/archive/README-lab-notebook-2026-09-27.md`.
