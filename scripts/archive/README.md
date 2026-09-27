# Archived scripts (moved Sep 24, 2026)

One-off helpers from the Sep 9-16 phases, moved here at the Assembly101 close so `scripts/`
holds only what is still run. Each still runs from this directory with
`uv run python scripts/archive/<name>.py`; the run roots they read or wrote are cited by
`docs/archive/method-ledger.md` and are kept on disk (none is in the Sep 24 archive list,
`docs/archive/qa/runs-archive-list-2026-09-24.md`, and none was deleted in the Sep 24 cleanup pass).

- `compare_frame_rate_arms.py`: the 30 vs 60 fps arm comparison keyed on source seconds
  (Sep 13 evening); read the three `runs/muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260913t22{3959,4508,4556}z`
  arms and wrote `runs/frame_rate_comparison.json`, copied to
  `docs/archive/frame-rate-comparison-2026-09-13.json`.
- `report_e4_candidate.py`: continuity measures (`e4_candidate_report.json`) for the Sep 9
  e4-only 60-second candidate, `runs/muggledsam-sam3-g4-e4-candidate-ego-hmc21179183-20260909t033519z`.
- `report_ego_viewpoint_screen.py`: `viewpoint_screen_report.json` for the Sep 9 four-view
  ego-viewpoint screen, `runs/muggledsam-sam3-ego-viewpoint-screen-20260909t033053z`.
- `render_overnight_v3_audits.py`: the raw/stabilized/fallback and mask/contact audit sheets
  under `runs/interaction-review-overnight-v3/audits`, from the WiLoR stabilized-v2-r3 and
  audited-source-state runs and the Sep 16 four-part focused C10379 run.
- `profile_view_route.py`: timing of the calibration browser's display-only view-aid route
  (`/api/view-filters/render`); serves a throwaway workspace from a temporary directory and
  reads no run. The numbers it produced are the timing table in the README's calibration
  workspace section.
