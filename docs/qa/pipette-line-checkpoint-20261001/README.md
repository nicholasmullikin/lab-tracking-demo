# Pipette camera and 3D line comparison

The three-click recipe's model-top masks do not improve overall 3D line consistency on these samples. The resting pipette is strong; held masks still cause axis bias and endpoint overshoot. Manual candidate selection helps one difficult frame, but does not resolve physical tip placement.

| Mask set | 3D fits / 7 | Directed / 7 | Matched median dropped-camera residual, px | Median angle, degrees |
|---|---:|---:|---:|---:|
| earlier-selected | 7 | 5 | 6.62 | 0.90 |
| recipe-model-top | 7 | 5 | 6.99 | 0.90 |
| recipe-visual-best | 7 | 5 | 5.98 | 0.84 |

All sets share 39 evaluable dropped-camera cells out of 42 possible camera/time cells. Each set has 39 local axes. The exact no-axis views are 100/T4, 400/T4, 500/T4. These compact views still contribute centroid rays to the fit. Unavailable dropped-camera fits are retained explicitly in coverage metadata.

Model-top substitutions improve four matched cells by more than 0.1px, worsen 13, and leave 22 within 0.1px. Visually selected substitutions improve six, worsen eight, and leave 25 within 0.1px. The median paired changes are about 0.008px and 0.000px respectively. These are descriptive differences; the 0.1px band is a reporting tolerance, not a statistical test. The aggregate median residual difference is not the median paired change.

At raw frame 200, the per-frame median residual goes from 5.91px to 15.12px with the recipe's top candidate and to 4.13px with its visually selected candidate. The top T4 candidate's axis follows rack/background influence. The fitted visible extent remains roughly 34-35cm across all variants, exceeding the visually clear opaque shaft. This does not establish the true tip length.

At frames 0, 200, 300, 400 and 500, the rendered arrows point toward the visibly identified dispensing side in direct review. At 100 and 600 the colour and taper cues disagree, so the review abstains. Endpoint indices can reverse when the fitter's arbitrary direction sign reverses; an index change alone is not a physical tip flip. This sparse colour/taper policy is not the full adopted temporal orientation vote and is not an accuracy estimate.

## Inputs and method

Blue pipette only, raw frames 0 through 600 every 100, all six cameras. Earlier selected masks are the five-pass every100 review. The two recipe variants replace only seven study images: frame 0 T1/T2/FPV, 100 T2, 200 T4, 300 T3, 500 T1. This is a partial-substitution comparison, not a full-recipe six-camera run. No segmentation inference was run.

The existing mask-axis estimator and 3D plane/ray fitter are unchanged. Disconnected components are preserved. All earlier masks, including needs-review masks, participate under the same policy. Fits use original pixel dimensions, raw-frame camera poses and the full-recording camera config. No clip-offset subtraction is used for FPV. A missing FPV pose excludes that view from 3D but retains its image. No missing poses occurred in these seven samples.

Visible extents use mask terminal centroids without a length prior, length completion or clamping. Direction uses the strongest camera per existing colour/taper cue, with camera-to-world endpoint correspondence. Positive cues must agree and their summed log odds must exceed the existing threshold. No hand cue, gravity cue or episode accumulation is used.

The input camera config is configs/finebio/cameras/P03_03_01.json. Its fixed-view marker residuals range from 0.72px to 6.71px. Cross-view offsets cannot all be assigned to segmentation or calibration from this comparison. No camera solve was performed. All source-image/mask and camera-config hashes, cue evidence, fit conditioning, local axes, world endpoints and dropped-camera failures are preserved in results.json.

## Direct review and limitations

All seven six-camera before/after sheets were directly inspected against originals, covering 126 variant overlays. Separate notes for every view are in visual-review.json. The actual Rerun rendering was also inspected. Yellow is the local axis, green the projected 3D visible extent, white the world endpoint indices, magenta the dispensing arrow, and cyan the mask. Crops are for display only and preserve aspect ratio; full-frame images remain available.

Review found rack spill, support-tube contamination, body/shaft offsets and uncertain transparent or occluded endpoints. No independent reference masks, human shaft clicks or physical tip labels were created. Residuals compare reconstructed lines with the input masks' own axes. Lower residuals can reflect shared errors and are not ground-truth 3D accuracy. The seven correlated times do not test unseen frames or temporal tracking.

## Open and reproduce

Open review.html to switch frames, or run:

```bash
.venv/bin/rerun runs/finebio-pipette-improvement-20260930/line-comparison/pipette-lines.rrd runs/finebio-pipette-improvement-20260930/line-comparison/pipette-lines.rbl
```

Build the comparison from the saved masks with `.venv/bin/python -m battle.pipette_line_review`, then run this directory's summarize.py with the project interpreter to regenerate the explicit review notes and chart. The review driver is versioned; images and recordings remain local under runs/.

## Next work

Do not adopt the model-top recipe for lines yet. First diagnose candidate selection using local shaft alignment and cross-view disagreement; the frame-200 T4 failure is the clearest example. Refine rack/tube masks only where they damage the axis or extent. Keep physical tip localization separate from shaft fitting. Validate on unseen frames before changing temporal tracking. Production masks, tracking policy and the raw-frame-600 start remain unchanged.

## Checkpoint and continuation

Implementation: `src/battle/pipette_line_review.py`; focused tests: `tests/test_pipette_line_review.py`. This checkpoint records the completed first-pass comparison. Production behavior is unchanged. The input checkpoint is `docs/qa/pipette-checkpoint-20261001/README.md`.

Full local outputs are in `/home/nick/src/battle/runs/finebio-pipette-improvement-20260930/line-comparison/`. Restore the saved input archive from the earlier checkpoint before rebuilding if those directories are absent. This checkpoint's summarize.py is a snapshot: copy it into the line-comparison run directory before running it.

The new output archive is `/home/nick/src/battle/runs/checkpoints/pipette-lines-20261001/pipette-line-comparison.tar.gz`. It contains final recordings, images, provenance, review notes and the report. Its checksum and per-file verification are in archive-verification.json. Original raw videos, model weights and environments remain outside the archive. These are local archives on this machine, not off-machine backups.

Run `.venv/bin/python -m pytest tests/test_pipette_line_review.py tests/test_finebio_observations.py tests/test_multiview_lines.py tests/test_finebio_orientation.py -q` for the comparison's checks. The completed run passed 64 tests; two real-data tests were deselected by the project's default test policy. Rerun screenshots at raw frames 0 and 200 and viewer-state evidence are kept locally.
