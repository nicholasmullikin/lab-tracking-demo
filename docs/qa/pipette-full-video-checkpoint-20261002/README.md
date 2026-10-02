# Full-video pipette checkpoint

The revised prompt recipes do not solve identity drift over the complete recording.
Direct inspection found masks on paper, gloves, racks and different pipettes.
The per-second rates measure apparent tracker motion and cannot establish physical pipette speeds.

## Scope and protocol

Recording P03_03_01 has 8,492 native frames, raw 0–8491, in each of six cameras.
Its frame rate is 30000/1001 and its duration is 283.349733 seconds.
Inference processes every native frame; the export shows 284 samples per camera, once per second.
The final second contains ten frames.

This is an isolated blue-pipette propagation comparison, not the production detector and multi-object tracker.
All variants start at raw 0. Later camera-specific corrections occur at T2 raw 100,
T4 raw 200, T3 raw 300 and T1 raw 500. T5 and FPV have no later corrections.
The earlier control receives its masks at the same times as the revised variants.
There is no reset at raw 600. No further prompts are supplied after raw 500.
Input verification checks provenance and hashes; it does not certify segmentation quality.

SAM3.1 uses the existing video tracker's 504-pixel image side, one prompt slot and four frame slots.
Each distinct variant history has an independent one-object memory bank.
Identical schedules share propagation, giving 14 distinct histories and 118,888 native masks.
The six videos contribute 50,952 native camera frames.
Full-resolution output masks feed the existing axis, calibrated line and color/taper direction methods.
FPV camera poses use native raw frame indices.

## Whole-video consistency

The residual drops a camera, fits from the other cameras and measures error in the dropped camera.
It measures geometric consistency, not ground-truth segmentation or tip accuracy.

| Variant | Fit fraction | Directed fraction | Median paired change, px | Measured against |
|---|---:|---:|---:|---|
| Earlier selected | 0.9783 | 0.7989 | 0 | Its own residual cells |
| Model-top recipe | 0.9980 | 0.7865 | +0.0540 | 29,640 matching earlier camera/frame cells |
| Visual-best recipe | 0.9978 | 0.8132 | +0.0236 | 30,924 matching earlier camera/frame cells |

Positive paired change means worse consistency. These tiny median differences are essentially unchanged;
individual wins and losses are mixed. Medians of separate residual groups differ from the median
of paired differences. Visual-best uses reviewed candidate selection and is not an automatic policy.
Missing fits and differing camera support must be considered alongside residuals.

## Change per second

[per-second.csv](per-second.csv) has 852 rows: three variants in each of 284 one-second bins.
Each row includes valid sample counts, medians and 90th percentiles for translation and rotation.
Both native adjacent-frame rates and 30-frame-lag rates are retained.
The actual lag is 30/fps, about 1.001 seconds.
Position measures the fitted visible midpoint in cm/s. Axis rotation ignores vector sign flips.
Directed rotation requires resolved directions at both ends of the interval.
Missing measurements remain blank, not zero.

| Variant | Apparent translation, cm/s | Unoriented rotation, degrees/s | Directed rotation, degrees/s | Measured against |
|---|---:|---:|---:|---|
| Earlier selected | 9.80 | 18.08 | 24.47 | Median of valid per-second median 30-frame-lag rates |
| Model-top recipe | 9.16 | 13.62 | 17.96 | Same statistic |
| Visual-best recipe | 10.49 | 14.31 | 24.77 | Same statistic |

These are tracker estimates, not measured physical speeds. Occlusion, changing visible extent,
wrong object association and calibration error affect them. Correction flags identify native intervals
spanning mask resets; they include intervals without valid motion samples and do not filter rates.
The CSV also records changes from the preceding second in consistency and fit/direction coverage.

## Direct image review

[visual-review.json](visual-review.json) records seven later inspected seconds across all six cameras.
Together with the early review, I compared 78 original camera images and 234 variant overlays at 13 times.
The full-frame context at 244 seconds and the actual full Rerun screenshot at 120 seconds were also inspected.
This is sparse qualitative inspection, not a count of failure frequency.

At 120 seconds, T1 masks a paper sheet, T4 includes a nearby tool and T5/FPV select table pipettes
as the blue target is held. Later frames contain a held yellow-marked pipette and resting pipettes.
A held object is therefore not automatically the original blue target.
Cross-view identity must be established before combining observations into a world line.

Sparse reference agreement uses 21 reviewed approximate SAM drafts in the first 20 seconds.
All seed/correction images and references needing review are excluded for every variant.
Mean overlap is 0.212 for earlier masks, 0.193 for model-top and 0.208 for visual-best.
These references are not independent ground truth or validation of the rest of the recording.
See [draft-agreement-summary.json](draft-agreement-summary.json).

## Reproduce and inspect

From the repository root:

```bash
.venv/bin/python -m battle.pipette_video_run --workers 4
.venv/bin/python -m battle.pipette_video_review
.venv/bin/rerun runs/finebio-pipette-improvement-20260930/full-video-line-comparison/pipette-full-video.rrd runs/finebio-pipette-improvement-20260930/full-video-line-comparison/pipette-full-video.rbl
```

The standalone model worker uses the muggled_sam Python environment and cached checkpoint.
To export every frame from the saved results, run:

```bash
.venv/bin/python -m battle.pipette_video_review --every-frame
.venv/bin/rerun runs/finebio-pipette-improvement-20260930/full-video-line-comparison/pipette-full-video-native.rrd runs/finebio-pipette-improvement-20260930/full-video-line-comparison/pipette-full-video-native.rbl
```

The full native-frame recording is about 26 GB and was too memory-intensive in the viewer.
For the first 30 seconds at full frame rate, use this smaller recording:

```bash
.venv/bin/python -m battle.pipette_video_review --every-frame --frames 900
.venv/bin/rerun --memory-limit 8GB runs/finebio-pipette-improvement-20260930/full-video-line-comparison/pipette-full-video-native-raw0-899.rrd runs/finebio-pipette-improvement-20260930/full-video-line-comparison/pipette-full-video-native-raw0-899.rbl
```

This separate recording displays all native frames and retains the per-second measurement curves.
It leaves the sampled HTML review and recorded measurements intact.
The completion marker is `native-export-complete.json` in the local run directory.

[config.json](config.json) records inputs, prompt schedules and model paths.
The environment files record versions and source/input hashes. Raw inputs and masks remain local.
No full artifact archive was created for this run; this tracked checkpoint preserves measurements and notes.
Superseded shared-reset and 1280-pixel attempts are excluded from the final comparison.

All six complete markers, contiguous native records, mask names, geometry frames and second bins were checked.
[verification.json](verification.json) records the counts. Focused tests pass, including sign-invariant motion,
correction flags, reference exclusions and clipping extreme projected lines for display.
Clipping changes rendering only; raw geometry and measurements remain available.

Evidence: `runs/finebio-pipette-improvement-20260930/full-video-line-comparison/`.

Validation: 112 focused tests passed, two deselected; Ruff passed for changed pipeline and test files.
The 37 documentation checks also passed after the README and checkpoint updates.
