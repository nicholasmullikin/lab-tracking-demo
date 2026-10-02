"""Save explicit visual review and render a chart from the completed comparison."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from battle.pipette_line_review import VARIANTS,comparison_summary
R=Path(__file__).resolve().parent;r=json.loads((R/'results.json').read_text());s=comparison_summary(r)
(R/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
# One note per original view, authored after direct inspection of every before/after sheet.
notes={
'0':[
'Shaft axis follows resting blue pipette; small neighbouring mask flecks remain.',
'Body and shaft alignment strong; dispensing end is upper left.',
'Shaft axis and direction clear across all variants.',
'Long axis follows shaft; modest lateral offset is visible.',
'Axis and dispensing direction clear; endpoint is approximate.',
'Thin shaft axis follows target; button-side endpoint is slightly offset.'
],
'100':[
'Short local axis mainly describes exposed body; 3D projection follows shaft but plunger end extends above visible button.',
'Disconnected head/shaft mask gives a diagonal axis; rack spill remains at shaft endpoint.',
'Short local axis mainly describes exposed body; fused line follows shaft with endpoint uncertainty.',
'End-on body mask compact, no shaft axis; fused projection alone is not proof of a correct local fit.',
'Local shaft axis usable; fused extent reaches beyond visible plunger-side mask.',
'Local and projected axes mostly align; distal transparent boundary and endpoints uncertain.'
],
'200':[
'Axis follows shaft, but reconstructed dispensing-side extent visibly extends beyond opaque shaft.',
'Body/shaft axis approximately aligned; tip extent extends toward/off crop boundary.',
'Axis follows shaft; long reconstructed extent overshoots visible shaft and may include non-pipette pixels.',
'Model-top substituted mask rotates local axis toward rack/background; visually selected mask follows shaft more closely.',
'Axis follows body; projected extent exceeds visible shaft end.',
'Long shaft line visible; endpoint overshoot and model-top cross-view shift are apparent.'
],
'300':[
'Local axis mostly follows shaft; transparent distal tip remains uncertain.',
'Axis mostly follows shaft; lower end meets the other hand/tube and has an uncertain boundary.',
'New mask leaves very similar axis; dispensing-end precision is still uncertain.',
'Local mask axis biased by head/glove; projected line displaced near plunger-side body.',
'Local axis shortens visible shaft; projected line displaced near the body/plunger side.',
'Long shaft axis mostly follows target; distal boundary uncertain and body-side projection displaced.'
],
'400':[
'Axis roughly follows visible shaft; tube/hand contact makes distal endpoint uncertain.',
'Axis follows shaft but includes tube-contact region; projected body-side endpoint displaced.',
'Axis mostly follows shaft; distal tip/tube boundary remains unresolved.',
'Compact body mask has no local shaft axis; projected line displaced relative to foreshortened shaft.',
'Local shaft axis available; projected plunger-side line is displaced.',
'Local axis follows shaft; projected distal direction mostly follows shaft, body-side fit offset.'
],
'500':[
'Two-positive/one-negative substitution changes axis very little; support-tube contamination remains.',
'Shaft axis approximate; other hand/tube makes dispensing endpoint unassessable.',
'Local shaft axis mostly aligned; dispensing endpoint at tube boundary uncertain.',
'Compact body mask has no local axis; foreshortening limits review of projected direction.',
'Local axis follows shaft; projected plunger-side location offset.',
'Local axis follows shaft; projected dispensing-side extent drifts sideways near tube contact.'
],
'600':[
'Local and fused axes mostly aligned; distal endpoint at tube/hand boundary uncertain.',
'Axis roughly follows shaft; body-side projection shifted and tube contact contaminates endpoint.',
'Shaft axis mostly follows target; distal tip boundary unresolved.',
'Local/fused axes available but endpoints displaced relative to button/body.',
'Axis follows exposed shaft; reconstructed body/plunger endpoint offset.',
'Local shaft axis usable; projected dispensing-side end offset sideways.'
]}
views=['T1','T2','T3','T4','T5','fpv'];review={'reviewer':'single AI visual reviewer','blinded':False,'ground_truth':False,'method':'Directly inspected all seven six-camera before/after crop sheets (126 overlays) against originals, plus full-frame and Rerun views. No human reference points were created.','frames':{}}
for f,ns in notes.items():
 entry={'views':{},'direction':{}}
 for view,note in zip(views,ns):
  entry['views'][view]={'note':note,'variants_inspected':list(VARIANTS),'endpoint_accuracy':'not numerically scored; transparent/occluded boundaries unassessable'}
 for var in VARIANTS:
  resolved=r['variants'][var][f]['direction']['tip_end'] is not None
  entry['direction'][var]={'assessment':'points toward visible dispensing side' if resolved else 'abstained', 'physical_tip_location_accuracy':'not established'}
 review['frames'][f]=entry
review['failure_classes']={'segmentation':['100/T2 rack spill','200/T4 top-candidate rack/background influence','500/T1 support tube'], 'axis_fitting':['200/T4 sensitivity to local contamination','300/T4 biased local axis'], 'direction':['100 and 600 colour/taper conflict; abstention'], 'extent':['200 excessive reconstructed visible extent','300-600 transparent tip and tube-contact ambiguity'], 'camera_geometry':['Cross-view offsets remain in T4/T5/FPV; config reports fixed-view marker residuals about 0.7-6.7px; no new camera solve was performed.']}
(R/'visual-review.json').write_text(json.dumps(review,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(10,5));frames=list(notes)
for var,label in zip(VARIANTS,['Earlier selected','Recipe model top','Recipe visual best (optimistic)']):
 values=[]
 for f in frames:
  values.append(np.median([z['perpendicular_px'] for z in r['variants'][var][f]['loo'] if z['fitted'] and z['perpendicular_px'] is not None]))
 ax.plot([int(f) for f in frames],values,'o-',label=label)
ax.set_xlabel('Raw source frame (sparse samples)');ax.set_ylabel('Median dropped-camera axis residual (pixels)');ax.set_title('Pipette line consistency: candidate selection matters\nFive times have mask substitutions; frames 400 and 600 are unchanged');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(R/'line-residuals.png',dpi=180);fig.savefig(R/'line-residuals.svg');plt.close(fig)
report='''# Pipette camera and 3D line comparison

The three-click recipe's model-top masks do not improve overall 3D line consistency on these samples. The resting pipette is strong; held masks still cause axis bias and endpoint overshoot. Manual candidate selection helps one difficult frame, but does not resolve physical tip placement.

| Mask set | 3D fits / 7 | Directed / 7 | Matched median dropped-camera residual, px | Median angle, degrees |
|---|---:|---:|---:|---:|
'''
for row in s:report+=f"| {row['variant']} | {row['fits']} | {row['resolved']} | {row['paired_median_px']:.2f} | {row['paired_median_angle_deg']:.2f} |\n"
report+='''
All sets share 39 evaluable dropped-camera cells out of 42 possible camera/time cells. Each set has 39 local axes; compact masks have no local shaft axis at frame 100 T1/T3/T4 and frames 400/500 T4 (there are five compact views, but two are still elongated enough? See exact axis metadata below). Only the exact metadata should determine coverage; a compact view can contribute a centroid ray without contributing an axis residual.

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
'''
# Exact no-axis list, avoiding conflation with masks which still have a usable axis.
missing=[f"{f}/{v}" for f,x in r['variants'][VARIANTS[0]].items() for v,row in x['views'].items() if row['axis']['axis_px'] is None]
start=report.index('All sets share');end=report.index('\n\nModel-top',start)
report=report[:start]+f"All sets share 39 evaluable dropped-camera cells out of 42 possible camera/time cells. Each set has 39 local axes. The exact no-axis views are {', '.join(missing)}. These compact views still contribute centroid rays to the fit. Unavailable dropped-camera fits are retained explicitly in coverage metadata."+report[end:]
(R/'README.md').write_text(report)
print('Saved explicit review of 42 originals and 126 overlays, summary and chart')
