# Pipette point-prompt mini study

The best tested default is a good box, two positives (body interior and exposed opaque shaft), and one targeted negative on a nearby confuser: three points total, plus the box. This is a preliminary recipe for these views, not a universal optimum.

## Practical guidance

1. Check the box first. Include the visible pipette and exclude neighbouring pipettes where possible. A good box alone worked well on the clear resting controls.
2. Start with one positive inside the clearly visible body. Avoid the silhouette boundary, glove, uncertain transparent tip, and occluded parts.
3. Add a positive near the centre of an exposed shaft segment if the shaft is missing. Do not place points on the support tube or infer hidden shaft pixels.
4. Place a negative inside an unwanted object actually included by the mask, such as the glove, rack, or neighbouring pipette. Prefer the interior of that error region over an arbitrary background point.
5. Add a cap/hook point only when it is clearly visible and missing. Cap-only prompting was weaker here, particularly in a small occluded cap view.
6. Inspect all candidates if practical. Stop once the visible object is represented well; blindly adding five or eight positives often made masks worse.

## Results

“Mostly correct” means visual grade 4 or 5; grade 4 can still include small errors or an uncertain tip and does not mean production-ready.

| Condition | Positives / negatives | Mostly correct, model top / 7 | Mostly correct, visual best / 7 |
|---|---:|---:|---:|
| box-only | 0 / 0 | 2 | 4 |
| p1-body | 1 / 0 | 3 | 6 |
| p1-shaft | 1 / 0 | 3 | 5 |
| p1-cap | 1 / 0 | 2 | 2 |
| p2-body-shaft | 2 / 0 | 4 | 5 |
| p3-parts | 3 / 0 | 3 | 4 |
| p5-spread | 5 / 0 | 2 | 3 |
| p8-spread | 8 / 0 | 2 | 2 |
| p3-n1 | 3 / 1 | 3 | 6 |
| p3-n3 | 3 / 3 | 4 | 6 |
| p5-n3 | 5 / 3 | 3 | 4 |
| p8-n3 | 8 / 3 | 2 | 3 |
| p3-far3 | 3 / 3 | 2 | 2 |
| points-only-p3 | 3 / 0 | 2 | 3 |
| points-only-p3-n3 | 3 / 3 | 3 | 5 |
| p3-wrong-positive | 4 / 0 | 1 | 2 |
| p1-body-n1 | 1 / 1 | 5 | 6 |
| p2-body-shaft-n1 | 2 / 1 | 6 | 6 |
| p2-body-shaft-n3 | 2 / 3 | 4 | 6 |

The two-positive/one-negative recipe gave a mostly correct model-top mask in six of seven views. Body-only gave three of seven model-top masks, despite six of seven views having a better available candidate. This candidate-ranking gap matters: prompt quality should be judged on the mask the workflow actually chooses, not only the best mask found after manual inspection.

One body positive plus one negative also performed well (five of seven model-top masks). Adding three negatives to the body/shaft pair gave four of seven, versus six with one negative. More negatives are therefore not automatically better; use them to correct specific remaining errors.

Remaining problems: frame 100 T2 still has rack/shaft ambiguity; frame 500 T1 still has support-tube contamination; the crowded frame 0 T1 has neighbouring-pipette flecks. Transparent distal-tip boundaries remain uncertain. None of these findings proves downstream tracking accuracy.

## Method and limitations

Seven hard-biased views from one recording and one blue pipette: frames 0 (T2, T1, FPV), 100 (T2), 200 (T4), 300 (T3), and 500 (T1). The samples are correlated and do not cover every camera, pipette colour, or pose. No held-out test set or human ground-truth masks were used.

Local SAM3.1 multiplex checkpoint; original full-frame images encoded at maximum side length 1280; the same box was held fixed within each view except explicit points-only conditions. All points were manually chosen on enlarged originals. Each of 19 conditions was decoded independently with all its points supplied simultaneously. No previous mask logits were fed back. Display crops are for review only, not inference. There were 133 decodes and four masks per decode, totalling 532 directly reviewed candidates.

A follow-up of three simpler recipes was added after inspecting the original 16 conditions. This adaptive design makes the winner exploratory; validate it on unseen views before adopting a fixed policy. The extra five/eight-positive conditions densify visible parts and are not an experiment in eight sequential corrective clicks.

Grades: 1 wrong/failed; 2 major non-target inclusion; 3 notable leakage/missing part; 4 mostly correct with small errors or uncertain tip; 5 clean approximate visible pixels. One AI reviewer inspected all candidate sheets against the originals. The review was not blinded. Model-top means highest estimated IoU score, not measured IoU. Best-of-four is an optimistic manual-selection comparison.

All 532 masks were checked for native image dimensions and binary values. Full-frame inspection of the winning recipe found no mask pixels outside the display crops. A separate outside-crop audit is in validation.json; other conditions contain small remote islands, largest 380 pixels. Production labels and tracking settings were not changed.

## General SAM guidance

[The original SAM paper](https://arxiv.org/html/2304.02643v1) uses interior first clicks and corrective clicks inside false-positive or false-negative regions. Its reported diminishing returns around eight iteratively sampled points includes previous-mask feedback; that does not establish eight simultaneous positive clicks as an optimal recipe for these pipettes. The body/shaft/negative recommendation is our inference from this local comparison.

## Review and reproduce

- Open review.html to switch view and prompt, compare all four candidates with the original, and inspect full-resolution binary masks.
- results-chart.png / .svg: aggregate counts; case-grades.png: every case and condition.
- design.py and design.json: point locations and fixed conditions.
- run.py: inference; rate.py: explicit visual judgments; report.py: these deliverables.
- manifest.json records source hashes, prompts, candidate scores and model selection; ratings.csv contains all 133 ratings.

These are study outputs only. No masks from this experiment have been promoted into the existing annotation set.
