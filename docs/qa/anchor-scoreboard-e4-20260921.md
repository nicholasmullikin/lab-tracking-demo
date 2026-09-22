# Anchor scoreboard: e4 (HMC_21179183), 26 frames (52 labelled / 23 hidden / 29 skipped), Sep 21 four-part rerun

Copied verbatim from `runs/multiview-reprompt-20260921/anchor_iou_e4.md` (generated 2026-09-22T03:42:40.516531Z by
`battle-anchor-iou --view ego-hmc21179183`; `runs/` is not tracked). Anchor record:
`docs/qa/first-minute-review-anchors-ego-hmc21179183.human-record.json`; counts {'labeled': 52, 'hidden': 23, 'unlabeled': 29}.
Review evidence, not ground truth: one person's choice of SAM3 image-decoder mask per visible
part on a handful of frames of one camera, used to rank arms against each other; not a dataset
and no accuracy claim. A run without a slot for a part scores 0 on that part's labelled cells
(`missing`), so `IoU all` is comparable across rows. CC BY-NC 4.0 applies to the frames.

Runs scored:

- `r1280`: `runs/sam3-views-r1280-20260919/views/HMC_21179183/muggledsam-sam3-four-part-multiview-first-minute-ego-hmc21179183-20260920t023927z-r1280`
- `seeded-3part`: `runs/sam3-views-r1280-pmappend-seeded-20260920/views/HMC_21179183/muggledsam-sam3-four-part-multiview-first-minute-ego-hmc21179183-20260920t155813z-r1280-pm-append`
- `human-accepted-4part`: `runs/sam3-views-r1280-4part-20260921/views/HMC_21179183/muggledsam-sam3-four-part-multiview-first-minute-ego-hmc21179183-20260922t021759z-r1280-pm-append`

| arm | IoU chassis | IoU interior | IoU rear_body | IoU cabin | IoU all | IoU 279-408 | IoU 573-722 | IoU 1020-1172 | IoU outside | missing | hidden FP (px) | scored / unlabeled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r1280 | 0.845 | 0.000 | 0.262 | 0.831 | 0.558 | 0.633 | 0.538 | 0.525 | 0.558 | 10 | 8 (48705) | 42 / 29 |
| seeded-3part | 0.842 | 0.000 | 0.262 | 0.940 | 0.567 | 0.637 | 0.541 | 0.535 | 0.568 | 10 | 8 (20083) | 42 / 29 |
| human-accepted-4part | 0.851 | 0.000 | 0.349 | 0.938 | 0.589 | 0.644 | 0.542 | 0.542 | 0.597 | 10 | 7 (16474) | 42 / 29 |

Anchors: 52 labeled, 23 hidden, 29 unlabeled (skipped). Windows use the anchor frames inside each; `outside` is the anchor frames in no window. human_review_anchor masks are review evidence for scoring tracker arms against each other on a handful of frames; they are not a dataset, not ground truth, and support no accuracy claim. A human chose one SAM3 image-decoder mask per visible part, or marked the part hidden; the mask boundary is the decoder's, the choice is the human's.
