# Anchor scoreboard: e4 (HMC_21179183, ego, monochrome), 26 frames, labelled Sep 21

Copied verbatim from `runs/anchor-scoreboard-e4-20260920/anchor_iou.md` (generated 2026-09-22T00:35:29.570910Z by
`battle-anchor-iou --view ego-hmc21179183`; the JSON beside it is the source of these numbers; `runs/`
is not tracked). Anchor record: `docs/qa/first-minute-review-anchors-ego-hmc21179183.human-record.json`;
counts {'labeled': 52, 'hidden': 23, 'unlabeled': 29}. Review evidence, not ground truth: one person's choice of SAM3
image-decoder mask per visible part on a handful of frames of one camera, used to rank arms
against each other; not a dataset and no accuracy claim. CC BY-NC 4.0 applies to the frames.

Runs scored:

- `HMC_21179183`: `runs/sam3-views-r1280-20260919/views/HMC_21179183/muggledsam-sam3-four-part-multiview-first-minute-ego-hmc21179183-20260920t023927z-r1280`

| arm | IoU chassis | IoU interior | IoU rear_body | IoU cabin | IoU all | IoU 279-408 | IoU 573-722 | IoU 1020-1172 | IoU outside | missing | hidden FP (px) | scored / unlabeled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HMC_21179183 | 0.845 | 0.000 | 0.262 | 0.831 | 0.558 | 0.633 | 0.538 | 0.525 | 0.558 | 10 | 8 (48705) | 42 / 29 |

Anchors: 52 labeled, 23 hidden, 29 unlabeled (skipped). Windows use the anchor frames inside each; `outside` is the anchor frames in no window. human_review_anchor masks are review evidence for scoring tracker arms against each other on a handful of frames; they are not a dataset, not ground truth, and support no accuracy claim. A human chose one SAM3 image-decoder mask per visible part, or marked the part hidden; the mask boundary is the decoder's, the choice is the human's.
