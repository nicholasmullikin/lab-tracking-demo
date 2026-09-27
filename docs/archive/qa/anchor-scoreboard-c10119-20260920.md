# Anchor scoreboard: C10119 (top-down static camera), 26 frames, labelled Sep 21

Copied verbatim from `runs/anchor-scoreboard-c10119-20260920/anchor_iou.md` (generated 2026-09-22T00:13:16.913800Z by
`battle-anchor-iou --view static-c10119`; the JSON beside it is the source of these numbers; `runs/`
is not tracked). Anchor record: `docs/qa/first-minute-review-anchors-static-c10119.human-record.json`;
counts {'labeled': 63, 'hidden': 1, 'unlabeled': 40}. Review evidence, not ground truth: one person's choice of SAM3
image-decoder mask per visible part on a handful of frames of one camera, used to rank arms
against each other; not a dataset and no accuracy claim. CC BY-NC 4.0 applies to the frames.

Runs scored:

- `r1280`: `runs/sam3-views-r1280-20260919/views/C10119/muggledsam-sam3-four-part-multiview-first-minute-static-c10119-20260920t021449z-r1280`
- `seeded`: `runs/sam3-views-r1280-pmappend-seeded-20260920/views/C10119/muggledsam-sam3-four-part-multiview-first-minute-static-c10119-20260920t153339z-r1280-pm-append`
- `consensus-only`: `runs/multiview-reprompt-20260920/C10119/iter1/arms/consensus-only/muggledsam-sam3-four-part-multiview-first-minute-static-c10119-20260920t181354z-r1280-pm-append`
- `consensus-only-ds`: `runs/multiview-reprompt-20260920/variants/decoder-score/C10119/iter1/arms/consensus-only/muggledsam-sam3-four-part-multiview-first-minute-static-c10119-20260920t191112z-r1280-pm-append`

| arm | IoU chassis | IoU interior | IoU rear_body | IoU cabin | IoU all | IoU 279-408 | IoU 573-722 | IoU 1020-1172 | IoU outside | missing | hidden FP (px) | scored / unlabeled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r1280 | 0.758 | 0.000 | 0.839 | 0.859 | 0.474 | 0.436 | 0.364 | 0.335 | 0.510 | 25 | 0 (0) | 38 / 40 |
| seeded | 0.758 | 0.000 | 0.842 | 0.948 | 0.478 | 0.436 | 0.364 | 0.335 | 0.516 | 25 | 0 (0) | 38 / 40 |
| consensus-only | 0.758 | 0.000 | 0.270 | 0.862 | 0.392 | 0.436 | 0.364 | 0.335 | 0.399 | 25 | 0 (0) | 38 / 40 |
| consensus-only-ds | 0.758 | 0.000 | 0.270 | 0.862 | 0.392 | 0.436 | 0.364 | 0.335 | 0.399 | 25 | 0 (0) | 38 / 40 |

Anchors: 63 labeled, 1 hidden, 40 unlabeled (skipped). Windows use the anchor frames inside each; `outside` is the anchor frames in no window. human_review_anchor masks are review evidence for scoring tracker arms against each other on a handful of frames; they are not a dataset, not ground truth, and support no accuracy claim. A human chose one SAM3 image-decoder mask per visible part, or marked the part hidden; the mask boundary is the decoder's, the choice is the human's.
