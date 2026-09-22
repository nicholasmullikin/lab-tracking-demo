# Anchor scoreboard: C10119, 26 frames (63 labelled / 1 hidden / 40 skipped), Sep 21 four-part rerun and the guarded re-prompt

Copied verbatim from `runs/multiview-reprompt-20260921/anchor_iou_c10119.md` (generated 2026-09-22T03:42:39.186505Z by
`battle-anchor-iou --view static-c10119`; `runs/` is not tracked). Anchor record:
`docs/qa/first-minute-review-anchors-static-c10119.human-record.json`; counts {'labeled': 63, 'hidden': 1, 'unlabeled': 40}.
Review evidence, not ground truth: one person's choice of SAM3 image-decoder mask per visible
part on a handful of frames of one camera, used to rank arms against each other; not a dataset
and no accuracy claim. A run without a slot for a part scores 0 on that part's labelled cells
(`missing`), so `IoU all` is comparable across rows. CC BY-NC 4.0 applies to the frames.

Runs scored:

- `r1280`: `runs/sam3-views-r1280-20260919/views/C10119/muggledsam-sam3-four-part-multiview-first-minute-static-c10119-20260920t021449z-r1280`
- `seeded-3part`: `runs/sam3-views-r1280-pmappend-seeded-20260920/views/C10119/muggledsam-sam3-four-part-multiview-first-minute-static-c10119-20260920t153339z-r1280-pm-append`
- `consensus-only-ds`: `runs/multiview-reprompt-20260920/variants/decoder-score/C10119/iter1/arms/consensus-only/muggledsam-sam3-four-part-multiview-first-minute-static-c10119-20260920t191112z-r1280-pm-append`
- `human-accepted-4part`: `runs/sam3-views-r1280-4part-20260921/views/C10119/muggledsam-sam3-four-part-multiview-first-minute-static-c10119-20260922t014959z-r1280-pm-append`
- `4part-guard-consensus-only-ds`: `runs/multiview-reprompt-20260921/C10119/iter1/arms/consensus-only/muggledsam-sam3-four-part-multiview-first-minute-static-c10119-20260922t033448z-r1280-pm-append`

| arm | IoU chassis | IoU interior | IoU rear_body | IoU cabin | IoU all | IoU 279-408 | IoU 573-722 | IoU 1020-1172 | IoU outside | missing | hidden FP (px) | scored / unlabeled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r1280 | 0.758 | 0.000 | 0.839 | 0.859 | 0.474 | 0.436 | 0.364 | 0.335 | 0.510 | 25 | 0 (0) | 38 / 40 |
| seeded-3part | 0.758 | 0.000 | 0.842 | 0.948 | 0.478 | 0.436 | 0.364 | 0.335 | 0.516 | 25 | 0 (0) | 38 / 40 |
| consensus-only-ds | 0.758 | 0.000 | 0.270 | 0.862 | 0.392 | 0.436 | 0.364 | 0.335 | 0.399 | 25 | 0 (0) | 38 / 40 |
| human-accepted-4part | 0.339 | 0.114 | 0.838 | 0.949 | 0.350 | 0.155 | 0.028 | 0.176 | 0.436 | 0 | 1 (5133) | 63 / 40 |
| 4part-guard-consensus-only-ds | 0.667 | 0.290 | 0.833 | 0.945 | 0.554 | 0.487 | 0.361 | 0.330 | 0.616 | 0 | 1 (3795) | 63 / 40 |

Anchors: 63 labeled, 1 hidden, 40 unlabeled (skipped). Windows use the anchor frames inside each; `outside` is the anchor frames in no window. human_review_anchor masks are review evidence for scoring tracker arms against each other on a handful of frames; they are not a dataset, not ground truth, and support no accuracy claim. A human chose one SAM3 image-decoder mask per visible part, or marked the part hidden; the mask boundary is the decoder's, the choice is the human's.
