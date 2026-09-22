# Anchor scoreboard: C10379 re-prompt arms, Sep 21 (51 labelled cells + 1 hidden, 13 frames)

Copied verbatim from `runs/multiview-reprompt-20260921/anchor_iou_arms.md` (generated 2026-09-22T03:42:36.777850Z; `runs/` is not
tracked). Anchor record: `docs/qa/first-minute-review-anchors.human-record.json`. Review evidence
for ranking arms against each other on one view, not accuracy, not a dataset. CC BY-NC 4.0.

Runs scored:

- `pm-append`: `runs/sam3-memory-arms-20260919/arms/pm-append/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260920t033947z-r1280-pm-append`
- `seed-only-1280`: `runs/multiview-reprompt-20260920/C10379/iter1/arms/seed-only-1280/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260920t165430z-r1280-pm-append-seed0`
- `consensus-only-iter1`: `runs/multiview-reprompt-20260920/C10379/iter1/arms/consensus-only/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260920t164049z-r1280-pm-append`
- `consensus-only-iter2`: `runs/multiview-reprompt-20260920/C10379/iter2/arms/consensus-only/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260920t170137z-r1280-pm-append`
- `human-plus-consensus`: `runs/multiview-reprompt-20260920/C10379/iter1/arms/human-plus-consensus/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260920t164734z-r1280-pm-append`
- `consensus-only-ds-iter1`: `runs/multiview-reprompt-20260920/variants/decoder-score/C10379/iter1/arms/consensus-only/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260920t190133z-r1280-pm-append`
- `consensus-only-ds-4part-iter1`: `runs/multiview-reprompt-20260921/C10379/iter1/arms/consensus-only/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260922t024742z-r1280-pm-append`
- `human-plus-consensus-ds-4part-iter1`: `runs/multiview-reprompt-20260921/C10379/iter1/arms/human-plus-consensus/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260922t025550z-r1280-pm-append`
- `consensus-only-ds-4part-noguard-iter1`: `runs/multiview-reprompt-20260921/variants/no-guard/C10379/iter1/arms/consensus-only/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260922t030420z-r1280-pm-append`
- `human-plus-consensus-ds-4part-noguard-iter1`: `runs/multiview-reprompt-20260921/variants/no-guard/C10379/iter1/arms/human-plus-consensus/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260922t031236z-r1280-pm-append`

| arm | IoU chassis | IoU interior | IoU rear_body | IoU cabin | IoU all | IoU 279-408 | IoU 573-722 | IoU 1020-1172 | IoU outside | missing | hidden FP (px) | scored / unlabeled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pm-append | 0.637 | 0.585 | 0.790 | 0.962 | 0.743 | 0.853 | 0.687 | 0.616 | 0.800 | 0 | 1 (1265) | 51 / 0 |
| seed-only-1280 | 0.404 | 0.228 | 0.789 | 0.959 | 0.591 | 0.648 | 0.628 | 0.579 | 0.527 | 0 | 1 (1252) | 51 / 0 |
| consensus-only-iter1 | 0.291 | 0.316 | 0.810 | 0.959 | 0.590 | 0.573 | 0.651 | 0.637 | 0.517 | 0 | 1 (1226) | 51 / 0 |
| consensus-only-iter2 | 0.519 | 0.397 | 0.787 | 0.959 | 0.663 | 0.653 | 0.661 | 0.747 | 0.606 | 0 | 1 (1251) | 51 / 0 |
| human-plus-consensus | 0.591 | 0.585 | 0.791 | 0.960 | 0.730 | 0.823 | 0.679 | 0.560 | 0.834 | 0 | 1 (1207) | 51 / 0 |
| consensus-only-ds-iter1 | 0.609 | 0.288 | 0.788 | 0.959 | 0.659 | 0.720 | 0.694 | 0.677 | 0.566 | 0 | 1 (1266) | 51 / 0 |
| consensus-only-ds-4part-iter1 | 0.534 | 0.363 | 0.790 | 0.959 | 0.659 | 0.648 | 0.649 | 0.749 | 0.603 | 0 | 1 (1219) | 51 / 0 |
| human-plus-consensus-ds-4part-iter1 | 0.636 | 0.583 | 0.790 | 0.961 | 0.742 | 0.853 | 0.663 | 0.609 | 0.822 | 0 | 1 (1261) | 51 / 0 |
| consensus-only-ds-4part-noguard-iter1 | 0.543 | 0.356 | 0.789 | 0.958 | 0.659 | 0.648 | 0.649 | 0.686 | 0.656 | 0 | 1 (1241) | 51 / 0 |
| human-plus-consensus-ds-4part-noguard-iter1 | 0.638 | 0.580 | 0.790 | 0.960 | 0.741 | 0.853 | 0.663 | 0.607 | 0.821 | 0 | 1 (1274) | 51 / 0 |

Anchors: 51 labeled, 1 hidden, 0 unlabeled (skipped). Windows use the anchor frames inside each; `outside` is the anchor frames in no window. human_review_anchor masks are review evidence for scoring tracker arms against each other on a handful of frames; they are not a dataset, not ground truth, and support no accuracy claim. A human chose one SAM3 image-decoder mask per visible part, or marked the part hidden; the mask boundary is the decoder's, the choice is the human's.
