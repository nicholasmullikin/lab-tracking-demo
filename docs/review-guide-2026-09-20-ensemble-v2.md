# Review guide: ensemble reference v2, the candidate (Sep 20)

Written for the human who labelled the 52 anchors on Sep 19 and left the follow-up plan
running. Everything here is pulled from the `### Sep 19:` and `### Sep 20:` sections of
[`method-ledger.md`](method-ledger.md) and from the manifests on disk; every path existed when
this guide was committed. `runs/` is gitignored, so the paths are for this machine. Nothing in
this guide runs a model; the GPU is idle.

Standing rule: the anchors are **13 frames on one view** (C10379: 300, 370, 400, 600, 650, 700,
900, 1050, 1100, 1150, 1200, 1500, 1700), one accepted SAM3 decoder mask per visible part or a
hidden mark. They rank arms against each other; they are not a dataset, not ground truth, and
support no accuracy claim. Means within ~0.02 do not rank; two arms within 0.01 on `all` are
tied. The ensemble's **arms were chosen by these anchors**, so its anchor score is
selection-biased; its **fallback intervals were not** (cross-view consensus only). Ensemble v2 is
a candidate. Adopting it as the review reference is your disposition, not the agent's.

## The recording

```bash
rerun /home/nick/src/battle/runs/interaction-review-first-minute-v5/interaction_review_first_minute_v4.rrd
```

100 MB, all layers, built on `runs/ensemble-reference-first-minute-v2/` (policy
`configs/ensemble_reference/first_minute_v2.json`) with the eight-view consensus rebuilt on the
same primary run (`runs/multiview-part-consensus-first-minute-r1280-pm-append/`). The v4 package
you may still have open is untouched. Same blueprint as v4: the provenance layer
`diagnostics/reference_provenance/<part>` (1 = SAM3 `pm-append`, 2 = DAM4SAM fallback; there is
no 3 = hidden any more), magenta `primary/reference_provenance_overlay/<part>` boxes on the 48
DAM4SAM frames, and `diagnostics/segmentation_contact_eligible/<part>`.

What is in it, in one line each:

- **Primary** `pm-append`: SAM3 at 1280 px, policy off, your four corrections `[327, 900, 1172,
  1235]`, with every correction *appended* to the prompt bank instead of replacing the seed
  (`--prompt-memory-semantics append`). 0.743 on the anchors; the old reference is 0.668.
- **Fallback** `dam4sam-large-1024-sched-60s`: DAM4SAM with the large SAM2.1 checkpoint, one
  shared predictor, the same four corrections. 0.715 on the anchors. Used only inside the
  label-free intervals below and only when a rule fires on the SAM3 mask and the DAM4SAM mask
  passes the v1 sanity bounds: **chassis 46 frames** (`[311,313)`, `[480,494)`, 512, 514,
  `[1057,1085)`), **rear_body 2 frames** (1762, 1776), interior and cabin none.
- **No hidden interior.** The v1 label that blanked the interior over `[1024,1172)` is
  withdrawn because you drew the interior at 1050, 1100 and 1150.
- **Frame 1700 screwdriver.** rear_body `[1660,1800)` is `not_contact_eligible` with
  `failure_case: distractor_confusion` and your rationale; the mask is displayed as tracked (no
  substitution, DAM4SAM has almost no rear_body mask after 1635). Bounds are label-free: the
  primary's rear_body centroid leaves the table piece at 1660 and never comes back.

## The three numbers

| number | old reference (`off-r720-sched`) | `pm-append` alone | **ensemble v2** |
| --- | --- | --- | --- |
| anchor IoU, all 51 visible cells | 0.668 | 0.743 | **0.743** (byte-equal to `pm-append` on every cell) |
| `[573,722)` / `[1020,1172)` window means | 0.628 / 0.533 | 0.687 / 0.616 | **0.687 / 0.616** |
| hidden-FP px at (1700, rear_body) | 1306 | 1265 | **1265** |

Why v2 equals its primary on the anchors: the label-free fallback intervals contain exactly one
anchor frame, 1050, and there the DAM4SAM chassis (7.3k px; it would have scored 0.87 where
`pm-append` scores 0.00) was **declined** by the continuity sanity check (IoU 0.07, centroid jump
73 px against a last-accepted SAM3 chassis that had itself collapsed to 961 px at 1048; the
allowance only reaches 75 px at 1057). Every other substituted frame has no anchor. So the
anchors cannot see the ensemble's contribution at all; they see `pm-append`. The scoreboard
(`runs/anchor-scoreboard-20260919/anchor_iou.md`, 20 arms) states this and the selection bias in
its footnotes; `anchors_vs_reference_vs_ensemble_v2.png` beside it is anchors | old reference |
v2, 13 rows.

Label-free corroboration of the same story: with `pm-append` as the C10379 reference the other
seven views plus e4 contradict the chassis on 78 frames instead of 174 (everything inside
`[573,722)` is gone), and the hull-vs-mask IoU in `[573,722)` goes 0.000 -> 0.675 while
`[1020,1172)` goes 0.466 -> 0.362, the same directions your anchors gave `pm-append` against the
1280 baseline (`runs/multiview-part-consensus-first-minute-r1280-pm-append/summary.md`).

## Frames to scrub, and what to expect

Per-cell anchor IoU is quoted as old reference / v2 / DAM4SAM-large where an anchor exists.

- **327** (your first correction). No anchor; the appended prompt bank keeps the frame-0 seed
  alive alongside this correction from here on. Around it, 300 / 370 / 400: chassis 0.75 / 0.68
  / 0.71 at 300 is the one early cell where v2 is below the old reference; interior 0.78 -> 0.88
  and rear_body 0.80 -> 0.93 at 300 are gains. Expect no visible provenance change (SAM3 only);
  the chassis `[296,313)` fallback interval fired on 311-312 only.
- **480-494 and 512-514** (label-free chassis fallback, no anchor). The DAM4SAM chassis replaces
  a SAM3 chassis that had shrunk below half its rolling median. **Look for the overlap**: the
  DAM4SAM chassis includes the region the SAM3 interior occupies (IoU 0.41-0.59 between the two),
  which is why the metrics report two new chassis/interior swap episodes here. The two sources
  disagree on where the interior ends; the ensemble does not resolve that, it shows it.
- **573-722** (the window the appended prompt bank fixes). Chassis at 600 / 650 / 700: 0.27 / 0.51
  / 0.00 -> 0.25 / 0.20 / 0.71; interior 0.54 / 0.58 / 0.20 -> 0.59 / 0.62 / 0.41. Frame 700 is
  the gain (the old reference had lost the chassis entirely); 650 is a loss (0.51 -> 0.20). No
  fallback fires here: with `pm-append` the other views no longer contradict C10379 in this
  window, so there is no label-free interval, although the label-free area-anomaly detector
  still flags the chassis as undersized over 572-695 (4.0x, was 5.9x on v1).
- **1020-1172** (the swap window). Chassis at 1050 / 1100 / 1150: 0.27 / 0.00 / 0.00 -> **0.00**
  / 0.56 / 0.60 (DAM4SAM-large: 0.87 / 0.80 / 0.50); interior 0.14 / 0.15 / 0.40 -> 0.09 / 0.26
  / 0.48 and now **visible** (v1 blanked it). Scrub 1048-1057: the SAM3 chassis collapses to
  200-300 px, the substitution is declined for eight frames (the SAM3 mask is kept and shown as
  unsane, contact-ineligible), then from 1057 to 1084 the magenta DAM4SAM chassis takes over
  (28 frames; the top-ranked v5 metrics episode is this 18.6x chassis growth at 1057-1071).
  The DAM4SAM chassis overlaps the SAM3 interior at IoU 0.5-0.8 through those frames. From
  1085 the SAM3 chassis is back (0.56 at 1100, 0.60 at 1150). Contact eligibility: chassis
  `[0,1049)` and `[1057,1200)`; interior `[0,1200)` again.
- **1235** (your screwdriver-vs-rear_body correction). No anchor between 1200 and 1500; the
  correction is in the bank. Nothing to compare against except your eyes.
- **1469** (where the *tiny* scheduled DAM4SAM arm lost the chassis for good, the reason it was
  not the fallback). The large fallback keeps a chassis here; the primary is SAM3 anyway. At
  1500: chassis 0.81 -> 0.88, interior 0.10 -> **0.68**, rear_body 0.00 in every SAM3 arm (the
  label is on the table piece while the body is attached; this is why `[1200,1800)` stays
  contact-ineligible).
- **1700** (your hidden mark). rear_body is on the yellow screwdriver in every arm: 1306 px in
  the old reference, 1265 px here. From 1660 the provenance layer shows the frames as
  `not_contact_eligible` / `distractor_confusion`; the mask is still drawn so you can see the
  failure. Chassis 0.65 -> 0.87 and interior 0.30 -> 0.87 at 1700 are the largest late gains of
  the appended prompt bank. Also 1748-1761: the rear_body mask collapses to ~270-500 px (an
  11.4x area anomaly, second in the v5 rank) still on the screwdriver.

## v5 vs v4, label-free

| | v4 (ensemble v1) | v5 (ensemble v2) |
| --- | --- | --- |
| reference primary / fallback | `off-r720-sched` (001210z) / DAM4SAM tiny seed-only | `pm-append` (1280, append) / DAM4SAM large + schedule |
| fallback frames chassis / interior / rear_body / cabin | 111 / 0 (148 hidden) / 0 / 0 | 46 / 0 / 2 / 0 |
| contact-eligible chassis | `[0,1020)` `[1055,1056)` `[1062,1200)` | `[0,1049)` `[1057,1200)` |
| contact-eligible interior | `[0,1024)` `[1172,1200)` | `[0,1200)` |
| contact rows observed / invalid_mask / missing_hand | 7,598 / 5,178 / 1,624 | 7,820 / 4,816 / 1,764 |
| debounced contact candidates / events | 3,265 / 168 | 3,417 / 185 |
| segmentation triggers / episodes (index) | 183 / 35 | 206 / 43 |
| multiview layer: views / C10379 contradicted frames | 8 / 262 | 9 / 115 |
| review-metrics episodes (`runs/review-metrics-first-minute-v4-current` vs `-v5`) | 187 | 185 |
| area anomalies / identity swaps / label crossings | 26 / 2 / 1 | 18 / 4 / 2 |
| contact intervals (sub-5-frame; median; p90; max) | 138 (47; 10; 55.0; 297) | 148 (47; 10; 53.6; 323) |
| part-area medians chassis / interior / rear_body / cabin | 7369 / 2503 / 2139 / 16451 | 7054 / 2872 / 2079 / 16117 |

The review metrics were re-run on the current v4 index because the existing
`review-metrics-first-minute-v2` package measured a Sep 17 index; both packages have their own
`metrics_report.md` and top-12 contact sheets.

## Caveats, again

- 13 frames on one view. Nothing before 300, nothing between 1200 and 1500, so 327, 1235 and
  1469 are unscored, and a single cell moves a window mean by 0.1.
- The arm choice saw the anchors; the ensemble's anchor row is optimistic by construction. The
  fallback intervals, the hidden-label withdrawal and the distractor bounds did not see them.
- The substituted DAM4SAM chassis overlaps the SAM3 interior inside both substituted intervals;
  the ensemble shows it rather than hiding it. A symmetric overlap bound would also decline the
  `[1057,1085)` substitution your anchors favour, so it was not added.
- Everything is comparison evidence on the first minute of one CC BY-NC 4.0 recording. If you
  adopt v2, the v4 builder's default reference flips from
  `runs/ensemble-reference-first-minute-v1` to `-v2` in one constant; until then the default is
  unchanged.
