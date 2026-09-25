# `runs/` archive list, Sep 24, 2026 (nothing deleted)

Written at the Assembly101 close (ledger: "Sep 24: Assembly101 phase closed"). This is the
dry-run listing of `uv run python scripts/prune_runs.py` (which never deletes) plus the run
sub-directories the close-out plan named for archiving, each with its size on disk. **Nothing
has been deleted or moved; the user decides.** Sizes are the sum of file sizes under each
directory (`Path.rglob`), in decimal MB / GB, measured on Sep 24. `runs/` is gitignored; the
names are cited so the evidence can be found, not because it is tracked.

`prune_runs.py`: 255 run directories under `runs/`; 161 cited by
tracked files; 18 cited only by another run's manifest (kept by the script; listed in
section 7); **76 unreferenced, 1.94 GB**.

## Totals per group

| Group | Directories | Bytes | Size |
| --- | --- | --- | --- |
| 1. Unreferenced runs (prune_runs) | 76 | 1,938,450,588 | 1.94 GB |
| 2. `blocked-by-gpu-guard/` | 1 | 75,514 | 0.1 MB |
| 3. `failed-worker-edit-*` | 1 | 91,352 | 0.1 MB |
| 4. `code-snapshot-*` (git-archive snapshots inside run roots) | 22 | 75,881,653 | 75.9 MB |
| 5. `runs/dedup-equivalence/` (all 17 sub-roots; `pass3-*` = 10 of them, 773,321,258 bytes, 773.3 MB) | 1 | 1,165,708,288 | 1.17 GB |
| 6. `runs/dedup-pass2-smokes-*` | 1 | 620,400,580 | 620.4 MB |
| Superseded scratch roots `runs/*-local` | 0 | 0 | none exist on Sep 24 (the `interaction-review-first-minute-v4-local` root named in the Sep 18 open items is gone) |
| **All groups, de-duplicated** | | **3,793,709,039** | **3.79 GB** |

Overlaps, so the groups are not simply additive: the two `code-snapshot-*` directories inside
`runs/dedup-pass2-smokes-20260922/` (6,898,936 bytes) are counted in groups 4 and 6 and
subtracted once in the de-duplicated total; the `pass3-*` roots are a subset of group 5 and are
not added again. Groups 2-6 all live inside run roots that tracked files cite, so none of them
appears in group 1. Every group is kept for now; the plan said archive list, not deletion.

Reading aid per group: (1) nothing committed names these runs (README, ledger, QA records,
configs) and no other run's manifest does either; most are Sep 9-16 smokes, calibration
scratch and superseded review packages. (2) three view runs (C10395, C10404, HMC_21179183) of
the Sep 21 four-part rerun refused by the GPU guard (`worker_result.json`: "concurrent GPU model
process(es) detected", the calibration workspace's worker); manifests and logs only, the views
were re-run in the same pass. (3) seven empty run directories of the Sep 19 r1280 pass whose
SAM3 worker died at import during a concurrent live edit of `muggled_worker.py` (ledger, Sep 19
"SAM3 at 1280 px on all eight views", incident); the retry ran from a code snapshot. (4) `git archive` snapshots of `src/battle`
the overnight queue ran each GPU job from; every one is reproducible from the commit in its
name. (5) the before/after artifacts of the three Sep 22 dedup passes (the r1280 pm-append
consensus, the 19-arm scoreboard, the review v4 with `rerun rrd verify`, the v6 package before
and after group 1, five contact sheets); the equivalence is recorded in the ledger, the bytes
are only needed to re-check it. (6) the byte-identical GPU smoke masks of the six video drivers
for dedup pass 2.

## 1. Unreferenced runs (`prune_runs.py`, sorted by size)

| Bytes | Size | Run |
| --- | --- | --- |
| 244,113,282 | 244.1 MB | `muggledsam-sam3-full-ego-manual-seed-multiplexed-ego-hmc21179183-20260910t015418z` |
| 158,733,862 | 158.7 MB | `grounding-dino-sam2-open-vocabulary-four-part-20s-20260916t1002z` |
| 147,945,990 | 147.9 MB | `samurai-four-part-reviewed-seed-20s-20260916t1006z` |
| 96,203,600 | 96.2 MB | `muggledsam-sam3-smoke-static-c10379-20260909t025738z` |
| 79,845,692 | 79.8 MB | `muggledsam-sam3-static-black-prompt-sweep-20260914t021116z` |
| 73,579,793 | 73.6 MB | `mediapipe-hands-static-60s-tight-roi-th035-20260916t0433z` |
| 70,650,480 | 70.7 MB | `dam4sam_video_smoke-10s-20260916t052422z` |
| 69,898,191 | 69.9 MB | `review-snapshots` |
| 61,611,265 | 61.6 MB | `wilor-hands-stabilized-60s-v4` |
| 57,043,060 | 57.0 MB | `kineo-nlf-headless-60s-v4-postreboot` |
| 52,365,760 | 52.4 MB | `muggledsam-sam3-smoke-ego-hmc21110305-20260909t025755z` |
| 41,355,718 | 41.4 MB | `wilor-hands-static-20s-20260916t0438z` |
| 40,716,103 | 40.7 MB | `muggledsam-sam3-four-part-frame-zero-calibration-20260915t0221z` |
| 32,850,504 | 32.9 MB | `mediapipe-hands-static-20s-full-plus-roi-th035-20260916t0426z` |
| 30,822,973 | 30.8 MB | `muggledsam-sam3-four-part-static-pilot-static-c10379-20260916t011641z` |
| 30,548,555 | 30.5 MB | `mediapipe-hands-static-20s-full-plus-roi-v3-20260916t0424z` |
| 30,346,110 | 30.3 MB | `muggledsam-sam3-four-part-static-pilot-static-c10379-20260915t230953z` |
| 29,884,218 | 29.9 MB | `muggledsam-sam3-e4-four-target-keyframes-20260910t031614z` |
| 27,270,059 | 27.3 MB | `muggledsam-sam3-e4-four-target-keyframes-20260913t203413z` |
| 26,589,784 | 26.6 MB | `mediapipe-hands-static-20s-full-plus-roi-v2-20260916t0422z` |
| 26,165,692 | 26.2 MB | `mediapipe-hands-static-20s-full-plus-roi-20260916t0420z` |
| 25,983,947 | 26.0 MB | `mediapipe-hands-static-20s-20260916t034731z` |
| 25,956,703 | 26.0 MB | `mediapipe-hands-static-20s-20260916t034906z` |
| 25,539,836 | 25.5 MB | `mediapipe-hands-static-20s-roi-045-035-up2-20260916t0418z` |
| 22,400,046 | 22.4 MB | `muggledsam-sam3-e4-four-target-keyframes-20260913t195828z` |
| 20,827,260 | 20.8 MB | `muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260914t000716z` |
| 20,810,822 | 20.8 MB | `muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260913t233715z` |
| 20,804,720 | 20.8 MB | `muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260914t000819z` |
| 20,753,418 | 20.8 MB | `muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260913t233812z` |
| 20,721,377 | 20.7 MB | `wilor-hands-static-10s-test` |
| 19,441,689 | 19.4 MB | `wilor-hands-stabilized-20s-overnight-v2-r2` |
| 18,509,220 | 18.5 MB | `kineo-nlf-headless-20s-20260916t0535z` |
| 18,375,837 | 18.4 MB | `muggledsam-sam3-four-part-frame-zero-source190-calibration-20260915t1714z` |
| 16,770,333 | 16.8 MB | `muggledsam-sam3-smoke-hybrid-static-static-c10379-20260915t005001z` |
| 16,766,699 | 16.8 MB | `muggledsam-sam3-smoke-hybrid-static-static-c10379-20260915t004454z` |
| 16,736,181 | 16.7 MB | `muggledsam-sam3-smoke-hybrid-static-static-c10379-20260915t004720z` |
| 16,152,511 | 16.2 MB | `athena-hands-first-minute-mediapipe-rigdlt` |
| 14,825,179 | 14.8 MB | `muggledsam-sam3-smoke-static-c10379-20260914t014456z` |
| 13,729,199 | 13.7 MB | `muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260913t233645z` |
| 13,729,187 | 13.7 MB | `muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260914t000644z` |
| 12,760,345 | 12.8 MB | `muggledsam-sam3-e4-four-target-keyframes-20260913t211155z` |
| 11,119,126 | 11.1 MB | `samurai-static-20s-smoke-20260916t0510z` |
| 10,490,987 | 10.5 MB | `muggledsam-sam3-e4-web-calibration-left-hand-yellow-toy-top-black-toy-top-base-20260910T013118Z` |
| 10,146,471 | 10.1 MB | `muggledsam-sam3-e4-four-target-keyframes-20260910t182438z` |
| 9,274,662 | 9.3 MB | `muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260913t224300z` |
| 9,262,574 | 9.3 MB | `muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260913t224347z` |
| 8,904,165 | 8.9 MB | `drop-dtw-static-20s-20260916t0450z` |
| 8,887,733 | 8.9 MB | `grounding-dino-sam2-open-vocabulary-four-part-20s-20260916t1000z` |
| 8,386,883 | 8.4 MB | `muggledsam-sam3-smoke-multi-keyframe-corrections-ego-hmc21179183-20260913t222916z` |
| 8,374,810 | 8.4 MB | `muggledsam-sam3-smoke-multi-keyframe-corrections-ego-hmc21179183-20260913t222850z` |
| 8,203,784 | 8.2 MB | `muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260910t023602z` |
| 7,808,653 | 7.8 MB | `muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260910t015234z` |
| 4,329,653 | 4.3 MB | `muggledsam-sam3-four-part-static-corrections-20260915t2320z` |
| 3,443,142 | 3.4 MB | `boxmot-yolo-static-0.1s-20260916t054619z` |
| 3,440,470 | 3.4 MB | `mediapipe-hands-static-0.1s-20260916t054618z` |
| 3,051,744 | 3.1 MB | `interaction-review-first-minute-v4-audit` |
| 2,645,515 | 2.6 MB | `multiview-part-consensus-first-minute-r1280-4part-excl-c10119-20260921` |
| 2,639,210 | 2.6 MB | `multiview-part-consensus-first-minute-r1280-4part-others-only-20260921` |
| 2,451,018 | 2.5 MB | `muggledsam-sam3-e4-four-target-keyframes-20260913t201454z` |
| 2,412,770 | 2.4 MB | `muggledsam-sam3-e4-four-target-keyframes-20260913t200611z` |
| 2,256,981 | 2.3 MB | `ego-four-part-focused-visibility-review` |
| 239,749 | 0.2 MB | `muggledsam-sam3-four-part-source187-mask-test-20260915t0234z` |
| 235,027 | 0.2 MB | `muggledsam-sam3-four-part-frame-zero-human-calibration-20260915t0242z` |
| 106,259 | 0.1 MB | `muggledsam-sam3-e4-four-target-keyframes-20260913t203334z` |
| 105,685 | 0.1 MB | `rapid-calibration-headless-validation-20260909t0428z` |
| 19,022 | 0.0 MB | `muggledsam-sam3-smoke-multi-keyframe-corrections-ego-hmc21179183-20260913t222836z` |
| 19,020 | 0.0 MB | `muggledsam-sam3-smoke-multi-keyframe-corrections-ego-hmc21179183-20260913t222835z` |
| 19,018 | 0.0 MB | `muggledsam-sam3-smoke-multi-keyframe-corrections-ego-hmc21179183-20260913t221426z` |
| 16,797 | 0.0 MB | `muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260909t223751z` |
| 16,424 | 0.0 MB | `muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260910t015126z` |
| 2,411 | 0.0 MB | `muggledsam-sam3-e4-web-calibration-smoke` |
| 2,254 | 0.0 MB | `muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260909t222959z` |
| 1,838 | 0.0 MB | `muggledsam-sam3-four-part-frame-zero-source190-calibration-redo-20260915t1728z` |
| 1,533 | 0.0 MB | `muggledsam-sam3-e4-box-calibration-20260909t035137z` |
| 0 | 0.0 MB | `muggledsam-sam3-four-part-focused-calibration-20260916t015937z` |
| 0 | 0.0 MB | `muggledsam-sam3-e4-web-calibration-left-hand-yellow-toy-top-black-toy-top-base-20260909T234323Z` |
| **1,938,450,588** | **1.94 GB** | **76 runs** |

## 2. `blocked-by-gpu-guard/`

| Bytes | Size | Directory |
| --- | --- | --- |
| 75,514 | 0.1 MB | `runs/sam3-views-r1280-4part-20260921/blocked-by-gpu-guard` |

## 3. `failed-worker-edit-*`

| Bytes | Size | Directory |
| --- | --- | --- |
| 91,352 | 0.1 MB | `runs/sam3-views-r1280-20260919/failed-worker-edit-20260920t0156z` |

## 4. `code-snapshot-*` (`find runs -maxdepth 3 -type d -name "code-snapshot-*"`)

| Bytes | Size | Directory |
| --- | --- | --- |
| 4,060,456 | 4.1 MB | `runs/correction-acceptance-search-20260920/code-snapshot-12f48e9` |
| 2,632,358 | 2.6 MB | `runs/dam4sam-arms-20260919/code-snapshot-6eed418` |
| 2,631,637 | 2.6 MB | `runs/dam4sam-arms-20260919/code-snapshot-7380a9e` |
| 3,438,961 | 3.4 MB | `runs/dedup-pass2-smokes-20260922/code-snapshot-2a90ed2` |
| 3,459,975 | 3.5 MB | `runs/dedup-pass2-smokes-20260922/code-snapshot-6e7f1df` |
| 3,523,894 | 3.5 MB | `runs/multiview-reprompt-20260920/code-snapshot-0c62799` |
| 3,799,637 | 3.8 MB | `runs/multiview-reprompt-20260920/code-snapshot-625321a` |
| 3,970,361 | 4.0 MB | `runs/multiview-reprompt-20260921/code-snapshot-259b4f7` |
| 4,024,903 | 4.0 MB | `runs/multiview-seeds-human-accepted-20260921/code-snapshot-4551b73` |
| 3,716,189 | 3.7 MB | `runs/rec2-automatic-20260920/code-snapshot-21da378` |
| 4,006,804 | 4.0 MB | `runs/rec2-seed-proposals-20260920/code-snapshot-78f2eca` |
| 2,991,108 | 3.0 MB | `runs/sam3-exemplar-20260922/code-snapshot-0ef1868` |
| 4,419,628 | 4.4 MB | `runs/sam3-exemplar-20260922/code-snapshot-7f608e5` |
| 4,629,232 | 4.6 MB | `runs/sam3-exemplar-20260922/code-snapshot-d5298a8` |
| 3,073,513 | 3.1 MB | `runs/sam3-exemplar-20260922/code-snapshot-d914bef` |
| 3,015,932 | 3.0 MB | `runs/sam3-exemplar-20260922/code-snapshot-fb426c2` |
| 2,539,902 | 2.5 MB | `runs/sam3-memory-arms-20260919/code-snapshot-7380a9e` |
| 2,666,775 | 2.7 MB | `runs/sam3-views-r1280-20260919/code-snapshot-072fe0b` |
| 3,450,068 | 3.5 MB | `runs/sam3-views-r1280-4part-20260921/code-snapshot-eceae60` |
| 3,012,084 | 3.0 MB | `runs/sam3-views-r1280-pmappend-seeded-20260920/code-snapshot-df1edff` |
| 3,311,743 | 3.3 MB | `runs/seed-search-20260920/code-snapshot-20c843b` |
| 3,506,493 | 3.5 MB | `runs/seed-search-20260920/code-snapshot-df1edff` |
| **75,881,653** | **75.9 MB** | **22 snapshots** |

## 5. `runs/dedup-equivalence/` sub-roots

| Bytes | Size | Directory |
| --- | --- | --- |
| 55,549,904 | 55.5 MB | `runs/dedup-equivalence/after` |
| 56,142,382 | 56.1 MB | `runs/dedup-equivalence/before` |
| 56,140,768 | 56.1 MB | `runs/dedup-equivalence/pass2-a` |
| 56,141,007 | 56.1 MB | `runs/dedup-equivalence/pass2-after` |
| 56,120,884 | 56.1 MB | `runs/dedup-equivalence/pass2-b` |
| 56,145,592 | 56.1 MB | `runs/dedup-equivalence/pass2-before` |
| 56,146,493 | 56.1 MB | `runs/dedup-equivalence/pass2-c` |
| 56,093,986 | 56.1 MB | `runs/dedup-equivalence/pass3-after` |
| 56,096,111 | 56.1 MB | `runs/dedup-equivalence/pass3-before` |
| 56,095,702 | 56.1 MB | `runs/dedup-equivalence/pass3-g1` |
| 56,090,823 | 56.1 MB | `runs/dedup-equivalence/pass3-g2` |
| 56,091,008 | 56.1 MB | `runs/dedup-equivalence/pass3-g3` |
| 56,084,849 | 56.1 MB | `runs/dedup-equivalence/pass3-g4` |
| 56,094,795 | 56.1 MB | `runs/dedup-equivalence/pass3-g5` |
| 56,275,756 | 56.3 MB | `runs/dedup-equivalence/pass3-sheets` |
| 160,919,001 | 160.9 MB | `runs/dedup-equivalence/pass3-v6` |
| 163,479,227 | 163.5 MB | `runs/dedup-equivalence/pass3-v6-before` |
| **1,165,708,288** | **1.17 GB** | **whole root** (`pass3-*`: 773,321,258 bytes, 773.3 MB) |

## 6. `runs/dedup-pass2-smokes-*`

| Bytes | Size | Directory |
| --- | --- | --- |
| 620,400,580 | 620.4 MB | `runs/dedup-pass2-smokes-20260922` |

## 7. Cited only by another run's manifest (kept by `prune_runs.py`, listed for completeness)

| Run | Cited by |
| --- | --- |
| `mediapipe-hands-c10095-60s-20260918` | `athena-hands-first-minute-mediapipe`, `athena-hands-first-minute-mediapipe-ego`, `athena-hands-first-minute-mediapipe-rigdlt` |
| `mediapipe-hands-c10115-60s-20260918` | `athena-hands-first-minute-mediapipe`, `athena-hands-first-minute-mediapipe-ego`, `athena-hands-first-minute-mediapipe-rigdlt` |
| `mediapipe-hands-c10118-60s-20260918` | `athena-hands-first-minute-mediapipe`, `athena-hands-first-minute-mediapipe-ego`, `athena-hands-first-minute-mediapipe-rigdlt` |
| `mediapipe-hands-c10119-60s-20260918` | `athena-hands-first-minute-mediapipe`, `athena-hands-first-minute-mediapipe-ego`, `athena-hands-first-minute-mediapipe-rigdlt` |
| `mediapipe-hands-c10390-60s-20260918` | `athena-hands-first-minute-mediapipe`, `athena-hands-first-minute-mediapipe-ego`, `athena-hands-first-minute-mediapipe-rigdlt` |
| `mediapipe-hands-c10395-60s-20260918` | `athena-hands-first-minute-mediapipe`, `athena-hands-first-minute-mediapipe-ego`, `athena-hands-first-minute-mediapipe-rigdlt` |
| `mediapipe-hands-c10404-60s-20260918` | `athena-hands-first-minute-mediapipe`, `athena-hands-first-minute-mediapipe-ego`, `athena-hands-first-minute-mediapipe-rigdlt` |
| `muggledsam-sam3-e4-web-calibration-left-hand-yellow-toy-top-black-toy-top-base-20260910T014322Z` | `muggledsam-sam3-full-ego-manual-seed-multiplexed-ego-hmc21179183-20260910t015418z`, `muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260910t015126z`, `muggledsam-sam3-smoke-manual-seed-multiplexed-ego-hmc21179183-20260910t015234z` |
| `muggledsam-sam3-four-part-focused-calibration-20260916t015945z` | `muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260916t020716z` |
| `muggledsam-sam3-four-part-frame-zero-source190-calibration-redo-20260915t1749z` | `muggledsam-sam3-four-part-static-pilot-static-c10379-20260915t230953z` |
| `muggledsam-sam3-four-part-multiview-first-minute-static-c10095-20260918t055420z` | `multiview-part-consensus-first-minute`, `multiview-part-consensus-first-minute-with-e4`, `multiview-visual-hull-first-minute`, `multiview-visual-hull-first-minute-with-e4` |
| `muggledsam-sam3-four-part-multiview-first-minute-static-c10115-20260918t055658z` | `multiview-part-consensus-first-minute`, `multiview-part-consensus-first-minute-with-e4`, `multiview-visual-hull-first-minute`, `multiview-visual-hull-first-minute-with-e4` |
| `muggledsam-sam3-four-part-multiview-first-minute-static-c10118-20260918t055938z` | `multiview-part-consensus-first-minute`, `multiview-part-consensus-first-minute-with-e4`, `multiview-visual-hull-first-minute`, `multiview-visual-hull-first-minute-with-e4` |
| `muggledsam-sam3-four-part-multiview-first-minute-static-c10119-20260918t060222z` | `multiview-part-consensus-first-minute`, `multiview-part-consensus-first-minute-with-e4`, `multiview-visual-hull-first-minute`, `multiview-visual-hull-first-minute-with-e4` |
| `muggledsam-sam3-four-part-multiview-first-minute-static-c10390-20260918t060505z` | `multiview-part-consensus-first-minute`, `multiview-part-consensus-first-minute-with-e4`, `multiview-visual-hull-first-minute`, `multiview-visual-hull-first-minute-with-e4` |
| `muggledsam-sam3-four-part-multiview-first-minute-static-c10395-20260918t060741z` | `multiview-part-consensus-first-minute`, `multiview-part-consensus-first-minute-with-e4`, `multiview-visual-hull-first-minute`, `multiview-visual-hull-first-minute-with-e4` |
| `muggledsam-sam3-four-part-multiview-first-minute-static-c10404-20260918t061020z` | `multiview-part-consensus-first-minute`, `multiview-part-consensus-first-minute-with-e4`, `multiview-visual-hull-first-minute`, `multiview-visual-hull-first-minute-with-e4` |
| `muggledsam-sam3-four-part-static-corrections-frames-0-36-65-162-20260915t2331z` | `muggledsam-sam3-four-part-static-full-exploratory-static-c10379-20260916t012945z`, `muggledsam-sam3-four-part-static-pilot-static-c10379-20260916t011641z` |

## The command the script prints (copy, edit, run by hand; not run)

```bash
uv run python scripts/prune_runs.py        # the listing above, table form
uv run python scripts/prune_runs.py --json # the same as JSON
# The script ends with an `rm -rf runs/...` line covering group 1 only. It was not run.
```
