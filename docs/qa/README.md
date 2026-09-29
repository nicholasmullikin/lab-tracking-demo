# Fixed two-timestamp human QA

This directory holds small, tracked JSON decisions, never recordings, masks, checkpoints,
contact-sheet images or other generated payloads. Visual evidence remains, outside Git, under
`runs/<run-id>/human_qa/`; each tracked record refers to it by repository-relative URI and
SHA-256.

The preserved Sep 8 plan defines the fixed rule, not literal times: every completed method
must be reviewed at the same two source-timeline instants, one representative easy
manipulation and one hard/occluded manipulation
([plan lines 70–77](../archive/plan-2026-09-08-assembly-rerun-lab.md#pre-accuracy-success-measures)).
It also says those times belong in the clip manifest, but the approved manifests predate
that field and do not contain them. Existing full-run sheets use three convenience samples
at proxy times 0.000, 90.000 and 179.967 seconds; they are not evidence that I selected any sample
as the planned easy or hard checkpoint.

I shifted the original candidates to the nearest historical mask-bearing frames on Sep 13 and
selected:

- easy/clear: analysis frame 984, source frame 14,868, source `247.8` seconds
  (clip-relative `32.8` seconds);
- hard/occluded: analysis frame 4,416, source frame 21,732, source `362.2` seconds
  (clip-relative `147.2` seconds).

Both analysis frames are divisible by six, the historical 5 fps sampling period at the
30 fps analysis rate. The source values above are the exact frame-derived canonical values.

The two aligned canonical completed SAM3 baselines awaiting this gate are:

- `muggledsam-sam3-g3-full-hybrid-static-static-c10379-20260915t005919z`: the aligned
  four-target full static hybrid baseline.
- `muggledsam-sam3-full-ego-manual-seed-multiplexed-ego-hmc21179183-20260910t024052z`:
  the selected full ego four-target baseline, seeded by me.

The earlier `muggledsam-sam3-g3-full-static-c10379-20260909t030710z` zero-shot record
is preserved as historical three-target/mismatched-contract evidence and is not rewritten.

Generate a raw, synchronized selection aid before choosing the semantic timestamps:

```bash
uv run battle-human-qa-candidates \
  runs/muggledsam-sam3-g3-full-static-c10379-20260909t030710z \
  runs/muggledsam-sam3-full-ego-manual-seed-multiplexed-ego-hmc21179183-20260910t024052z
```

This writes a 12-candidate contact sheet, outside Git, under `artifacts/qa/`. Each candidate
shows static and ego proxy frames side by side with its exact source time, clip-relative
time and analysis frame. It uses only the shared manifest mapping and raw proxy frames:
there are no model overlays or suggested easy/hard labels.

The pending records and their inference-free two-column sheets were prepared with:

```bash
uv run battle-prepare-human-qa \
  runs/muggledsam-sam3-g3-full-hybrid-static-static-c10379-20260915t005919z \
  runs/muggledsam-sam3-full-ego-manual-seed-multiplexed-ego-hmc21179183-20260910t024052z \
  --easy-source-seconds 247.8 \
  --hard-source-seconds 362.2 \
  --output-directory docs/qa
```

The tracked records are:

- `muggledsam-sam3-g3-full-hybrid-static-static-c10379-20260915t005919z.human-qa.json`;
- `muggledsam-sam3-full-ego-manual-seed-multiplexed-ego-hmc21179183-20260910t024052z.human-qa.json`.

The historical zero-shot static record remains alongside them without transferred
dispositions.

The timestamps must lie on both completed runs' 30 fps analysis grid within source interval
`[215.000, 395.000)` seconds. The command validates each run manifest and config
fingerprint, writes only `pending` decisions, fingerprints the generated evidence, refuses
to overwrite any record I have reviewed and prints the next manual action. It never invokes
SAM3.

For each JSON record, inspect both evidence columns and the corresponding raw-clip/Rerun
overlay. I then change each checkpoint from `pending` to `pass`, `flag` or `fail`, may add a
note, supply `reviewed_by` and a timezone-aware `reviewed_at`, and update `overall_status` to
the schema-derived conservative value. Validate the edited record with:

```bash
uv run python -c 'from pathlib import Path; from battle.schemas import FixedTimestampHumanQARecord; [FixedTimestampHumanQARecord.model_validate_json(p.read_text()) for p in Path("docs/qa").glob("*.human-qa.json")]'
```

These dispositions are a semantic sanity gate only. They are not ground truth, and every
record fixes `ground_truth_accuracy_claim` to `false`.

## Human review anchors (labeled Sep 19)

`first-minute-review-anchors.human-record.json` (`ReviewAnchorHumanRecord` in
`battle.review_anchors`) is written by `battle-anchor-export export` from the uncommitted
workspace `runs/human-review-anchors-first-minute/`: one entry per anchor frame × part
(13 × 4) with its state (`labeled`, `hidden`, `unlabeled`) and the SHA-256 of each accepted
mask, plus fingerprints of the anchor config, the mask set and the calibration manifest.
It is my decision record for the anchors: `author` and `reviewed_at` come from the
`provenance` block of `configs/qa/first_minute_review_anchors.json` (filled after the Sep 19
session; `reviewed_at` is the workspace manifest's last autosave, since the workspace stores no
per-acceptance timestamps). The committed file holds 51 `labeled` and 1 `hidden` (rear_body at
1700) cells, 0 `unlabeled`. The masks themselves stay in `runs/`; they are `human_review_anchor`
review evidence for scoring tracker arms, not a dataset and not ground truth. Scores against
them: `runs/sam3-policy-ablation-20260918/anchor_iou.{json,md}` and the ledger's "Anchor IoU"
subsection under "Sep 18: slot exclusivity and score-gated memory".

## FineBio gate 1, seed decisions on trial 1 (held Sep 26)

`finebio-P03_03_01-seed-decisions.human-record.json` is the no-pixel record of the soft seed
gate of the FineBio 3D-tracking plan (`p2-gate1`): one entry per SAM3 seed slot of trial 1
`P03_03_01` (60, plus the six plan-driven plate slots that had no decision) with my
`accept` / `reject` / `null`, my `note`, the slot's `label`, `class`, `rule`, `seed_frame`,
its index in both seed runs, the seed mask's SHA-256 and the SHA-256 of the `decisions.json`
that `battle-detector-seed apply-decisions` consumed (52 accept / 6 reject / 2 null). The
sheets, masks and filtered worker files stay, outside Git, under
`runs/finebio-seeds-P03_03_01-20260925/` (FineBio license). It is a seed filter, not an anchor
and not ground truth; the ledger entry "Sep 26: gate 1 held, seed decisions on trial 1
(p2-gate1)" describes the fields and the tile check.

## FineBio gate 2, review anchors on trial 1 (held Sep 27)

`finebio-P03_03_01-review-anchors.human-record.json` is written by `battle-finebio-anchors
export` from the uncommitted workspace `runs/finebio-anchors-P03_03_01-20260925/`: one entry per
anchor cell (440 = 40 frame-views × 11 slots; frame 916 in six views, 12 disagreement and 5
random frames on the fpv and T4) with its `state` (`mask` 325, `hidden` 4, `none_fits` 8,
`unlabeled` 103: the cells without a detector box), the accepted candidate's index and kind
(`arm_b_tight` 307, `tight_rank2` 8, `box_point` 6, `margin` 4), the SHA-256 and pixel area of
the accepted mask PNG, and my `instance_identity` (six names); plus the counts, the
anchor config's and the `decisions.json`'s URI and SHA-256, `author` and the claim boundary. The
masks, sheets and the decisions file stay under `runs/` (FineBio license). It is my
decision record for ranking the tracking arms (a) to (d) against each other, my choice among
decoder masks on 18 frames of one trial, not a dataset and not ground truth; the scoreboard
against it and the read-back are in the ledger entry "Sep 27: gate 2 held, the anchor scoreboard
on trial 1 (p6-anchors)".

## FineBio tip clicks on trial 1 (clicked Sep 28)

**The tip clicks give 38 three-dimensional tip anchors on 30 frames of trial 1, and no
tracker meets the pre-registered 2 cm median.** `finebio-P03_03_01-tip-clicks.human-record.json`
is written by `battle-finebio-tips export` from the uncommitted workspace
`runs/finebio-tips-P03_03_01-20260928/`. It holds one entry per cell (167: ten rest, ten held
and ten low frames, each pipette in two or three views) with my click in full-frame pixels
(`tip_px`), the crop I clicked in, the hidden flag, the identity key and a note. The file also
carries the counts, the frame convention, the protocol below, the URI and SHA-256 of
`cells.json` and of the `decisions.json`, and the claim boundary. The crops and the decisions
file stay under `runs/` (FineBio license).

**I clicked to accept, not to place.** On every cell I clicked only when the dashed marker
(the mask axis end away from the hand) sat on the correct tip, and pressed `h` otherwise. So
`hidden` on 25 cells means the marker was wrong or the tip was hidden, not that the tip was
hidden. Each of the 138 clicks is my acceptance of the axis end as the tip. That is why 84 of
them (61%) fall within 3 px of the marker: the protocol, not a bias toward the suggestion.
Four cells stayed undecided. I pressed no identity key, by design. The record's `protocol`
field states this.

`battle-finebio-tips score` triangulates the clicks through the rig. Frames with one clicked
view (10), none (4) or clicks that disagree by more than 30 px after one view is dropped (12)
get no anchor. Eight of the 12 are the 8-channel pipette, whose accepted marker sat on the
plunger end in one view (T3 or T5 on seven of them). The 38 anchors that remain reproject to a
median of 3.7 px (p90 6.7 px). Each tracker's tip is measured against them raw and split along
and across the track's axis. The along part is the one a disposable tip would show as a steady
positive offset. On these clicks its median is minus 0.5 cm, so the misses are not tip length.
They are tracks that name the wrong end, or accepted markers that sat on the plunger end (the
8-channel in T3 and T5, the red pipette at frame 3311). For the 8-channel the axis end away
from the hand was often the plunger, so its anchors are unreliable as tip anchors. The clicks
are my clicks on one trial: they rank trackers against each other and are not ground truth.

Evidence: `runs/finebio-tips-P03_03_01-20260928/scoreboard/tip_scoreboard.md`, plan todo
`p3-tip-anchors`.
