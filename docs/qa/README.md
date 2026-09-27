# Fixed two-timestamp human QA

This directory holds small, tracked JSON decisions, never recordings, masks, checkpoints,
contact-sheet images, or other generated payloads. Visual evidence remains under gitignored
`runs/<run-id>/human_qa/`; each tracked record refers to it by repository-relative URI and
SHA-256.

The preserved Sep 8 plan defines the fixed rule, not literal times: every completed method
must be reviewed at the same two source-timeline instants, one representative easy
manipulation and one hard/occluded manipulation
([plan lines 70–77](../archive/plan-2026-09-08-assembly-rerun-lab.md#pre-accuracy-success-measures)).
It also says those times belong in the clip manifest, but the approved manifests predate
that field and do not contain them. Existing full-run sheets use three convenience samples
at proxy times 0.000, 90.000, and 179.967 seconds; they are not evidence that any sample was
human-selected as the planned easy or hard checkpoint.

On Sep 13 the human shifted the original candidates to the nearest historical mask-bearing
frames and selected:

- easy/clear: analysis frame 984, source frame 14,868, source `247.8` seconds
  (clip-relative `32.8` seconds);
- hard/occluded: analysis frame 4,416, source frame 21,732, source `362.2` seconds
  (clip-relative `147.2` seconds).

Both analysis frames are divisible by six, the historical 5 FPS sampling period at the
30 FPS analysis rate. The source values above are the exact frame-derived canonical values.

The two aligned canonical completed SAM3 baselines awaiting this gate are:

- `muggledsam-sam3-g3-full-hybrid-static-static-c10379-20260915t005919z` — aligned
  four-target full static hybrid baseline.
- `muggledsam-sam3-full-ego-manual-seed-multiplexed-ego-hmc21179183-20260910t024052z`
  — selected full ego four-target, human-seeded baseline.

The earlier `muggledsam-sam3-g3-full-static-c10379-20260909t030710z` zero-shot record
is preserved as historical three-target/mismatched-contract evidence and is not rewritten.

Generate a raw, synchronized selection aid before choosing the semantic timestamps:

```bash
uv run battle-human-qa-candidates \
  runs/muggledsam-sam3-g3-full-static-c10379-20260909t030710z \
  runs/muggledsam-sam3-full-ego-manual-seed-multiplexed-ego-hmc21179183-20260910t024052z
```

This writes a gitignored 12-candidate contact sheet under `artifacts/qa/`. Each candidate
shows static and ego proxy frames side-by-side with its exact source time, clip-relative
time, and analysis frame. It uses only the shared manifest mapping and raw proxy frames:
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

The timestamps must lie on both completed runs' 30 FPS analysis grid within source interval
`[215.000, 395.000)` seconds. The command validates each run manifest and config
fingerprint, writes only `pending` decisions, fingerprints the generated evidence, refuses
to overwrite any human-reviewed record, and prints the next manual action. It never invokes
SAM3.

For each JSON record, inspect both evidence columns and the corresponding raw-clip/Rerun
overlay. A human then changes each checkpoint from `pending` to `pass`, `flag`, or `fail`,
optionally adds a note, supplies `reviewed_by` and a timezone-aware `reviewed_at`, and updates
`overall_status` to the schema-derived conservative value. Validate the edited record with:

```bash
uv run python -c 'from pathlib import Path; from battle.schemas import FixedTimestampHumanQARecord; [FixedTimestampHumanQARecord.model_validate_json(p.read_text()) for p in Path("docs/qa").glob("*.human-qa.json")]'
```

These dispositions are a semantic sanity gate only. They are not ground truth, and every
record fixes `ground_truth_accuracy_claim` to `false`.

## Human review anchors (labelled Sep 19)

`first-minute-review-anchors.human-record.json` (`ReviewAnchorHumanRecord` in
`battle.review_anchors`) is written by `battle-anchor-export export` from the gitignored
workspace `runs/human-review-anchors-first-minute/`: one entry per anchor frame x part
(13 x 4) with its state (`labeled`, `hidden`, `unlabeled`) and the SHA-256 of each accepted
mask, plus fingerprints of the anchor config, the mask set and the calibration manifest.
It is the **human's decision record** for the anchors: `author` and `reviewed_at` come from the
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
`P03_03_01` (60, plus the six plan-driven plate slots that had no decision) with the human's
`accept` / `reject` / `null`, their `note`, the slot's `label`, `class`, `rule`, `seed_frame`,
its index in both seed runs, the seed mask's SHA-256, and the SHA-256 of the `decisions.json`
that `battle-detector-seed apply-decisions` consumed (52 accept / 6 reject / 2 null). The
sheets, masks and filtered worker files stay under gitignored
`runs/finebio-seeds-P03_03_01-20260925/` (FineBio licence). It is a seed filter, not an anchor
and not ground truth; the ledger entry "Sep 26: gate 1 held, seed decisions on trial 1
(p2-gate1)" describes the fields and the tile check.

## FineBio gate 2, review anchors on trial 1 (held Sep 27)

`finebio-P03_03_01-review-anchors.human-record.json` is written by `battle-finebio-anchors
export` from the gitignored workspace `runs/finebio-anchors-P03_03_01-20260925/`: one entry per
anchor cell (440 = 40 frame-views x 11 slots; frame 916 in six views, 12 disagreement and 5
random frames on the fpv and T4) with its `state` (`mask` 325, `hidden` 4, `none_fits` 8,
`unlabeled` 103: the cells without a detector box), the accepted candidate's index and kind
(`arm_b_tight` 307, `tight_rank2` 8, `box_point` 6, `margin` 4), the SHA-256 and pixel area of
the accepted mask PNG, and the human's `instance_identity` (six names); plus the counts, the
anchor config's and the `decisions.json`'s URI and SHA-256, `author`, and the claim boundary. The
masks, sheets and the decisions file stay under `runs/` (FineBio licence). It is the human's
decision record for ranking the tracking arms (a)-(d) against each other, one person's choice
among decoder masks on 18 frames of one trial, not a dataset and not ground truth; the scoreboard
against it and the read-back are in the ledger entry "Sep 27: gate 2 held, the anchor scoreboard
on trial 1 (p6-anchors)".
