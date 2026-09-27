# Gate 1 (soft), Sep 25: the FineBio trial-1 seed shortlist, 15 minutes, no GPU

The `p2-gate1` todo of the FineBio 3D-tracking plan
([`plan-2026-09-25-finebio-3d-tracking.md`](plan-2026-09-25-finebio-3d-tracking.md)). **Nothing
waits for this sitting.** The seeds below ran auto-accepted (`provenance: auto`) and the GPU
lane continues on them; your accept/reject is applied afterwards as a filter
(`battle-detector-seed apply-decisions`), and only the slots you reject are removed (and, if
you want them replaced, re-run). Nothing is drawn, no mask is edited; the decoder's masks are
the decoder's, the choice is yours. FineBio is non-commercial research data: the sheets stay
under `runs/`, never in the repository.

Standing claim boundary: the slots were opened by rules over the FineBio DINO detector's boxes
(the detector was trained on this lab's objects and cameras) and each seed was accepted because
the SAM3 mask's bounding box agrees with the detector box (IoU >= 0.6). Your decisions are
review evidence about which per-view SAM3 slots deserve to exist; they are not ground truth and
support no accuracy claim.

## What to look at

Run directory: `runs/finebio-seeds-P03_03_01-20260925/` (trial 1 `P03_03_01`, raw frames
[600, 4200), the plan's window; produced by `battle-detector-seed` on
`runs/finebio-detect-P03_03_01-600-4200-20260925/dino`).

- `sheets/<view>.jpg` for `T1`, `T2`, `T3`, `T4`, `T5`, `fpv`: one tile per accepted seed at
  its seed frame, cropped around the box with context. Each tile shows the SAM3 mask (filled,
  in the class family's colour: pipettes orange, tubes magenta, racks yellow, machines
  blue-cyan, groups white), the detector box, and three text lines: `<label>  slot <n>`,
  `<rule> / <role>  f<seed frame>`, `det <detector score>  dec <decoder IoU prediction>  bbox
  IoU <mask-bbox IoU vs box> <tight|margin>`. No tile is missing: all 60 seeds decoded.
- `seeds.md`: the same 60 seeds as a table, one line per view underneath with what stayed
  `detector_only` (never a slot) and what was `capped` (would have been a slot beyond the
  10-per-view cap).
- `decisions.template.json`: one entry per slot (`view`, `slot`, `label`, `class`, `rule`,
  `role`, `status`, `seed_frame`, `decision: null`, `note: ""`). Copy it to `decisions.json`
  and fill `decision` with `"accept"`, `"reject"` or leave `null`.

## The shortlist the rule produced (per view, in slot order)

| view | landmarks | objects that moved | groups | held only |
|---|---|---|---|---|
| T1 | centrifuge, vortex_mixer, pcr_machine | 8_channel_pipette, blue_pipette, 50ml_tube#0 (frames 1118-1553, in a hand 60%) | micro_tube_group (29 tubes in the micro-tube rack), 15ml_tube_group (3), 50ml_tube_group#0 (4), 50ml_tube_group#1 (3) | - |
| T2 | centrifuge, vortex_mixer (the PCR machine is never detected at score >= 0.2 in T2: it is outside or at the edge of that camera's view) | blue_pipette, yellow_pipette, 15ml_tube#0, 50ml_tube#0 (from 2043), 50ml_tube#1 (moves 410 px), micro_tube#0 (1100-1171), 50ml_tube#2 (1970-2034) | micro_tube_group | - |
| T3 | centrifuge, vortex_mixer, pcr_machine | yellow_pipette, 8_channel_pipette, blue_pipette, 50ml_tube#0, micro_tube#0 (895-2086, in a hand 41%), 50ml_tube#1 (1898-2078) | micro_tube_group (38) | - |
| T4 | centrifuge, vortex_mixer (held 14%), pcr_machine | trash_can, yellow_pipette, red_pipette, blue_pipette, 8_tube_stripes#0, 50ml_tube#0, 50ml_tube#1 (from 2564) | (capped) | - |
| T5 | centrifuge, vortex_mixer, pcr_machine | yellow_pipette, red_pipette, 8_channel_pipette, blue_pipette | micro_tube_group (54), 50ml_tube_group (5) | micro_tube#0 (2817-3004, in a hand 100%) |
| fpv | centrifuge, vortex_mixer, pcr_machine | yellow_pipette, red_pipette, trash_can, 8_channel_pipette, blue_pipette, 15ml_tube#0 (903-1881), 15ml_tube#1 (2045-2564) | - | - |

Against the plan's shortlist (`p2-gate1`: plate, pipettes in use, tubes in use, tip rack in
use, centrifuge, vortex, pcr_machine): the pipettes, the tubes and the three machines are
there in every view that detects them. Two departures you should know about, both from the
data rather than from a choice:

- **The cell-culture plate is not a slot in trial 1.** In protocol 03 the plate never moves
  in any fixed view (it is `detector_only` in all five, rule "static, not in a hand, not a
  named container"); in the fpv it "moves" only with the head and no fixed camera corroborates
  it. Rejecting or accepting nothing changes this; if you want the plate tracked anyway, say so
  and it is one flag (`--container-classes` with the plate added opens it as a static volume).
- **The tip racks are not in the top 10 of any view.** The racks did move in this trial (the
  blue and yellow tip racks by 25-40 px in T1/T5), but the cap ranks the machines, the moving
  objects and the tube groups above them; they are listed under `capped` in `seeds.md` with
  their numbers. Raising the cap (`--slot-cap 14`) would take them in; the memory-free arm's
  cost does not depend on the slot count, the video-memory arm's does.

## What to decide, per tile (15 minutes for 60 tiles)

Accept (or leave `null`, which means the same) when the mask is on the object the label names.
Reject when:

1. the mask is on a different object than the label (a pipette read in the wrong colour: the
   detector confuses blue/yellow/red pipettes, so `yellow_pipette#0` in one view may be the
   blue pipette; the class name matters less than the object being one consistent thing, so
   reject only if the *box* is on the wrong object, not because the colour word is wrong);
2. the mask covers a hand or the bench rather than the object (look at the blue pipette tiles:
   it is held in most views, the mask should be the pipette, not the glove);
3. the seed frame shows the object occluded or half out of the crop so that SAM3 would start
   from a partial mask;
4. two slots in one view are the same physical object (the merge joins fragments of one class
   by overlap and joins every instance of a singleton class; `50ml_tube#0`, `#1`, `#2` in T2
   are meant to be different tubes, `micro_tube#0` in T3 a tube out of its rack; if two tiles
   are visibly one tube, reject the shorter-lived one and note it).

A group tile (`micro_tube_group#0` etc.) shows the union box of the tubes in one rack and the
mask SAM3 produced for it: with the tubes packed in the rack this is the rack-with-tubes, and
that is the plan's "one track per rack until a member leaves". Accept it if the box is the
rack the tubes stand in.

Put a word in `note` when useful ("held pipette read as yellow", "same tube as #0"). Leave the
`status`, `rule`, `role` fields alone; they are the record of what the rule did.

## How the decisions are applied

```bash
cd /home/nick/src/battle
cp runs/finebio-seeds-P03_03_01-20260925/decisions.template.json \
   runs/finebio-seeds-P03_03_01-20260925/decisions.json
# edit decisions.json, then:
uv run battle-detector-seed apply-decisions \
  --output runs/finebio-seeds-P03_03_01-20260925 \
  --decisions runs/finebio-seeds-P03_03_01-20260925/decisions.json
```

This writes `filtered/box_streams/<view>.jsonl`, `filtered/schedules/<view>.json` and
`filtered/seeds.json` with the rejected slots removed and the remaining slots renumbered
contiguously (the SAM3 worker needs slots 0..N-1); every kept slot carries `provenance:
human_accepted` (you said accept) or `auto` (you left null), the file itself `provenance:
human_filtered` plus the SHA-256 of your decisions file. Arms run from `filtered/` when it
exists, else from the auto files; a rejected slot is simply absent from both worker files
(per-frame box decode and video memory alike). If you want a rejected slot *replaced* (say
the held pipette under its right colour), note it: the replacement is a re-run of `select`
with the class list adjusted, not a hand-drawn box.

The default when you do nothing: every seed stays, `provenance: auto`, and the ledger says so.

## Where the numbers came from (so you can check a tile against them)

- Rule per slot (`seeds.md`, column `rule`): `moves` = the object's median box centre over one
  second differs from its median over the next by more than 20 px at 1920 (40 px in the head
  camera, measured against the head's own motion and only for classes some fixed camera also
  saw move or held); `in_hand` = its centre sat inside a `left_hand`/`right_hand` box on more
  than 10% of its detected frames; `container` = a named static volume, opened once; `group`
  = three or more identical instances standing in one rack.
- `det` = the detector's score on the seed frame (the plan's floor is 0.3; the in-hand
  pipettes are the weakest at 0.31-0.44, as the preflight predicted); `dec` = SAM3's own IoU
  prediction for its chosen candidate; `bbox IoU` = IoU between the mask's bounding box and
  the detector box, the acceptance test (>= 0.6, all 60 passed at 0.63-0.99; the fill ratio is
  reported in `seeds.json` and is not a test because the transparent plate fills its box to
  0.5 when correct).
- Start frames: 43 of the 60 slots start at the window start (frame 600, analysis frame 0);
  the rest start when the object first persists (`50ml_tube#0` in T2 at 2043, the held micro
  tube in T5 at 2817, ...), which is the frame the video-memory worker seeds them at.

# Gate 2 (soft), Sep 25: review anchors on trial 1, about 1.5 hours, no GPU

The `p6-anchors` todo of the FineBio 3D-tracking plan
([`plan-2026-09-25-finebio-3d-tracking.md`](plan-2026-09-25-finebio-3d-tracking.md)). **Nothing
waits for this sitting either.** The frames were chosen and the candidate masks decoded as soon
as arm (b) existed (`battle-finebio-anchors select` / `workspace`); the scoreboard
(`battle-finebio-anchors score`) runs on an empty record today, says how many cells are
labelled, and re-runs as more land. Labelling is optional: without it the arms are ranked on the
label-free measures only, and the ledger says so. Nothing is drawn: you pick one SAM3
image-decoder mask per cell (or say the object is hidden), the mask boundary is the decoder's,
the choice is yours. FineBio is non-commercial research data: sheets and masks stay under
`runs/`, never in the repository; the anchor config (frame numbers, slot labels) and your record
(states, hashes, identities) carry no pixels.

Standing claim boundary: the anchors are one person's choice among decoder masks on 18 frames of
two views plus one six-view frame of one trial. They rank arms (a)-(d) against each other and are
not ground truth, not a dataset, and support no accuracy claim; the detector that proposed every
box was trained on this lab's objects and cameras.

## What was chosen, and why (`configs/qa/finebio_P03_03_01_review_anchors.json`)

- **Views.** The head camera `fpv` and one fixed view, **`T4`**: it has the most individually
  tracked slots (11 non-group; T1 7 + 4 group slots, T2 10 + 1, T3 10 + 1, T5 9 + 2) and its
  plate slot has a mask on 97.1% of the window (det-box IoU median 0.906). Group slots (racks of
  tubes) have no single detector box and sit outside the disagreement pool, which is why T5, the
  top-down camera with two group slots, was not taken. Frame **916**, the one frame the FineBio
  authors annotated in every camera, is labelled in **all six views**.
- **18 frames** (raw frame; proxy frame = raw - 600):
  - `916` (proxy 316): six-view annotated, all six views.
  - **12 disagreement frames**, fpv + T4: the frames where arm (b)'s mask disagrees most with the
    detector box that prompted it (lowest `detector_box_iou` per frame, pooled over the non-group
    slots of both views), lowest first, at least 90 frames (3 s) apart from every other chosen
    frame, with the two centrifuge cycles' neighbourhoods ([1176, 1228) and [3224, 3311), padded
    by 90 frames on each side) served first until three were inside them:
    `1120` (T4 centrifuge 0.19, lid opening before the first spin), `1294` (T4 yellow pipette
    0.21, after the first spin), `3372` (T4 vortex 0.22, after the second spin) in the
    neighbourhoods; `1521` (fpv blue pipette 0.17), `1739` (fpv 8-channel pipette 0.13), `1966`
    (fpv red pipette 0.22), `2380` (fpv blue pipette 0.13), `2748` (T4 blue pipette 0.21), `3054`
    (fpv blue pipette 0.14), `3481` (fpv blue pipette 0.22), `3571` (T4 blue pipette 0.23), `4032`
    (T4 blue pipette 0.06) window-wide. The held blue pipette is the object the two models
    disagree on most, as the arms scoreboard already said.
  - **5 random frames**, fpv + T4, `numpy.default_rng(20260925)` uniform over the eligible frames
    (a SAM3 row in both views, valid fpv pose), same spacing: `637`, `1635`, `2132`, `2240`, `4122`.
- **Cells.** 40 frame-views x 11 slots = **440 cells**, of which **337 have candidates** (the slot
  had a detector box on that frame in arm (b)) and 103 have none (no box: the tube had not
  arrived or had left, the pipette was out of frame): those 103 take only `hidden` or
  `none_fits`.

## The workspace (`runs/finebio-anchors-P03_03_01-20260925/`)

Built by `battle-finebio-anchors workspace` (40 image encodes, 1011 decodes, 33 s on the GPU at
2.57 GiB peak, guard record in `results.json`). The Assembly101 calibration web workspace was
**not** generalised: it is bound to the Assembly101 clip configs and the four-part target policy,
and gate 2 follows the gate-1 pattern instead, static sheets plus a decisions JSON.

- `sheets/f<raw>_<view>.jpg`, one per frame-view (40 files, 1280 px wide): **one row per slot**,
  columns = the context crop with the reference box (arm (b)'s prompt, the detector box), then
  the candidates: **c0** = arm (b)'s own mask (tight box), **c1** = the 0.15-margin box, **c2** =
  the box plus a positive point (arm (b)'s mask centroid, else the box centre), **c3** = the tight
  box's second-ranked decoder output. A greyed tile captioned `= cK` is a duplicate (IoU >= 0.97)
  of an earlier candidate and needs no look; 556 of the 1011 alternates are duplicates, so a
  typical row has two or three distinct masks (67 rows have one, 144 two, 67 three, 59 four).
  Under each tile: `dec` = the decoder's own IoU prediction, `bbox IoU` = the mask's bounding box
  against the reference box, the area.
- `sheets/f<raw>_<view>_overview.jpg`: the whole frame with every slot's box and number, to see
  where a slot sits and which tube is which.
- `decisions.template.json`: one entry per cell (`raw_frame`, `proxy_frame`, `view`, `slot`,
  `label`, `class`, `candidates` = the indices that exist, `sheet`, `decision: null`,
  `instance_identity`, `note`). Copy it to `decisions.json`, fill it in, put your name in
  `author`.
- `workspace.json`, `requests.json`, `results.json`: the record of what was asked and decoded.

Viewing: the sheets are plain JPEGs, and the web workspace below serves the same tiles as pages
with one keypress per row. The static sheets stay as the fallback: if you want only them over
the tailnet the way the Assembly101 workspaces were served, a read-only static server on this
machine's Tailscale address is enough (nothing here needs a GPU):

```bash
cd /home/nick/src/battle/runs/finebio-anchors-P03_03_01-20260925
python3 -m http.server 8766 --bind "$(tailscale ip -4)" --directory sheets
# then http://<tailscale ip>:8766/ from any device on the tailnet; Ctrl-C when done
```

### Web workspace (`battle-finebio-anchors-web`, Sep 26)

Added because clicking through 40 sheets and hand-editing a 7,866-line JSON is the slow part.
The same workspace, served as pages; nothing decoded, no GPU, no browser opened by the tool:

```bash
cd /home/nick/src/battle
uv run battle-finebio-anchors-web --workspace runs/finebio-anchors-P03_03_01-20260925 --tailscale
# prints http://<tailscale ip>:8766/ (and the MagicDNS name); Ctrl-C when done
# loopback only: --bind 127.0.0.1 (the default); another record: --record <file>; --author <name>
```

- **Index**: the 40 frame-views in the order that pays (916 in all six views, then 1120 / 1294 /
  3372 in the centrifuge neighbourhoods, then the rest), each with its labelled / 11 count, and
  the total out of 440. The author field writes into the record.
- **Frame page**: the overview at the top (click or `o` to enlarge), then **one row per slot**:
  label and class, the reference-box crop, the candidate tiles c0..c3 cut from the sheet with
  their `dec` / `bbox IoU`; missing candidates are absent, duplicates greyed with their `= cK`;
  the current decision in yellow; an `instance_identity` field that autocompletes over every
  identity already in the record; a `note` field. The 103 rows without a box offer only
  `hidden` / `none_fits`.
- **Keys** (no modifier; when no text field has focus): `0`-`3` choose that candidate for the
  current row; `h` hidden, `b` box, `n` none_fits, `x` clear (back to null); `j` / `k` (or the
  arrows) move the current row; `]` / `[` next / previous frame-view in the index order (`]` on
  the last page returns to the index); **`a` accepts c0 for the current row and advances**, the
  fast path: when arm (b)'s own mask is right, a page is eleven presses of `a`; `i` focuses the
  identity field, Enter or Esc leaves it; `-` / `=` shrink / enlarge the tiles. Clicking a tile
  or a button does the same as its key.
- **Saving**: every change is written at once to `decisions.json` (a temporary file renamed into
  place; `author` and `updated_at` at the top) in the template's own schema, so `score` and
  `export` read it unchanged, and the template is never modified. The page updates in place and
  keeps its current row; the record is the only state, so stopping and restarting the server, or
  editing the JSON by hand in between, loses nothing.
- **Score now** (index and frame pages) runs `battle-finebio-anchors score` against the trial's
  four arms into `<workspace>/scoreboard/` (about 20 s) and shows the table inline with links to
  the `.md` / `.json`; the button is disabled while a run is in progress and shows the error if
  one fails.

## What to do per frame (about 1.5 h for 440 cells, most of them one glance)

Open the frame's page in the web workspace (or its sheet and overview side by side). Per row
(slot):

1. **Pick the candidate whose mask is the object the label names**: write its index (`0`-`3`)
   into `decision`. `0` is arm (b)'s mask; if it is right, `0` is the answer and takes a second.
   Prefer the mask that covers the whole visible object and nothing else: for the transparent
   plate the mask should be the plate's footprint (its fill is about half the box when right);
   for a held pipette the pipette, not the glove; for a tube in a rack the tube, not its
   neighbours. Ignore the colour word in a pipette label (the detector confuses blue / yellow /
   red); the question is whether the mask is on the object the box is on.
2. **`"hidden"`** when the object is not visible in this view on this frame (fully occluded, out
   of frame, inside the closed centrifuge). Any arm mask there counts as a false positive. This
   is the only answer besides `none_fits` for the 103 rows without a box.
3. **`"box"`** when the reference box is the object but no candidate mask is acceptable (the
   arms are then scored by the overlap of their mask's bounding box with that box).
4. **`"none_fits"`** when the object is visible, no candidate fits, and the box is wrong too. The
   cell is excluded and counted.
5. Leave `null` when you do not want to decide; it is skipped and counted as unlabelled.
6. **`instance_identity`**: a short name for the physical object, the same string wherever that
   object appears, across views on frame 916 and across frames (`tube_A`, `tube_B`,
   `pipette_blue`, ...). It is pre-filled for the bench's singletons (centrifuge, vortex, PCR
   machine, plate, trash can); fill it for the tubes and pipettes where you can tell, leave it
   null where you cannot (a tube in the rack you cannot tell from its neighbours). Group slots
   (`micro_tube_group#0`, a rack of tubes as one object) can carry the rack's name.
7. `note`: a word when useful ("mask on glove", "lid half open").

Order that pays: frame 916 first in all six views (the six-view identity is what cross-camera
IDF1 needs), then the three centrifuge-neighbourhood frames, then the rest. Stopping early is
fine: the scoreboard reports on whatever exists.

## What the scoreboard computes (`battle-finebio-anchors score`)

```bash
cd /home/nick/src/battle
uv run battle-finebio-anchors score \
  --workspace runs/finebio-anchors-P03_03_01-20260925 \
  --record runs/finebio-anchors-P03_03_01-20260925/decisions.json \
  --arms a=runs/finebio-arms-P03_03_01-20260925/a-boxes-only,b=runs/finebio-arms-P03_03_01-20260925/b-box-decode-arm,c=runs/finebio-arms-P03_03_01-20260925/c-video-memory-arm,d=runs/finebio-arms-P03_03_01-20260925/d-video-memory-arm \
  --output runs/finebio-anchors-P03_03_01-20260925/scoreboard
# the committed skeleton of your record (states, hashes, identities; no pixels):
uv run battle-finebio-anchors export \
  --workspace runs/finebio-anchors-P03_03_01-20260925 \
  --record runs/finebio-anchors-P03_03_01-20260925/decisions.json \
  --output docs/qa/finebio-P03_03_01-review-anchors.human-record.json
```

Per arm and labelled cell: **mask IoU** between the arm's mask (the same speckle rule as the
arms' measures) and the accepted candidate; **box IoU** between the arm's mask bounding box, or
its detector box for the boxes-only arm (a), and the candidate's bbox (the reference box for
`box` accepts); `missing` when the arm has no row for the cell; `hidden FP` when an arm has a mask
on a cell you marked hidden; per class, per view, per origin (six-view / disagreement / random)
and inside the centrifuge neighbourhoods. **Identity**: on every labelled cell with an
`instance_identity`, the tracker's 3D track id behind the arm's row is looked up
(`tracks.jsonl`, `support_slots`), and identity F1 is computed between your names and the track
ids (maximum one-to-one matching; IDF1 = 2 IDTP / (cells with a name + cells with a track)), on
frame 916 across the six cameras and pooled over every labelled frame, with the number of names
split across several ids, ids covering several names, and named cells the tracker has no track
for. The table's first line says how many of the 440 cells are labelled. Arm (b) scoring 1.0 on
every cell where you chose `0` is the built-in check that the scorer read the right masks.

Where the numbers on the tiles come from: the reference box is the FineBio DINO box arm (b) was
prompted with on that frame (its score under `det`); `dec` is SAM3's own IoU prediction for the
candidate; `bbox IoU` is the candidate's bounding box against the reference box, the same measure
the seeds were accepted on (>= 0.6) and the arms were compared on.
