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
