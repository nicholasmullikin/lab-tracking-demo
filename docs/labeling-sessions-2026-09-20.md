# Labeling sessions prepared on Sep 20 (nothing started, nothing labelled)

Four sessions for the human, each with the exact command, what to label, and the vocabulary.
Everything below was prepared on CPU; no server is running and no GPU decode has happened.
The calibration workspace decodes on the GPU when you label, so start a session only when the
GPU queue is idle (`nvidia-smi` shows nothing but the idle 1.2 GB). Standing rule, unchanged:
anchor masks are review evidence for ranking arms against each other on a handful of frames;
they are not a dataset, not ground truth, and support no accuracy claim. CC BY-NC 4.0 applies to
every frame and to everything derived from it.

Vocabulary the workspace records, per frame x part:

- **accepted mask** (`labeled`): you drew a box (plus optional `F`/`N` clicks), the decoder
  returned candidates, you accepted exactly one with `1`-`4`. The boundary is the decoder's,
  the choice is yours.
- **hidden**: the part has no visible surface on this frame (occluded, out of frame, or inside
  another part). Press the small **hidden** button in the part's status cell; draw nothing.
- **distractor**: not a workspace button. When the visible thing a tracker would latch onto is
  *not* the part (the yellow screwdriver next to the rear body at C10379 frame 1700), mark the
  part **hidden** and note the distractor in the config's `provenance.notes` or in the record
  after export; the scorer then counts any run mask there as a hidden false positive.
- Do not click **Finalize tracking plan**, do not mark eligibility or correction keyframes.

Per-frame procedure (identical to the Sep 19 C10379 session, ~4-5 minutes per frame):
`runs/human-review-anchors-first-minute/README.md` section 2.

## (a) C10119 anchors, 26 frames (top-down camera; 0.3 px consensus error)

Workspace prepared by `battle-anchor-export prepare --view static-c10119` at
`runs/human-review-anchors-first-minute-static-c10119/` from
`configs/qa/first_minute_review_anchors_static_c10119.json` (26 frames: the 13 C10379 anchors
mapped through the clock rules with shift +1, C10119 frames 301, 371, 401, 601, 651, 701, 901,
1051, 1101, 1151, 1201, 1501, 1701; 8 detector-selected extras 221, 481, 501, 551, 1541, 1651,
1731, 1751; 5 random extras 41, 81, 861, 1411, 1771). Start it, headless, on the tailnet:

```bash
cd /home/nick/src/battle
uv run battle-muggled-calibration-web \
  --config configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_all_static_g2.json \
  --view static-c10119 \
  --manual-seed-target-config configs/muggledsam_static_four_part_reassembly_focused_manual_seed_static_c10119.json \
  --output-dir runs/human-review-anchors-first-minute-static-c10119 --resume --tailscale
# prints http://100.64.0.7:8765/ and http://host.example.ts.net:8765/; open either on the
# tailnet device. No login inside the tailnet; stop the server when done.
# afterwards:
uv run battle-anchor-export export --view static-c10119
uv run battle-anchor-iou --view static-c10119 \
  --run runs/sam3-views-r1280-20260919/views/C10119 \
  --run seeded=runs/sam3-views-r1280-pmappend-seeded-20260920/views/C10119 \
  --output runs/anchor-scoreboard-c10119-20260920/anchor_iou.json
```

What to label. On each of the 26 frames accept one mask for **chassis**, **rear_body** and
**cabin** wherever the part has a visible surface, and for the **interior** whenever you can see
it (from about frame 323 it is attached and seen through the cabin from C10379; the top-down view
may show it more or less than that; if you cannot see it, mark it hidden rather than guessing).
The 13 mapped frames score the agent-seeded C10119 run against the human-corrected C10379 run on
the same pose instants; the 8 detector-selected frames (220 interior, 1750 / 1540 / 1730
rear_body, 480 / 550 chassis, 500 / 1650 interior on the C10379 clock) test whether the label-free
detectors' most suspicious frames are failures; the 5 random frames test the detectors rather than
confirm them. Use hidden for the interior before it is placed and for any part under a hand;
use hidden plus a note when a distractor (screwdriver, another yellow part) is what is visible
where the part should be. The workspace header counts `n/104 done`.

## (b) e4 (HMC_21179183) anchors and the human interior seed

Two workspaces. The anchor workspace is prepared
(`runs/human-review-anchors-first-minute-ego-hmc21179183/`, config
`configs/qa/first_minute_review_anchors_ego_hmc21179183.json`, 26 frames at shift +4: 304, 374,
404, 604, 654, 704, 904, 1054, 1104, 1154, 1204, 1504, 1704 mapped; extras 224, 484, 504, 554,
1544, 1654, 1734, 1754; random 44, 84, 864, 1414, 1774):

```bash
cd /home/nick/src/battle
uv run battle-muggled-calibration-web \
  --config configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_ego_e4_g2.json \
  --view ego-hmc21179183 \
  --manual-seed-target-config configs/muggledsam_ego_four_part_reassembly_focused_manual_seed_ego_hmc21179183.json \
  --output-dir runs/human-review-anchors-first-minute-ego-hmc21179183 --resume --tailscale
# afterwards:
uv run battle-anchor-export export --view ego-hmc21179183
uv run battle-anchor-iou --view ego-hmc21179183 \
  --run runs/sam3-views-r1280-20260919/views/HMC_21179183 \
  --output runs/anchor-scoreboard-e4-20260920/anchor_iou.json
```

The interior seed is a separate one-off calibration workspace (not an anchor set): e4 frame 4 is
the same pose instant as C10379 frame 0 (shift +4), and e4 frame 430 is the interior "rest" frame
the seed search found (hands clear of the chassis by 40 mm, C10379 frame 426). Both timestamps in
one workspace:

```bash
uv run battle-muggled-calibration-web \
  --config configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_ego_e4_g2.json \
  --view ego-hmc21179183 \
  --manual-seed-target-config configs/muggledsam_ego_four_part_reassembly_focused_manual_seed_ego_hmc21179183.json \
  --timestamps 0.133333,14.333333 \
  --output-dir runs/human-e4-interior-seed-20260920 --tailscale
```

What to label. Anchor workspace: the same four parts as (a) on the 26 frames, with the ego
camera's own caveats: the head moves, parts leave the frame (hidden when out of frame), and the
video is monochrome, so the black chassis against the grey interior is the hard pair; mark hidden
rather than accept a mask you cannot verify. Seed workspace: on frame 4 accept one mask for each
of the four parts that is in view (this is the human e4 seed the plan keeps as the comparison
arm), and on frame 430 accept the **interior** (and the other parts if convenient). The automatic
interior candidates the seed search proposed at e4 frame 430 are in session (c); labelling the
interior here by hand is what the automatic seed will be compared against (agreement within 0.02
IoU on the same views would make the automatic seed the run condition).

## (c) Seed-search proposals: 24 accept/reject decisions (about ten minutes)

`battle-seed-proposal-sheets` rendered one contact sheet per view under
`runs/labeling-sessions-20260920/proposal_sheets/<VIEW>.png` (C10095, C10115, C10118, C10119,
C10390, C10395, C10404, HMC_21179183; 0.8-1.3 MB each). Each sheet has three rows, one per
proposal cell (chassis at frame 0, rear_body at frame 0, interior at the rest frame 427-430):
the full frame with every candidate outlined and the crop box, then one zoomed crop per candidate
with that candidate filled, its strategy, area and the decoder's own IoU estimate. The same
outlines are drawn on the camera tiles of the v6 recording at those frames
(`multiview.rbl`, label `proposal: accept/reject pending`), if you prefer to judge them with the
video around them.

Record decisions in `configs/qa/seed_proposal_decisions.template.json` (copy it to
`configs/qa/seed_proposal_decisions.json`, fill `author`, `reviewed_at`, and per cell
`decision` (`accept` / `reject` / `unsure`) with `accepted_candidate` = the index of the one
candidate that is the part and only the part, else null). The agent then writes each decision
into the cell's `proposal.json` (`human_decision`) and reports the acceptance rate per view and
part, which is the number the seeding plan asked for. Reading aid: chassis and interior are
proposals because their C10379 held-out IoU failed the 0.6 gate (0.525 / 0.465), not because the
candidates looked wrong; rear_body cells are the accepted B3 seeds shown for confirmation
(`decision_for_b3: accepted_seed` in the row header); cabin never disagreed across strategies and
has no proposal.

## (d) Recording 2 (`nusar_9061`), one view

Being prepared by the GPU worker. The expected config is
`configs/qa/nusar_9061_review_anchors_c10379.json`; it did **not** exist when this document was
written, so no command is given here. When it lands, the session is the same as (a) with
`--recording nusar_9061`'s clip config
(`configs/clips/assembly101_nusar_9061_four_part_reassembly_focused_all_static_g2.json`) and its
per-view target policy, and the frames should be chosen from the contact sheet
`data/derived/assembly101/nusar-2021_action_both_9061-c02a_9061_user_id_2021-02-09_141537/contact_sheet_374.000-454.000.png`
(the four parts lie apart on the table only from proxy frame ~75 to ~420; a fifth part, the rear
bumper, is handled inside the window and is not a target).

## Order and time

(c) first (ten minutes, no GPU); then (a) (about 2 h at 4-5 minutes per frame, GPU decodes on
demand); then (b) (about 2 h plus ten minutes for the seed workspace); (d) when its config exists.
After each anchor session run the `export` and `anchor-iou` commands above; the scoreboards go
beside `runs/anchor-scoreboard-20260919/` and the ledger gets the numbers with their counts.
