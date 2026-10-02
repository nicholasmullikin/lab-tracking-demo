# Multiplex pipette study

Saved T2 configuration recovered and replayed: 30 frames, 240/240 pixel-identical masks with unchanged slot IDs. These identities reproduce the old run; they do not independently establish physical identity.

Reviewed native raw-0 seeds are frozen in `runs/pipette-multiplex-20261002/seeds/frozen/`; blue=0, yellow=1, red=2, multichannel=3. Three decoding rounds and their rejected candidates are preserved. Initial blue geometry uses six camera axes. Multichannel shaft direction remains ambiguous in head-dominated/occluded views.

The first 300-native-frame (10.01-second) recording is complete and directly reviewed. All four labeled pipettes have mask observations and extracted 2D axes where fits are available; only blue has 3D geometry and a conservative dispensing-end vote. Both ends were inspected in an actual Rerun viewer capped at 8 GB. No production defaults have changed.

Blue follows pickup, while yellow/red/multichannel remain on their resting objects. FPV glove leakage at raw 200 corrupts blue axis/geometry; raw 200 direction is unresolved, and raw 299 projections overshoot/misalign. Fit coverage must not be read as correct identity or geometry. Multichannel shaft direction remains unresolved in head-dominated or occluded observations.

Open the recording with:

```bash
.venv/bin/rerun --memory-limit 8GB runs/pipette-multiplex-20261002/baseline/clips/raw-000000-000299-labeled.rrd runs/pipette-multiplex-20261002/baseline/clips/raw-000000-000299.rbl
```

[Protocol](protocol.json), [frozen config](config.json), [seed review](seed-review.json), [10-second review](review-300.json), [replay summary](replay-summary.json).

The 900-frame (30.03-second) stage is complete and reviewed. T1 blue drifts onto a resting shaft fragment at raw 899; T4 blue includes forearm pixels. The 150-second extension is in progress, resuming compatible checkpoints without corrective prompts. All 24 raw-299 resumed masks are pixel-identical to their saved stage boundary.

[30-second review](review-900.json). Three final native-rate 10-second clips (raw 0–299, 300–599 and 600–899) are verified and inspected at both ends in a fresh 8 GB viewer. Absent masks use explicit transparent layers; earlier exports are archived.

Published JSON paths are normalized relative to the repository (or the sibling model checkout). Original snapshots and their recorded hashes remain under the local run.
