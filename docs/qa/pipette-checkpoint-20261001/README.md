# Pipette tracking checkpoint — 2026-10-01

Saved at 2026-10-02T03:14:28.785677+00:00. Working tree before checkpoint: master, existing HEAD `aa417d3e742cead7be86427aeff1864e3313f62b`, one commit ahead of origin/master. Unrelated untracked `test.sh` was left alone. This checkpoint is local; it does not publish results or change production labels.

## Resume here

The point-prompt mini study is finished. The next useful experiment is validation of the three-click recipe on unseen views, followed by sequential corrective prompting with previous-mask feedback. Do not repeat the existing 133 decodes merely to recover context.

Read `point-prompt-study/README.md` in this checkpoint for methods and limitations, then open the local review:

`/home/nick/src/battle/runs/finebio-pipette-improvement-20260930/point-prompt-study/review.html`

No study inference is running. No study masks were promoted into the annotation set. The latest study is in HTML and charts; the existing Rerun recording remains the earlier every-100-frame refinement review.

## Where the thread stands

- Original Cursor plan: `/home/nick/.cursor/plans/orientation_vote_and_reprojection_prompting_0c532b17.plan.md`; conversation request ID `63af6df7-590a-449b-8118-234d1ad83609`. A snapshot is included in the local archive when present.
- Earlier implementation commits: `31701cf` (FineBio reprojection prompting), then `aa417d3` (supplemental reprojected-mask evidence). This checkpoint does not certify every original-plan criterion; retain the existing tracking policy until a validated improvement is demonstrated.
- Production tracking still starts at raw frame 600. Offline initialization/annotation experiments include raw zero-based frames 0 through 600 every 100. Sampling earlier frames did not change production start time.
- Prior blue-pipette refinement: all six views at seven sampled times, 42 images. Five passes, 130 decodes / 520 candidate masks; 26/42 approximate usable drafts, 16 still need review. This is a different qualitative rubric from the new study's visual grades.
- Rerun: `runs/finebio-pipette-improvement-20260930/every100-review/refinement.rrd` with `labels.rbl`; `refinement.rrd` and `labels.rrd` were hardlinks. Use the distinct refinement filename because the viewer cached the earlier labels filename. Viewer port was 9876. Recording ID `raw0-600-blue-refinement-five-passes`, application `finebio-every100-blue`, timeline `source_frame`.
- Current study: seven representative blue-pipette views from the same recording (0 T2/T1/FPV, 100 T2, 200 T4, 300 T3, 500 T1), 19 conditions, 133 independent decodes and 532 directly inspected candidates. Full original-image encoding at maximum side 1280, same box within each view except points-only conditions, no previous-mask feedback.

## Findings to preserve

Best tested default: good box + positive inside visible body + positive at exposed shaft centre + one targeted negative inside actual unwanted inclusion. Three points total, plus the box. Begin with a box or body click and stop earlier if already good.

Model-top masks graded mostly correct: box only 2/7; body only 3/7; body/shaft pair 4/7; body + one negative 5/7; body/shaft + one negative 6/7; five positives 2/7; eight positives 2/7. A better candidate often exists below SAM's top estimated-IoU score. Inspect candidate selection as well as prompt locations.

Points should be safely inside visible pipette pixels, away from boundaries, gloves, support tubes and uncertain transparent tips. Target negatives to nearby leakage; distant background clicks were less useful. Use cap/hook points only when clearly visible and missing. More points were not consistently better.

These are exploratory, unblinded judgments by one AI reviewer, without reference masks or measured IoU. Seven correlated, hard-biased views do not establish a universal optimum or tracking accuracy. Frame 100 T2 retains rack/shaft ambiguity; frame 500 T1 retains support-tube contamination; crowded frame 0 T1 has neighbouring-pipette flecks. A grade of 4 is not a production-ready label.

## Saved material and restoration

Versioned checkpoint files include study design, scripts, grades, report, validation and a SHA-256 inventory, plus the earlier selected-label metadata and export scripts. No dataset pixels are committed. Scripts are snapshots: restore them to their original run directories before executing, because they resolve dependencies relative to their location.

Local full archive:

`/home/nick/src/battle/runs/checkpoints/pipette-20261001/pipette-review-and-study.tar.gz`

It contains the complete `point-prompt-study`, `every100-review` and `frame-150-labeling` directories, including originals, all candidate masks, review pages, GIFs and Rerun files. `SHA256SUMS` alongside the archive verifies the archive; `artifact-inventory.json` verifies individual files. This is a local snapshot on the same disk, not an off-machine backup. The raw videos, model checkpoint and Python environments are dependencies outside this archive.

Restore the archive into an empty checkout or recovery directory using `tar -xzf ARCHIVE -C DESTINATION`. Existing run directories already remain intact. Do not overwrite newer work blindly.

Model: `/home/nick/src/muggled_sam/model_weights/sam3.1_multiplex.pt`; model Python: `/home/nick/.pyenv/versions/muggled_sam/bin/python`; project Python: `.venv/bin/python`. The model environment runs inference; the project environment builds charts and Rerun exports. Keep the muggled_sam checkout available. `frame-150-labeling/iterate.py` is the study's imported worker.

## Suggested continuation

1. Select unseen frames/cameras and pipette types; compare box/body, body-plus-negative, and body/shaft-plus-negative under the same candidate-selection policy.
2. Establish independently reviewed reference masks, or at least fixed visible-body/shaft and leakage judgments before comparing recipes.
3. Test sequential correction with previous-mask logits separately from simultaneous point sets. The current result does not evaluate that workflow.
4. Resolve rack and support-tube failures before adopting an automatic initialization policy.
5. Evaluate temporal propagation and downstream tracking after mask quality is established. Keep the production frame-600 start and current policy until validated.
