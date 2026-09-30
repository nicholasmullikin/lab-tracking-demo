# Pipette tracking improvement, 2026-09-30

Requested: inspect the actual Rerun outputs, plan improvements, review the plan twice, implement and evaluate overnight. Preserve the earlier Cursor plan and frozen evaluation anchors.

## Evidence before changes

The adopted orientation-vote baseline v5a has P03 clean held-tip median 4.271 cm, 47/48 matched anchors and one wrong-end case. Reprojection v5 worsens to 10.981 cm, seven wrong ends, and 40 active-class identities instead of 21. P20 identities rise 20 to 26. Added class-mask coverage is not support for the same physical pipette. Accepted examples include forearms and identical resting masks requested under different colours. Direct endpoint panels show raw 2409 T2 reversing the endpoint on a resting pipette and raw 3066 T2 leaving its shaft. These examples also expose that tracker `held` is a geometric state, not verified human-held identity.

Rerun review: completed with the verified six-camera recording and a focused three-camera before/after blueprint. See `rerun_review.json` and settled screenshots. Review raw 1204, 1799, 1969, 2409, 2416, 3066, 4033 and 4068; include resting and moving intervals rather than only worst errors. Record screenshots and observations, and distinguish original background masks from v5 track projections.

## Proposed implementation

1. Add an opt-in `--line-reprojection-policy supplemental` to the tracker. Keep legacy behavior reproducible and adopt no experiment automatically.
2. Treat reprojection masks as supplemental evidence, not independent detections: retain original detector fallback, associate originals first, forbid new births from supplemental masks, and never use prompt labels as new colour votes.
3. Hard-reject supplemental masks marked as disagreeing with the prompting line. Require a current, same-class line track, projected axis agreement within the existing angle gate, broadside visibility, interval overlap, and a unique plausible recipient. Do not equate source-track ids across reruns.
4. Supply supplemental masks only for missing axis views, with one independent primary observation on the recipient track. Prefer an existing original mask. Reject ambiguous attribution instead of forcing a match. Add reason counters to identity metrics.
5. Test the failure mechanisms on the real camera fixture geometry: detector fallback, negative controls, no synthetic births, crossed classes, ambiguous recipients, and successful extra-view recovery. Verify no-reprojection input is byte-identical under the new policy.
6. Run locked ablations in both rooms: supplemental policy alone, and policy plus steep single-view prior. Reuse the existing decoded masks and colour cache; no new labels, training or costly segmentation passes. Run the clean fixed anchor scorer and frozen held-support scorer, report both numerical deltas and sample counts. Compare v5a and v5 as fixed references.
7. Directly inspect the winner's projections at the same frames and all clean held anchors. Produce an updated Rerun comparison and a readout. Adopt only with lower held-tip error, no matched-anchor loss, no increase in wrong ends, fewer identities than v5 and no >1 px camera-consistency regression in either room. Coverage may not be sacrificed below v5a; report failures honestly.

## Review 1: causal and identity review

Reviewed by the implementing agent before code changes. The proposed hard rejection alone is insufficient: accepted masks at raw 2416 under three labels already agree geometrically with a resting shaft. Track ids from v5a are unsafe in a rerun with different births. Revised the design to require a unique current same-class recipient with independent support, perform original association first, and forbid supplemental births. A detector observation in a missing-mask camera must survive source selection; otherwise even a rejected supplement changes the baseline. Prompted colour labels must not contribute to identity votes. Masks that fit several tracks must abstain even if one recipient already has support; assignment availability is not identity evidence. No claim of physical identity comes from these constraints alone.

## Review 2: evaluation and execution review

Reviewed again by the implementing agent before code changes, against the scorer and source selection code. Frozen `held` frames are the baseline tracker's geometric state; several displayed anchors lie on resting pipettes. Keep that label and its limitations rather than silently redefining the denominator. Evaluate all 48 clean anchors, paired matched cells, wrong ends and rest/held subsets, not just survivor medians. Preserve original/cache hashes and run the identical policy in P20 without tuning. Policy-only and policy-plus-prior isolate whether the gravity prior caused damage. No-reprojection output must be byte-identical; explicit baseline replay guards inadvertent source changes. Both negative controls and synthetic multi-instance ambiguity tests must be included. Fix gates before viewing candidate scores; do not search thresholds on the tip clicks. Keep v5a as the adopted method if the policy fails any gate. Direct Rerun inspection and same-frame endpoint panels are required before the final verdict.

## Deferred follow-up

If conservative supplementation cannot recover real held shafts, the next experiment is object-specific video mask propagation with independent appearance matching and hand-relative motion. It requires a new prompt/propagation experiment, not tuning the current clean tip anchors. Do not claim current colour labels prove identity.

## Implementation audit

Implemented the policy and 13 mechanism tests. An upgraded detector remains consumed before the birth pass; prompted slots are excluded from the SAM3 identity proxy. Baseline replay tracks, oriented tracks, events and residuals are byte-identical in both rooms. Fixed protocols are in `locked_protocol.json`; full comparison and control commands are stored beside each result. No masks or click labels were regenerated.

## Completion and verdict

All seven implementation/evaluation steps are complete. The updated Rerun recording has been opened and inspected. Both rooms ran the locked variants and paired 30-frame normal/shifted controls; input hashes and baseline byte comparisons passed. All 13 baseline-matched held-anchor panels were directly inspected.

Neither variant passes the adoption clauses. The prior variant cuts the rejected v5 held-tip median from 10.981 to 5.152 cm, but the accepted v5a baseline is 4.271 cm. Both candidates lose the held-state match at raw 3066, slot 0. On the same 12 matched held anchors, v5a is 3.734 cm against 5.152 cm for either candidate. The prior variant has two wrong ends against one for v5a. Its P20 support reaches 50.0%, but P03 stays at 10.4%. Default tracking and the adopted v5a method remain unchanged; the policy is explicit and experimental.

Tests: 13 new mechanism tests; 1,235 suite tests passed, 14 skipped and 73 deselected. The unrelated remote archive probe was excluded after stalling. Ruff passed. Evidence and full tables: `runs/finebio-pipette-improvement-20260930/readout.md` and `readout.json`. Open `comparison.rrd` with `comparison.rbl` to review scored before/after endpoints on the original camera videos.
