# Blue pipette initialization sweep and refinement

Raw zero-based frames 0,100,200,300,400,500,600 in all six cameras. Blue only, physical_id blue; frame 0 resting, later held. The legacy candidate filenames say blue-held even at 0.

Five passes: {'1': 42, '2': 14, '3': 27, '4': 27, '5': 20}; 130 decodes / 520 candidate masks. Retried all 27 previously flagged images in passes 3–5: 74 decodes / 296 new candidates. Pass 3 used cropped box plus points; pass 4 cropped points only; pass 5 whole-image 1280 encoding with corrected positives within the visually reviewed main pipette mask. Corrected misplaced prompt points after checking enlarged originals. All candidates and all 27 original/before/after comparisons directly inspected.

Approximate usable drafts: 15→26/42; 16 still need review. Qualitative visible-pixel judgment, not reference labels or measured tracking accuracy. Support-tube separation, neighbouring pipettes, gloves and thin shafts remain difficult. Earlier selections retained where new masks regressed. Explicit selections and notes are in selections.json, with full provenance and counts in labels.json. Binary masks remain source-sized.

No temporal propagation or production start change: existing tracking starts at 600. Frame 0, especially T2–T5, remains the best tested initialization candidate. before-refinement/ archives the prior review and selections. GIFs contain sparse examples, not continuous video.

Reproduce with model-environment Python: extract.py; run.py; run.py round2; run.py round3; run.py round4; run.py round5; export.py. Project .venv Python: export_rerun.py. Review review.html, refinement-examples.gif and the refinement-before-after.jpg sheets under each frame.
