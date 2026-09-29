# Writing style for the live pages

The live pages are for a technical startup founder with five minutes: `README.md`, `docs/story.md`,
`docs/results.md`, `docs/pipeline.md` and `docs/README.md`. They say what works, what it was measured
against and where the evidence sits, in that order. The archive under `docs/archive/` is the append-only
lab record, and its prose stays exactly as written.

## The ten rules

1. **Result, evidence, caveat, in that order.** The first sentence of every section is the verdict.
2. **One idea per sentence.** Aim under 25 words on average. No semicolon chains. Parentheses only for a unit or a path.
3. **Numbers carry their meaning.** Write "track births fell 40% (308 to 186)", never a bare slash-list.
   Three or more numbers become a table with a "measured against" column.
4. **Provenance lives at the end.** One `Evidence:` line per section names the run directory and the archive entry.
   No commit hashes, ledger anchors or dates inside sentences.
5. **Plain names, defined once.** The glossary below maps the lab's words to plain ones. After the first use, write the plain name.
6. **Named actors, active voice.** Write "I", "the tracker", "SAM3", "the detector". The one person who drew the seeds and held both gates is "I". Never "the human", "the user" or "the agent".
7. **Headings are topics.** Dates appear only in `docs/story.md`, where the date is the topic.
8. **Caveats once.** The claim boundaries live in one section of the README and one of `docs/results.md`.
   A number elsewhere gets at most one clause, such as "against the detector's box, not ground truth".
9. **Bold is for verdicts.** Code font is for commands, paths, flags and identifiers. Nothing else is styled.
10. **Every page opens with what it tells you.** Two or three sentences. Every result says why it matters to someone building on it.

## Glossary

| The lab says | Plain words |
|---|---|
| arm (a) | boxes only: the detector's boxes tracked with no masks, the baseline |
| arm (b) | per-frame box decode: SAM3 draws a fresh mask from the detector's box on every frame, with no memory. The default arm |
| arm (c) | SAM3 video memory: SAM3 tracks one seed box across frames, remembering what it saw |
| arm (d) | video memory plus re-seeding from the detector when the mask drifts |
| slot | one object in one camera |
| seed | the detector box that starts a slot's mask |
| gate 1 | human review of the seeds: each seed box accepted or rejected |
| gate 2 | human review of the masks: on a sample of frames, which mask is the object, or the box is wrong |
| anchor | one human-checked mask on one frame in one camera. The anchors are the yardstick the arms are scored against |
| LOO | leave-one-camera-out residual: drop one camera, triangulate from the rest, measure the error in the dropped one in pixels |
| det-box IoU | overlap between a mask and the detector's box. Model against model, not accuracy |
| fpv | the head camera |
| T1-T5 | the five fixed cameras |
| with-plate | the seed set that includes the cell culture plate (66 slots, not 60) |
| pm-append | SAM3's "append prompts to memory" setting, the best Assembly101 arm |
| proxy | the working copy of a camera's video that every stage reads |
| rig | the six cameras solved into one world frame, with the pixel gates derived from them |
| coasting | a track kept alive on prediction while no camera sees it |
| held | a track whose object is in a hand |
| contained | a track whose object is inside a container, such as the centrifuge |
| storyboard | the bookmarked frames in a recording that walk through the trial |
| preset | a saved viewer layout: World, Cameras or Evidence |
| tracks / tracks-ext | the core tracker's output / the same tracker with the four extensions (motion model, containers, groups, held objects) |
| trial 1 | recording `P03_03_01`, room 1. Both human gates were held here |
| trial 2 | recording `P20_03_01`, room 2. The same build with nothing tuned and no human gate |
| DINO | FineBio's shipped detector, trained on these cameras. It supplies every box |
| DDETR | FineBio's second shipped detector, used only to flag disagreement with DINO |
| MuggledSAM / SAM3.1 | the wrapper I run / the mask model inside it |
| video memory vs per-frame box decode | the two ways SAM3 makes masks. Memory carries a mask across frames (arm (c)). Box decode starts fresh from the box on each frame (arm (b)) |
| negative control | the same checks run on a camera pose known to be wrong, to show the checks can fail |
| claim boundary | what a number is measured against, and what it does not show |

## Before and after

From the scorecard row on multi-camera tracking.

> Before: "Done with its limits named. Per-trial camera solve (camera 6 re-solved in room 1, cameras 1-4 in room 2, 0.7-2.1 px; no view dropped), rig gates as formulas (association 30.1 / 27.5 px), the seven cross-checks and a negative control on every build; a new 3D tracker (birth by triangulation, predict-project-gate, coasting / lost, confirmed re-acquisition, hand-off re-seed) plus four extensions on inventory evidence: tracks born 285 / 308 / 270 -> 191 / 186 / 147 and ambiguities 126 / 158 / 133 -> 33 / 54 / 22 on trial 1 [...] **the in-hand pipette fragments in both rooms** (41 / 59 / 20 and 24 / 48 / 21 ids with the motion model) [...]"

> After: "**Every static object on the bench gets one 3D identity, in both rooms.** Six cameras are solved into one world frame to within 2 px, and any object two cameras see is triangulated and tracked. Four tracker extensions (motion model, containers, groups, held objects) cut spurious track births by 40% (308 to 186) and identity ambiguities by two thirds (158 to 54). A tube keeps its identity through a closed centrifuge lid. The one failure is the pipette in the hand, which splits into about 60 identities over two minutes; I read that as an observation problem, not a tracking one. Evidence: `runs/finebio-arms-P03_03_01-filtered-20260927/scoreboard/`, archive entry 'p3-tracker-ext'."

## Checks

A test applies two mechanical checks to every live page. No sentence may run over 60 words. None of the
strings "the human", "the user" or "the agent" may appear. This page is the one exception, because rule 6
and the before example have to quote what the other pages may not contain.
