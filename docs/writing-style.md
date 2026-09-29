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
| IoU | intersection over union: the overlap of two masks or boxes as a fraction of the area they cover together, 1.0 when identical |
| PnP | perspective-n-point: solving a camera's pose from the printed markers' known corners |
| IDF1 | identity F1: how often a tracker gives one object the same id across cells, 1.0 when it never splits or merges |
| AUROC | area under the ROC curve: how well a score ranks failures above successes, 0.5 for chance |
| p90 | the 90th percentile: the value nine in ten measurements sit under |
| MuggledSAM / SAM3.1 | the wrapper I run / the mask model inside it |
| video memory vs per-frame box decode | the two ways SAM3 makes masks. Memory carries a mask across frames (arm (c)). Box decode starts fresh from the box on each frame (arm (b)) |
| negative control | the same checks run on a camera pose known to be wrong, to show the checks can fail |
| claim boundary | what a number is measured against, and what it does not show |

## House style: The Economist Style Guide, 11th edition

The live pages follow The Economist Style Guide (11th edition, 2015) wherever it goes beyond the
ten rules above. The rules below are paraphrased from the guide's entries, named in brackets, and
are the ones that changed something on these pages. One deliberate departure: I am American and
the pages keep American spelling (color, labeled, center, license, synchronization) and American
compound forms (setup, email) where the guide prescribes British ones. Everything else applies.

### Words

1. [short words] Use the short word: about, after, before, but, enough, let, show, use, set up,
   take part. Not approximately, following, prior to, however, sufficient, permit, demonstrate,
   utilize, establish, participate.
2. [unnecessary words] Cut every word that can go. Test "very" by leaving it out. Currently,
   actually and really seldom earn a place. "The fact that" is "that". Shed the preposition after
   a verb: cut, not cut back; meet, not meet with.
3. [active or passive] Use the active voice with a named actor: "I picked", "the tracker lost",
   not "was picked", "was lost".
4. [jargon] Where an everyday word exists, use it. A technical term that must stay is explained
   once, in the glossary or in brackets on first use: PnP (perspective-n-point).
5. [abbreviations] Spell a term out on first appearance unless the short form is the better
   known one (DNA, IoU). Unit abbreviations are lower case (px, fps, kg).
6. [nouns as verbs] Do not verb nouns or coin verbs: no "productise", "disposition", "impact",
   "leverage", "source", "action", "trial". Turning something into a product is not productising
   it.
7. [metrics] Metrics is the theory of measurement. The numbers themselves are measures.
8. [key] Key is a noun and an overused one. Not "key numbers", never "the choice is key".
9. [horrible words; clichés] No ongoing, upcoming, proactive, facilitate, showcase, "grow the
   business", "likely" for probably, silver bullets or green lights.
10. [compare] Compared with, to mark a difference. Compared to only to claim a likeness.
11. [identical with] Not identical to.
12. [different from] Not different to or than.
13. [fewer, less] Fewer for things counted, less for quantities and proportions.
14. [data, media] Both are plural. Plural nouns take plural verbs.
15. [none] None takes a singular verb.
16. [due to] Only after the noun it modifies; otherwise because of or owing to.
17. [while] Temporal only, not a substitute for although.
18. [Latin] No Latin where English serves: through, not via; such as, not e.g.; against, not vs;
    a year, not per annum.
19. [only] Put only next to the word it qualifies.

### Figures

20. [figures] Words for one to ten, figures from 11. Exceptions that stay figures: percentages
    (4%), a number with a unit (94 px, 2.3 h), a decimal (0.928), and a set of numbers of which
    one exceeds ten (52, 6 and 2). Never start a sentence with a figure. The same rule for
    ordinals: the 11th slot, the first minute.
21. [fractions] Hyphenated and spelled out: two-thirds, one-half.
22. [per cent, percentage points] Use % in figures; write percentage in full. The gap between
    99.1% and 79.3% is 20 percentage points, named as such, not "20 points" and not "20%".
23. [ranges] Figures in a range take an en rule with no spaces: 0.7–2.1 px, 5–6%. In words, "to".
    Never "from 1947–50" or "between 1961–65": from 1947 to 1950, between 1961 and 1965.
24. [million] m for million (47.7m parameters); billion and trillion in words.
25. [dates] Month, day, year, no commas: Sep 25 2026. Do not open a sentence with a date unless
    the date is the point.
26. [thousands] Counts of four digits and more take a comma: 3,600 frames, 148,362 masks. Frame
    indices, years and size labels are labels and do not: raw 3031, 1280 px.
27. [ratios and dimensions] Spell out "to" in a ratio, and use × for a dimension: 13 × 4, 4×4.

### Hyphens

28. [hyphens: compound adjectives] Two or more words acting as one adjective before a noun take
    hyphens: six-camera tracker, five-minute version, top-down view, wet-lab bench, per-frame
    decode, label-free.
29. [hyphens: adverbs] An adverb is not hyphenated to what it modifies, especially one ending in
    -ly: separately managed, nearly redundant, side by side.
30. [hyphens: prefixes] No hyphen after a short prefix (rearrange, reopen) unless it avoids a
    confusion: re-solve (solve again), re-run (beside the tool named Rerun), non-commercial.
31. [hyphens: numbers] A unit abbreviation used adjectivally takes no hyphen (a 20 s clip, a
    120 s window); a unit written as a word does (the 300-frame clip, a 52-frame closure).

### Punctuation

32. [commas in lists] No comma before the final "and" or "or" in a list unless an item itself
    contains an "and".
33. [semi-colons] A pause longer than a comma, shorter than a full stop, and not many of them.
    After a colon, only where commas would not separate the items.
34. [dashes] At most one pair a sentence, for a parenthesis. Never a dash where a colon, a comma
    or a full stop serves. The README uses none.
35. [full stops] Use plenty. They keep sentences short. None at the end of a heading.
36. [inverted commas] Double quotation marks; single only inside a quotation. Punctuation goes
    inside the closing mark only when it belongs to the quoted words. No quotation marks around a
    term that is not a quotation: an archive entry is named "p6-anchors", a column is described,
    not quoted.
37. [brackets] A whole sentence inside brackets keeps its full stop inside.
38. [italics] Only for foreign words not yet anglicized and newspaper titles. Never for emphasis
    and never in a heading.
39. [capitals] Lower case unless it looks absurd. Proper names take capitals (Git, Rerun,
    FineBio); informal names and common nouns do not (gate 1, the detector, the founder).
40. [headings] Sentence case, no full stop, no pun or catchphrase, and no figure at the start.

## Before and after

From the scorecard row on multi-camera tracking.

> Before: "Done with its limits named. Per-trial camera solve (camera 6 re-solved in room 1, cameras 1-4 in room 2, 0.7-2.1 px; no view dropped), rig gates as formulas (association 30.1 / 27.5 px), the seven cross-checks and a negative control on every build; a new 3D tracker (birth by triangulation, predict-project-gate, coasting / lost, confirmed re-acquisition, hand-off re-seed) plus four extensions on inventory evidence: tracks born 285 / 308 / 270 -> 191 / 186 / 147 and ambiguities 126 / 158 / 133 -> 33 / 54 / 22 on trial 1 [...] **the in-hand pipette fragments in both rooms** (41 / 59 / 20 and 24 / 48 / 21 ids with the motion model) [...]"

> After: "**Every static object on the bench gets one 3D identity, in both rooms.** Six cameras are solved into one world frame to within 2 px, and any object two cameras see is triangulated and tracked. Four tracker extensions (motion model, containers, groups, held objects) cut spurious track births by 40% (308 to 186) and identity ambiguities by two thirds (158 to 54). A tube keeps its identity through a closed centrifuge lid. The one failure is the pipette in the hand, which splits into about 60 identities over two minutes; I read that as an observation problem, not a tracking one. Evidence: `runs/finebio-arms-P03_03_01-filtered-20260927/scoreboard/`, archive entry 'p3-tracker-ext'."

## Checks

A test applies two mechanical checks to every live page. No sentence may run over 60 words. None of the
strings "the human", "the user" or "the agent" may appear. This page is the one exception, because rule 6
and the before example have to quote what the other pages may not contain.
