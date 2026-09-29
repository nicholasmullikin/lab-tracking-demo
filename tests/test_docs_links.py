"""Link, media and style checks on the documentation pages.

Every relative link and image path on the root README, the live `docs/*.md` pages, the
`docs/archive/` and `docs/qa/` indexes and the script and fixture READMEs must resolve to a
file, and a `#fragment` must name a heading in the target page under GitHub's anchor rules.
Every `media/story/` path the story and the README show must exist, and every output the
renderer's manifest lists must appear in the story.  The live pages also obey two rules of
`docs/writing-style.md`: no sentence over 60 words and none of the actor names the style guide
forbids.  Text is read once per page; no network.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import cache
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
STORY = DOCS / "story.md"
README = ROOT / "README.md"
MANIFEST = ROOT / "media" / "story" / "manifest.json"

LINK_PAGES = sorted(
    [README, *DOCS.glob("*.md"), DOCS / "archive" / "README.md", DOCS / "qa" / "README.md"]
    + [*ROOT.glob("scripts/*/README.md"), *ROOT.glob("tests/fixtures/*/README.md")]
)
STYLE_PAGES = [
    README,
    STORY,
    DOCS / "results.md",
    DOCS / "pipeline.md",
    DOCS / "README.md",
    DOCS / "archive" / "README.md",
]
MEDIA_PAGES = [README, STORY]

MAX_SENTENCE_WORDS = 60
# Rule 6 of the style guide, matched across line wraps and as a prefix so that "the user's"
# and "the humans" fail too.  `docs/writing-style.md` quotes these on purpose and is not checked.
BANNED_ACTORS = re.compile(r"\bthe\s+(?:human|user|agent)", re.IGNORECASE)

FENCE = re.compile(r"^(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
MD_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
MD_REFERENCE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)", re.MULTILINE)
HTML_TARGET = re.compile(r"""(?:src|href)\s*=\s*["']([^"']+)["']""")
URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
MEDIA_PATH = re.compile(r"media/story/[\w.-]+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][)\"'\]])\s+|(?<=[.!?]\*\*)\s+")
LIST_ITEM = re.compile(r"^(?:[-*+]|\d+[.)])\s")


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


@cache
def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@cache
def _prose_lines(path: Path) -> tuple[str, ...]:
    """The page's lines with every fenced code block blanked, so line numbers are kept."""
    out: list[str] = []
    in_fence = False
    for line in _text(path).splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else line)
    return tuple(out)


def _heading_text(raw: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", raw)
    text = re.sub(r"<[^>]+>", "", text)
    return text.replace("`", "").replace("**", "").replace("*", "")


def github_anchor(heading: str) -> str:
    """GitHub's slug for one heading, before de-duplication: lowercase, punctuation dropped,
    spaces to hyphens.  Letters, digits, `_` and `-` survive; every other character goes."""
    text = _heading_text(heading).lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


@cache
def anchors_of(path: Path) -> frozenset[str]:
    seen: Counter[str] = Counter()
    anchors: set[str] = set()
    for line in _prose_lines(path):
        match = HEADING.match(line)
        if match is None:
            continue
        slug = github_anchor(match.group(2))
        anchors.add(slug if seen[slug] == 0 else f"{slug}-{seen[slug]}")
        seen[slug] += 1
    return frozenset(anchors)


def link_targets(path: Path) -> list[tuple[int, str]]:
    """Every link and image target on the page outside code fences, with its 1-based line."""
    targets: list[tuple[int, str]] = []
    for number, line in enumerate(_prose_lines(path), start=1):
        for pattern in (MD_LINK, HTML_TARGET):
            targets.extend((number, m.group(1)) for m in pattern.finditer(line))
        reference = MD_REFERENCE.match(line)
        if reference is not None:
            targets.append((number, reference.group(1)))
    return targets


def resolve_link(page: Path, target: str) -> str | None:
    """None when the target resolves, otherwise the reason it does not."""
    if URL_SCHEME.match(target):
        return None
    path_part, _, fragment = target.partition("#")
    path_part = unquote(path_part)
    if path_part == "":
        file = page
    elif path_part.startswith("/"):
        file = ROOT / path_part.lstrip("/")
    else:
        file = (page.parent / path_part).resolve()
    if not file.is_file():
        return f"missing file {path_part or _rel(page)}"
    if fragment:
        if file.suffix != ".md":
            return f"fragment #{fragment} on a non-markdown target {path_part}"
        if fragment not in anchors_of(file):
            return f"no heading for #{fragment} in {_rel(file)}"
    return None


def sentences(path: Path) -> list[tuple[int, str]]:
    """Sentences of the page's prose with the line each starts on.  Wrapped paragraph lines are
    joined; headings, list items, block quotes and each table cell stand alone."""
    units: list[tuple[int, str]] = []
    buffer: list[str] = []
    start = 0

    def flush() -> None:
        if buffer:
            units.append((start, " ".join(buffer)))
            buffer.clear()

    for number, line in enumerate(_prose_lines(path), start=1):
        stripped = line.strip().lstrip(">").strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("|"):
            flush()
            units.extend((number, cell) for cell in stripped.split("|"))
            continue
        if stripped.startswith(("#", "![")) or LIST_ITEM.match(stripped):
            flush()
            start = number
            stripped = LIST_ITEM.sub("", stripped).lstrip("#").strip()
        elif not buffer:
            start = number
        buffer.append(stripped)
    flush()

    found: list[tuple[int, str]] = []
    for number, unit in units:
        unit = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", unit)
        unit = re.sub(r"<[^>]+>", " ", unit)
        found.extend((number, s.strip()) for s in SENTENCE_END.split(unit) if s.strip())
    return found


# --------------------------------------------------------------------------- the checks


@pytest.mark.parametrize(
    ("heading", "slug"),
    [
        ("Sep 22 and 23: cleanup, then nothing", "sep-22-and-23-cleanup-then-nothing"),
        ("Sep 17–18: leak-onset attempt", "sep-1718-leak-onset-attempt"),
        ("Trial `P20_03_01` (room 2)", "trial-p20_03_01-room-2"),
        ("**Bold** and [linked](x.md) words", "bold-and-linked-words"),
        ("Committed media (Sep 27)", "committed-media-sep-27"),
    ],
)
def test_github_anchor_follows_the_rules(heading: str, slug: str) -> None:
    assert github_anchor(heading) == slug


def test_anchors_deduplicate_repeated_headings(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text("# Same\n\n```\n# not a heading\n```\n\n## Same\n\n## Other ##\n", "utf-8")
    assert anchors_of(page) == frozenset({"same", "same-1", "other"})


def test_sentences_join_wrapped_lines_and_split_cells_and_items(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        "# A heading\n\nOne sentence wraps\nonto a second line. Then another! And a\n"
        "[link](x.md) too.\n\n- item one\n  continues here\n- item two\n\n"
        "| a b | c d e |\n|---|---|\n\n```\nnot prose. not prose. not prose.\n```\n"
        + " ".join(["word"] * 61)
        + ".\n",
        "utf-8",
    )
    found = sentences(page)
    assert found[0] == (1, "A heading")
    assert (3, "One sentence wraps onto a second line.") in found
    assert (3, "Then another!") in found and (3, "And a link too.") in found
    assert (7, "item one continues here") in found and (9, "item two") in found
    assert (11, "a b") in found and (11, "c d e") in found
    assert not any("not prose" in text for _, text in found)
    assert [len(text.split()) for _, text in found if text.startswith("word")] == [61]
    assert BANNED_ACTORS.search("in which the\nUser and the  agent") is not None
    assert BANNED_ACTORS.search("the users' choice") is not None
    assert BANNED_ACTORS.search("bathe humans; a human, one user, our agent") is None


def test_link_extractor_sees_the_readme_links() -> None:
    targets = {target for _, target in link_targets(README)}
    assert "docs/story.md" in targets and "media/story/2026-09-25-finebio-3d.gif" in targets
    assert any(target.startswith("docs/story.md#sep-") for target in targets), "table links"
    assert len(targets) > 20


@pytest.mark.parametrize("page", LINK_PAGES, ids=_rel)
def test_relative_links_and_images_resolve(page: Path) -> None:
    targets = link_targets(page)
    problems = [
        f"{_rel(page)}:{number}: {target}: {reason}"
        for number, target in targets
        if (reason := resolve_link(page, target)) is not None
    ]
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("page", MEDIA_PAGES, ids=_rel)
def test_story_media_named_on_the_page_exists(page: Path) -> None:
    named = sorted(set(MEDIA_PATH.findall(_text(page))))
    assert named, f"{_rel(page)} names no media/story/ file"
    missing = [name for name in named if not (ROOT / name).is_file()]
    assert not missing, f"{_rel(page)} names media that is not on disk: {missing}"


def test_every_manifest_output_is_shown_in_the_story() -> None:
    manifest = json.loads(_text(MANIFEST))
    story = _text(STORY)
    outputs = [entry["output"] for entry in manifest["entries"]]
    assert outputs, "the story-media manifest lists no entries"
    unused = [output for output in outputs if output not in story]
    assert not unused, f"rendered but not shown in docs/story.md: {unused}"


@pytest.mark.parametrize("page", STYLE_PAGES, ids=_rel)
def test_no_sentence_runs_over_sixty_words(page: Path) -> None:
    found = sentences(page)
    assert found, f"{_rel(page)} has no prose"
    long = [
        f"{_rel(page)}:{number}: {len(text.split())} words: {text[:90]}..."
        for number, text in found
        if len(text.split()) > MAX_SENTENCE_WORDS
    ]
    assert not long, "\n".join(long)


@pytest.mark.parametrize("page", STYLE_PAGES, ids=_rel)
def test_no_forbidden_actor_names(page: Path) -> None:
    text = _text(page)
    hits = [
        f"{_rel(page)}:{text.count(chr(10), 0, m.start()) + 1}: {m.group(0)!r}"
        for m in BANNED_ACTORS.finditer(text)
    ]
    assert not hits, "\n".join(hits)
