"""Short forms of a fact for list views (search results, get_entity).

Measured 2026-10-09 on 86 labelled queries (scratch/memeval/trunc): the
`[src: ...]` tag was a fifth of search output and of get_entity output, and
mostly repeats a date the fact already states. Long facts were most of the
rest: cut to their first ~300 characters, every must-have fact was still
recognisable (1 of 58 needed the full text; 4 fetches in 86 searches) while
search output fell from 494 to ~330 tokens with both changes.

The full fact and its full source stay one call away: get_entity with the
observation id, or output_format="json".
"""

import re

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

# Facts up to CLIP_AT + CLIP_SLACK characters are shown whole: cutting a few
# words saves less than the marker costs.
CLIP_AT = 300
CLIP_SLACK = 40


def source_tag(source: str, content: str) -> str:
    """' (YYYY-MM-DD)' from the source, or '' when the fact already carries a
    date or the source has none."""
    if not source or _ISO_DATE.search(content or ""):
        return ""
    m = _ISO_DATE.search(source)
    return f" ({m.group(0)})" if m else ""


def clip(content: str, obs_id: str, match: str = "") -> str:
    """The first ~CLIP_AT characters of a long fact, cut at a word, then a
    marker naming how much is hidden and the id that fetches it. With `match`
    (an exact-substring hit), the shown window is moved so the match is in it."""
    content = content or ""
    if len(content) <= CLIP_AT + CLIP_SLACK:
        return content
    start = 0
    if match:
        pos = content.lower().find(match.lower())
        if pos >= 0 and pos + len(match) > CLIP_AT:
            start = max(0, pos - CLIP_AT // 3)
            start = content.rfind(" ", 0, start) + 1 if start else 0
    end = start + CLIP_AT
    if end >= len(content) - CLIP_SLACK:
        shown = content[start:]
    else:
        cut = content.rfind(" ", start, end)
        shown = content[start:cut if cut > start else end]
    hidden = len(content[:start].split()) + len(content[start + len(shown):].split())
    if not hidden:
        return content
    lead = "… " if start else ""
    tail = " …" if start + len(shown) < len(content) else ""
    return f"{lead}{shown}{tail} [+{hidden} words, obs={obs_id}]"
