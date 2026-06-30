"""Speaker attribution for quoted dialogue lines.

Given the text surrounding a quote, decide which character said it using
`said`/`replied`/... attribution patterns. Lines with no resolvable
attribution are left as ambiguous so they can be sent to an LLM cleanup
pass (see resolve.py) instead of guessed at with regex.
"""

import re
from dataclasses import dataclass

UNKNOWN_SPEAKER = "UNKNOWN"

_PRONOUNS = {
    "He",
    "She",
    "They",
    "It",
    "I",
    "We",
    "You",
    "Him",
    "Her",
    "Them",
}

_ATTRIBUTION_VERBS = (
    "said|replied|shouted|whispered|asked|muttered|yelled|cried|exclaimed|"
    "answered|snapped|growled|sighed|murmured|gasped|groaned|sneered|"
    "hissed|roared|bellowed|laughed|sobbed|called|continued|added|"
    "interrupted|breathed|panted|thought"
)

_NAME = r"[A-Z][\w'.\-]*(?:\s+[A-Z][\w'.\-]*){0,2}"

QUOTE = re.compile(r'"([^"]+)"')

# "...," said Harry  /  "..." said Harry.
_TRAILING_ATTRIBUTION = re.compile(
    rf"^\s*,?\s*(?:{_ATTRIBUTION_VERBS})\s+(?P<name>{_NAME})\b"
)

# Harry said, "..."
_LEADING_ATTRIBUTION = re.compile(
    rf"(?P<name>{_NAME})\s+(?:{_ATTRIBUTION_VERBS})\b[^.\"]{{0,40}},?\s*$"
)


@dataclass
class AttributedQuote:
    text: str
    speaker: str | None  # None => ambiguous, needs LLM cleanup
    before: str
    after: str


def _clean_name(name: str) -> str | None:
    name = name.strip()
    first_word = name.split()[0]
    if first_word in _PRONOUNS:
        return None
    return name


def attribute_speaker(before: str, after: str) -> str | None:
    """Resolve a speaker name from the text surrounding a quote.

    Checks, in order:
      1. trailing attribution right after the quote ("..." said Harry)
      2. leading attribution right before the quote (Harry said, "...")
      3. trailing attribution in the gap before the quote, for the second
         half of split dialogue ("...," said Harry, "...")
    Returns None (ambiguous) when no pattern matches or only a pronoun
    is found, e.g. unattributed narration or bare thought-speech ("He
    thought, '...'").
    """
    match = _TRAILING_ATTRIBUTION.match(after.strip())
    if match:
        return _clean_name(match.group("name"))

    match = _LEADING_ATTRIBUTION.search(before.strip())
    if match:
        return _clean_name(match.group("name"))

    match = _TRAILING_ATTRIBUTION.match(before.strip())
    if match:
        return _clean_name(match.group("name"))

    return None


def extract_quotes(text: str, window: int = 80) -> list[AttributedQuote]:
    """Find every quoted span in `text` and attribute a speaker to each."""
    matches = list(QUOTE.finditer(text))
    results = []
    for i, m in enumerate(matches):
        prev_end = matches[i - 1].end() if i > 0 else max(0, m.start() - window)
        next_start = matches[i + 1].start() if i + 1 < len(matches) else min(
            len(text), m.end() + window
        )
        before = text[prev_end : m.start()]
        after = text[m.end() : next_start]
        speaker = attribute_speaker(before, after)
        results.append(AttributedQuote(text=m.group(1), speaker=speaker, before=before, after=after))
    return results
