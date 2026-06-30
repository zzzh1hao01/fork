"""Book and chapter boundary detection for the combined HP corpus PDF.

Book boundaries are inferred from chapter numbering resets (each book's
chapters restart at "Chapter One") rather than matching book title text.
The combined corpus PDF's front-matter table of contents lists all 7 book
titles before any real chapter content, which makes title-text matching
fire early and mis-tag every subsequent chapter -- the chapter-reset signal
doesn't have that failure mode.
"""

import re
from dataclasses import dataclass

_WORD_NUMBERS = {
    word: i + 1
    for i, word in enumerate(
        [
            "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
            "twenty", "twenty-one", "twenty-two", "twenty-three",
            "twenty-four", "twenty-five", "twenty-six", "twenty-seven",
            "twenty-eight", "twenty-nine", "thirty", "thirty-one",
            "thirty-two", "thirty-three", "thirty-four", "thirty-five",
            "thirty-six", "thirty-seven", "thirty-eight",
        ]
    )
}

_CHAPTER_HEADING = re.compile(
    r"^\s*chapter\s+(?P<num>[a-z\-]+|\d+)\s*$", re.IGNORECASE
)


@dataclass
class ChapterSpan:
    book: int
    chapter: int
    text: str


def detect_chapter_start(line: str) -> int | None:
    match = _CHAPTER_HEADING.match(line.strip())
    if not match:
        return None
    raw = match.group("num").strip().lower()
    if raw.isdigit():
        return int(raw)
    return _WORD_NUMBERS.get(raw)


def split_into_chapters(full_text: str) -> list[ChapterSpan]:
    """Walk the corpus line by line, tagging each chapter's text with its
    book and chapter number. A chapter heading numbered 1 starts a new book;
    any other chapter heading continues the current book."""
    spans: list[ChapterSpan] = []
    current_book = 0
    current_chapter = 0
    buffer: list[str] = []

    def flush() -> None:
        if buffer and current_book and current_chapter:
            spans.append(
                ChapterSpan(
                    book=current_book,
                    chapter=current_chapter,
                    text="\n".join(buffer),
                )
            )
        buffer.clear()

    for line in full_text.splitlines():
        chapter = detect_chapter_start(line)
        if chapter is not None:
            flush()
            if chapter == 1:
                current_book += 1
            current_chapter = chapter
            continue

        buffer.append(line)

    flush()
    return spans
