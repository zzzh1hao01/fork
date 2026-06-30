"""Stage 1: parse the HP corpus PDF into speaker-attributed dialogue JSONL.

Usage:
    python -m pipeline.stage1_parser.parser data/harrypotter.pdf data/dialogue.jsonl
"""

import argparse
import json
from pathlib import Path

from pipeline.common.schemas import DialogueLine
from pipeline.stage1_parser.attribution import extract_quotes
from pipeline.stage1_parser.chapters import split_into_chapters
from pipeline.stage1_parser.resolve import LLMResolver, resolve_ambiguous


def extract_pdf_text(pdf_path: Path) -> str:
    return extract_pdf_text_bytes(pdf_path.read_bytes())


def extract_pdf_text_bytes(pdf_bytes: bytes) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


_QUOTE_NORMALIZATION = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
})


def normalize_quotes(text: str) -> str:
    """Convert curly quotes/apostrophes to straight ones.

    Published book PDFs are typeset with curly quotes; every downstream
    regex in this package (chapter headings, dialogue extraction, name
    matching) is written against straight quotes, so this runs once up
    front rather than every module re-handling both styles.
    """
    return text.translate(_QUOTE_NORMALIZATION)


def parse_corpus(
    full_text: str, resolver: LLMResolver | None = None
) -> list[DialogueLine]:
    full_text = normalize_quotes(full_text)
    lines: list[DialogueLine] = []
    line_index = 0
    for span in split_into_chapters(full_text):
        quotes = extract_quotes(span.text)
        resolve_ambiguous(quotes, resolver)
        for quote in quotes:
            text = quote.text.strip()
            if not text:
                continue
            lines.append(
                DialogueLine(
                    text=text,
                    speaker=quote.speaker,
                    book=span.book,
                    chapter=span.chapter,
                    line_index=line_index,
                )
            )
            line_index += 1
    return lines


def write_jsonl(lines: list[DialogueLine], output_path: Path) -> None:
    with output_path.open("w") as f:
        for line in lines:
            f.write(line.model_dump_json() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    full_text = extract_pdf_text(args.pdf_path)
    lines = parse_corpus(full_text)
    write_jsonl(lines, args.output_path)
    print(f"Wrote {len(lines)} dialogue lines to {args.output_path}")


if __name__ == "__main__":
    main()
