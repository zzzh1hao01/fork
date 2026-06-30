"""Pure prompt construction for Stage 2 persona extraction.

Builds the instruction + dialogue context sent to the vLLM-served model
(see endpoint.py for which model and the model call itself). No network or
model imports here so this is fully unit-testable.
"""

from pipeline.common.aliases import resolve_aliases
from pipeline.common.schemas import DialogueLine

_INSTRUCTIONS = """You are a literary analyst building a structured persona card for a \
fictional character based only on their dialogue lines from the Harry Potter \
series.

Return a single JSON object with exactly these fields, and no other text:
- "name": the character's full name (string)
- "speech_style": a short description of how the character speaks (string)
- "vocabulary_fingerprint": distinctive words/phrases the character favors (list of strings)
- "core_values": the character's core values (list of strings)
- "key_relationships": the character's key relationships, one entry per relationship (list of strings)
- "knowledge_by_book": for each book number the character appears in, a list of \
the major things the character knows by the end of that book (object mapping \
book number to list of strings)
- "canonical_quotes": a handful of verbatim quotes from the dialogue below that \
best capture the character's voice (list of strings)

Base every field strictly on the dialogue lines provided. Do not invent facts \
not supported by the lines. Respond with JSON only, no markdown fences, no \
commentary."""


def _format_line(line: DialogueLine) -> str:
    return f"[Book {line.book}, Chapter {line.chapter}] {line.text}"


def build_persona_prompt(character: str, lines: list[DialogueLine]) -> str:
    """Build the persona-extraction prompt for one character.

    `lines` may contain dialogue from other speakers; only lines spoken by
    `character` (or one of their known aliases, e.g. young Voldemort's
    dialogue tagged "Riddle") are included in the prompt context.
    """
    aliases = resolve_aliases(character)
    character_lines = [line for line in lines if line.speaker in aliases]
    formatted = "\n".join(_format_line(line) for line in character_lines)
    return (
        f"{_INSTRUCTIONS}\n\n"
        f"Character: {character}\n\n"
        f"Dialogue lines spoken by {character}:\n{formatted}\n"
    )
