"""Pure response-parsing for Stage 2 persona extraction.

Turns the raw text completion from the vLLM model into a validated
PersonaCard. No network or model imports here so this is fully
unit-testable (see endpoint.py for the model call).
"""

import json
import re

from pydantic import ValidationError

from pipeline.common.schemas import PersonaCard

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class PersonaParseError(ValueError):
    """Raised when the model's raw output can't be parsed into a PersonaCard."""


def _strip_code_fence(text: str) -> str:
    return _FENCE.sub("", text.strip()).strip()


def _extract_json_object(text: str) -> str:
    text = _strip_code_fence(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise PersonaParseError(f"No JSON object found in model output: {text!r}")
    return text[start : end + 1]


def _names_match(a: str, b: str) -> bool:
    return bool(set(a.lower().split()) & set(b.lower().split()))


def parse_persona_response(raw_text: str, expected_character: str | None = None) -> PersonaCard:
    """Parse a raw LLM completion into a validated PersonaCard.

    Tolerates markdown code fences and leading/trailing commentary around
    the JSON object; raises PersonaParseError for anything that isn't
    valid JSON or doesn't satisfy the PersonaCard schema.

    If `expected_character` is given, also raises PersonaParseError when the
    card's `name` shares no word with it -- catches the model hallucinating
    a persona for the wrong character.
    """
    json_text = _extract_json_object(raw_text)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise PersonaParseError(f"Malformed JSON in model output: {exc}") from exc

    try:
        persona_card = PersonaCard.model_validate(data)
    except ValidationError as exc:
        raise PersonaParseError(f"Model output failed PersonaCard validation: {exc}") from exc

    if expected_character is not None and not _names_match(persona_card.name, expected_character):
        raise PersonaParseError(
            f"Model returned persona for {persona_card.name!r}, expected {expected_character!r}"
        )

    return persona_card
