import pytest

from pipeline.common.schemas import DialogueLine, PersonaCard
from pipeline.stage2_persona.parse_response import PersonaParseError, parse_persona_response
from pipeline.stage2_persona.prompt import build_persona_prompt


def _line(text, speaker, book=1, chapter=1, line_index=0):
    return DialogueLine(text=text, speaker=speaker, book=book, chapter=chapter, line_index=line_index)


def test_prompt_includes_target_character_lines():
    lines = [
        _line("I solemnly swear that I am up to no good.", "Harry Potter"),
        _line("It's LeviOsa, not LevioSA.", "Hermione Granger"),
    ]
    prompt = build_persona_prompt("Harry Potter", lines)
    assert "I solemnly swear that I am up to no good." in prompt
    assert "LeviOsa" not in prompt


def test_prompt_names_target_character():
    lines = [_line("Hello.", "Harry Potter")]
    prompt = build_persona_prompt("Harry Potter", lines)
    assert "Harry Potter" in prompt


def test_prompt_requests_all_persona_card_fields():
    lines = [_line("Hello.", "Harry Potter")]
    prompt = build_persona_prompt("Harry Potter", lines)
    for field in [
        "name",
        "speech_style",
        "vocabulary_fingerprint",
        "core_values",
        "key_relationships",
        "knowledge_by_book",
        "canonical_quotes",
    ]:
        assert field in prompt


def test_prompt_includes_book_and_chapter_context():
    lines = [_line("Hello.", "Harry Potter", book=3, chapter=5)]
    prompt = build_persona_prompt("Harry Potter", lines)
    assert "Book 3" in prompt
    assert "Chapter 5" in prompt


def test_prompt_with_no_matching_lines_is_still_well_formed():
    lines = [_line("Hello.", "Ron Weasley")]
    prompt = build_persona_prompt("Harry Potter", lines)
    assert "Harry Potter" in prompt
    assert "Hello." not in prompt


_VALID_PERSONA_JSON = """{
    "name": "Harry Potter",
    "speech_style": "Plainspoken and earnest, prone to blunt questions.",
    "vocabulary_fingerprint": ["blimey", "reckon"],
    "core_values": ["loyalty", "courage"],
    "key_relationships": ["Ron Weasley - best friend", "Hermione Granger - best friend"],
    "knowledge_by_book": {"1": ["He is a wizard"], "2": ["The Chamber of Secrets exists"]},
    "canonical_quotes": ["I solemnly swear that I am up to no good."]
}"""


def test_parse_persona_response_returns_valid_persona_card():
    card = parse_persona_response(_VALID_PERSONA_JSON)
    assert isinstance(card, PersonaCard)
    assert card.name == "Harry Potter"
    assert card.knowledge_by_book[1] == ["He is a wizard"]
    assert "I solemnly swear that I am up to no good." in card.canonical_quotes


def test_parse_persona_response_strips_markdown_code_fence():
    fenced = f"```json\n{_VALID_PERSONA_JSON}\n```"
    card = parse_persona_response(fenced)
    assert card.name == "Harry Potter"


def test_parse_persona_response_strips_leading_commentary():
    chatty = f"Sure, here is the persona card:\n{_VALID_PERSONA_JSON}"
    card = parse_persona_response(chatty)
    assert card.name == "Harry Potter"


def test_parse_persona_response_raises_on_malformed_json():
    with pytest.raises(PersonaParseError):
        parse_persona_response("{not valid json")


def test_parse_persona_response_raises_when_no_json_object_present():
    with pytest.raises(PersonaParseError):
        parse_persona_response("I cannot extract a persona card for this character.")


def test_parse_persona_response_raises_on_schema_mismatch():
    with pytest.raises(PersonaParseError):
        parse_persona_response('{"name": "Harry Potter"}')


def test_parse_persona_response_accepts_matching_expected_character():
    card = parse_persona_response(_VALID_PERSONA_JSON, expected_character="Harry Potter")
    assert card.name == "Harry Potter"


def test_parse_persona_response_raises_on_character_mismatch():
    with pytest.raises(PersonaParseError):
        parse_persona_response(_VALID_PERSONA_JSON, expected_character="Hermione Granger")
