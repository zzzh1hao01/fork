import pytest

from pipeline.common.schemas import PersonaCard, ServeRequest
from pipeline.stage4_serve.orchestrate import assemble_response, resolve_fork


def _persona(name="Harry Potter") -> PersonaCard:
    return PersonaCard(
        name=name,
        speech_style="Plainspoken and earnest.",
        vocabulary_fingerprint=["blimey", "reckon"],
        core_values=["loyalty", "courage"],
        key_relationships=["Ron Weasley - best friend"],
        knowledge_by_book={1: ["He is a wizard"]},
        canonical_quotes=["I solemnly swear that I am up to no good."],
    )


def _payload(book, chapter, text="Hello there.", speaker="Harry Potter", line_index=0):
    return {
        "speaker": speaker,
        "book_number": book,
        "chapter_number": chapter,
        "line_index": line_index,
        "raw_text": text,
    }


def _fake_generate(system_prompt: str, user_message: str) -> str:
    return "A grounded, in-character reply."


def test_chat_response_has_nonempty_text_and_valid_citations():
    request = ServeRequest(
        character="Harry Potter",
        user_message="How do you feel about Snape?",
        mode="chat",
        book_number=5,
    )
    payloads = [_payload(3, 12, "Snape's a git."), _payload(5, 2, "I don't trust him.")]
    response = assemble_response(request, _persona(), payloads, _fake_generate)

    assert isinstance(response.response_text, str)
    assert response.response_text != ""
    assert len(response.citations) >= 1
    for citation in response.citations:
        assert citation.text
        assert isinstance(citation.book, int)
        assert isinstance(citation.chapter, int)


def test_fork_mode_response_citations_respect_fork_book_ceiling():
    request = ServeRequest(
        character="Harry Potter",
        user_message="What happens next?",
        mode="fork",
        book_number=1,
        fork_id="slytherin-sorting",
    )
    fork = resolve_fork(request)
    payloads = [
        _payload(1, 6, "I hope I'm in Gryffindor."),
        _payload(1, 7, "Not Slytherin, eh?"),
        _payload(4, 20, "Voldemort has returned."),
    ]
    response = assemble_response(request, _persona(), payloads, _fake_generate, fork=fork)

    assert response.response_text != ""
    assert len(response.citations) >= 1
    assert all(citation.book <= fork.fork_book for citation in response.citations)


def test_resolve_fork_returns_none_for_chat_mode():
    request = ServeRequest(character="Harry Potter", user_message="Hi", mode="chat", book_number=1)
    assert resolve_fork(request) is None


def test_resolve_fork_raises_for_missing_fork_id():
    request = ServeRequest(character="Harry Potter", user_message="Hi", mode="fork", book_number=1)
    with pytest.raises(ValueError):
        resolve_fork(request)


def test_resolve_fork_raises_for_unknown_fork_id():
    request = ServeRequest(
        character="Harry Potter", user_message="Hi", mode="fork", book_number=1, fork_id="not-a-real-fork"
    )
    with pytest.raises(ValueError):
        resolve_fork(request)


def test_fanfic_mode_response_is_nonempty():
    request = ServeRequest(
        character="Hermione Granger",
        user_message="",
        mode="fanfic",
        book_number=7,
        custom_prompt="Hermione finds a hidden room in the library.",
    )
    payloads = [_payload(2, 5, "We should look in the library.", speaker="Hermione Granger")]
    response = assemble_response(request, _persona("Hermione Granger"), payloads, _fake_generate)

    assert response.response_text != ""
