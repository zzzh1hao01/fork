"""Pure orchestration logic for Stage 4: persona + retrieved context -> system
prompt, and retrieved payloads -> citations. No network calls (Qdrant, GPT-4o)
here so this stays fully unit-testable; see endpoint.py for those calls.
"""

from typing import Callable

from pipeline.common.schemas import Citation, ForkPoint, PersonaCard, ServeRequest, ServeResponse
from pipeline.stage4_serve.forks import FORK_POINTS

GenerateFn = Callable[[str, str], str]  # (system_prompt, user_message) -> response_text


def resolve_fork(request: ServeRequest) -> ForkPoint | None:
    """Returns the ForkPoint for a 'fork' mode request, or None otherwise.

    Raises ValueError if mode is 'fork' but fork_id is missing or unknown.
    """
    if request.mode != "fork":
        return None
    if request.fork_id is None:
        raise ValueError("mode='fork' requires fork_id")
    fork = FORK_POINTS.get(request.fork_id)
    if fork is None:
        raise ValueError(f"Unknown fork_id: {request.fork_id!r}")
    return fork


def retrieval_character(request: ServeRequest, fork: ForkPoint | None) -> str:
    """The character whose dialogue/persona should ground retrieval and the
    system prompt -- the fork's POV character takes precedence in fork mode."""
    return fork.character_pov if fork is not None else request.character


def retrieval_book_ceiling(request: ServeRequest, fork: ForkPoint | None) -> int:
    """The timeline filter ceiling: canon up to the fork's book in fork mode,
    otherwise whatever book the user has reached."""
    return fork.fork_book if fork is not None else request.book_number


def citations_from_payloads(
    payloads: list[dict], max_book_number: int | None = None
) -> list[Citation]:
    """Builds Citations from Qdrant payload dicts (as stored by Stage 3:
    speaker/book_number/chapter_number/line_index/raw_text), dropping any
    that exceed max_book_number as a safety net on top of the retrieval
    filter itself."""
    citations = []
    for payload in payloads:
        if max_book_number is not None and payload["book_number"] > max_book_number:
            continue
        citations.append(
            Citation(
                text=payload["raw_text"],
                book=payload["book_number"],
                chapter=payload["chapter_number"],
            )
        )
    return citations


def build_system_prompt(
    persona_card: PersonaCard,
    request: ServeRequest,
    retrieved_payloads: list[dict],
    fork: ForkPoint | None = None,
) -> str:
    persona_block = (
        f"You are {persona_card.name}.\n"
        f"Speech style: {persona_card.speech_style}\n"
        f"Core values: {', '.join(persona_card.core_values)}\n"
        f"Key relationships: {', '.join(persona_card.key_relationships)}"
    )

    context_block = "Canon context retrieved for this conversation:\n" + "\n".join(
        f"[Book {p['book_number']}, Chapter {p['chapter_number']}] {p['raw_text']}"
        for p in retrieved_payloads
    )

    if request.mode == "chat":
        mode_block = (
            "Respond in character to the user's message, staying strictly "
            "grounded in the canon context above. Only reference events the "
            "character could plausibly know at this point in the story."
        )
    elif request.mode == "fork":
        if fork is None:
            raise ValueError("mode='fork' requires a resolved ForkPoint")
        mode_block = fork.system_prompt_override
    elif request.mode == "fanfic":
        mode_block = (
            f"Write a short scene in {persona_card.name}'s voice based on this "
            f"prompt: {request.custom_prompt}. Stay consistent with the "
            "character's established voice and relationships from the canon "
            "context above."
        )
    else:
        raise ValueError(f"Unknown mode: {request.mode}")

    return f"{persona_block}\n\n{context_block}\n\n{mode_block}"


def assemble_response(
    request: ServeRequest,
    persona_card: PersonaCard,
    retrieved_payloads: list[dict],
    generate_fn: GenerateFn,
    fork: ForkPoint | None = None,
) -> ServeResponse:
    """Ties prompt-building, citation extraction, and generation together.

    `generate_fn` is injected (rather than calling GPT-4o directly) so this
    full assembly path is testable without a network call -- see endpoint.py
    for the real OpenAI-backed generate_fn.
    """
    book_ceiling = retrieval_book_ceiling(request, fork)
    citations = citations_from_payloads(retrieved_payloads, max_book_number=book_ceiling)
    system_prompt = build_system_prompt(persona_card, request, retrieved_payloads, fork=fork)
    response_text = generate_fn(system_prompt, request.user_message)
    return ServeResponse(response_text=response_text, citations=citations)
