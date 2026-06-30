from typing import Literal

from pydantic import BaseModel


class DialogueLine(BaseModel):
    text: str
    speaker: str
    book: int
    chapter: int
    line_index: int


class PersonaCard(BaseModel):
    name: str
    speech_style: str
    vocabulary_fingerprint: list[str]
    core_values: list[str]
    key_relationships: list[str]
    knowledge_by_book: dict[int, list[str]]
    canonical_quotes: list[str]


class Citation(BaseModel):
    text: str
    book: int
    chapter: int


class ServeRequest(BaseModel):
    character: str
    user_message: str
    mode: Literal["chat", "fork", "fanfic"]
    book_number: int
    fork_id: str | None = None
    custom_prompt: str | None = None


class ServeResponse(BaseModel):
    response_text: str
    citations: list[Citation]


class ForkPoint(BaseModel):
    id: str
    title: str
    setup_summary: str
    fork_chapter: int
    fork_book: int
    character_pov: str
    system_prompt_override: str
