import hashlib

from qdrant_client import QdrantClient

from pipeline.common.schemas import DialogueLine
from pipeline.stage3_embed.embed import embed_lines
from pipeline.stage3_embed.retrieval import search

VECTOR_SIZE = 16


def fake_embed_fn(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255 for b in digest[:VECTOR_SIZE]]


def fake_batch_embed_fn(texts: list[str]) -> list[list[float]]:
    return [fake_embed_fn(text) for text in texts]


def _client_with_lines(lines: list[DialogueLine]) -> QdrantClient:
    client = QdrantClient(":memory:")
    embed_lines(lines, fake_batch_embed_fn, client)
    return client


def _corpus() -> list[DialogueLine]:
    return [
        DialogueLine(text="Wingardium Leviosa, Ron.", speaker="Hermione", book=1, chapter=10, line_index=0),
        DialogueLine(text="I solemnly swear I am up to no good.", speaker="Harry", book=3, chapter=4, line_index=1),
        DialogueLine(text="We should look in the library.", speaker="Hermione", book=2, chapter=5, line_index=2),
        DialogueLine(text="Kreacher's mission is complete.", speaker="Voldemort", book=6, chapter=1, line_index=3),
        DialogueLine(text="The Resurrection Stone is mine at last.", speaker="Voldemort", book=7, chapter=36, line_index=4),
        DialogueLine(text="Horcruxes must be destroyed, Harry.", speaker="Hermione", book=7, chapter=6, line_index=5),
        DialogueLine(text="I'm the Boy Who Lived.", speaker="Harry", book=1, chapter=17, line_index=6),
        DialogueLine(text="Ron, you're amazing sometimes.", speaker="Hermione", book=4, chapter=22, line_index=7),
    ]


def test_retrieval_filters_by_speaker_and_book_number():
    client = _client_with_lines(_corpus())
    matches = search(
        speaker="Hermione",
        max_book_number=4,
        query_text="What should we do next?",
        qdrant_client=client,
        embed_fn=fake_embed_fn,
        top_k=10,
    )
    assert len(matches) > 0
    for match in matches:
        assert match.payload["speaker"] == "Hermione"
        assert match.payload["book_number"] <= 4


def test_book_seven_query_returns_nothing_under_book_three_filter():
    client = _client_with_lines(_corpus())
    matches = search(
        speaker="Voldemort",
        max_book_number=3,
        query_text="The Resurrection Stone is mine at last.",
        qdrant_client=client,
        embed_fn=fake_embed_fn,
        top_k=10,
    )
    assert matches == []


def test_embed_lines_calls_embed_fn_once_per_batch_not_once_per_line():
    calls = []

    def counting_batch_embed_fn(texts: list[str]) -> list[list[float]]:
        calls.append(len(texts))
        return fake_batch_embed_fn(texts)

    lines = _corpus()
    client = QdrantClient(":memory:")
    embed_lines(lines, counting_batch_embed_fn, client, batch_size=3)

    assert calls == [3, 3, 2]
