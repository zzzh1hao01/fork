"""Stage 3: timeline-filtered retrieval over the hp_dialogue Qdrant collection."""

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, Range, ScoredPoint

from pipeline.common.aliases import resolve_aliases
from pipeline.stage3_embed.embed import COLLECTION_NAME, EmbedFn


def search(
    speaker: str,
    max_book_number: int,
    query_text: str,
    qdrant_client: QdrantClient,
    embed_fn: EmbedFn,
    top_k: int = 5,
) -> list[ScoredPoint]:
    """Searches hp_dialogue for lines matching speaker and book_number <= max_book_number.

    This is the timeline filter described in the PRD: retrieval is always
    filtered by speaker and book_number <= current_book so a character never
    surfaces lines from books they haven't lived through yet. `speaker` is
    resolved through known aliases (e.g. "Lord Voldemort" also matches
    lines tagged "Riddle") so a character's full line set is retrievable
    under their canonical name.
    """
    query_vector = embed_fn(query_text)
    query_filter = Filter(
        must=[
            FieldCondition(key="speaker", match=MatchAny(any=resolve_aliases(speaker))),
            FieldCondition(key="book_number", range=Range(lte=max_book_number)),
        ]
    )
    result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )
    return result.points
