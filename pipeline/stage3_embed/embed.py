"""Stage 3: embed dialogue lines and load them into Qdrant.

Usage:
    python -m pipeline.stage3_embed.embed data/dialogue.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import Callable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from pipeline.common.schemas import DialogueLine

COLLECTION_NAME = "hp_dialogue"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

EmbedFn = Callable[[str], list[float]]
BatchEmbedFn = Callable[[list[str]], list[list[float]]]


def openai_embed_fn(api_key: str | None = None) -> EmbedFn:
    """Returns a single-text embed_fn backed by the real OpenAI API, for
    embedding one query at a time (see retrieval.search). Kept separate from
    the core upsert logic so that logic stays mockable/testable without
    requiring an OpenAI API key.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def embed(text: str) -> list[float]:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return response.data[0].embedding

    return embed


def openai_batch_embed_fn(api_key: str | None = None) -> BatchEmbedFn:
    """Returns a batch embed_fn backed by the real OpenAI API, for bulk
    corpus loading in embed_lines(). Batches every text in a Qdrant upsert
    batch into a single API request instead of one call per line -- the
    full HP corpus is ~37k dialogue lines, and one call each would mean
    tens of thousands of network round-trips.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in response.data]

    return embed


def ensure_collection(qdrant_client: QdrantClient, vector_size: int = EMBEDDING_DIM) -> None:
    if qdrant_client.collection_exists(COLLECTION_NAME):
        return
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    # Qdrant Cloud requires an explicit payload index to filter on these
    # fields -- without it, retrieval.search()'s speaker/book_number filter
    # raises a 400 (discovered when Stage 4 hit this for real).
    from qdrant_client.models import PayloadSchemaType

    qdrant_client.create_payload_index(
        COLLECTION_NAME, field_name="speaker", field_schema=PayloadSchemaType.KEYWORD
    )
    qdrant_client.create_payload_index(
        COLLECTION_NAME, field_name="book_number", field_schema=PayloadSchemaType.INTEGER
    )


def embed_lines(
    lines: list[DialogueLine],
    embed_fn: BatchEmbedFn,
    qdrant_client: QdrantClient,
    batch_size: int = 100,
    start_id: int = 0,
) -> int:
    """Embeds dialogue lines and upserts them into the hp_dialogue collection.

    `embed_fn` is called once per Qdrant upsert batch (not once per line) --
    see openai_batch_embed_fn(). Returns the number of points upserted.
    """
    if not lines:
        return 0

    collection_ready = False
    count = 0
    for batch_start in range(0, len(lines), batch_size):
        batch = lines[batch_start : batch_start + batch_size]
        vectors = embed_fn([line.text for line in batch])

        if not collection_ready:
            ensure_collection(qdrant_client, vector_size=len(vectors[0]))
            collection_ready = True

        points = [
            PointStruct(
                id=start_id + batch_start + offset,
                vector=vector,
                payload={
                    "speaker": line.speaker,
                    "book_number": line.book,
                    "chapter_number": line.chapter,
                    "line_index": line.line_index,
                    "raw_text": line.text,
                },
            )
            for offset, (line, vector) in enumerate(zip(batch, vectors))
        ]
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        count += len(points)
    return count


def read_jsonl(input_path: Path) -> list[DialogueLine]:
    lines: list[DialogueLine] = []
    with input_path.open() as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            lines.append(DialogueLine(**json.loads(raw_line)))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    args = parser.parse_args()

    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise SystemExit(
            "QDRANT_URL and QDRANT_API_KEY environment variables must be set"
        )

    lines = read_jsonl(args.input_path)
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30)
    embed_fn = openai_batch_embed_fn()
    count = embed_lines(lines, embed_fn, qdrant_client)
    print(f"Embedded and upserted {count} dialogue lines into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
