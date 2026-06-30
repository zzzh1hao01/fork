"""Stage 4 Flash endpoint: orchestration + Qdrant retrieval + GPT-4o.

A Runpod Flash endpoint (Mode 2, load-balanced) kept warm (workers min=1)
to avoid cold-start latency during the live demo, per the PRD.

Local dev:
    flash dev
    # read the route table flash dev prints to get the exact local URL/path
    curl -s "$URL/respond" -d '{"data": {"character": "Harry Potter", \
"user_message": "How do you feel about Snape?", "mode": "chat", "book_number": 5}}'

Deploy:
    flash deploy
"""

from runpod_flash import Endpoint, GpuGroup

stage4 = Endpoint(
    name="fork-stage4-serve",
    gpu=GpuGroup.ANY,
    # (1, N) keeps a GPU warm continuously -- matches the PRD's no-cold-start
    # requirement for the live demo, at the cost of paying to idle.
    workers=(1, 3),
    dependencies=["pydantic", "qdrant-client", "openai"],
)

TOP_K = 8
GPT_MODEL = "gpt-4o"


@stage4.post("/respond")
async def respond(data: dict) -> dict:
    """data: a ServeRequest-shaped dict (character, user_message, mode,
    book_number, fork_id?, custom_prompt?). Returns a ServeResponse dict."""
    import os

    from openai import OpenAI
    from qdrant_client import QdrantClient

    from pipeline.common.schemas import ServeRequest
    from pipeline.stage3_embed.embed import openai_embed_fn
    from pipeline.stage3_embed.retrieval import search
    from pipeline.stage4_serve.orchestrate import (
        assemble_response,
        resolve_fork,
        retrieval_book_ceiling,
        retrieval_character,
    )
    from pipeline.stage4_serve.personas import load_persona_card

    request = ServeRequest.model_validate(data)
    fork = resolve_fork(request)
    character = retrieval_character(request, fork)
    book_ceiling = retrieval_book_ceiling(request, fork)
    persona_card = load_persona_card(character)

    qdrant_client = QdrantClient(
        url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"]
    )
    embed_fn = openai_embed_fn()
    matches = search(
        speaker=character,
        max_book_number=book_ceiling,
        query_text=request.user_message,
        qdrant_client=qdrant_client,
        embed_fn=embed_fn,
        top_k=TOP_K,
    )
    payloads = [m.payload for m in matches]

    openai_client = OpenAI()

    def generate(system_prompt: str, user_message: str) -> str:
        completion = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return completion.choices[0].message.content

    response = assemble_response(request, persona_card, payloads, generate, fork=fork)
    return response.model_dump()


@stage4.get("/health")
async def health() -> dict:
    return {"status": "ok"}
