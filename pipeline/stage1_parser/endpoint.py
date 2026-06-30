"""Stage 1 Flash endpoint: PDF -> speaker-attributed dialogue JSONL.

A Runpod Flash CPU endpoint, load-balanced (no GPU needed) per the PRD.

Local dev:
    flash dev
    # read the route table flash dev prints to get the exact local URL/path
    curl -s "$URL/parse" -d '{"data": {"pdf_url": "https://.../harrypotter.pdf"}}'

Deploy:
    flash deploy
"""

from runpod_flash import CpuInstanceType, Endpoint

stage1 = Endpoint(
    name="fork-stage1-preprocess",
    cpu=CpuInstanceType.CPU5C_4_8,
    workers=(0, 3),
    dependencies=["pymupdf", "pydantic", "httpx"],
)


@stage1.post("/parse")
async def parse(data: dict) -> dict:
    """data: {"pdf_url": str}.

    Downloads the corpus PDF, runs the Stage 1 parser, and returns the
    speaker-attributed dialogue lines. Takes a URL rather than the raw PDF
    bytes because of Flash's 10MB request payload limit.
    """
    import httpx

    from pipeline.stage1_parser.parser import extract_pdf_text_bytes, parse_corpus

    pdf_url = data["pdf_url"]
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        response = await client.get(pdf_url)
        response.raise_for_status()
        pdf_bytes = response.content

    full_text = extract_pdf_text_bytes(pdf_bytes)
    lines = parse_corpus(full_text)
    return {"count": len(lines), "lines": [line.model_dump() for line in lines]}


@stage1.get("/health")
async def health() -> dict:
    return {"status": "ok"}
