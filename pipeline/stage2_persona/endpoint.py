"""Stage 2: dialogue lines for one character -> persona card.

Backed by RunPod's officially maintained `runpod-workers/worker-vllm` image
(Flash's external-image/Client mode) instead of pip-installing vllm into a
generic Flash worker -- that approach hit a wall of CUDA/torch/torchvision
version-skew errors (the worker-vllm image bundles its own self-consistent
stack and is what RunPod actually tests their GPU nodes against). The
endpoint still bursts up to `workers=(0, 3)` (one worker per roster
character) and scales to zero after -- the PRD's Flash showcase behavior,
just via a maintained image rather than custom pip dependencies.

Usage (batch run for the roster, writes persona JSON for Stage 4):
    python -m pipeline.stage2_persona.endpoint data/dialogue.jsonl
"""

import argparse
import asyncio
import json
from pathlib import Path

from runpod_flash import CudaVersion, Endpoint, GpuGroup

from pipeline.common.schemas import DialogueLine, PersonaCard
from pipeline.stage2_persona.parse_response import parse_persona_response
from pipeline.stage2_persona.prompt import build_persona_prompt

stage2_vllm = Endpoint(
    name="fork-stage2-persona",
    image="registry.runpod.net/runpod-workers-worker-vllm-main-dockerfile:9e1c48313",
    # The Hub's own recommended tier for this specific image build (it
    # requires CUDA 13.0) -- AMPERE_24 + CUDA 13 had zero matching supply,
    # the job sat queued for 300s and was never picked up. Costs more than
    # the 24GB tier; revisit once the pipeline is proven end-to-end. A
    # single GPU type, not a list -- Flash only auto-switches across a
    # list of GPU types when workers max >= 5, and ours is 3.
    gpu=GpuGroup.AMPERE_80,
    min_cuda_version=CudaVersion.V13_0,
    workers=(0, 3),
    env={"MODEL_NAME": "Qwen/Qwen2.5-7B-Instruct"},
)

ROSTER = ["Harry Potter", "Ron Weasley", "Hermione Granger"]


async def extract_persona(character: str, lines: list[DialogueLine]) -> PersonaCard:
    """Builds the persona prompt, runs it through the worker-vllm endpoint's
    RunPod-native chat completions format, and parses the result."""
    prompt = build_persona_prompt(character, lines)
    job = await stage2_vllm.run(
        {
            "messages": [{"role": "user", "content": prompt}],
            "sampling_params": {"temperature": 0.2, "max_tokens": 2048},
        }
    )
    await job.wait(timeout=300)
    # worker-vllm's RunPod-native output is a list of streamed batches even
    # for non-streaming requests; the final batch holds the complete response.
    output = job.output[-1] if isinstance(job.output, list) else job.output
    raw_text = output["choices"][0]["message"]["content"]
    return parse_persona_response(raw_text, expected_character=character)


def _slug(character: str) -> str:
    return character.lower().replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dialogue_path", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("pipeline/stage4_serve/personas")
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if a persona JSON already exists for a character.",
    )
    args = parser.parse_args()

    lines = [
        DialogueLine(**json.loads(raw_line))
        for raw_line in args.dialogue_path.read_text().splitlines()
        if raw_line.strip()
    ]

    async def run_all() -> None:
        for character in ROSTER:
            output_path = args.output_dir / f"{_slug(character)}.json"
            if output_path.exists() and not args.force:
                print(f"Skipping {character}, {output_path} already exists (use --force to redo)")
                continue
            print(f"Extracting persona for {character}...")
            persona_card = await extract_persona(character, lines)
            output_path.write_text(persona_card.model_dump_json(indent=2))
            print(f"  wrote {output_path}")

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
