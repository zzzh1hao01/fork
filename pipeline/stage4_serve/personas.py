"""Loads the persona cards Stage 2 produced, keyed by character name."""

from pathlib import Path

from pipeline.common.schemas import PersonaCard

DEFAULT_PERSONA_DIR = Path(__file__).parent / "personas"


def _slug(character: str) -> str:
    return character.lower().replace(" ", "_")


def load_persona_card(character: str, persona_dir: Path = DEFAULT_PERSONA_DIR) -> PersonaCard:
    path = persona_dir / f"{_slug(character)}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No persona card found for {character!r} at {path} "
            "(run Stage 2 persona extraction first)"
        )
    return PersonaCard.model_validate_json(path.read_text())
