"""Maps roster character names to every raw speaker label Stage 1's
attribution produces for that character.

Stage 1 tags dialogue with whatever name appears in the book's own
attribution verb ("said Harry", "said Riddle"), not a canonical identity --
so the same character can be split across multiple labels (e.g. young
Voldemort's diary/Pensieve-memory dialogue is tagged "Riddle", not
"Voldemort"). Anything that queries by roster character name (persona
extraction, retrieval) needs to resolve aliases first or it will silently
undercount that character's lines.
"""

CHARACTER_ALIASES: dict[str, list[str]] = {
    "Harry Potter": ["Harry"],
    "Ron Weasley": ["Ron"],
    "Hermione Granger": ["Hermione"],
}


def resolve_aliases(character: str) -> list[str]:
    """Returns every raw speaker label that should count as `character`,
    always including `character` itself alongside any registered aliases
    (data may use the canonical name directly). Safe to call for any
    character, not just the roster -- names with no registered aliases
    just resolve to [character].
    """
    aliases = CHARACTER_ALIASES.get(character, [])
    return [character] + [a for a in aliases if a != character]
