"""LLM cleanup pass for dialogue lines the regex attributor couldn't resolve."""

from collections.abc import Callable

from pipeline.stage1_parser.attribution import UNKNOWN_SPEAKER, AttributedQuote

# (quote_text, before_context, after_context) -> speaker name, or None if the
# model also can't attribute it (e.g. true narrator text, unsigned letters).
LLMResolver = Callable[[str, str, str], str | None]


def resolve_ambiguous(
    quotes: list[AttributedQuote], resolver: LLMResolver | None
) -> None:
    """Fill in `speaker` on ambiguous quotes in place, using `resolver` when
    given, otherwise falling back to UNKNOWN_SPEAKER."""
    for quote in quotes:
        if quote.speaker is not None:
            continue
        resolved = resolver(quote.text, quote.before, quote.after) if resolver else None
        quote.speaker = resolved or UNKNOWN_SPEAKER
