from pipeline.stage1_parser.attribution import UNKNOWN_SPEAKER, extract_quotes
from pipeline.stage1_parser.chapters import split_into_chapters
from pipeline.stage1_parser.parser import normalize_quotes, parse_corpus
from pipeline.stage1_parser.resolve import resolve_ambiguous


def _corpus(body: str, chapter: str = "Chapter One") -> str:
    return f"{chapter}\n{body}\n"


def test_trailing_attribution():
    [quote] = extract_quotes('"I solemnly swear that I am up to no good," said Harry.')
    assert quote.speaker == "Harry"
    assert quote.text == "I solemnly swear that I am up to no good,"


def test_leading_attribution():
    [quote] = extract_quotes('Hermione said, "It\'s LeviOsa, not LevioSA."')
    assert quote.speaker == "Hermione"


def test_split_dialogue_carries_speaker_to_second_half():
    quotes = extract_quotes('"You know," said Ron, "this is a bad idea."')
    assert [q.speaker for q in quotes] == ["Ron", "Ron"]


def test_nested_single_quote_inside_dialogue():
    [quote] = extract_quotes(
        '"She told me, \'I\'ll be there,\' and then left," said Hermione.'
    )
    assert quote.speaker == "Hermione"
    assert "I'll be there" in quote.text


def test_thought_speech_with_bare_pronoun_is_ambiguous():
    [quote] = extract_quotes('He thought, "This is never going to work."')
    assert quote.speaker is None


def test_thought_speech_with_named_subject_resolves():
    [quote] = extract_quotes('Harry thought, "This is never going to work."')
    assert quote.speaker == "Harry"


def test_letter_with_no_attribution_is_ambiguous():
    [quote] = extract_quotes(
        'The letter read: "Dear Mr. Potter, we are pleased to inform you."'
    )
    assert quote.speaker is None


def test_resolve_ambiguous_falls_back_to_unknown_without_resolver():
    quotes = extract_quotes('"Dear Mr. Potter, we are pleased to inform you."')
    resolve_ambiguous(quotes, resolver=None)
    assert quotes[0].speaker == UNKNOWN_SPEAKER


def test_resolve_ambiguous_uses_llm_resolver_when_provided():
    quotes = extract_quotes('He thought, "This is never going to work."')
    resolve_ambiguous(quotes, resolver=lambda text, before, after: "Harry")
    assert quotes[0].speaker == "Harry"


def test_split_into_chapters_tags_chapter_within_a_book():
    text = (
        "Chapter One\n"
        'Body line one with "a quote," said Harry.\n'
        "Chapter Two\n"
        'Another "quote here," said Ron.\n'
    )
    spans = split_into_chapters(text)
    assert [(s.book, s.chapter) for s in spans] == [(1, 1), (1, 2)]


def test_split_into_chapters_increments_book_on_chapter_one_reset():
    text = (
        "Chapter One\n"
        'Book one stuff, "hello," said Harry.\n'
        "Chapter Two\n"
        'More book one, "hi," said Ron.\n'
        "Chapter One\n"
        'Book two stuff, "hey," said Hermione.\n'
    )
    spans = split_into_chapters(text)
    assert [(s.book, s.chapter) for s in spans] == [(1, 1), (1, 2), (2, 1)]


def test_normalize_quotes_converts_curly_to_straight():
    assert normalize_quotes("“Hello,” said Harry. He didn’t know.") == (
        '"Hello," said Harry. He didn\'t know.'
    )


def test_parse_corpus_handles_curly_quotes_from_real_pdf_typesetting():
    text = _corpus("“I solemnly swear that I am up to no good,” said Harry.")
    [line] = parse_corpus(text)
    assert line.speaker == "Harry"
    assert line.text == "I solemnly swear that I am up to no good,"


def test_parse_corpus_end_to_end_produces_attributed_dialogue_lines():
    text = _corpus('"I solemnly swear that I am up to no good," said Harry.')
    lines = parse_corpus(text)
    assert len(lines) == 1
    line = lines[0]
    assert line.speaker == "Harry"
    assert line.book == 1
    assert line.chapter == 1
    assert line.text == "I solemnly swear that I am up to no good,"
    assert line.line_index == 0


def test_parse_corpus_skips_whitespace_only_quotes():
    text = _corpus('Harry stared at the empty quote, "\n   \n" said no one.')
    lines = parse_corpus(text)
    assert lines == []


def test_parse_corpus_marks_unattributed_lines_unknown_by_default():
    text = _corpus('The letter read: "Dear Mr. Potter, we are pleased to inform you."')
    [line] = parse_corpus(text)
    assert line.speaker == UNKNOWN_SPEAKER


def test_parse_corpus_skips_narrator_lines_without_quotes():
    text = _corpus("Harry walked down the corridor, thinking about the match.")
    lines = parse_corpus(text)
    assert lines == []
