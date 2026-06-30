from pipeline.common.aliases import resolve_aliases


def test_resolve_aliases_includes_canonical_name_and_known_aliases():
    assert resolve_aliases("Ron Weasley") == ["Ron Weasley", "Ron"]


def test_resolve_aliases_falls_back_to_bare_name_for_unregistered_character():
    assert resolve_aliases("Albus Dumbledore") == ["Albus Dumbledore"]


def test_resolve_aliases_matches_raw_attribution_label():
    assert "Harry" in resolve_aliases("Harry Potter")
