"""Tests for the left-recursion robustness of Grammar.parse."""

from rvnn_text.grammar import Grammar


def _cyclic_grammar() -> Grammar:
    """Grammar with an indirect left-recursion cycle (NP -> ADJP -> ... -> NP)."""
    productions = {
        "ROOT": [("S",)],
        "S": [("NP", "VP")],
        "NP": [("DT", "NN"), ("DT", "NP"), ("JJ", "NN"), ("ADJP",)],
        "ADJP": [("ADVP", "JJ")],
        "ADVP": [("CC", "RB")],
        "CC": [("DT",), ("NP",)],      # cycle edge: NP -> ADJP -> ADVP -> CC -> NP
        "DT": [("the",), ("a",)],
        "NN": [("cat",), ("dog",)],
        "VP": [("VBZ", "NP")],
        "VBZ": [("likes",), ("sees",)],
        "JJ": [("big",), ("small",)],
        "RB": [("very",), ("really",)],
    }
    return Grammar(productions, start="ROOT")


def test_parse_terminates_on_left_recursive_grammar():
    g = _cyclic_grammar()
    assert g.parse_sentence("the cat likes a dog") is not None
    assert g.parse_sentence("the big cat likes a small dog") is not None


def test_parse_returns_none_for_out_of_grammar():
    g = _cyclic_grammar()
    assert g.parse_sentence("the cat likes") is None
    assert g.parse_sentence("zzz zzz zzz zzz") is None
