"""Tests for the multi-sentence story grammar."""

import random

from rvnn_text.grammar import flatten, make_story_grammar


def _count_s(node) -> int:
    """Count how many S sentences a (story) tree contains."""
    return (1 if node.symbol == "S" else 0) + sum(_count_s(c) for c in node.children)


def test_story_grammar_start_and_symbols():
    g = make_story_grammar()
    assert g.start == "Story"
    assert "Story" in g.internal
    assert "S" in g.internal
    assert g.preterminals


def test_story_sample_parse_roundtrip():
    g = make_story_grammar()
    rng = random.Random(0)
    for _ in range(300):
        tree = g.sample_tree(rng)
        assert g.parse(flatten(tree)) is not None


def test_story_can_have_multiple_sentences():
    g = make_story_grammar()
    rng = random.Random(0)
    multi = 0
    for _ in range(500):
        if _count_s(g.sample_tree(rng)) > 1:
            multi += 1
    assert multi > 0, "sampling 500 stories should produce some multi-sentence ones"


def test_story_parses_single_and_multi_sentence():
    g = make_story_grammar()
    single = g.parse_sentence("the happy cat sees a dog")
    assert single is not None
    assert single.symbol == "Story"
    assert _count_s(single) == 1
    multi = g.parse_sentence("Alice likes a cat Bob chases the dog")
    assert multi is not None
    assert _count_s(multi) == 2
