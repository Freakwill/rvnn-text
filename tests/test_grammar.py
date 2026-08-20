import random

from rvnn_text.grammar import (
    DEFAULT_PRODUCTIONS,
    Grammar,
    flatten,
    render,
    to_sentence,
)


def test_grammar_validates_and_classifies_symbols():
    g = Grammar(DEFAULT_PRODUCTIONS)
    assert g.start == "S"
    assert set(g.internal) == {"S", "NP", "NOM", "VP"}
    assert set(g.preterminals) == {"Det", "Proper", "Adj", "Noun", "Verb", "Adv"}


def test_sample_tree_is_valid_and_parse_roundtrip():
    g = Grammar(DEFAULT_PRODUCTIONS)
    rng = random.Random(0)
    for _ in range(300):
        tree = g.sample_tree(rng)
        words = flatten(tree)
        assert words, "sampled tree must not be empty"
        assert g.parse(words) is not None


def test_parse_known_sentence():
    g = Grammar(DEFAULT_PRODUCTIONS)
    tree = g.parse_sentence("the happy cat sees a dog")
    assert tree is not None
    assert tree.symbol == "S"
    assert flatten(tree) == ["the", "happy", "cat", "sees", "a", "dog"]


def test_parse_rejects_unknown_word():
    g = Grammar(DEFAULT_PRODUCTIONS)
    assert g.parse_sentence("the cat sees a dragon") is None


def test_parse_rejects_ungrammatical_order():
    g = Grammar(DEFAULT_PRODUCTIONS)
    assert g.parse_sentence("cat the sees dog") is None


def test_render_contains_symbols_and_words():
    g = Grammar(DEFAULT_PRODUCTIONS)
    tree = g.parse_sentence("Alice likes every dog")
    text = render(tree)
    assert "S" in text
    assert "Alice" in text
    assert "VP" in text


def test_to_sentence_roundtrip():
    g = Grammar(DEFAULT_PRODUCTIONS)
    tree = g.parse_sentence("Bob eats a robot")
    assert to_sentence(tree) == "Bob eats a robot"
