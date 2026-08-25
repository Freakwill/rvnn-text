"""Tests for the SST-5 sentiment experiment (tree parsing / binarization)."""

from rvnn_text.grammar import flatten
from rvnn_text.sst import binarize, build_grammar, parse_tree


def _max_arity(node) -> int:
    if node.is_leaf:
        return 0
    return max([len(node.children)] + [_max_arity(c) for c in node.children])


def test_parse_tree_labels_and_words():
    tree = parse_tree("(3 (2 (2 The) (2 Rock)) (4 (3 (2 is) (2 great))) (2 .))")
    assert tree.label == 3
    assert not tree.is_leaf
    np_node = tree.children[0]               # (2 (2 The) (2 Rock))
    assert np_node.label == 2 and not np_node.is_leaf
    assert np_node.children[0].is_leaf and np_node.children[0].word == "The"
    assert flatten(tree) == ["The", "Rock", "is", "great", "."]


def test_parse_single_word_wraps_in_unary():
    tree = parse_tree("(1 Boring)")
    assert not tree.is_leaf          # wrapped in a unary N node
    assert tree.label == 1
    assert tree.children[0].is_leaf and tree.children[0].word == "Boring"


def test_binarize_reduces_arity_and_propagates_label():
    tree = parse_tree("(4 (2 a) (3 b) (1 c) (0 d))")
    b = binarize(tree)
    assert _max_arity(b) <= 2
    assert flatten(b) == ["a", "b", "c", "d"]
    # every internal node keeps the root's label after binarization
    labels = []

    def walk(n):
        if not n.is_leaf:
            labels.append(n.label)
            for c in n.children:
                walk(c)

    walk(b)
    assert labels and all(l == 4 for l in labels)


def test_build_grammar_generic_symbols():
    trees = [
        binarize(parse_tree("(3 (2 a) (4 (2 b) (2 c)))")),
        binarize(parse_tree("(1 d)")),
    ]
    g = build_grammar(trees)
    assert g.start == "N"
    assert g.internal == ["N"]
    assert g.preterminals == ["W"]
    assert ("W",) in g.productions["N"]        # unary rule for single words
    assert ("a",) in g.productions["W"]
