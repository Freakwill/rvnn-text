"""Tests for the GENIA treebank experiment (PTB parsing / grammar induction)."""

from rvnn_text.genia import _prune_punct, build_grammar, parse_ptb
from rvnn_text.grammar import flatten


def test_parse_ptb_categories_and_words():
    tree = parse_ptb("(S1 (S (NP (DT the) (NN cat)) (VP (VBP is) (JJ small))))")
    assert tree.symbol == "S1"
    np_node = tree.children[0].children[0]
    assert np_node.symbol == "NP"
    assert np_node.children[0].symbol == "DT" and np_node.children[0].word == "the"
    assert flatten(tree) == ["the", "cat", "is", "small"]


def test_prune_punct_removes_punctuation():
    tree = parse_ptb("(S1 (S (NP (DT the) (NN cat)) (. .) (VP (VBZ is) (JJ fat))))")
    pruned = _prune_punct(tree)
    assert pruned is not None
    assert flatten(pruned) == ["the", "cat", "is", "fat"]
    assert all(not n.is_leaf or n.symbol not in {".", ",", ":"} for n in _nodes(pruned))


def test_build_grammar_real_categories():
    trees = [
        _prune_punct(parse_ptb("(S1 (S (NP (DT the) (NN cat)) (VP (VBZ is) (JJ fat))))")),
        _prune_punct(parse_ptb("(S1 (S (NP (NNP John)) (VP (VBD saw) (NP (DT the) (NN dog)))))")),
    ]
    trees = [t for t in trees if t is not None]
    g = build_grammar(trees)
    assert g.start == "S1"
    assert "NP" in g.internal and "VP" in g.internal
    assert "DT" in g.preterminals and "NN" in g.preterminals
    assert ("DT", "NN") in g.productions["NP"]
    # punctuation is gone and <unk> is available under every preterminal
    assert all(p not in g.preterminals for p in [".", ",", ":"])
    assert all(("<unk>",) in g.productions[p] for p in g.preterminals)


def _nodes(node):
    yield node
    for c in node.children:
        yield from _nodes(c)
