"""Tests for the Simple Wikipedia plain-style experiment."""

from rvnn_text.genia import _prune_punct, parse_ptb
from rvnn_text.grammar import flatten
from rvnn_text.simplewiki import build_grammar, is_plain, load_plain_trees  # noqa: F401
from rvnn_text.sst import binarize


def _tree(text: str):
    t = _prune_punct(parse_ptb(text))
    assert t is not None
    return binarize(t)


def test_is_plain_accepts_plain_svo():
    tree = _tree(
        "(ROOT (S (NP (DT A) (NN bridge)) (VP (VBZ spans) (NP (DT this) (NN river))) (. .)))"
    )
    assert is_plain(tree)
    assert flatten(tree) == ["A", "bridge", "spans", "this", "river"]


def test_is_plain_rejects_embedded_clause():
    tree = _tree(
        "(ROOT (S (NP (DT The) (NN girl)) (VP (VBZ thinks) (SBAR (IN that) (S (NP (PRP she)) (VP (VBZ is) (JJ smart))))) (. .)))"
    )
    assert not is_plain(tree)


def test_is_plain_rejects_proper_noun():
    tree = _tree(
        "(ROOT (S (NP (NNP John)) (VP (VBD saw) (NP (DT the) (NN dog))) (. .)))"
    )
    assert not is_plain(tree)


def test_is_plain_rejects_punct_miscast_as_word():
    # parser artifact: the quote char tagged as JJ slips past POS pruning
    tree = _tree(
        "(ROOT (S (NP (DT A) (JJ '') (NN knot)) (VP (VBZ is) (NP (DT a) (NN unit))) (. .)))"
    )
    assert not is_plain(tree)


def test_is_plain_rejects_too_long():
    tree = _tree(
        "(ROOT (S (NP (DT A) (NN bridge)) (VP (VBZ spans) (NP (DT this) (JJ long) (JJ wide) (JJ old) (JJ cold) (JJ deep) (NN river))) (. .)))"
    )
    assert not is_plain(tree)


def test_grammar_from_plain_trees_has_real_categories():
    trees = [
        _tree("(ROOT (S (NP (DT A) (NN bridge)) (VP (VBZ spans) (NP (DT this) (NN river))) (. .)))"),
        _tree("(ROOT (S (NP (DT The) (NN cat)) (VP (VBD ate) (NP (DT a) (JJ small) (NN fish))) (. .)))"),
    ]
    g = build_grammar(trees)
    assert g.start == "ROOT"
    assert "NP" in g.internal and "VP" in g.internal
    assert ("DT", "NN") in g.productions["NP"]
    assert ("VBZ", "NP") in g.productions["VP"] or ("VBZ", "VP") in g.productions["VP"]
    assert all((("<unk>",)) in g.productions[p] for p in g.preterminals)
