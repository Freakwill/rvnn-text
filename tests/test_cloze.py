"""Tests for cloze and auto-completion (masked-word reconstruction)."""

import torch

from rvnn_text.data import make_corpus
from rvnn_text.grammar import flatten, make_story_grammar
from rvnn_text.train import train_model
from rvnn_text.utils import set_seed


def _trained_model(n=400, epochs=10, mask_frac=0.2):
    set_seed(0)
    g = make_story_grammar()
    corpus = make_corpus(g, n, seed=0)
    model, _ = train_model(
        g, corpus, dim=16, epochs=epochs, lr=1e-2,
        device=torch.device("cpu"), quiet=True, mask_frac=mask_frac,
    )
    model.eval()
    return model


def test_cloze_preserves_unmasked_words_and_fills_valid_word():
    model = _trained_model()
    r = model.cloze("the happy cat sees a dog", mask_indices=[1], greedy=True)
    assert r is not None
    words = r["output"].split()
    assert len(words) == 6
    assert words[0] == "the" and words[2] == "cat"
    assert words[3] == "sees" and words[4] == "a" and words[5] == "dog"
    adj_words = {rhs[0] for rhs in model.grammar.productions["Adj"]}
    assert words[1] in adj_words, f"masked Adj position filled with {words[1]!r}"


def test_cloze_fills_all_masked_positions():
    model = _trained_model()
    r = model.cloze("every clever girl likes the robot", mask_indices=[1, 3], greedy=True)
    assert r is not None
    words = r["output"].split()
    assert len(words) == 6
    assert words[0] == "every" and words[2] == "girl"
    assert words[4] == "the" and words[5] == "robot"
    assert r["input"].count("[MASK]") == 2


def test_cloze_returns_none_for_unparseable():
    model = _trained_model()
    assert model.cloze("the cat sees a dragon", mask_indices=[1]) is None


def test_continue_sentence_keeps_prefix():
    model = _trained_model()
    r = model.continue_sentence("the happy cat sees a dog", keep=4, greedy=True)
    assert r is not None
    words = r["output"].split()
    assert words[:4] == ["the", "happy", "cat", "sees"]
    assert len(words) == 6
    assert words[4] in model.grammar.words and words[5] in model.grammar.words


def test_continue_sentence_story_level():
    model = _trained_model()
    r = model.continue_sentence("Alice likes a cat Bob chases the dog", keep=4, greedy=True)
    assert r is not None
    words = r["output"].split()
    assert words[:4] == ["Alice", "likes", "a", "cat"]
    assert len(words) == 8
    assert all(w in model.grammar.words for w in words[4:])


def test_masked_training_still_learns_grammar():
    set_seed(0)
    g = make_story_grammar()
    corpus = make_corpus(g, 400, seed=0)
    model, losses = train_model(
        g, corpus, dim=16, epochs=10, lr=1e-2,
        device=torch.device("cpu"), quiet=True, mask_frac=0.2,
    )
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    acc = model.evaluate(corpus)
    assert acc["rule_acc"] > 0.9
    assert acc["word_acc"] > 0.9


def test_generate_from_prompt_opens_with_prompt_and_is_grammatical():
    model = _trained_model()
    tree = model.generate_from_prompt(
        "every clever girl likes a cat", temperature=0.9
    )
    assert tree is not None
    assert tree.symbol == "Story"
    words = flatten(tree)
    # prompt sentence must be spliced in verbatim at the start
    assert " ".join(words[:6]) == "every clever girl likes a cat"
    # the whole thing must parse under the grammar
    assert model.grammar.parse(words) is not None


def test_generate_from_prompt_rejects_unparseable():
    model = _trained_model()
    assert model.generate_from_prompt("a dragon flies", temperature=0.9) is None
