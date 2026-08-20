import torch

from rvnn_text.data import make_corpus
from rvnn_text.grammar import DEFAULT_PRODUCTIONS, Grammar, flatten
from rvnn_text.train import train_model
from rvnn_text.utils import set_seed


def test_training_decreases_loss():
    set_seed(0)
    g = Grammar(DEFAULT_PRODUCTIONS)
    corpus = make_corpus(g, 400, seed=0)
    _, losses = train_model(
        g, corpus, dim=16, epochs=8, lr=1e-2, device=torch.device("cpu"), quiet=True
    )
    assert len(losses) == 8
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_trained_model_generates_grammatical_sentences():
    set_seed(0)
    g = Grammar(DEFAULT_PRODUCTIONS)
    corpus = make_corpus(g, 400, seed=0)
    model, _ = train_model(
        g, corpus, dim=16, epochs=8, lr=1e-2, device=torch.device("cpu"), quiet=True
    )
    model.eval()
    with torch.no_grad():
        for _ in range(30):
            tree = model.generate(max_depth=20)
            assert g.parse(flatten(tree)) is not None


def test_evaluate_learns_the_grammar():
    set_seed(0)
    g = Grammar(DEFAULT_PRODUCTIONS)
    corpus = make_corpus(g, 300, seed=0)
    model, _ = train_model(
        g, corpus, dim=16, epochs=8, lr=1e-2, device=torch.device("cpu"), quiet=True
    )
    model.eval()
    acc = model.evaluate(corpus)
    assert 0.0 <= acc["rule_acc"] <= 1.0
    assert 0.0 <= acc["word_acc"] <= 1.0
    assert acc["rule_acc"] > 0.9
    assert acc["word_acc"] > 0.9
