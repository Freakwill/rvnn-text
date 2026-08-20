import torch

from rvnn_text.grammar import DEFAULT_PRODUCTIONS, Grammar, flatten
from rvnn_text.model import RvNNText


def _model(dim: int = 16) -> tuple[RvNNText, Grammar]:
    g = Grammar(DEFAULT_PRODUCTIONS)
    return RvNNText(g, dim=dim), g


def test_encode_shape():
    model, g = _model()
    tree = g.parse_sentence("the cat sees a dog")
    h = model.encode(tree)
    assert h.shape == (16,)
    assert h.dtype == torch.float32


def test_training_loss_is_scalar_with_grad():
    model, g = _model()
    tree = g.parse_sentence("the cat sees a dog")
    loss, metrics = model.training_loss(tree)
    assert loss.dim() == 0
    assert set(metrics) == {"rule", "word", "recon", "total"}
    loss.backward()  # must not raise
    assert model.start_embedding.grad is not None
    assert model.word_emb.weight.grad is not None


def test_generate_returns_grammatical_tree():
    model, g = _model()
    model.eval()
    with torch.no_grad():
        for _ in range(50):
            tree = model.generate(max_depth=20)
            words = flatten(tree)
            assert words
            assert g.parse(words) is not None


def test_generate_sentence_is_string():
    model, _ = _model()
    model.eval()
    with torch.no_grad():
        sent = model.generate_sentence()
    assert isinstance(sent, str)
    assert len(sent.split()) >= 2
