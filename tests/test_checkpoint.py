import torch

from rvnn_text.checkpoint import load_model, save_checkpoint
from rvnn_text.grammar import Grammar
from rvnn_text.model import RvNNText


def test_checkpoint_roundtrip(tmp_path):
    g = Grammar()
    model = RvNNText(g, dim=16)
    path = save_checkpoint(model, tmp_path / "model.pt")
    assert path.exists()

    loaded = load_model(path, device=torch.device("cpu"))
    assert loaded.dim == 16
    assert loaded.grammar.to_dict() == g.to_dict()

    for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2)

    loaded.eval()
    sent = loaded.generate_sentence()
    assert isinstance(sent, str)
    assert len(sent.split()) >= 2
