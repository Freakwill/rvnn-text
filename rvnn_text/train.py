"""Training loop for the RvNN text model (CLI via fire)."""

from __future__ import annotations

import random

import fire
import torch

from .checkpoint import save_checkpoint
from .data import make_corpus, train_test_split
from .grammar import DEFAULT_PRODUCTIONS, Grammar, Node
from .model import RvNNText
from .utils import get_device, set_seed


@torch.no_grad()
def _eval_loss(model: RvNNText, trees: list[Node], recon_weight: float) -> float:
    """Average total loss over a set of trees (no gradient)."""
    model.eval()
    total = 0.0
    for tree in trees:
        _, m = model.training_loss(tree, recon_weight=recon_weight)
        total += m["total"]
    model.train()
    return total / max(len(trees), 1)


def train_model(
    grammar: Grammar,
    train_trees: list[Node],
    *,
    val_trees: list[Node] | None = None,
    dim: int = 64,
    epochs: int = 30,
    lr: float = 1e-3,
    recon_weight: float = 1.0,
    l2_weight: float = 0.0,
    device: torch.device | None = None,
    sample_interval: int = 0,
    quiet: bool = False,
) -> tuple[RvNNText, list[float]]:
    """Train an RvNN text model on parse trees.

    Returns the trained model (in ``train`` mode) and the list of per-epoch
    average training losses.
    """
    device = device or get_device()
    model = RvNNText(grammar, dim=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []

    model.train()
    for epoch in range(1, epochs + 1):
        trees = train_trees[:]
        random.Random(epoch).shuffle(trees)
        epoch_loss = 0.0
        epoch_rule = 0.0
        epoch_word = 0.0
        epoch_recon = 0.0
        for tree in trees:
            optimizer.zero_grad(set_to_none=True)
            loss, m = model.training_loss(tree, recon_weight=recon_weight, l2_weight=l2_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += m["total"]
            epoch_rule += m["rule"]
            epoch_word += m["word"]
            epoch_recon += m["recon"]
        n = len(trees)
        avg = epoch_loss / n
        losses.append(avg)
        if not quiet:
            line = (
                f"epoch {epoch:3d}/{epochs}  loss={avg:.4f}  "
                f"rule={epoch_rule / n:.4f}  word={epoch_word / n:.4f}  "
                f"recon={epoch_recon / n:.4f}"
            )
            if val_trees is not None:
                line += f"  val={_eval_loss(model, val_trees, recon_weight):.4f}"
            print(line)
            if sample_interval and epoch % sample_interval == 0:
                model.eval()
                print(f"        sample: {model.generate_sentence(greedy=True)}")
                model.train()

    return model, losses


def main(
    num_sentences: int = 4000,
    epochs: int = 30,
    dim: int = 64,
    lr: float = 1e-3,
    recon_weight: float = 1.0,
    l2_weight: float = 0.0,
    seed: int = 42,
    val_frac: float = 0.1,
    save: bool = True,
    out_dir: str = "checkpoints",
    sample_interval: int = 5,
) -> None:
    """Train the RvNN text model and (optionally) save a checkpoint.

    Args:
        num_sentences: number of synthetic sentences to generate for training.
        epochs: number of passes over the corpus.
        dim: embedding dimension.
        lr: Adam learning rate.
        recon_weight: weight of the reconstruction term relative to CE terms.
        l2_weight: L2 regularization on word embeddings (0 to disable).
        seed: random seed for corpus generation and weight init.
        val_frac: fraction of the corpus held out for validation.
        save: whether to write a checkpoint under ``out_dir``.
        out_dir: directory for the checkpoint.
        sample_interval: print a greedy sample every N epochs (0 to disable).
    """
    set_seed(seed)
    device = get_device()
    grammar = Grammar(DEFAULT_PRODUCTIONS)
    corpus = make_corpus(grammar, num_sentences, seed=seed)
    train_trees, val_trees = train_test_split(corpus, val_frac=val_frac, seed=seed)
    print(
        f"corpus: {len(corpus)} trees "
        f"({len(train_trees)} train / {len(val_trees)} val), device={device}"
    )

    model, _ = train_model(
        grammar,
        train_trees,
        val_trees=val_trees,
        dim=dim,
        epochs=epochs,
        lr=lr,
        recon_weight=recon_weight,
        l2_weight=l2_weight,
        device=device,
        sample_interval=sample_interval,
    )

    if save:
        path = save_checkpoint(model, f"{out_dir}/rvnn_text.pt")
        print(f"saved checkpoint to {path}")


if __name__ == "__main__":
    fire.Fire(main)


def cli() -> None:
    """Console-script entry point."""
    fire.Fire(main)
