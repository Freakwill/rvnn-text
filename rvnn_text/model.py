"""Recursive Neural Network (RvNN) for grammar-driven text generation.

Unlike a Recurrent Neural Network (RNN), which consumes a flat left-to-right
sequence, a Recursive Neural Network consumes the *parse tree* of a sentence
and composes child vectors into a parent vector bottom-up, applying the same
composition function at every internal node. This module couples the RvNN with
a context-free grammar: an encoder composes a tree into a sentence vector, and
a recursive decoder expands that vector back into a grammatical sentence.

Training follows the recursive-autoencoder recipe:
  * a supervised cross-entropy (rule + word) computed on the *true* bottom-up
    embeddings trains the encoder (word embeddings + composition weights);
  * a reconstruction loss with detached targets trains the decoder's top-down
    projections, which invert the composition and enable generation.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grammar import Grammar, Node, to_sentence


class RvNNText(nn.Module):
    """A recursive autoencoder over parse trees, with a grammar-constrained decoder.

    Encoder (bottom-up): each leaf is a word embedding; every internal node is
    produced by a shared composition function ``tanh(W_l h_l + W_r h_r + b)``.

    Decoder (top-down): starting from a learned root vector, at every internal
    node the model (a) picks a production rule and (b) for each preterminal
    child samples a word from the parent embedding, projecting the parent
    vector into an embedding for each internal child and recursing. Because
    every choice is restricted to valid productions of the grammar, the
    generated sentence is always grammatical.
    """

    def __init__(self, grammar: Grammar, dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.grammar = grammar
        self.dim = dim
        self.words = grammar.words
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.preterminal_set = set(grammar.preterminals)
        self.internal_set = set(grammar.internal)

        # production-rule index: internal nonterminal -> {rhs tuple -> index}
        self.rule_index: dict[str, dict[tuple[str, ...], int]] = {
            x: {rhs: i for i, rhs in enumerate(grammar.productions[x])}
            for x in grammar.internal
        }
        # word index: preterminal -> {word -> local index}
        self.word_index: dict[str, dict[str, int]] = {
            x: {rhs[0]: i for i, rhs in enumerate(grammar.productions[x])}
            for x in grammar.preterminals
        }
        self.preterm_words: dict[str, list[str]] = {
            x: [rhs[0] for rhs in grammar.productions[x]] for x in grammar.preterminals
        }

        self.word_emb = nn.Embedding(len(self.words), dim)

        # Encoder: shared recursive composition weights (one function, applied
        # recursively at every internal node -- the defining RvNN property).
        self.W_left = nn.Linear(dim, dim, bias=False)
        self.W_right = nn.Linear(dim, dim, bias=False)
        self.W_unary = nn.Linear(dim, dim, bias=False)
        self.b_binary = nn.Parameter(torch.zeros(dim))
        self.b_unary = nn.Parameter(torch.zeros(dim))

        # Decoder: per-nonterminal production-rule predictors, applied to the
        # node's own embedding.
        self.prod_predictors = nn.ModuleDict(
            {x: nn.Linear(dim, len(grammar.productions[x])) for x in grammar.internal}
        )
        # Decoder: per-preterminal word predictors, applied to the PARENT's
        # embedding (predicting a child word from its parent is non-degenerate,
        # unlike predicting a word from its own embedding).
        self.word_predictors = nn.ModuleDict(
            {x: nn.Linear(dim, len(grammar.productions[x])) for x in grammar.preterminals}
        )
        # Decoder: shared top-down projection (inverse of composition), used to
        # produce an embedding for each internal child. D_unary is only used if
        # the grammar contains unary internal rules (this demo grammar does not).
        self.D_left = nn.Linear(dim, dim)
        self.D_right = nn.Linear(dim, dim)
        self.D_unary = nn.Linear(dim, dim)

        # Learned root vector that generation starts from.
        self.start_embedding = nn.Parameter(torch.randn(dim) * 0.1)

        # Mask embedding: used in cloze / auto-completion. A masked leaf is
        # encoded with this vector, so its parent must predict the word from
        # the surrounding context only (the tree analogue of BERT's [MASK]).
        self.mask_embedding = nn.Parameter(torch.randn(dim) * 0.1)

        self.dropout = nn.Dropout(dropout)

    # -- encoding ----------------------------------------------------------

    def compose(self, hs: list[torch.Tensor]) -> torch.Tensor:
        """Compose child vectors into a parent vector (shared, recursive)."""
        if len(hs) == 1:
            return torch.tanh(self.W_unary(hs[0]) + self.b_unary)
        if len(hs) == 2:
            return torch.tanh(self.W_left(hs[0]) + self.W_right(hs[1]) + self.b_binary)
        raise ValueError(f"unsupported arity: {len(hs)}")

    def encode(self, node: Node) -> torch.Tensor:
        """Encode a tree bottom-up into a single vector at its root."""
        device = self.word_emb.weight.device
        if node.is_leaf:
            idx = torch.tensor(self.word_to_idx[node.word], device=device)
            return self.word_emb(idx)
        return self.compose([self.encode(c) for c in node.children])

    def encode_sentence(self, sentence: str) -> torch.Tensor | None:
        """Parse a sentence and return its root encoding (``None`` if unparseable)."""
        tree = self.grammar.parse_sentence(sentence)
        if tree is None:
            return None
        return self.encode(tree)

    def _encode_tree(self, root: Node) -> dict[int, torch.Tensor]:
        """Encode every node bottom-up, keyed by ``id(node)``."""
        device = self.word_emb.weight.device
        true_h: dict[int, torch.Tensor] = {}

        def bottom_up(node: Node) -> torch.Tensor:
            if node.is_leaf:
                idx = torch.tensor(self.word_to_idx[node.word], device=device)
                h = self.word_emb(idx)
            else:
                h = self.compose([bottom_up(c) for c in node.children])
            true_h[id(node)] = h
            return h

        bottom_up(root)
        return true_h

    # -- masked encoding (cloze / auto-completion) -------------------------

    def _encode_tree_masked(self, root: Node, masked: set[int]) -> dict[int, torch.Tensor]:
        """Encode every node bottom-up; leaves whose ``id()`` is in ``masked``
        use the mask embedding instead of their word embedding."""
        device = self.word_emb.weight.device
        true_h: dict[int, torch.Tensor] = {}

        def bottom_up(node: Node) -> torch.Tensor:
            if node.is_leaf:
                if id(node) in masked:
                    return self.mask_embedding
                assert node.word is not None  # a leaf always carries a word
                return self.word_emb(
                    torch.tensor(self.word_to_idx[node.word], device=device)
                )
            h = self.compose([bottom_up(c) for c in node.children])
            true_h[id(node)] = h
            return h

        bottom_up(root)
        return true_h

    def _leaf_pairs(self, root: Node) -> list[tuple[Node, Node]]:
        """Return ``(leaf, parent)`` pairs in left-to-right order.

        The parent is the leaf's immediate parent node, whose embedding the
        word predictor is conditioned on.
        """
        pairs: list[tuple[Node, Node]] = []

        def rec(node: Node) -> None:
            if node.is_leaf:
                return
            for child in node.children:
                if child.is_leaf:
                    pairs.append((child, node))
                else:
                    rec(child)

        rec(root)
        return pairs

    def _fill_masked(self, tree: Node, mask_indices: list[int], greedy: bool,
                     temperature: float) -> tuple[list[str], list[str], list[str]]:
        """Core of cloze/auto-completion: mask leaves, encode, predict each
        masked word from its parent embedding.

        Returns ``(original_words, filled_words, masked_markup)`` where
        ``masked_markup`` renders masked positions as ``[MASK]``.
        """
        pairs = self._leaf_pairs(tree)
        masked = {id(pairs[i][0]) for i in mask_indices}
        true_h = self._encode_tree_masked(tree, masked)
        original: list[str] = []
        filled: list[str] = []
        markup: list[str] = []
        for i, (leaf, parent) in enumerate(pairs):
            assert leaf.word is not None  # a leaf always carries a word
            original.append(leaf.word)
            if i in mask_indices:
                logits = self.word_predictors[leaf.symbol](true_h[id(parent)])
                idx = self._sample(logits, temperature, greedy)
                filled.append(self.preterm_words[leaf.symbol][idx])
                markup.append("[MASK]")
            else:
                filled.append(leaf.word)
                markup.append(leaf.word)
        return original, filled, markup

    @torch.no_grad()
    def cloze(self, sentence: str, mask_indices: list[int] | None = None,
              greedy: bool = True, temperature: float = 1.0) -> dict | None:
        """Fill masked words via encode-decode (tree analogue of MLM).

        Args:
            sentence: a sentence (or story) in the grammar's vocabulary.
            mask_indices: 0-based leaf positions to mask; defaults to a couple
                of interior words (skipping the first and last leaf).
            greedy: take the argmax word (True) or sample (False).
            temperature: sampling temperature when ``greedy`` is False.

        Returns a dict with ``input`` (``[MASK]`` markup), ``output`` (filled
        sentence) and ``original``, or ``None`` if the sentence is unparseable.
        """
        tree = self.grammar.parse_sentence(sentence)
        if tree is None:
            return None
        pairs = self._leaf_pairs(tree)
        if not pairs:
            return None
        if mask_indices is None:
            mask_indices = [i for i in range(1, len(pairs) - 1)][:2]
        mask_indices = [i for i in mask_indices if 0 <= i < len(pairs)]
        if not mask_indices:
            return None
        original, filled, markup = self._fill_masked(tree, mask_indices, greedy, temperature)
        return {
            "input": " ".join(markup),
            "output": " ".join(filled),
            "original": " ".join(original),
            "mask_indices": mask_indices,
        }

    @torch.no_grad()
    def continue_sentence(self, sentence: str, keep: float | int = 0.5,
                          greedy: bool = True, temperature: float = 1.0) -> dict | None:
        """Auto-complete a sentence/story by masking its tail and filling it.

        Args:
            sentence: a sentence (or story) in the grammar's vocabulary.
            keep: how many leading words to keep — a fraction (``0.5``) or an
                absolute count (``4``). Everything after is masked and filled,
                which is the auto-continuation (续写) behaviour.
            greedy / temperature: as in :meth:`cloze`.

        Returns a dict with ``input`` (kept words + ``[MASK]`` tail), ``output``
        (completed text) and ``original``, or ``None`` if unparseable.
        """
        tree = self.grammar.parse_sentence(sentence)
        if tree is None:
            return None
        pairs = self._leaf_pairs(tree)
        n = len(pairs)
        cut = int(keep * n) if isinstance(keep, float) else int(keep)
        cut = max(1, min(cut, n - 1))
        original, filled, markup = self._fill_masked(
            tree, list(range(cut, n)), greedy, temperature
        )
        return {
            "input": " ".join(markup),
            "output": " ".join(filled),
            "original": " ".join(original),
            "mask_indices": list(range(cut, n)),
        }

    @torch.no_grad()
    def evaluate(self, trees: list[Node]) -> dict[str, float]:
        """Return held-out rule- and word-prediction accuracy.

        This quantifies how well the RvNN has internalized the grammar: the
        rule predictor re-derives each node's production rule from its composed
        embedding (a parsing task), and the word predictor fills each preterminal.
        """
        n_rule = rule_correct = 0
        n_word = word_correct = 0
        for root in trees:
            true_h = self._encode_tree(root)

            def check(node: Node) -> None:
                nonlocal n_rule, rule_correct, n_word, word_correct
                if node.is_leaf:
                    return
                h = true_h[id(node)]
                pred = int(self.prod_predictors[node.symbol](h).argmax().item())
                truth = self.rule_index[node.symbol][tuple(c.symbol for c in node.children)]
                n_rule += 1
                rule_correct += pred == truth
                for child in node.children:
                    if child.is_leaf:
                        p = int(self.word_predictors[child.symbol](h).argmax().item())
                        t = self.word_index[child.symbol][child.word]
                        n_word += 1
                        word_correct += p == t
                for child in node.children:
                    check(child)

            check(root)
        return {
            "rule_acc": rule_correct / max(n_rule, 1),
            "word_acc": word_correct / max(n_word, 1),
        }

    # -- training objective ------------------------------------------------

    def training_loss(
        self, root: Node, recon_weight: float = 1.0, l2_weight: float = 0.0,
        mask_frac: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the training loss for one tree plus its component breakdown.

        The supervised rule/word cross-entropies are computed on true bottom-up
        embeddings (training the encoder); the reconstruction loss uses detached
        targets (training the decoder projections) so the two objectives cannot
        interact to collapse the embeddings.

        When ``mask_frac > 0``, a random subset of leaves is masked (tree
        analogue of BERT's MLM) and the word predictor must recover the masked
        words from the *masked* bottom-up context, teaching cloze /
        auto-completion.  Unmasked words, rule prediction and reconstruction
        still use the clean embeddings, so evaluation accuracy is unaffected.
        """
        device = self.word_emb.weight.device
        true_h = self._encode_tree(root)
        root_h = true_h[id(root)]

        # Masked pass: only for the word loss on masked leaves.
        masked_ids: set[int] = set()
        if mask_frac > 0:
            for leaf, _ in self._leaf_pairs(root):
                if random.random() < mask_frac:
                    masked_ids.add(id(leaf))
        masked_h = self._encode_tree_masked(root, masked_ids) if masked_ids else true_h

        rule_loss = torch.zeros((), device=device)
        word_loss = torch.zeros((), device=device)
        recon_loss = torch.zeros((), device=device)
        n_rule = 0
        n_word = 0
        n_recon = 0

        def collect(node: Node) -> None:
            nonlocal rule_loss, word_loss, recon_loss, n_rule, n_word, n_recon
            if node.is_leaf:
                return
            h_node = true_h[id(node)]
            rule = tuple(c.symbol for c in node.children)
            rule_loss = rule_loss + F.cross_entropy(
                self.prod_predictors[node.symbol](h_node).unsqueeze(0),
                torch.tensor([self.rule_index[node.symbol][rule]], device=device),
            )
            n_rule += 1
            for pos, child in enumerate(node.children):
                if child.is_leaf:
                    # Predict the child's word from the parent's embedding. For
                    # masked leaves the parent embedding comes from the masked
                    # pass (the context the model must fill in from).
                    h_parent = (
                        masked_h[id(node)]
                        if id(child) in masked_ids
                        else h_node
                    )
                    word_loss = word_loss + F.cross_entropy(
                        self.word_predictors[child.symbol](h_parent).unsqueeze(0),
                        torch.tensor([self.word_index[child.symbol][child.word]], device=device),
                    )
                    n_word += 1
                else:
                    # Project the parent embedding down to the internal child
                    # (detached target keeps this a pure decoder objective).
                    if len(node.children) == 1:
                        proj = self.D_unary(h_node)
                    else:
                        proj = self.D_left(h_node) if pos == 0 else self.D_right(h_node)
                    recon_loss = recon_loss + F.mse_loss(proj, true_h[id(child)].detach())
                    n_recon += 1
            for child in node.children:
                collect(child)

        collect(root)
        # Anchor the learned root vector toward the corpus's root embeddings.
        recon_loss = recon_loss + F.mse_loss(self.start_embedding, root_h.detach())
        n_recon += 1

        total = (
            rule_loss / max(n_rule, 1)
            + word_loss / max(n_word, 1)
            + recon_weight * recon_loss / max(n_recon, 1)
        )
        if l2_weight > 0.0:
            total = total + l2_weight * (self.word_emb.weight**2).sum()

        metrics = {
            "rule": (rule_loss / max(n_rule, 1)).item(),
            "word": (word_loss / max(n_word, 1)).item(),
            "recon": (recon_loss / max(n_recon, 1)).item(),
            "total": total.item(),
        }
        return total, metrics

    # -- generation --------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        temperature: float = 1.0,
        greedy: bool = False,
        max_depth: int = 12,
        seed_noise: float = 0.0,
        root_embedding: torch.Tensor | None = None,
    ) -> Node:
        """Generate a parse tree (and hence a sentence) from the decoder.

        Args:
            temperature: softmax temperature for sampling rules and words.
            greedy: if True, always take the argmax instead of sampling.
            max_depth: maximum tree depth (guards the right-recursive NOM rule).
            seed_noise: std of Gaussian noise added to the start vector.
            root_embedding: optional explicit root vector (e.g. an encoded sentence).
        """
        if root_embedding is None:
            h = self.start_embedding
            if seed_noise > 0.0:
                h = h + seed_noise * torch.randn_like(h)
        else:
            h = root_embedding
        return self._generate_node(self.grammar.start, h, temperature, greedy, max_depth)

    def _generate_node(
        self, symbol: str, h: torch.Tensor, temperature: float, greedy: bool, max_depth: int
    ) -> Node:
        # ``symbol`` is always an internal non-terminal; words are sampled at
        # the parent level from the preterminal child's predictor.
        rules = self.grammar.productions[symbol]
        logits = self.prod_predictors[symbol](h)
        if max_depth <= 0:
            # Depth guard: prefer rules whose children are all preterminals, so
            # expansion terminates in one more step; fall back to argmax.
            candidates = [
                i for i, r in enumerate(rules) if all(s in self.preterminal_set for s in r)
            ]
            if candidates:
                idx = candidates[int(logits[candidates].argmax().item())]
            else:
                idx = int(logits.argmax().item())
        else:
            idx = self._sample(logits, temperature, greedy)

        rule = rules[idx]
        children: list[Node] = []
        for pos, child_symbol in enumerate(rule):
            if child_symbol in self.preterminal_set:
                word = self._sample_word(child_symbol, h, temperature, greedy)
                children.append(Node(symbol=child_symbol, word=word))
            else:
                if len(rule) == 1:
                    h_child = self.D_unary(h)
                else:
                    h_child = self.D_left(h) if pos == 0 else self.D_right(h)
                children.append(
                    self._generate_node(child_symbol, h_child, temperature, greedy, max_depth - 1)
                )
        return Node(symbol=symbol, children=children)

    def _sample_word(self, symbol: str, parent_h: torch.Tensor, temperature: float, greedy: bool) -> str:
        """Sample a word for preterminal ``symbol`` from its parent embedding."""
        idx = self._sample(self.word_predictors[symbol](parent_h), temperature, greedy)
        return self.preterm_words[symbol][idx]

    def generate_sentence(self, **kwargs) -> str:
        """Generate a sentence string (convenience wrapper)."""
        return to_sentence(self.generate(**kwargs))

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, greedy: bool) -> int:
        if greedy or temperature <= 0.0:
            return int(logits.argmax().item())
        probs = F.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())
