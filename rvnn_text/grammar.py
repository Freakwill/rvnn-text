"""Context-free grammar, parse trees, and sentence sampling/parsing.

A Recursive Neural Network (RvNN) does not consume text as a flat sequence.
It consumes the *parse tree* of a sentence and composes child vectors into a
parent vector, bottom-up. This module supplies the grammar and the tree data
structure that the RvNN operates on; it is also the bridge to classical
grammar/parsing analysis (context-free grammar, constituency trees).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Node:
    """A node in a constituency parse tree.

    A leaf node represents a preterminal that directly yields a word: it has
    ``word`` set and no children (e.g. ``Node("Noun", word="cat")``). An
    internal node has ``word=None`` and one or two child ``Node`` objects
    (e.g. ``Node("S", children=[np, vp])``).
    """

    symbol: str
    word: str | None = None
    children: list["Node"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        """True if this node yields a terminal word directly."""
        return self.word is not None


# A small English context-free grammar. Title-case symbols are non-terminals;
# lower-case symbols inside preterminal rules are terminal words.
DEFAULT_PRODUCTIONS: dict[str, list[tuple[str, ...]]] = {
    "S": [("NP", "VP")],
    "NP": [("Det", "NOM"), ("Proper",)],
    "NOM": [("Noun",), ("Adj", "NOM")],
    "VP": [("Verb",), ("Verb", "NP"), ("Verb", "Adv")],
    "Det": [("the",), ("a",), ("every",)],
    "Proper": [("Alice",), ("Bob",), ("Mary",)],
    "Adj": [("happy",), ("red",), ("small",), ("clever",)],
    "Noun": [("cat",), ("dog",), ("robot",), ("tree",), ("girl",), ("boy",)],
    "Verb": [("sees",), ("likes",), ("chases",), ("eats",)],
    "Adv": [("quickly",), ("slowly",), ("loudly",)],
}

# Story grammar: wraps the sentence grammar into multi-sentence paragraphs.
# ``Story -> S | S Story`` (right-recursive, so the top-down parser terminates)
# makes the RvNN recursive at the paragraph level too.
STORY_PRODUCTIONS: dict[str, list[tuple[str, ...]]] = {
    "Story": [("S",), ("S", "Story")],
    **DEFAULT_PRODUCTIONS,
}


def make_story_grammar() -> "Grammar":
    """Return a grammar whose start symbol generates 1..N sentence paragraphs."""
    return Grammar(STORY_PRODUCTIONS, start="Story")


class Grammar:
    """A context-free grammar (CFG) plus helpers to sample, parse and render trees.

    The grammar is constrained to arity <= 2 (unary or binary rules), which is
    the natural input for a recursive binary composition function.
    """

    def __init__(
        self,
        productions: dict[str, list[tuple[str, ...]]] | None = None,
        start: str = "S",
    ) -> None:
        if productions is None:
            productions = DEFAULT_PRODUCTIONS
        self.productions: dict[str, list[tuple[str, ...]]] = {
            lhs: [tuple(rhs) for rhs in rules] for lhs, rules in productions.items()
        }
        self.start = start
        self.nonterminals: list[str] = sorted(self.productions)
        preterminal_set = {
            lhs
            for lhs, rules in self.productions.items()
            if all(len(rhs) == 1 and rhs[0] not in self.productions for rhs in rules)
        }
        self.preterminals: list[str] = sorted(preterminal_set)
        self.internal: list[str] = [
            x for x in self.nonterminals if x not in preterminal_set
        ]
        self.words: list[str] = sorted(
            {rhs[0] for lhs in self.preterminals for rhs in self.productions[lhs]}
        )
        self.validate()

    def validate(self) -> None:
        """Raise if the grammar is malformed or has unsupported arity."""
        for lhs, rules in self.productions.items():
            if not rules:
                raise ValueError(f"non-terminal {lhs!r} has no productions")
            for rhs in rules:
                if len(rhs) == 1 and rhs[0] not in self.productions:
                    continue  # preterminal -> word
                for sym in rhs:
                    if sym not in self.productions:
                        raise ValueError(
                            f"{lhs} -> {rhs}: {sym!r} is neither a non-terminal "
                            "nor a preterminal word"
                        )
                if len(rhs) > 2:
                    raise ValueError(
                        f"{lhs} -> {rhs}: arity > 2; binarize the grammar first"
                    )

    # -- sampling ----------------------------------------------------------

    def sample_tree(self, rng: random.Random | None = None) -> Node:
        """Sample a random parse tree from the grammar."""
        rng = rng or random.Random()
        return self._expand(self.start, rng)

    def _expand(self, symbol: str, rng: random.Random) -> Node:
        rhs = rng.choice(self.productions[symbol])
        if len(rhs) == 1 and rhs[0] not in self.productions:
            return Node(symbol=symbol, word=rhs[0])  # preterminal -> word
        return Node(symbol=symbol, children=[self._expand(s, rng) for s in rhs])

    def sample_sentence(self, rng: random.Random | None = None) -> str:
        """Sample a random sentence (flat string of terminal words)."""
        return to_sentence(self.sample_tree(rng))

    # -- parsing -----------------------------------------------------------

    def parse(self, tokens: list[str] | tuple[str, ...]) -> Node | None:
        """Parse a token sequence into a tree (top-down with memoization).

        Returns ``None`` if the sentence is outside the grammar. The grammar is
        assumed to be free of left recursion.
        """
        tokens = tuple(tokens)
        memo: dict[tuple[str, int], list[tuple[Node, int]]] = {}

        def rec(symbol: str, pos: int) -> list[tuple[Node, int]]:
            key = (symbol, pos)
            if key in memo:
                return memo[key]
            results: list[tuple[Node, int]] = []
            for rhs in self.productions[symbol]:
                if len(rhs) == 1 and rhs[0] not in self.productions:
                    if pos < len(tokens) and tokens[pos] == rhs[0]:
                        results.append((Node(symbol, word=rhs[0]), pos + 1))
                else:
                    states: list[tuple[int, list[Node]]] = [(pos, [])]
                    for s in rhs:
                        next_states: list[tuple[int, list[Node]]] = []
                        for cur, children in states:
                            for child, end in rec(s, cur):
                                next_states.append((end, children + [child]))
                        states = next_states
                    for end, children in states:
                        results.append((Node(symbol, children=children), end))
            memo[key] = results
            return results

        for node, end in rec(self.start, 0):
            if end == len(tokens):
                return node
        return None

    def parse_sentence(self, sentence: str) -> Node | None:
        """Parse a whitespace-separated sentence into a tree, or ``None``."""
        return self.parse(sentence.split())

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "productions": {k: [list(r) for r in v] for k, v in self.productions.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Grammar":
        return cls(
            {k: [tuple(r) for r in v] for k, v in data["productions"].items()},
            start=data.get("start", "S"),
        )


def flatten(node: Node) -> list[str]:
    """Return the terminal words of a tree in left-to-right order."""
    if node.is_leaf:
        return [node.word]  # type: ignore[list-item]
    return [w for child in node.children for w in flatten(child)]


def to_sentence(node: Node) -> str:
    """Return the sentence (joined terminal words) of a tree."""
    return " ".join(flatten(node))


def render(node: Node) -> str:
    """Render a tree as an ASCII sketch (for notebooks and demos)."""
    lines: list[str] = [node.symbol]

    def rec(n: Node, prefix: str, is_last: bool) -> None:
        branch = "└── " if is_last else "├── "
        label = f"{n.symbol}: {n.word}" if n.is_leaf else n.symbol
        lines.append(prefix + branch + label)
        extension = "    " if is_last else "│   "
        for i, child in enumerate(n.children):
            rec(child, prefix + extension, i == len(n.children) - 1)

    for i, child in enumerate(node.children):
        rec(child, "", i == len(node.children) - 1)
    return "\n".join(lines)
