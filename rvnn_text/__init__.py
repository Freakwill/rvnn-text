"""rvnn-text: a Recursive Neural Network (RvNN) for grammar-driven text generation.

Unlike a Recurrent Neural Network (RNN), which reads text as a flat
left-to-right sequence, an RvNN consumes the *parse tree* of a sentence and
composes child vectors into parent vectors recursively. This package couples
the RvNN with a context-free grammar so that generation is always
grammatically valid.
"""

from .data import make_corpus, train_test_split
from .grammar import (
    DEFAULT_PRODUCTIONS,
    STORY_PRODUCTIONS,
    Grammar,
    Node,
    flatten,
    make_story_grammar,
    render,
    to_sentence,
)
from .model import RvNNText
from .utils import get_device, set_seed

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_PRODUCTIONS",
    "STORY_PRODUCTIONS",
    "Grammar",
    "Node",
    "RvNNText",
    "flatten",
    "get_device",
    "make_corpus",
    "make_story_grammar",
    "render",
    "set_seed",
    "to_sentence",
    "train_test_split",
]
