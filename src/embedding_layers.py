"""
GENERAL TODO:
1) Replace _emb_i,v,f with an EmbeddingBag for efficiency
"""

import math
import torch
from torch import Tensor
import torch.nn as nn
from typing import Union

from .dictionaries import ThreeHotDict, AlphabetDict, ThreeHotDictArbitraryOrdering


class AlphabetEmbedding(nn.Module):
    """
    An embedding layer for one hot encoding dictionaries. Just wraps `nn.Embedding` so that it can
    be initialized directly from an `AlphabetDict`.
    """

    def __init__(self, alphabet_dict: AlphabetDict, emb_size):
        """Initialize the Alphabet embedding table

        Args:
            dictionary: an AlphabetDict
            emb_size: the embedding size
        """
        super(AlphabetEmbedding, self).__init__()
        self.alphabet_dict = alphabet_dict
        self._pad_idx = self.alphabet_dict.pad()
        self.emb_size = emb_size
        self.embedding = nn.Embedding(
            len(self.alphabet_dict), emb_size, padding_idx=self._pad_idx
        )

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class ThreeHotEmbedding(nn.Module):
    """
    A threehot embedding layer. Acts roughly like an `nn.EmbeddingBag`, but allows
    multiple padding indices. The embeddings are averaged and scaled by `sqrt(emb_size)`.
    """

    def __init__(
        self, dictionary: Union[ThreeHotDict, ThreeHotDictArbitraryOrdering], emb_size
    ):
        """Initialize the ThreeHot embedding table

        Args:
            dictionary: a `ThreeHotDict` or `ThreeHotDictArbitraryOrdering`
            emb_size: the embedding size
        """
        super(ThreeHotEmbedding, self).__init__()

        size_i, size_v, size_f = dictionary.sizes()
        pad_i, pad_v, pad_f = dictionary.pad()
        self.pad_f = pad_f

        self._emb_i = nn.Embedding(size_i, emb_size, padding_idx=pad_i)
        self._emb_v = nn.Embedding(size_v, emb_size, padding_idx=pad_v)
        self._emb_f = nn.Embedding(size_f, emb_size, padding_idx=pad_f)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        """Embed the input tensor. The `tokens` must be in threehot form.

        Args:
            tokens: a threehot tensor. that is, the shape is (X, Y, ... , 3)
            where [..., 0] is the initial class, [..., 1] is the vowel class,
            etc.

        Returns:
            A combined representation of the input embeddings
        """
        assert len(tokens.size()) > 1 and tokens.size(-1) == 3, "must be threehot"

        i, v, f = tokens[..., 0], tokens[..., 1], tokens[..., 2]
        return (
            (self._emb_i(i) + self._emb_v(v) + self._emb_f(f))
            * math.sqrt(self.emb_size)
            / 3
        )


class ThreeHotConcatEmbedding(nn.Module):
    """
    A threehot embedding layer. This is different from `ThreeHotEmbedding` in that the sub-embeddings
    are concatenated together before passed through a dense layer to give a final `emb_size` output.
    """

    def __init__(
        self, dictionary: Union[ThreeHotDict, ThreeHotDictArbitraryOrdering], emb_size
    ):
        """Initialize the ThreeHot Concat embedding table

        Args:
            dictionary: a `ThreeHotDict` or `ThreeHotDictArbitraryOrdering`
            emb_size: the embedding size
        """
        super(ThreeHotConcatEmbedding, self).__init__()

        size_i, size_v, size_f = dictionary.sizes()
        pad_i, pad_v, pad_f = dictionary.pad()
        self.pad_f = pad_f

        self._emb_i = nn.Embedding(size_i, emb_size, padding_idx=pad_i)
        self._emb_v = nn.Embedding(size_v, emb_size, padding_idx=pad_v)
        self._emb_f = nn.Embedding(size_f, emb_size, padding_idx=pad_f)

        self._combiner = nn.Linear(3 * emb_size, emb_size)

        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        """Embed the input tensor. The `tokens` must be in threehot form.

        Args:
            tokens: a threehot tensor. that is, the shape is (X, Y, ... , 3)
            where [..., 0] is the initial class, [..., 1] is the vowel class,
            etc.

        Returns:
            A combined representation of the input embeddings
        """
        assert len(tokens.size()) > 1 and tokens.size(-1) == 3, "must be threehot"

        i, v, f = tokens[..., 0], tokens[..., 1], tokens[..., 2]
        i, v, f = self._emb_i(i), self._emb_v(v), self._emb_f(f)

        return self._combiner(torch.cat([i, v, f], axis=-1)) * math.sqrt(self.emb_size)
