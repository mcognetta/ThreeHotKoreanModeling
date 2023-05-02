"""
GENERAL TODO:

1) make UnrolledRNN/UnrolledDiagonalRNN share code (all but the init are the same)
2) make sure every decoder layer outputs (logits), (preds) (or just remove the preds cause we don't use it)
"""

import torch
import torch.nn as nn

from typing import Union

from .dictionaries import *

class AlphabetDecoder(nn.Module):
    """
    A one-hot decoder based on an `AlphabetDict`. Returns the class logits and `argmax` prediction.
    """

    def __init__(self, dictionary: AlphabetDict, hidden_size):
        super(AlphabetDecoder, self).__init__()
        self._projection = nn.Linear(hidden_size, len(dictionary))

    def forward(self, hidden, **_):
        logits = self._projection(hidden)
        return logits, torch.argmax(logits, dim=-1).detach()


class UnrolledRNNDecoder(nn.Module):
    """An unrolled rnn that computes 3 ticks of an rnn loop to predict each of
    the three subcharacters in a syllable.

    This uses a full sized RNN and manually computes the `tanh(h*Wh + x*Wx)`
    term three times.
    """

    def __init__(
        self,
        dictionary: Union[ThreeHotDict, ThreeHotDictArbitraryOrdering],
        hidden_size,
    ):
        """Initialize the unrolled RNN decoder

        Args:
            dictionary: a `ThreeHotDict` or `ThreeHotDictArbitraryOrdering`
            hidden_size: the hidden dimension of the model
            inner_embedding_size: the internal RNN's hidden dimension size. Currently unused.
        """
        super(UnrolledRNNDecoder, self).__init__()

        self.hidden_size = hidden_size
        size_i, size_v, size_f = dictionary.sizes()
        self.size_i, self.size_v, self.size_f = size_i, size_v, size_f
        pad_i, pad_v, _ = dictionary.pad()

        self._reembed_i = nn.Embedding(size_i, hidden_size, padding_idx=pad_i)
        self._reembed_v = nn.Embedding(size_v, hidden_size, padding_idx=pad_v)

        self._fc_i = nn.Linear(hidden_size, size_i)
        self._fc_v = nn.Linear(hidden_size, size_v)
        self._fc_f = nn.Linear(hidden_size, size_f)

        self._A = nn.Linear(hidden_size, hidden_size)
        self._B = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, force=None):
        """Predict the threehot triplet.

        This is meant for training and uses teacher forcing to predict the
        syllables. The `force` parameter contains threehot subcharacter labels
        for the syllable to be predicted, so it should have shape (seq, batch, 3)
        where index [..., 0] is the first subcharacter class, etc.

        If `force` is `None`, the models own predictions are used to generate
        the next subcharacter.

        Args:
            hidden: a hidden context vector
            force: a threehot tensor containing teacher forcing values for the prediction.

        Returns:
            logits: a 3-tuple the unnormalized logits for each subcharacter (i, v, f)
            preds: the argmax of each of the logit vectors
        """
        if force is not None:
            # hidden is (seq, batch, H)
            # force is (seq, batch, 3)

            i_labels = force[..., 0]  # (seq, batch)
            v_labels = force[..., 1]  # (seq, batch)

            emb_i = self._reembed_i(i_labels)  # (seq, batch, h)
            emb_v = self._reembed_v(v_labels)  # (seq, batch, h)

            # unroll an RNN
            h_0 = torch.zeros_like(hidden)
            h_1 = torch.tanh(self._A(hidden) + self._B(h_0))
            h_2 = torch.tanh(self._A(emb_i) + self._B(h_1))
            h_3 = torch.tanh(self._A(emb_v) + self._B(h_2))

            logits_i = self._fc_i(h_1)
            logits_v = self._fc_v(h_2)
            logits_f = self._fc_f(h_3)

            pred_i = torch.argmax(logits_i, dim=-1).detach()
            pred_v = torch.argmax(logits_v, dim=-1).detach()
            pred_f = torch.argmax(logits_f, dim=-1).detach()

            return (logits_i, logits_v, logits_f), (pred_i, pred_v, pred_f)

        else:
            # hidden is (seq, batch, H)
            # force is (seq, batch, 3)

            emb_i = self._reembed_i(i_labels)  # (seq, batch, h)
            emb_v = self._reembed_v(v_labels)  # (seq, batch, h)

            # unroll an RNN
            h_0 = torch.zeros_like(hidden)

            h_1 = torch.tanh(self._A(hidden) + self._B(h_0))
            logits_i = self._fc_i(h_1)
            pred_i = torch.argmax(logits_i, dim=-1).detach()
            emb_i = self._reembed_i(pred_i)

            h_2 = torch.tanh(self._A(emb_i) + self._B(h_1))
            logits_v = self._fc_v(h_2)
            pred_v = torch.argmax(logits_v, dim=-1).detach()
            emb_v = self._reembed_v(pred_v)

            h_3 = torch.tanh(self._A(emb_v) + self._B(h_2))
            logits_f = self._fc_f(h_3)
            pred_f = torch.argmax(logits_f, dim=-1).detach()

            return (logits_i, logits_v, logits_f), (pred_i, pred_v, pred_f)

class DiagonalLinear(nn.Module):
    def __init__(self, hidden_size):
        super(DiagonalLinear, self).__init__()
        self._diag = nn.Parameter(torch.randn(hidden_size))
        self._bias = nn.Parameter(torch.randn(hidden_size))

    def forward(self, input):
        # return self._diag * input + self._bias
        return torch.addcmul(self._bias, input, self._diag)


# TODO make this a subclass of UnrolledRNNDecoder since they are entirely the
# same other than the diagonal layers.
class UnrolledDiagonalRNNDecoder(nn.Module):
    def __init__(
        self,
        dictionary: Union[ThreeHotDict, ThreeHotDictArbitraryOrdering],
        hidden_size,
    ):
        super(UnrolledDiagonalRNNDecoder, self).__init__()

        self.hidden_size = hidden_size
        size_i, size_v, size_f = dictionary.sizes()
        pad_i, pad_v, _ = dictionary.pad()

        self._reembed_i = nn.Embedding(size_i, hidden_size, padding_idx=pad_i)
        self._reembed_v = nn.Embedding(size_v, hidden_size, padding_idx=pad_v)

        self._fc_i = nn.Linear(hidden_size, size_i)
        self._fc_v = nn.Linear(hidden_size, size_v)
        self._fc_f = nn.Linear(hidden_size, size_f)

        # the diagonal rnn paper (`DIAGONAL RNNS IN SYMBOLIC MUSIC MODELING`)
        # has uses a full dense layer to go from the embedding to the inner
        # hidden dimension (since they don't need to be the same size) but a
        # diagonal layer for the inner hidden state update (since that is the
        # same size). Right now we always have the hid_dim = emb_dim.
        # self._A = nn.Linear(hidden_size, hidden_size)
        self._A = DiagonalLinear(hidden_size)
        self._B = DiagonalLinear(hidden_size)

    def forward(self, hidden, force=None):
        """Predict the threehot triplet.

        This is meant for training and uses teacher forcing to predict the
        syllables. The `force` parameter contains threehot subcharacter labels
        for the syllable to be predicted, so it should have shape (seq, batch, 3)
        where index [..., 0] is the first subcharacter class, etc.

        If `force` is `None`, the models own predictions are used to generate
        the next subcharacter.

        Args:
            hidden: a hidden context vector
            force: a threehot tensor containing teacher forcing values for the prediction.

        Returns:
            logits: a 3-tuple the unnormalized logits for each subcharacter (i, v, f)
            preds: the argmax of each of the logit vectors
        """
        if force is not None:
            # hidden is (seq, batch, H)
            # force is (seq, batch, 3)

            i_labels = force[..., 0]  # (seq, batch)
            v_labels = force[..., 1]  # (seq, batch)

            emb_i = self._reembed_i(i_labels)  # (seq, batch, h)
            emb_v = self._reembed_v(v_labels)  # (seq, batch, h)

            # unroll an RNN
            h_0 = torch.zeros_like(hidden)
            h_1 = torch.tanh(self._A(hidden) + self._B(h_0))
            h_2 = torch.tanh(self._A(emb_i) + self._B(h_1))
            h_3 = torch.tanh(self._A(emb_v) + self._B(h_2))

            logits_i = self._fc_i(h_1)
            logits_v = self._fc_v(h_2)
            logits_f = self._fc_f(h_3)

            pred_i = torch.argmax(logits_i, dim=-1).detach()
            pred_v = torch.argmax(logits_v, dim=-1).detach()
            pred_f = torch.argmax(logits_f, dim=-1).detach()

            return (logits_i, logits_v, logits_f), (pred_i, pred_v, pred_f)
        else:
            seq, bsz, _ = hidden.shape
            # hidden is (seq, batch, H)
            # force is (seq, batch, 3)

            emb_i = self._reembed_i(i_labels)  # (seq, batch, h)
            emb_v = self._reembed_v(v_labels)  # (seq, batch, h)

            # unroll an RNN
            h_0 = torch.zeros_like(hidden)

            h_1 = torch.tanh(self._A(hidden) + self._B(h_0))
            logits_i = self._fc_i(h_1)
            pred_i = torch.argmax(logits_i, dim=-1).detach()
            emb_i = self._reembed_i(pred_i)

            h_2 = torch.tanh(self._A(emb_i) + self._B(h_1))
            logits_v = self._fc_v(h_2)
            pred_v = torch.argmax(logits_v, dim=-1).detach()
            emb_v = self._reembed_v(pred_v)

            h_3 = torch.tanh(self._A(emb_v) + self._B(h_2))
            logits_f = self._fc_f(h_3)
            pred_f = torch.argmax(logits_f, dim=-1).detach()

            return (logits_i, logits_v, logits_f), (pred_i, pred_v, pred_f)

class ThreeHotIndependentDecoder(nn.Module):
    """
    An independent threehot decoder (all three jamo are predicted simultaneously and independently).
    """

    def __init__(
        self,
        dictionary: Union[ThreeHotDict, ThreeHotDictArbitraryOrdering],
        hidden_size,
        *_
    ):
        """Initialize the unrolled RNN decoder

        Args:
            dictionary: a `ThreeHotDict` or `ThreeHotDictArbitraryOrdering`
            hidden_size: the hidden dimension of the model
            inner_embedding_size: the internal RNN's hidden dimension size. Currently unused.
        """
        super(ThreeHotIndependentDecoder, self).__init__()

        size_i, size_v, size_f = dictionary.sizes()

        self._fc_i = nn.Linear(hidden_size, size_i)
        self._fc_v = nn.Linear(hidden_size, size_v)
        self._fc_f = nn.Linear(hidden_size, size_f)

    def forward(self, hidden, **_):
        """Predict the threehot triplet (independently)

        This predicts a syllable triplet by generating each subcharacter
        independently.

        Args:
            hidden: a hidden context vector

        Returns:
            logits: a 3-tuple the unnormalized logits for each subcharacter (i, v, f)
            preds: the argmax of each of the logit vectors
        """
        logits_i = self._fc_i(hidden)
        logits_v = self._fc_v(hidden)
        logits_f = self._fc_f(hidden)

        pred_i = torch.argmax(logits_i, dim=-1).detach()
        pred_v = torch.argmax(logits_v, dim=-1).detach()
        pred_f = torch.argmax(logits_f, dim=-1).detach()

        return (logits_i, logits_v, logits_f), (pred_i, pred_v, pred_f)