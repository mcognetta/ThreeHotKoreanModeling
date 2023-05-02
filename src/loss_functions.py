import warnings

import torch.nn as nn


def onehot_loss(logits, preds, tgt_dict):
    """
    A wrapper around cross entropy loss that takes a dictionary so that
    the padding index can be easily set.
    """
    crit = nn.CrossEntropyLoss(ignore_index=tgt_dict.pad())

    return crit(logits, preds)


def threehot_loss(i, v, f, I, V, F, tgt_dict):
    """
    Our model-agnostic perplexity measure is bits-per-subcharacter. This loss
    doesn't quite capture that relationship, as the losses are averaged over
    each subcharcter class, not the total number of subcharacters.

    `threehot_loss_per_subcharacter` matches the perplexity measure
    """
    pad_i, pad_v, pad_f = tgt_dict.pad()
    crit_i = nn.CrossEntropyLoss(ignore_index=pad_i, reduction="mean")
    crit_v = nn.CrossEntropyLoss(ignore_index=pad_v, reduction="mean")
    crit_f = nn.CrossEntropyLoss(ignore_index=pad_f, reduction="mean")

    return crit_i(i, I) + crit_v(v, V) + crit_f(f, F)


def threehot_loss_per_subcharacter(i, v, f, I, V, F, tgt_dict):
    """
    This loss computes the total crossentropy loss for each token and divides
    it over the total number of (non-padding) subcharacters. This gives us
    a more accurate loss function for our bits-per-subcharacter metric.
    """
    pad_i, pad_v, pad_f = tgt_dict.pad()
    crit_i = nn.CrossEntropyLoss(ignore_index=pad_i, reduction="sum")
    crit_v = nn.CrossEntropyLoss(ignore_index=pad_v, reduction="sum")
    crit_f = nn.CrossEntropyLoss(ignore_index=pad_f, reduction="sum")

    losses = crit_i(i, I) + crit_v(v, V) + crit_f(f, F)
    multiples = (I != pad_i).long() + (V != pad_v).long() + (F != pad_f).long()
    return losses / sum(multiples)


def threehot_loss_weighted(i, v, f, I, V, F, tgt_dict):
    warnings.warn(
        "the name `threehot_loss_weighted` is deprecatd, use `threehot_loss_per_subcharacter` instead",
        DeprecationWarning,
    )
    return threehot_loss_per_subcharacter(i, v, f, I, V, F, tgt_dict)
