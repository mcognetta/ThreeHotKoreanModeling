import torch
import torch.nn as nn
from .transformers import create_mask, create_mask_threehot
from .common import _NO_JAMO_TOKEN

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device(int(os.environ.get("CUDA_DEVICE_NUM")))
DEVICE = None


@torch.no_grad()
def threehot_perplexity_per_jamo_class(i, v, f, I, V, F, tgt_dict):
    """
    Get the perplexity and number of non-padding tokens for each class.

    This returns 3 tuples `(loss_i, non_pad_i)`, `(loss_v, non_pad_v)`, `(loss_f, non_pad_f)`.

    The overall perplexity can be calculated by summing the lossing and non-pad
    and dividing.
    """
    pad_i, pad_v, pad_f = tgt_dict.pad()
    crit_i = nn.CrossEntropyLoss(ignore_index=pad_i, reduction="sum")
    crit_v = nn.CrossEntropyLoss(ignore_index=pad_v, reduction="sum")
    crit_f = nn.CrossEntropyLoss(ignore_index=pad_f, reduction="sum")

    return (
        (crit_i(i, I), sum(I != pad_i)),
        (crit_v(v, V), sum(V != pad_v)),
        (crit_f(f, F), sum(F != pad_f)),
    )


@torch.no_grad()
def threehot_perplexity_per_jamo_class_ignore_non_jamo_token(
    i, v, f, I, V, F, tgt_dict
):
    """
    Get the perplexity and number of non-padding tokens for each class.
    This is teh same as `threehot_perplexity_per_jamo_class`, except we also
    count NON_JAMO (㋨) as padding so the loss for predicting non-Korean triplets
    only counts the non-Korean token. E.g., predicting `(a, ㋨, ㋨)` only accumulates
    the loss from the `a` prediction, and the ㋨'s are ignored.

    This returns 3 tuples `(loss_i, non_pad_i)`, `(loss_v, non_pad_v)`, `(loss_f, non_pad_f)`.

    The overall perplexity can be calculated by summing the lossing and non-pad
    and dividing.
    """
    pad_i, pad_v, pad_f = tgt_dict.pad()

    non_jamo_i, non_jamo_v, non_jamo_f = tgt_dict[_NO_JAMO_TOKEN]
    crit_i = nn.CrossEntropyLoss(ignore_index=pad_i, reduction="none")
    crit_v = nn.CrossEntropyLoss(ignore_index=pad_v, reduction="none")
    crit_f = nn.CrossEntropyLoss(ignore_index=pad_f, reduction="none")

    i_loss = crit_i(i, I)
    v_loss = crit_v(v, V)
    f_loss = crit_f(f, F)

    i_loss = sum(torch.where(I != non_jamo_i, i_loss, 0.0))
    v_loss = sum(torch.where(V != non_jamo_v, v_loss, 0.0))
    f_loss = sum(torch.where(F != non_jamo_f, f_loss, 0.0))

    return (
        (i_loss, sum(torch.logical_and(I != pad_i, I != non_jamo_i))),
        (v_loss, sum(torch.logical_and(V != pad_v, V != non_jamo_v))),
        (f_loss, sum(torch.logical_and(F != pad_f, F != non_jamo_f))),
    )


@torch.no_grad()
def perplexity_syllable_batched(
    model, data, src_dict, tgt_dict, batch_size=4000, device=None
):
    """
    Compute perplexity over a corpus in batches. The total loss is divided by
    the total number of tokens over all batches (not averaged per batch).

    This computes the syllable perplexity per subcharacter, so the number of tokens
    is 3*(number of syllables) + (number of non korean tokens)
    """
    model.eval()
    tot_loss = 0
    tot_lens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_dict.pad(), reduction="sum")

    batches = batch_by_target_size(data, batch_size)

    for idx, batch in enumerate(batches):
        src, tgt, multiples = collate_fn_syllable_perplexity(batch, src_dict, tgt_dict)
        src = torch.tensor(src, device=device)
        tgt = torch.tensor(tgt, device=device)
        multiples = torch.tensor(multiples, device=device)

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        multiples = multiples.transpose(0, 1)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, src_dict, tgt_dict, device=device
        )

        logits, _ = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        tgt_out = tgt[1:, :]
        multiples = multiples[1:, :].reshape(-1)

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        tot_loss += loss.item()
        tot_lens += sum(multiples).item()
        del loss

    return tot_loss / tot_lens


@torch.no_grad()
def perplexity_jamo_batched(model, data, src_dict, tgt_dict, batch_size=10000, device=None):
    """
    Compute perplexity over a corpus in batches. The total loss is divided by
    the total number of tokens over all batches (not averaged per batch).

    This computes the jamo perplexity. We normally use perplexity per-subcharacter,
    so no extra calculations need to be done, since the number of tokens
    predicted by a jamo model matches the number of subcharcters.
    """
    model.eval()
    tot_acc = 0
    tot_loss = 0
    tot_lens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_dict.pad(), reduction="sum")

    batches = batch_by_target_size(data, batch_size)

    for idx, batch in enumerate(batches):
        src, tgt = collate_fn(batch, src_dict, tgt_dict)
        src = torch.tensor(src, device=device)
        tgt = torch.tensor(tgt, device=device)

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, src_dict, tgt_dict, device=device
        )

        logits, _ = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        tot_acc += sum(
            torch.argmax(logits.reshape(-1, logits.shape[-1]), axis=1)
            == tgt_out.reshape(-1)
        ).item()
        tot_loss += loss.item()
        tot_lens += sum(tgt_out.reshape(-1) != tgt_dict.pad()).item()
        del loss
    return tot_loss / tot_lens  # , tot_acc / tot_lens


@torch.no_grad()
def _perplexity_threehot_per_class_batched_base(
    model,
    data,
    src_dict,
    tgt_dict,
    batch_size=4000,
    perp_fn=threehot_perplexity_per_jamo_class,
    device = None,
):
    """
    A base function for computing perplexity over a corpus in batches.
    This allows plugging in loss function variants. See:
     - `perplexity_threehot_per_class_batched`
     - `perplexity_threehot_per_class_ignore_non_jamo_token_batched`

    Compute perplexity over a corpus in batches. The total loss is divided by
    the total number of tokens over all batches (not averaged per batch).

    This computes the threehot perplexity per class. We return a tuple
    `(total, i, v, f)`, the total perplexity-per-subcharacter and the perplexity
    per token in each of the three subcharacter classes.

    This is the same as `perplexity_threehot_per_class_ignore_non_jamo_token_batched`
    except we count NON_JAMO tokens as padding. See `threehot_perplexity_per_jamo_class_ignore_non_jamo_token`
    """
    model.eval()
    tot_loss_i, tot_loss_v, tot_loss_f = 0, 0, 0
    tot_len_i, tot_len_v, tot_len_f = 0, 0, 0

    batches = batch_by_target_size(data, batch_size)

    for idx, batch in enumerate(batches):
        src, tgt = collate_fn(batch, src_dict, tgt_dict)
        src = torch.tensor(src, device=device)
        tgt = torch.tensor(tgt, device=device)

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask_threehot(
            src, tgt_input, src_dict, tgt_dict, device=device
        )
        tgt_out = tgt[1:, :]
        logits, _ = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
            force=tgt_out,
            teacher_force=True,
        )
        i, v, f = logits

        ((loss_i, len_i), (loss_v, len_v), (loss_f, len_f),) = perp_fn(
            i.reshape(-1, i.shape[-1]),
            v.reshape(-1, v.shape[-1]),
            f.reshape(-1, f.shape[-1]),
            tgt_out[..., 0].reshape(-1),
            tgt_out[..., 1].reshape(-1),
            tgt_out[..., 2].reshape(-1),
            tgt_dict,
        )

        tot_loss_i += loss_i.item()
        tot_len_i += len_i.item()
        tot_loss_v += loss_v.item()
        tot_len_v += len_v.item()
        tot_loss_f += loss_f.item()
        tot_len_f += len_f.item()

    return (
        (tot_loss_i + tot_loss_v + tot_loss_f) / (tot_len_i + tot_len_v + tot_len_f),
        (tot_loss_i / tot_len_i),
        (tot_loss_v / tot_len_v),
        (tot_loss_f / tot_len_f),
    )


@torch.no_grad()
def perplexity_threehot_per_class_batched(
    model, data, src_dict, tgt_dict, batch_size=4000, device=None
):
    """
    Compute perplexity over a corpus in batches. The total loss is divided by
    the total number of tokens over all batches (not averaged per batch).

    This computes the threehot perplexity per class. We return a tuple
    `(total, i, v, f)`, the total perplexity-per-subcharacter and the perplexity
    per token in each of the three subcharacter classes.
    """
    return _perplexity_threehot_per_class_batched_base(
        model,
        data,
        src_dict,
        tgt_dict,
        batch_size=batch_size,
        perp_fn=threehot_perplexity_per_jamo_class,
        device = device,
    )


@torch.no_grad()
def perplexity_threehot_per_class_ignore_non_jamo_token_batched(
    model, data, src_dict, tgt_dict, batch_size=4000, device=None
):
    """
    Compute perplexity over a corpus in batches. The total loss is divided by
    the total number of tokens over all batches (not averaged per batch).

    This computes the threehot perplexity per class. We return a tuple
    `(total, i, v, f)`, the total perplexity-per-subcharacter and the perplexity
    per token in each of the three subcharacter classes.

    This is the same as `perplexity_threehot_per_class_ignore_non_jamo_token_batched`
    except we count NON_JAMO tokens as padding. See `threehot_perplexity_per_jamo_class_ignore_non_jamo_token`
    """
    return _perplexity_threehot_per_class_batched_base(
        model,
        data,
        src_dict,
        tgt_dict,
        batch_size=batch_size,
        perp_fn=threehot_perplexity_per_jamo_class_ignore_non_jamo_token,
        device = device,
    )
