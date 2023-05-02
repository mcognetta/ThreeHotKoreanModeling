import torch
import torch.nn as nn
import torch.nn.functional as F


from .common import *
from data_batching import *
from .transformers import create_mask, create_mask_threehot
from .utils import load_model, serialize_model_and_opt
from .perplexity_functions import (
    perplexity_syllable_batched,
    perplexity_jamo_batched,
    perplexity_threehot_per_class_batched,
)
from .loss_functions import (
    threehot_loss,
)

import os, logging, random, time


def train_epoch_target_token_batch(
    model, optimizer, train_data, src_dict, tgt_dict, device=None
):
    """Train a model for a single epoch (syllable and jamo models only).
    Batches are formed by # of tokens in the target side.

    See `train_epoch_target_token_batch_threehot` for the threehot version.

    Args:
        model: a model
        optimizer: an optimizer
        train_data: a set of (src, tgt) pairs. for threehot/syllable this is
                    just syllable sequences on the tgt side. For jamo it is
                    jamo sequences. See `filtered_bpe_translation_train_pairs`
                    vs `filtered_bpe_jamo_translation_train_pairs`
        src_dict: the src language dictionary
        tgt_dict: the tgt language dictionary
        batch_size: the number of tokens for each batch (default 4000)
    """
    model.train()
    losses = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_dict.pad())

    random.shuffle(train_data)

    for batch in train_data:

        src, tgt = batch

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

        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        del loss

    return losses / len(train_data)


# works for both cond and ind
def train_epoch_target_token_batch_threehot(
    model,
    optimizer,
    train_data,
    src_dict,
    tgt_dict,
    loss_fn=threehot_loss,
    device = None,
):
    """Train a model for a single epoch (threehot only).
    Batches are formed by # of tokens (syllables) in the target side.

    Args:
        model: a model
        optimizer: an optimizer
        train_data: a set of (src, tgt) pairs. for threehot/syllable this is
                    just syllable sequences on the tgt side. For jamo it is
                    jamo sequences. See `filtered_bpe_translation_train_pairs`
                    vs `filtered_bpe_jamo_translation_train_pairs`
        src_dict: the src language dictionary
        tgt_dict: the tgt language dictionary
    """
    model.train()
    losses = 0
    # batches = batch_by_target_size(train_data, batch_size)
    random.shuffle(train_data)

    for batch in train_data:
        src, tgt = batch

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask_threehot(
            src, tgt_input, src_dict, tgt_dict, device=device,
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
            teacher_force=True,
            force=tgt_out,
        )
        i, v, f = logits
        optimizer.zero_grad()

        loss = loss_fn(
            i.reshape(-1, i.shape[-1]),
            v.reshape(-1, v.shape[-1]),
            f.reshape(-1, f.shape[-1]),
            tgt_out[..., 0].reshape(-1),
            tgt_out[..., 1].reshape(-1),
            tgt_out[..., 2].reshape(-1),
            tgt_dict,
        )

        loss.backward()

        optimizer.step()
        losses += loss.item()

        del loss

    return losses / len(train_data)


def train_syllable(
    epochs,
    model,
    optimizer,
    src_dict,
    tgt_dict,
    train_data,
    test_data,
    output_dir,
    output_name,
    batch_size=4000,
    output_every=1,
    device=None,
):
    """Train a syllable model for a given number of epochs.

    The model is serialized periodically, and a metrics dictionary is logged
    at each epoch.

    Args:
        epochs: the number of epochs to train for
        model: a model
        optimizer: an optimizer
        src_dict: the src language dictionary
        tgt_dict: the tgt language dictionary
        batch_size: the number of tokens for each batch (default 4000)
        train_data: a set of (src, tgt) pairs for training. See: `filtered_bpe_translation_train_pairs`
        test_data: a set of (src, tgt) pairs for testing. See: `filtered_bpe_translation_valid_pairs`
        output_dir: a directory to store seralized models
        output_name: a prefix to name the models (for example `SYLLABLE`). the
                     epoch will be appended to the model name (e.g. SYLLABLE_EPOCH_30.mod)
        batch_size: the batch size for training
        output_every: every `N` epochs, save and serialize the model
    """
    results = {
        "ACCURACY": [],
        "LOSS_PER_SUBCHAR_TEST": [],
        "LOSS_PER_SUBCHAR_TRAIN": [],
        "PERPLEXITY_TEST": [],
        "PERPLEXITY_TRAIN": [],
        "TIME": [],
    }

    path = os.path.join(output_dir, output_name)

    batched_data = batch_by_target_size(train_data, batch_size)
    random.shuffle(batched_data)

    tensor_batches = []

    for b in batched_data:
        s, t = collate_fn(b, src_dict, tgt_dict)

        tensor_batches.append(
            (torch.tensor(s, device=device), torch.tensor(t, device=device))
        )

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_epoch_target_token_batch(
            model,
            optimizer,
            tensor_batches,
            src_dict,
            tgt_dict,
            batch_size=batch_size,
            device=device,
        )
        results["TIME"].append(time.time() - start)

        perp_test = perplexity_syllable_batched(
            model, test_data, src_dict, tgt_dict, device=device
        )
        perp_train = perplexity_syllable_batched(
            model, random.sample(train_data, 5000), src_dict, tgt_dict, device=device
        )

        # results["ACCURACY"].append(acc)
        results["PERPLEXITY_TEST"].append(perp_test)
        results["PERPLEXITY_TRAIN"].append(perp_train)

        logging.info(f"name={output_name}, epoch={epoch}, metrics:{results}")

        if epoch % output_every == 0:
            serialize_model_and_opt(
                f"{path}_EPOCH_{epoch}.mod", model, optimizer, src_dict, tgt_dict, epoch
            )

    # output model one last time in case the final epoch was not divisible by `output_every`
    if epoch % output_every != 0:
        serialize_model_and_opt(
            f"{path}_EPOCH_{epoch}.mod", model, optimizer, src_dict, tgt_dict, epoch
        )

    return model, optimizer, results


def train_jamo(
    epochs,
    model,
    optimizer,
    src_dict,
    tgt_dict,
    train_data,
    test_data,
    output_dir,
    output_name,
    batch_size=10000,
    output_every=1,
    device = None,
):
    """Train a jamo model for a given number of epochs.

    The model is serialized periodically, and a metrics dictionary is logged
    at each epoch.

    Args:
        epochs: the number of epochs to train for
        model: a model
        optimizer: an optimizer
        src_dict: the src language dictionary
        tgt_dict: the tgt language dictionary
        batch_size: the number of tokens for each batch (default 10000)
        train_data: a set of (src, tgt) pairs for training. See: `filtered_bpe_jamo_translation_train_pairs`
        test_data: a set of (src, tgt) pairs for testing. See: `filtered_bpe_jamo_translation_valid_pairs`
        output_dir: a directory to store seralized models
        output_name: a prefix to name the models (for example `JAMO`). the
                     epoch will be appended to the model name (e.g. JAMO_EPOCH_30.mod)
        batch_size: the batch size for training
        output_every: every `N` epochs, save and serialize the model
    """
    results = {
        "ACCURACY": [],
        "LOSS_PER_SUBCHAR_TEST": [],
        "LOSS_PER_SUBCHAR_TRAIN": [],
        "PERPLEXITY_TEST": [],
        "PERPLEXITY_TRAIN": [],
        "TIME": [],
    }

    path = os.path.join(output_dir, output_name)
    

    batched_data = batch_by_target_size(train_data, batch_size)
    random.shuffle(batched_data)

    tensor_batches = []

    for b in batched_data:
        s, t = collate_fn(b, src_dict, tgt_dict)

        tensor_batches.append(
            (torch.tensor(s, device=device), torch.tensor(t, device=device))
        )

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_epoch_target_token_batch(
            model,
            optimizer,
            tensor_batches,
            src_dict,
            tgt_dict,
            batch_size=batch_size,
            device=device
        )
        results["TIME"].append(time.time() - start)

        perp_test = perplexity_jamo_batched(model, test_data, src_dict, tgt_dict, device=device)
        perp_train = perplexity_jamo_batched(model, random.sample(train_data, 6000), src_dict, tgt_dict, device=device)

        # results["ACCURACY"].append(acc)
        results["PERPLEXITY_TEST"].append(perp_test)
        results["PERPLEXITY_TRAIN"].append(perp_train)

        logging.info(f"name={output_name}, epoch={epoch}, metrics:{results}")

        if epoch % output_every == 0:
            serialize_model_and_opt(
                f"{path}_EPOCH_{epoch}.mod", model, optimizer, src_dict, tgt_dict, epoch
            )

    # output model one last time in case the final epoch was not divisible by `output_every`
    if epoch % output_every != 0:
        serialize_model_and_opt(
            f"{path}_EPOCH_{epoch}.mod", model, optimizer, src_dict, tgt_dict, epoch
        )

    return model, optimizer, results


def train_threehot(
    epochs,
    model,
    optimizer,
    src_dict,
    tgt_dict,
    train_data,
    test_data,
    output_dir,
    output_name,
    batch_size=4000,
    output_every=1,
    device=None,
    checkpoint_offset = 0,
):
    """Train a threehot model for a given number of epochs. This works for any
    threehot model (conditional or independent, any output order)

    The model is serialized periodically, and a metrics dictionary is logged
    at each epoch.

    Args:
        epochs: the number of epochs to train for
        model: a model
        optimizer: an optimizer
        src_dict: the src language dictionary
        tgt_dict: the tgt language dictionary
        batch_size: the number of tokens for each batch (default 4000)
        train_data: a set of (src, tgt) pairs for training. See: `filtered_bpe_translation_train_pairs`
        test_data: a set of (src, tgt) pairs for testing. See: `filtered_bpe_translation_valid_pairs`
        output_dir: a directory to store seralized models
        output_name: a prefix to name the models (for example `THREEHOT_FIV`). the
                     epoch will be appended to the model name (e.g. THREEHOT_FIV_EPOCH_30.mod)
        batch_size: the batch size for training
        output_every: every `N` epochs, save and serialize the model
    """
    results = {
        # "ACCURACY": [],
        # "LOSS_PER_SUBCHAR_TEST": [],
        # "LOSS_PER_SUBCHAR_TRAIN": [],
        # "LOSS_BY_SUBCHAR_CLASS_TEST": [],
        # "LOSS_BY_SUBCHAR_CLASS_TRAIN": [],
    }
    results = {
        "ACCURACY": [],
        "LOSS_PER_SUBCHAR_TEST": [],
        "LOSS_PER_SUBCHAR_TRAIN": [],
        "PERPLEXITY_TEST": [],
        "PERPLEXITY_TRAIN": [],
        "PERPLEXITY_TEST_NO_NON_JAMO": [],
        "PERPLEXITY_TRAIN_NO_NON_JAMO": [],
        "TIME": [],
    }

    path = os.path.join(output_dir, output_name)

    batched_data = batch_by_target_size(train_data, batch_size)
    random.shuffle(batched_data)

    tensor_batches = []

    for b in batched_data:
        s, t = collate_fn(b, src_dict, tgt_dict)

        tensor_batches.append(
            (torch.tensor(s, device=device), torch.tensor(t, device=device))
        )

    for epoch in range(checkpoint_offset + 1, epochs + 1):
        start = time.time()
        train_epoch_target_token_batch_threehot(
            model,
            optimizer,
            tensor_batches,
            src_dict,
            tgt_dict,
            batch_size=batch_size,
            device=device
        )
        results["TIME"].append(time.time() - start)

        # combine these so that they don't have to do the translations twice
        perp_test = perplexity_threehot_per_class_batched(
            model, test_data, src_dict, tgt_dict, device=device
        )
        # perp_train = perplexity_threehot_per_class_batched(
        #     model, random.sample(train_data, 6000), src_dict, tgt_dict, device=device
        # )
        # perp_test_no_non_jamo = (
        #     perplexity_threehot_per_class_ignore_non_jamo_token_batched(
        #         model, test_data, src_dict, tgt_dict
        #     )
        # )
        # perp_train_no_non_jamo = (
        #     perplexity_threehot_per_class_ignore_non_jamo_token_batched(
        #         model, train_data, src_dict, tgt_dict
        #     )
        # )
        # results["ACCURACY"].append(acc)
        results["PERPLEXITY_TEST"].append(perp_test)
        # results["PERPLEXITY_TRAIN"].append(perp_train)
        # results["PERPLEXITY_TEST_NO_NON_JAMO"].append(perp_test_no_non_jamo)
        # results["PERPLEXITY_TRAIN_NO_NON_JAMO"].append(perp_train_no_non_jamo)

        logging.info(f"name={output_name}, epoch={epoch}, metrics:{results}")

        if epoch % output_every == 0:
            serialize_model_and_opt(
                f"{path}_EPOCH_{epoch}.mod", model, optimizer, src_dict, tgt_dict, epoch
            )

    # output model one last time in case the final epoch was not divisible by `output_every`
    if epoch % output_every != 0:
        serialize_model_and_opt(
            f"{path}_EPOCH_{epoch}.mod", model, optimizer, src_dict, tgt_dict, epoch
        )

    return model, optimizer, results
