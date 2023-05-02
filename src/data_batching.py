import random

from .common import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, is_korean_syllable

def batch_by_target_size(data, max_tokens=4000):
    """
    This batches by the number of tokens in the target side. The sentences
    are sorted by (target side) length, but sentences of the same length
    are shuffled before batching, so that the batches are not the same each
    time.

    The batches are then greedily filled to a max of `max_tokens` tokens per
    batch. Thus, sentences of similar length are grouped together in a batch.
    """

    unstable_sorted = sorted(data, key=lambda x: (len(x[1]), random.random()))

    out = []
    running_batch = []
    cur_count = 0
    for (s, t) in unstable_sorted:
        if cur_count + len(t) > max_tokens:
            out.append(running_batch)
            running_batch = [(s, t)]
            cur_count = len(t)
        else:
            running_batch.append((s, t))
            cur_count += len(t)
    if running_batch != []:
        out.append(running_batch)
    return out


def collate_fn(batch, src_dict, tgt_dict):
    """
    Builds batched class label source and target sequences.
    """
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append([SOS_TOKEN] + src_sample + [EOS_TOKEN])
        tgt_batch.append([SOS_TOKEN] + tgt_sample + [EOS_TOKEN])

    src_batch = src_dict.encode_batch(src_batch)
    tgt_batch = tgt_dict.encode_batch(tgt_batch)
    return src_batch, tgt_batch


def _subchar_multiples(x):
    """
    This is just a helper method to determine the number of subcharacters in a token.
    0 for pads, 1 for non-Korean, 3 for Korean syllables.
    """
    if x == PAD_TOKEN:
        return 0
    elif is_korean_syllable(x):
        return 3
    else:
        return 1


def collate_fn_syllable_perplexity(batch, src_dict, tgt_dict):
    """
    Builds batched class label source and target sequences.
    This is used in the syllable decoder, which does not have subcharacter
    awareness. We also retain the number of subcharacters for each token
    (3 if korean, 1 otherwise) so that we can use the information to calculate
    bits-per-subcharacter in the perplexity calculation.
    """
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append([SOS_TOKEN] + src_sample + [EOS_TOKEN])
        tgt_batch.append([SOS_TOKEN] + tgt_sample + [EOS_TOKEN])
    src_batch = src_dict.encode_batch(src_batch)
    tgt_batch = tgt_dict.encode_batch(tgt_batch)
    subchars = [[_subchar_multiples(x) for x in tgt_dict.decode(t)] for t in tgt_batch]
    return src_batch, tgt_batch, subchars


def collate_fn_triple(batch, src_dict, tgt_input_dict, tgt_dict):
    """
    Builds batched class label source and target sequences.
    This is used in ablation studies when the target side encoder and decoder
    are different. For example a 3 hot encoder but syllable decoder or vice versa.

    The output is (source, target_input, target_output)
    """
    src_batch, tgt_input_batch, tgt_batch = [], [], []
    for src_sample, tgt_sample in batch:
        src_batch.append([SOS_TOKEN] + src_sample + [EOS_TOKEN])
        tgt_input_batch.append([SOS_TOKEN] + tgt_sample + [EOS_TOKEN])
        tgt_batch.append([SOS_TOKEN] + tgt_sample + [EOS_TOKEN])

    src_batch = src_dict.encode_batch(src_batch)
    tgt_input_batch = tgt_input_dict.encode_batch(tgt_batch)
    tgt_batch = tgt_dict.encode_batch(tgt_batch)
    return src_batch, tgt_input_batch, tgt_batch