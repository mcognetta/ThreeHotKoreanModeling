"""
TODO: when serializing models, serialze all instance variables so the model can be rebuilt no matter what
"""

import torch
import torch.nn as nn

import logging

from .transformers import Seq2SeqTransformer, Seq2SeqTransformerThreeHot
from .decoding_layers import (
    ThreeHotIndependentDecoder,
    UnrolledDiagonalRNNDecoder,
    UnrolledRNNDecoder,
)

DEVICE = None

def build_transformer(
    src_dict,
    tgt_dict,
    max_len=300,
    FFN_HID_DIM=512,
    ENCODER_LAYERS=6,
    DECODER_LAYERS=6,
    NHEADS=8,
    EMB_SIZE=512,
    tie_tgt_embeddings=False,
    device=None
):

    torch.manual_seed(0)

    transformer = Seq2SeqTransformer(
        src_dict,
        tgt_dict,
        ENCODER_LAYERS,
        DECODER_LAYERS,
        EMB_SIZE,
        NHEADS,
        FFN_HID_DIM,
        max_len=max_len,
        tie_tgt_embeddings=tie_tgt_embeddings,
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(device)

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-8
    )
    return src_dict, tgt_dict, transformer, optimizer


def build_transformer_threehot(
    src_dict,
    tgt_dict,
    FFN_HID_DIM=512,
    EMB_SIZE=512,
    ENCODER_LAYERS=6,
    DECODER_LAYERS=6,
    NHEADS=8,
    decoder_cls=UnrolledRNNDecoder,
    tie_tgt_embeddings=False,
    max_len=300,
    device=None,
):
    torch.manual_seed(0)

    transformer = Seq2SeqTransformerThreeHot(
        src_dict,
        tgt_dict,
        decoder_cls,
        ENCODER_LAYERS,
        DECODER_LAYERS,
        EMB_SIZE,
        NHEADS,
        FFN_HID_DIM,
        FFN_HID_DIM,
        tie_tgt_embeddings=tie_tgt_embeddings,
        max_len=max_len,
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(device)

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-8
    )
    return src_dict, tgt_dict, transformer, optimizer


three_hot_cond_builder = (
    lambda src_dict, tgt_dict, tie_target_embeddings: build_transformer_threehot(
        src_dict,
        tgt_dict,
        decoder_cls=UnrolledRNNDecoder,
        tie_tgt_embeddings=tie_target_embeddings,
    )
)
three_hot_cond_diag_builder = (
    lambda src_dict, tgt_dict, tie_target_embeddings: build_transformer_threehot(
        src_dict,
        tgt_dict,
        decoder_cls=UnrolledDiagonalRNNDecoder,
        tie_tgt_embeddings=tie_target_embeddings,
    )
)
three_hot_ind_builder = (
    lambda src_dict, tgt_dict, tie_target_embeddings: build_transformer_threehot(
        src_dict,
        tgt_dict,
        decoder_cls=ThreeHotIndependentDecoder,
        tie_tgt_embeddings=tie_target_embeddings,
    )
)
jamo_builder = lambda src_dict, tgt_dict, tie_target_embeddings: build_transformer(
    src_dict, tgt_dict, max_len=500, tie_tgt_embeddings=tie_target_embeddings
)
syllable_builder = lambda src_dict, tgt_dict, tie_target_embeddings: build_transformer(
    src_dict, tgt_dict, tie_tgt_embeddings=tie_target_embeddings
)
parameter_reallocation_builder = lambda src_dict, tgt_dict, tie_target_embedings: build_transformer_threehot(
    src_dict,
    tgt_dict,
    ENCODER_LAYERS=7,
    DECODER_LAYERS=7,
    FFN_HID_DIM=1024,
    decoder_cls=UnrolledDiagonalRNNDecoder,
)


def load_model(path, model_builder):
    checkpoint = torch.load(path, map_location='cpu')
    epoch = checkpoint["epoch"]
    tie_embeddings = checkpoint["tie_embeddings"]
    en_dict = checkpoint["en_dict"]
    kr_dict = checkpoint["kr_dict"]
    src_dict, tgt_dict, model, opt = model_builder(en_dict, kr_dict, tie_embeddings)
    model.load_state_dict(checkpoint["model_state_dict"])
    opt.load_state_dict(checkpoint["optimizer_state_dict"])

    return src_dict, tgt_dict, model, opt, epoch


def serialize_model_and_opt(path, model, opt, src_dict, tgt_dict, epoch):
    logging.info(f"serializing {path}")

    torch.save(
        {
            "tie_embeddings": model.tie_tgt_embeddings,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "en_dict": src_dict,
            "kr_dict": tgt_dict,
        },
        path,
    )
