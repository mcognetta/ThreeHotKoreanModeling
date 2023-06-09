This is the core code for `Parameter-Efficient Korean Character-Level Language Modeling`.

- Paper: https://aclanthology.org/2023.eacl-main.172/
- Poster: See repo.

# Overview
Here, we provide an embedding and decoding layer for Korean character (syllable) language models. We also provide a number of loss functions that we tried throughout the paper, an implementation of a prior three-hot model (`ThreeHotIndependentDecoder`), and some glue code to help ease the use of this with real life datasets.

Our main contributions are in `decoding_layers.py`. Namely, `UnrolledRNNDecoder` and `UnrolledDiagonalRNNDecoder`, which implement the 3-stage RNN decoder for jamo triplets, as described in our paper. We also include a dictionary implementation for threehot that allows for converting between syllables and jamo triplet ids and vice versa. It also allows for out-of-order generation (where a syllable is predicted like `vowel, initial, final` or some other non-standard ordering).

We also provide jamo- and syllable-level one hot encoding and decoding layers.

A basic seq2seq transformer model is provided for both one-hot and three-hot models, and it allows for weight sharing between the target-side embedding and decoding layers (as well as sharing between the three-hot RNN's embedding and output layers).

# Use

The three-hot models take in jamo triplets as `B x S x 3` vectors, where `input[..., 0]` is the initial jamo, `input[..., 1]` is the vowel jamo, and `input[..., 2]` is the final jamo (unless you are using an out-of-order dictionary, which just permutes these).

The threehot dictionaries take sequences of characters and converts it to jamo triplets. E.g.,

```python
non_korean_symbols = ['a', 'b', 'c']
D = ThreehotDictionary(non_korean_symbols)
D.encode("aa한국어") # -> [(19, 21, 28), (19, 21, 28), (18, 0, 3), (0, 13, 0), (11, 4, 27)]
```

You should provide your own tokenizer for the non-Korean side of the language model, but you can define a dictionary with a given vocabulary and then pass any pre-tokenized sequence into it to get the ids that can be used in the one-hot models.

## General setup

As an example setting of EN->Korean translation, you should perform the following steps to prepare the model for training.

1. Train an English tokenizer.
2. Make a list of (en, kr) sentence pairs, where the English side is pretokenized and the Korean side is character-by-character.
3. Make the English `AlphabetDict`
4. Collect all non-Korean characters that you want to be able to model in Korean (numbers, English letters/names, etc.) and make the Korean `ThreeHotDict`
5. Convert the sentence pairs to one-hot id's for English and Three-hot sequences for Korean (this is done by just calling the `encode` method on the input sentences with the English or Korean dictionaries).
6. Build the seq2seq model `Seq2SeqTransformerThreeHot`, which can take in an `AlphabetDict` and a `ThreehotDict` and properly build the embedding and decoding layers.

Training can be done as normal, using one of our loss functions.

# TODO
- Provide internal beam-search implementation.
    - The threehot decoder needs two beam search implementations to work, one that is internal (beam search over jamo to form triplets) and one that is external (beam search over jamo-triplets to form syllable sequences). Our implementation is currently not fully leveraging the GPU but a GPU-enabled is underway.
