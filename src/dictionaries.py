from .common import *
from .common import _NO_JAMO_TOKEN, _INITIAL_JAMO, _VOWEL_JAMO, _FINAL_JAMO

class ThreeHotDict:
    _RESERVED = set(SPECIAL_SYMBOLS + _INITIAL_JAMO + _VOWEL_JAMO + _FINAL_JAMO)

    def __init__(self, symbols):
        assert all(s not in self._RESERVED for s in symbols)
        self._ini = _INITIAL_JAMO + symbols + SPECIAL_SYMBOLS
        self._vow = _VOWEL_JAMO + [_NO_JAMO_TOKEN, PAD_TOKEN]
        self._fin = _FINAL_JAMO + [_NO_JAMO_TOKEN, PAD_TOKEN]

        self._ini2idx = {e: i for (i, e) in enumerate(self._ini)}
        self._idx2ini = {i: e for (i, e) in enumerate(self._ini)}
        self._vow2idx = {e: i for (i, e) in enumerate(self._vow)}
        self._idx2vow = {i: e for (i, e) in enumerate(self._vow)}
        self._fin2idx = {e: i for (i, e) in enumerate(self._fin)}
        self._idx2fin = {i: e for (i, e) in enumerate(self._fin)}

        self._sos = (
            self._ini2idx[SOS_TOKEN],
            self._vow2idx[_NO_JAMO_TOKEN],
            self._fin2idx[_NO_JAMO_TOKEN],
        )
        self._eos = (
            self._ini2idx[EOS_TOKEN],
            self._vow2idx[_NO_JAMO_TOKEN],
            self._fin2idx[_NO_JAMO_TOKEN],
        )
        self._unk = (
            self._ini2idx[UNK_TOKEN],
            self._vow2idx[_NO_JAMO_TOKEN],
            self._fin2idx[_NO_JAMO_TOKEN],
        )
        self._pad = (
            self._ini2idx[PAD_TOKEN],
            self._vow2idx[PAD_TOKEN],
            self._fin2idx[PAD_TOKEN],
        )

    def __getitem__(self, ele):
        if ele == PAD_TOKEN:
            return self._pad

        if not is_korean_syllable(ele):
            return (
                self._ini2idx.get(ele, self._ini2idx[UNK_TOKEN]),
                self._vow2idx[_NO_JAMO_TOKEN],
                self._fin2idx[_NO_JAMO_TOKEN],
            )

        i, v, f = get_jamo(ele)
        return self._ini2idx[i], self._vow2idx[v], self._fin2idx[f]

    def sizes(self):
        return len(self._ini), len(self._vow), len(self._fin)

    def encode(self, sentence, pad_len=None):
        if pad_len is not None:
            return [self[c] for c in sentence] + [self._pad] * (pad_len - len(sentence))
        else:
            return [self[c] for c in sentence]

    def encode_batch(self, sentences):
        pad_len = max(len(s) for s in sentences)
        return [self.encode(s, pad_len) for s in sentences]

    def decode(self, encoded):
        return [
            (self._idx2ini[i], self._idx2vow[v], self._idx2fin[f])
            for (i, v, f) in encoded
        ]

    def decode_batch(self, encoded_batch):
        return [self.decode(e) for e in encoded_batch]

    def __len__(self):
        return len(self._ini) + len(self._vow) + len(self._fin)

    def sos(self):
        return self._sos

    def eos(self):
        return self._eos

    def unk(self):
        return self._unk

    def pad(self):
        return self._pad


class ThreeHotDictArbitraryOrdering:
    _RESERVED = set(SPECIAL_SYMBOLS + _INITIAL_JAMO + _VOWEL_JAMO + _FINAL_JAMO)
    _JAMO = set(_INITIAL_JAMO + _VOWEL_JAMO + _FINAL_JAMO)
    _JAMO_GROUPS = {"i": _INITIAL_JAMO, "v": _VOWEL_JAMO, "f": _FINAL_JAMO}
    _ORDER_PERM = {
        ("i", "v", "f"): (0, 1, 2),
        ("i", "f", "v"): (0, 2, 1),
        ("v", "i", "f"): (1, 0, 2),
        ("v", "f", "i"): (1, 2, 0),
        ("f", "i", "v"): (2, 0, 1),
        ("f", "v", "i"): (2, 1, 0),
    }

    def __init__(self, symbols, ordering=("i", "v", "f")):
        assert all(s not in self._RESERVED for s in symbols)

        self._ordering = ordering
        self._order_perm = self._ORDER_PERM[self._ordering]
        first, second, third = self._ordering

        self._first = self._JAMO_GROUPS[first] + symbols + SPECIAL_SYMBOLS
        self._second = self._JAMO_GROUPS[second] + [_NO_JAMO_TOKEN, PAD_TOKEN]
        self._third = self._JAMO_GROUPS[third] + [_NO_JAMO_TOKEN, PAD_TOKEN]

        self._first2idx = {e: i for (i, e) in enumerate(self._first)}
        self._idx2first = {i: e for (i, e) in enumerate(self._first)}
        self._second2idx = {e: i for (i, e) in enumerate(self._second)}
        self._idx2second = {i: e for (i, e) in enumerate(self._second)}
        self._third2idx = {e: i for (i, e) in enumerate(self._third)}
        self._idx2third = {i: e for (i, e) in enumerate(self._third)}

        self._sos = (
            self._first2idx[SOS_TOKEN],
            self._second2idx[_NO_JAMO_TOKEN],
            self._third2idx[_NO_JAMO_TOKEN],
        )
        self._eos = (
            self._first2idx[EOS_TOKEN],
            self._second2idx[_NO_JAMO_TOKEN],
            self._third2idx[_NO_JAMO_TOKEN],
        )
        self._unk = (
            self._first2idx[UNK_TOKEN],
            self._second2idx[_NO_JAMO_TOKEN],
            self._third2idx[_NO_JAMO_TOKEN],
        )
        self._pad = (
            self._first2idx[PAD_TOKEN],
            self._second2idx[PAD_TOKEN],
            self._third2idx[PAD_TOKEN],
        )

    def __getitem__(self, ele):
        if ele is PAD_TOKEN:
            return self._pad

        if not is_korean_syllable(ele):
            return (
                self._first2idx.get(ele, self._first2idx[UNK_TOKEN]),
                self._second2idx[_NO_JAMO_TOKEN],
                self._third2idx[_NO_JAMO_TOKEN],
            )

        i, v, f = self._apply_perm(get_jamo(ele))
        return self._first2idx[i], self._second2idx[v], self._third2idx[f]

    def sizes(self):
        return len(self._first), len(self._second), len(self._third)

    def encode(self, sentence, pad_len=None):
        if pad_len is not None:
            return [self[c] for c in sentence] + [self._pad] * (pad_len - len(sentence))
        else:
            return [self[c] for c in sentence]

    def encode_batch(self, sentences):
        pad_len = max(len(s) for s in sentences)
        return [self.encode(s, pad_len) for s in sentences]

    def _apply_perm(self, l):
        return tuple(l[i] for i in self._order_perm)

    def _reverse_perm(self, l):
        out = [None, None, None]
        for i, v in enumerate(self._order_perm):
            out[v] = l[i]

        return tuple(out)

    def decode(self, encoded):
        return [
            (self._idx2first[i], self._idx2second[v], self._idx2third[f])
            if self._idx2first[i] not in self._JAMO
            else self._reverse_perm(
                (self._idx2first[i], self._idx2second[v], self._idx2third[f])
            )
            for (i, v, f) in encoded
        ]

    def decode_batch(self, encoded_batch):
        return [self.decode(e) for e in encoded_batch]

    def __len__(self):
        return len(self._first) + len(self._second) + len(self._third)

    def sos(self):
        return self._sos

    def eos(self):
        return self._eos

    def unk(self):
        return self._unk

    def pad(self):
        return self._pad


class AlphabetDict:
    _RESERVED = set(SPECIAL_SYMBOLS)

    def __init__(self, symbols):
        assert all(s not in self._RESERVED for s in symbols)
        self._alphabet = SPECIAL_SYMBOLS + symbols

        self._ele2idx = {e: i for (i, e) in enumerate(self._alphabet)}
        self._idx2ele = {i: e for (i, e) in enumerate(self._alphabet)}

        self._sos = self._ele2idx[SOS_TOKEN]
        self._eos = self._ele2idx[EOS_TOKEN]
        self._unk = self._ele2idx[UNK_TOKEN]
        self._pad = self._ele2idx[PAD_TOKEN]

    def __getitem__(self, ele):
        return self._ele2idx.get(ele, self._unk)

    def encode(self, sentence, pad_len=None):
        if pad_len is not None:
            return [self._ele2idx.get(c, self._unk) for c in sentence] + [self._pad] * (
                pad_len - len(sentence)
            )
        else:
            return [self._ele2idx.get(c, self._unk) for c in sentence]

    def encode_batch(self, sentences):
        pad_len = max(len(s) for s in sentences)
        return [self.encode(s, pad_len) for s in sentences]

    def decode(self, encoded):
        return [self._idx2ele[c] for c in encoded]

    def decode_batch(self, encoded_batch):
        return [self.decode(e) for e in encoded_batch]

    def __len__(self):
        return len(self._alphabet)

    def sos(self):
        return self._sos

    def eos(self):
        return self._eos

    def unk(self):
        return self._unk

    def pad(self):
        return self._pad
