import string

SOS_TOKEN, EOS_TOKEN, PUNC_TOKEN, NUMBER_TOKEN, OTHER_TOKEN, UNK_TOKEN, PAD_TOKEN = (
    "始",
    "末",
    "点",
    "数",
    "他",
    "不",
    "無",
)

SPECIAL_SYMBOLS = [
    SOS_TOKEN,
    EOS_TOKEN,
    PUNC_TOKEN,
    NUMBER_TOKEN,
    OTHER_TOKEN,
    UNK_TOKEN,
    PAD_TOKEN,
]

COMPATIBILITY_JAMO = [
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄸ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅃ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
    "ㅏ",
    "ㅐ",
    "ㅑ",
    "ㅒ",
    "ㅓ",
    "ㅔ",
    "ㅕ",
    "ㅖ",
    "ㅗ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅛ",
    "ㅜ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅠ",
    "ㅡ",
    "ㅢ",
    "ㅣ",
]

SYLLABLES = [chr(c) for c in range(0xAC00, 0xD7A3 + 1)]
_EMPTY_FINAL_JAMO = "☒"
_NO_JAMO_TOKEN = "㋨"

_INITIAL_JAMO = [chr(cp) for cp in range(0x1100, 0x1112 + 1)]
_VOWEL_JAMO = [chr(cp) for cp in range(0x1161, 0x1175 + 1)]
_FINAL_JAMO = [chr(cp) for cp in range(0x11A8, 0x11C2 + 1)] + ["☒"]

_INIT2COMPAT = {i: c for (i, c) in zip(_INITIAL_JAMO, "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")}
_VOW2COMPAT = {i: c for (i, c) in zip(_VOWEL_JAMO, "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")}
_FINAL2COMPAT = {i: c for (i, c) in zip(_FINAL_JAMO, "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ☒")}
_JAMO2COMPAT = {
    i: c
    for (i, c) in zip(
        _INITIAL_JAMO + _VOWEL_JAMO + _FINAL_JAMO,
        "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ☒",
    )
}

JAMO = _INITIAL_JAMO + _VOWEL_JAMO + _FINAL_JAMO

ENGLISH = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
PUNCTUATION = [" "] + list(string.punctuation)


def is_korean_syllable(c):
    return 44032 <= ord(c) <= 55203


def get_jamo(c):
    if not is_korean_syllable(c):
        return c
    cp = ord(c)
    final = (cp - 44032) % 28
    vowel = 1 + ((cp - 44032 - final) % 588) // 28
    initial = 1 + (cp - 44032) // 588

    if final == 0:
        return _INITIAL_JAMO[initial - 1], _VOWEL_JAMO[vowel - 1], _EMPTY_FINAL_JAMO
    else:
        return (
            _INITIAL_JAMO[initial - 1],
            _VOWEL_JAMO[vowel - 1],
            _FINAL_JAMO[final - 1],
        )


def char_to_triplet(c):
    if not is_korean_syllable(c):
        if c == PAD_TOKEN:
            return (PAD_TOKEN, PAD_TOKEN, PAD_TOKEN)
        else:
            return (c, _NO_JAMO_TOKEN, _NO_JAMO_TOKEN)
    return get_jamo(c)


def is_full_syllable(i, v, f):
    return i in _INITIAL_JAMO and v in _VOWEL_JAMO and f in _FINAL_JAMO


def is_jamo(c):
    return c in _INITIAL_JAMO or c in _VOWEL_JAMO or c in _FINAL_JAMO


def canonicalize_triplets(triplet_seq):
    '''
    Here we canonicalize triplets by:

    1) removing padding
    2) removing degenerate triples (like mix of jamo and non-jamo, padding and non-padding etc)
    3) removing _NO_JAMO_TOKENS from (x, ㋨, ㋨) triples
    4) converting to compatibility jamo so that we can get a jamo string.
    '''
    out = []
    for (i, v, f) in triplet_seq:

        if is_full_syllable(i, v, f):
            out.extend((i, v, f))
        else:
            # pad
            if i == PAD_TOKEN and v == PAD_TOKEN and f == PAD_TOKEN:
                pass
            # english
            elif v == _NO_JAMO_TOKEN and f == _NO_JAMO_TOKEN:
                if i == PAD_TOKEN or is_jamo(i):
                    out.append(UNK_TOKEN)
                else:
                    out.append(i)
            # degenerate
            else:
                out.append(UNK_TOKEN)
    return [_JAMO2COMPAT.get(c, c) for c in out]
