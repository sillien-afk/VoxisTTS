"""
voxistts.transcription
=====================
「拼音(声调) -> Kokoro IPA 音素串」的核心映射与转换逻辑。

移植自 misaki/transcription.py（MIT License，
原作 https://github.com/stefantaubert/pinyin-to-ipa ，Apache/MIT 改编）。
保留出处以符合许可证要求；逻辑完整移植，不依赖 misaki 包。

仅依赖：pypinyin(含 contrib.tone_convert) / ordered_set / 标准库。
无任何重型 NLP 依赖（无 spacy / phonemizer / thinc）。
"""

import itertools
from typing import Dict, Generator, List, Optional, Tuple

from ordered_set import OrderedSet
from pypinyin.contrib.tone_convert import to_finals, to_initials, to_normal, to_tone3

# ---------------------------------------------------------------------------
# 声母映射
# ---------------------------------------------------------------------------
INITIAL_MAPPING: Dict[str, List[Tuple[str, ...]]] = {
    "b": [("p",)],
    "c": [("ʦʰ",)],
    "ch": [("\uAB67ʰ",)],
    "d": [("t",)],
    "f": [("f",)],
    "g": [("k",)],
    "h": [("x",), ("h",)],
    "j": [("ʨ",)],
    "k": [("kʰ",)],
    "l": [("l",)],
    "m": [("m",)],
    "n": [("n",)],
    "p": [("pʰ",)],
    "q": [("ʨʰ",)],
    "r": [("ɻ",), ("ʐ",)],
    "s": [("s",)],
    "sh": [("ʂ",)],
    "t": [("tʰ",)],
    "x": [("ɕ",)],
    "z": [("ʦ",)],
    "zh": [("\uAB67",)],
}

INITIALS = INITIAL_MAPPING.keys()

# ---------------------------------------------------------------------------
# 音节化辅音（叹词/独立音节）
# ---------------------------------------------------------------------------
SYLLABIC_CONSONANT_MAPPINGS: Dict[str, List[Tuple[str, ...]]] = {
    "hm": [("h", "m0",)],
    "hng": [("h", "ŋ0",)],
    "m": [("m0",)],
    "n": [("n0",)],
    "ng": [("ŋ0",)],
}

SYLLABIC_CONSONANTS = SYLLABIC_CONSONANT_MAPPINGS.keys()

INTERJECTION_MAPPINGS: Dict[str, List[Tuple[str, ...]]] = {
    "io": [("j", "ɔ0")],
    "ê": [("ɛ0",)],
    "er": [("ɚ0",), ("aɚ̯0",)],
    "o": [("ɔ0",)],
}

INTERJECTIONS = INTERJECTION_MAPPINGS.keys()

# ---------------------------------------------------------------------------
# 韵母映射（“0” 为声调占位符，apply_tone 时替换为声调 IPA）
# ---------------------------------------------------------------------------
FINAL_MAPPING: Dict[str, List[Tuple[str, ...]]] = {
    "a": [("a0",)],
    "ai": [("ai̯0",)],
    "an": [("a0", "n")],
    "ang": [("a0", "ŋ")],
    "ao": [("au̯0",)],
    "e": [("ɤ0",)],
    "ei": [("ei̯0",)],
    "en": [("ə0", "n")],
    "eng": [("ə0", "ŋ")],
    "i": [("i0",)],
    "ia": [("j", "a0")],
    "ian": [("j", "ɛ0", "n")],
    "iang": [("j", "a0", "ŋ")],
    "iao": [("j", "au̯0")],
    "ie": [("j", "e0")],
    "in": [("i0", "n")],
    "iou": [("j", "ou̯0")],
    "ing": [("i0", "ŋ")],
    "iong": [("j", "ʊ0", "ŋ")],
    "ong": [("ʊ0", "ŋ")],
    "ou": [("ou̯0",)],
    "u": [("u0",)],
    "uei": [("w", "ei̯0")],
    "ua": [("w", "a0")],
    "uai": [("w", "ai̯0")],
    "uan": [("w", "a0", "n")],
    "uen": [("w", "ə0", "n")],
    "uang": [("w", "a0", "ŋ")],
    "ueng": [("w", "ə0", "ŋ")],
    "uo": [("w", "o0")],
    "o": [("w", "o0")],
    "ü": [("y0",)],
    "üe": [("ɥ", "e0")],
    "üan": [("ɥ", "ɛ0", "n")],
    "ün": [("y0", "n")],
}

FINALS = FINAL_MAPPING.keys()

FINAL_MAPPING_AFTER_ZH_CH_SH_R: Dict[str, List[Tuple[str, ...]]] = {
    "i": [("ɻ̩0",), ("ʐ̩0",)],
}

FINAL_MAPPING_AFTER_Z_C_S: Dict[str, List[Tuple[str, ...]]] = {
    "i": [("ɹ̩0",), ("z̩0",)],
}

# ---------------------------------------------------------------------------
# 声调映射（数字声调 -> IPA 声调符）
# ---------------------------------------------------------------------------
TONE_MAPPING = {
    1: "˥",
    2: "˧˥",
    3: "˧˩˧",
    4: "˥˩",
    5: "",
}


def get_tone(pinyin: str) -> int:
    pinyin_tone3 = to_tone3(pinyin, neutral_tone_with_five=True, v_to_u=True)
    if len(pinyin_tone3) == 0:
        raise ValueError("Parameter 'pinyin': Tone couldn't be detected!")
    tone_nr_str = pinyin_tone3[-1]
    try:
        tone_nr = int(tone_nr_str)
    except ValueError as error:
        raise ValueError(f"Parameter 'pinyin': Tone '{tone_nr_str}' couldn't be detected!") from error
    if tone_nr not in TONE_MAPPING:
        raise ValueError(f"Parameter 'pinyin': Tone '{tone_nr_str}' couldn't be detected!")
    return tone_nr


def get_syllabic_consonant(normal_pinyin: str) -> Optional[str]:
    if normal_pinyin in SYLLABIC_CONSONANTS:
        return normal_pinyin
    return None


def get_interjection(normal_pinyin: str) -> Optional[str]:
    if normal_pinyin in INTERJECTIONS:
        return normal_pinyin
    return None


def get_initials(normal_pinyin: str) -> Optional[str]:
    if normal_pinyin in SYLLABIC_CONSONANTS:
        return None
    if normal_pinyin in INTERJECTIONS:
        return None
    pinyin_initial = to_initials(normal_pinyin, strict=True)
    if pinyin_initial == "":
        return None
    if pinyin_initial not in INITIAL_MAPPING:
        raise ValueError(f"Parameter 'normal_pinyin': Initial '{pinyin_initial}' couldn't be detected!")
    return pinyin_initial


def get_finals(normal_pinyin: str) -> Optional[str]:
    if normal_pinyin in SYLLABIC_CONSONANTS:
        return None
    if normal_pinyin in INTERJECTIONS:
        return None
    pinyin_final = to_finals(normal_pinyin, strict=True, v_to_u=True)
    if pinyin_final == "":
        raise ValueError("Parameter 'normal_pinyin': Final couldn't be detected!")
    if pinyin_final not in FINAL_MAPPING:
        raise ValueError(f"Parameter 'normal_pinyin': Final '{pinyin_final}' couldn't be detected!")
    return pinyin_final


def apply_tone(variants: List[Tuple[str, ...]], tone: int) -> Generator[Tuple[str, ...], None, None]:
    tone_ipa = TONE_MAPPING[tone]
    yield from (
        tuple(phoneme.replace("0", tone_ipa) for phoneme in variant)
        for variant in variants
    )


def pinyin_to_ipa(pinyin: str) -> OrderedSet[Tuple[str, ...]]:
    """拼音(含声调数字) -> IPA 音素元组的有序集合（取 [0] 为首选）。"""
    tone_nr = get_tone(pinyin)
    pinyin_normal = to_normal(pinyin)

    interjection = get_interjection(pinyin_normal)
    if interjection is not None:
        interjection_ipa_mapping = INTERJECTION_MAPPINGS[pinyin_normal]
        interjection_ipa = OrderedSet(apply_tone(interjection_ipa_mapping, tone_nr))
        return interjection_ipa

    syllabic_consonant = get_syllabic_consonant(pinyin_normal)
    if syllabic_consonant is not None:
        syllabic_consonant_ipa_mapping = SYLLABIC_CONSONANT_MAPPINGS[syllabic_consonant]
        syllabic_consonant_ipa = OrderedSet(apply_tone(syllabic_consonant_ipa_mapping, tone_nr))
        return syllabic_consonant_ipa

    parts = []
    pinyin_initial = get_initials(pinyin_normal)
    pinyin_final = get_finals(pinyin_normal)
    assert pinyin_final is not None

    if pinyin_initial is not None:
        initial_phonemes = INITIAL_MAPPING[pinyin_initial]
        parts.append(initial_phonemes)

    final_phonemes: List[Tuple[str, ...]]
    if pinyin_initial in {"zh", "ch", "sh", "r"} and pinyin_final in FINAL_MAPPING_AFTER_ZH_CH_SH_R:
        final_phonemes = FINAL_MAPPING_AFTER_ZH_CH_SH_R[pinyin_final]
    elif pinyin_initial in {"z", "c", "s"} and pinyin_final in FINAL_MAPPING_AFTER_Z_C_S:
        final_phonemes = FINAL_MAPPING_AFTER_Z_C_S[pinyin_final]
    else:
        final_phonemes = FINAL_MAPPING[pinyin_final]

    final_phonemes = list(apply_tone(final_phonemes, tone_nr))
    parts.append(final_phonemes)

    assert len(parts) >= 1

    all_syllable_combinations = OrderedSet(
        tuple(itertools.chain.from_iterable(combination))
        for combination in itertools.product(*parts)
    )
    return all_syllable_combinations
