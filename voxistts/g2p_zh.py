"""
voxistts.g2p_zh
==============
中文 G2P：汉字文本 -> Kokoro IPA 音素串（仅用 vocab 内字符）。

流程（移植自 misaki.zh.ZHG2P.legacy_call，去除 misaki 包依赖）：
  1. cn2an.transform：阿拉伯数字 -> 中文（"2024" -> "二零二四"）
  2. map_punctuation：中文标点 -> ASCII 标点
  3. jieba 分词（仅对中文片段）
  4. 逐词：pypinyin 取拼音(声调数字) -> transcription.pinyin_to_ipa -> retone
  5. 末尾剥除 U+032F（̯，inverted breve below，Kokoro vocab 不需要）

retone 关键：把声调 IPA（˥/˧˥/˧˩˧/˥˩）转成 Kokoro vocab 里的
箭头符号（→/↗/↓/↘），并把 ɻ̩/ɹ̩ 折成 ɨ。输出字符全部落在 vocab 内。
"""

import logging
import re

import cn2an
import jieba
from pypinyin import Style, lazy_pinyin

from .transcription import pinyin_to_ipa
from .vocab import VALID_PHONEMES

# U+032F = ̯ (inverted breve below)，misaki 在末尾统一剥除
_BREVE_BELOW = chr(0x032F)


def retone(p: str) -> str:
    # 顺序很重要：先处理多字符声调符，最后再处理单字符一声（˥），
    # 避免把二/三/四声里的 ˥ 提前替换掉。
    p = p.replace("˧˩˧", "↓")  # 三声
    p = p.replace("˧˥", "↗")   # 二声
    p = p.replace("˥˩", "↘")   # 四声
    p = p.replace("˥", "→")    # 一声
    p = p.replace(chr(635) + chr(809), "ɨ").replace(chr(633) + chr(809), "ɨ")  # ɻ̩/ɹ̩ -> ɨ
    assert chr(809) not in p, p
    return p


def py2ipa(py: str) -> str:
    try:
        variants = pinyin_to_ipa(py)
        if not variants:
            return ""
        return "".join(retone(p) for p in variants[0])
    except Exception as e:  # noqa: BLE001
        logging.warning("VoxisTTS: 拼音 %r 无法转 IPA（%s），原样透传", py, e)
        return py


def word2ipa(w: str) -> str:
    pys = lazy_pinyin(w, style=Style.TONE3, neutral_tone_with_five=True)
    return "".join(py2ipa(py) for py in pys)


def map_punctuation(text: str) -> str:
    text = text.replace("、", ", ").replace("，", ", ")
    text = text.replace("。", ". ").replace("．", ". ")
    text = text.replace("！", "! ")
    text = text.replace("：", ": ")
    text = text.replace("；", "; ")
    text = text.replace("？", "? ")
    text = text.replace("«", ' "').replace("»", '" ')
    text = text.replace("《", ' "').replace("》", '" ')
    text = text.replace("「", ' "').replace("」", '" ')
    text = text.replace("【", ' "').replace("】", '" ')
    text = text.replace("（", " (").replace("）", ") ")
    return text.strip()


def legacy_call(text: str) -> str:
    is_zh = bool(re.match(r"[\u4E00-\u9FFF]", text[0]))
    result = ""
    for segment in re.findall(r"[\u4E00-\u9FFF]+|[^\u4E00-\u9FFF]+", text):
        if is_zh:
            words = jieba.lcut(segment, cut_all=False)
            segment = " ".join(word2ipa(w) for w in words)
        result += segment
        is_zh = not is_zh
    return result.replace(_BREVE_BELOW, "")


def g2p_zh(text: str) -> str:
    """中文文本 -> Kokoro IPA 音素串。"""
    if not text or not text.strip():
        return ""
    text = cn2an.transform(text, "an2cn")
    text = map_punctuation(text)
    ps = legacy_call(text)
    _validate(ps)
    return ps


def _validate(ps: str) -> None:
    bad = set(ps) - VALID_PHONEMES
    if bad:
        logging.warning(
            "VoxisTTS: G2P 输出含 vocab 外字符 %r，模型将丢弃这些音素",
            "".join(sorted(bad)),
        )
