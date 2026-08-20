"""
VoxisTTS —— 我们自己的轻量中英文 TTS
=====================================

架构：
  - 声学核心：沿用 Kokoro-82M 的 ISTFTNet（82M，轻量、已验证），但去掉 misaki 重型 G2P 链
  - G2P（本包自研，替代 misaki）：
      * 中文：pypinyin 取拼音 -> 拼音→Kokoro-IPA 映射（见 transcription.py）
      * 英文：espeak-ng（无 spacy / phonemizer / thinc）
  - 声学核心（acoustic.py，vendored 自 Kokoro-82M / StyleTTS2，Apache 2.0）：
        已去除对 kokoro 发行包的依赖，不再间接依赖 misaki / num2words
  - 去掉的依赖：spacy, phonemizer, thinc, preshed, murmurhash, srsly,
                smart-open, catalogue, blis, confection, proces, misaki, kokoro

设计约束（来自 Kokoro 的 VoxisModel.forward）：
  - 模型输入是一串「音素字符」，每个字符必须在 vocab 内；
    不在 vocab 的字符会被静默丢弃，导致发音错误。
  - 因此本包所有 G2P 输出都必须落在 VOCAB 字符集内，
    g2p 模块会对输出做 vocab 校验。
"""

__version__ = "0.1.0"
__author__ = "in-house"

from .vocab import VOCAB, VALID_PHONEMES
