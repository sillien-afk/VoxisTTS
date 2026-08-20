"""
voxistts.g2p_en
==============
轻量英文 G2P（我们自己的实现，**零 misaki / spacy / num2words / phonemizer 依赖**）。

核心思路：
  1. 直接用 espeak-ng 的 C 库（通过 espeakng_loader 提供的 libespeak-ng.dll）做英文
     音素化，得到带重音（ˈ/ˌ）与连音符（^）的 IPA 字符串；
  2. 套用 misaki.espeak.EspeakFallback 的 `E2M` 修正表 + 后处理，把 espeak 原生
     IPA 转成 Kokoro 声学模型所用的特定 IPA 字符集（见 acoustic.py 的 VOCAB）；
  3. 分词后逐词 phonemize，再补回 vocab 内的英文标点（逗号/句号/问号等），
     使模型能正确产生停顿。

对比 Kokoro 原英文 G2P（misaki.en.G2P）：原实现依赖 spacy(POS) + num2words(数字)
+ 词典 JSON + espeak 兜底，是一整条重型链。本实现只用 espeak-ng 作唯一音素源，
数字由 espeak-ng 直接拼读，因此彻底去掉了 spacy / num2words / phonemizer。

出处与许可：
  - E2M 修正表及后处理逻辑移植自 misaki（MIT，https://github.com/hexgrad/misaki）
    的 espeak.EspeakFallback，仅复用了「espeak 音素 -> Kokoro-IPA」的映射知识，
    未引入 misaki 运行时。
  - espeak-ng 本身为 GPLv3，经 espeakng_loader 以预编译 DLL 形式随包分发。
"""

import ctypes
import logging
import re

from .vocab import VALID_PHONEMES

# ---------------------------------------------------------------------------
# 1) espeak-ng 的轻量 ctypes 封装（移植自 phonemizer 的 text_to_phonemes，去依赖化）
#    关键事实（实测确认）：
#      - espeak_TextToPhonemes 的返回类型就是 c_char_p（直接返回音素串），
#        不是通过 outbuf 输出；
#      - 第一个参数必须是 char**（pointer(c_char_p(utf8_bytes))）；
#      - flags 用「旧格式」：0x02 | 0x80 | (ord(tie) << 8)，此 espeak-ng 构建才认。
# ---------------------------------------------------------------------------

_ES = None  # 单例


class EspeakNG:
    """极薄的 libespeak-ng 封装：给定英文文本，返回带重音/连音的 IPA 字符串。"""

    _LIB = None
    _DATA_PATH = None

    def __init__(self, voice: str = "en-us", tie: str = "^"):
        if EspeakNG._LIB is None:
            EspeakNG._load_library()
        self._lib = EspeakNG._LIB
        self._voice = voice
        self._tie = tie
        self._init()
        self.set_voice(voice)

    @staticmethod
    def _load_library():
        try:
            from espeakng_loader import load_library, get_data_path
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "voxistts.g2p_en 需要 espeak-ng（通过 pip install espeakng_loader 获得）"
            ) from e
        lib = load_library()
        if lib is None:
            raise RuntimeError("无法加载 libespeak-ng.dll（espeakng_loader 安装异常）")
        lib.espeak_Initialize.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
        lib.espeak_Initialize.restype = ctypes.c_int
        lib.espeak_SetVoiceByName.argtypes = (ctypes.c_char_p,)
        lib.espeak_SetVoiceByName.restype = ctypes.c_int
        lib.espeak_TextToPhonemes.argtypes = (ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int)
        lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
        EspeakNG._LIB = lib
        EspeakNG._DATA_PATH = get_data_path()

    def _init(self):
        # 2 = AUDIO_OUTPUT_SYNCHRONIZATION（不真正播放音频）
        r = self._lib.espeak_Initialize(2, 0, EspeakNG._DATA_PATH.encode("utf-8"), 0)
        if r <= 0:
            raise RuntimeError(f"espeak_Initialize 失败：{r}")

    def set_voice(self, voice: str):
        if self._lib.espeak_SetVoiceByName(voice.encode("utf-8")) != 0:
            raise RuntimeError(f"espeak 加载语音失败：{voice}")

    def phonemes(self, text: str) -> str:
        """英文文本 -> IPA 音素串（含重音 ˈ/ˌ 与连音符 ^）。"""
        text_ptr = ctypes.pointer(ctypes.c_char_p(text.encode("utf-8")))
        text_mode = 1  # espeakCHARS_UTF8
        phonemes_mode = 0x02 | 0x80 | (ord(self._tie) << 8)
        out = []
        # espeak 在消费完文本后把 *text_ptr 置空，循环自然结束
        while text_ptr.contents.value is not None:
            res = self._lib.espeak_TextToPhonemes(text_ptr, text_mode, phonemes_mode)
            if res:
                out.append(res.decode("utf-8"))
        return " ".join(out)


# ---------------------------------------------------------------------------
# 2) espeak 音素 -> Kokoro-IPA 的修正表（移植自 misaki.espeak.EspeakFallback.E2M）
#    按长度降序排列，保证最长匹配优先（如 'a^ɪ' 先于 'a'）。
# ---------------------------------------------------------------------------

E2M = sorted({
    'ʔˌn\u0329': 'tn', 'ʔn\u0329': 'tn', 'ʔn': 'tn', 'ʔ': 't',
    'a^ɪ': 'I', 'a^ʊ': 'W',
    'd^ʒ': 'ʤ',
    'e^ɪ': 'A', 'e': 'A',
    't^ʃ': 'ʧ',
    'ɔ^ɪ': 'Y',
    'ə^l': 'ᵊl',
    'ʲo': 'jo', 'ʲə': 'jə', 'ʲ': '',
    'ɚ': 'əɹ',
    'r': 'ɹ',
    'x': 'k', 'ç': 'k',
    'ɐ': 'ə',
    'ɬ': 'l',
    '\u0303': '',
}.items(), key=lambda kv: -len(kv[0]))


def _apply_espeak_to_kokoro(ps: str, british: bool = False) -> str:
    """把 espeak 原生 IPA 转成 Kokoro 声学模型所用的 IPA 字符集。"""
    for old, new in E2M:
        ps = ps.replace(old, new)
    # \u0329 = 组合用竖线下方符号（syllabic），\u0303 = 组合用波浪号
    ps = re.sub(r'(\S)\u0329', r'ᵊ\1', ps).replace('\u0309', '')
    if british:
        ps = ps.replace('e^ə', 'ɛː')
        ps = ps.replace('iə', 'ɪə')
        ps = ps.replace('ə^ʊ', 'Q')
    else:
        ps = ps.replace('o^ʊ', 'O')
        ps = ps.replace('ɜːɹ', 'ɜɹ')
        ps = ps.replace('ɜː', 'ɜɹ')
        ps = ps.replace('ɪə', 'iə')
        ps = ps.replace('ː', '')
    ps = ps.replace('o', 'ɔ')  # 兼容 espeak < 1.52
    return ps.replace('^', '')


# ---------------------------------------------------------------------------
# 3) 分词 + 逐词 phonemize + 补回标点
# ---------------------------------------------------------------------------

# 单词 = 字母/数字/撇号；其余非空白单字符视为标点；空白归一为空格
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]|\s+")

# 英文标点 -> Kokoro vocab 内的对应字符；不在 vocab 的直接丢弃
_PUNCT_MAP = {
    ',': ',', '.': '.', '!': '!', '?': '?', ';': ';', ':': ':',
    '(': '(', ')': ')', '"': '"', "'": "'", '“': '"', '”': '"',
    '—': '—', '…': '…', '-': ' ', '_': ' ',
}


def _map_punct(ch: str) -> str:
    return _PUNCT_MAP.get(ch, '')


def g2p_en(text: str, british: bool = False) -> str:
    """英文文本 -> Kokoro IPA 音素串（含重音、vocab 内标点）。"""
    global _ES
    if _ES is None:
        _ES = EspeakNG(voice="en-us", tie="^")

    if not text or not text.strip():
        return ""

    chunks = []
    for tok in _TOKEN_RE.findall(text):
        if tok.strip() == "":
            chunks.append(" ")  # 空白 -> 空格
        elif re.match(r"[A-Za-z0-9']+$", tok):
            raw = _ES.phonemes(tok)          # 逐词 phonemize
            raw = _apply_espeak_to_kokoro(raw, british=british)
            chunks.append(raw)
        else:
            mapped = _map_punct(tok)         # 标点
            if mapped:
                chunks.append(mapped)

    ps = "".join(chunks)
    # 折叠多余空格
    ps = re.sub(r"\s+", " ", ps).strip()
    # vocab 校验：保留 vocab 内字符与空格，丢弃模型会静默丢掉的字符（并记录）
    cleaned = "".join(c for c in ps if c in VALID_PHONEMES or c == " ")
    dropped = set(ps) - set(cleaned)
    if dropped:
        logging.warning("voxistts.g2p_en: 丢弃不在 vocab 内的字符 %r（已忽略）", sorted(dropped))
    return cleaned
