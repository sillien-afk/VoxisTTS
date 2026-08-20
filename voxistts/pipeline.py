"""
voxistts.pipeline
================
语言感知的 G2P + 推理编排（替代 kokoro.KPipeline）。

与 KPipeline 的区别：
  - G2P 改用本包自研实现（voxistts.g2p_zh / voxistts.g2p_en），不依赖 misaki
  - 类改名为 VoxisPipeline
  - 声学推理走 VoxisTTS（= vendored VoxisModel，见 acoustic.py，已断耦 kokoro 包）

用法：
    pipe = VoxisPipeline(lang_code='z')
    for graphemes, ps, audio in pipe("你好世界", voice='zf_xiaoxiao'):
        ...
"""

import re
import torch

from .model import VoxisTTS, DEFAULT_CONFIG, DEFAULT_MODEL
from . import g2p_zh, g2p_en

VOICES_DIR = r"E:/llm_models/kokoro-82m/voices"
SAMPLE_RATE = 24000
MAX_PHONEME_LEN = 510


class VoxisPipeline:
    def __init__(self, lang_code: str = "z", model: object = True, device: str = "cpu"):
        lang_code = lang_code.lower()
        self.lang_code = lang_code
        if isinstance(model, VoxisTTS):
            self.model = model
        elif model:
            self.model = VoxisTTS(device=device)
        else:
            self.model = None
        self.voices = {}

        if lang_code == "z":
            self.g2p = g2p_zh.g2p_zh
        else:
            self.g2p = g2p_en.g2p_en

    def load_voice(self, voice: str) -> torch.FloatTensor:
        if voice in self.voices:
            return self.voices[voice]
        if voice.endswith(".pt"):
            f = voice
        else:
            f = f"{VOICES_DIR}/{voice}.pt"
        pack = torch.load(f, weights_only=True)
        self.voices[voice] = pack
        return pack

    def __call__(self, text, voice: str = "zf_xiaoxiao", speed: float = 1.0):
        model = self.model
        pack = self.load_voice(voice).to(model.device) if model else None
        if isinstance(text, str):
            texts = re.split(r"\n+", text.strip())
        else:
            texts = text
        for graphemes in texts:
            ps = self.g2p(graphemes)
            if not ps:
                continue
            if len(ps) > MAX_PHONEME_LEN:
                ps = ps[:MAX_PHONEME_LEN]
            if model:
                # 与原 KPipeline.infer 保持一致：ref_s = pack[len(ps)-1]
                output = model(ps, pack[len(ps) - 1], speed, return_output=True)
                yield graphemes, ps, output.audio
            else:
                yield graphemes, ps, None
