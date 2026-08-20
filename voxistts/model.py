"""
voxistts.model
=============
声学模型封装。

后端为「我们自己的 TTS」声学核心 —— 直接复用 Kokoro-82M 的 ISTFTNet（82M，轻量、
已验证），但代码已 vendoring 进 voxistts.acoustic（见 acoustic.py），不再 import 任何
kokoro 发行包，从而彻底切断对 misaki / num2words 的间接依赖。

  - 默认从本地加载 config.json / model.pth（不联网下载）
  - 类改名为 VoxisTTS
  - 语言盲：只认 vocab 内的音素字符，G2P 由 VoxisTTS.g2p_* 负责
"""

from .acoustic import VoxisModel

DEFAULT_CONFIG = r"E:/llm_models/kokoro-82m/config.json"
DEFAULT_MODEL = r"E:/llm_models/kokoro-82m/model.pth"


class VoxisTTS(VoxisModel):
    # 改名占位：标识这是我们自己的模型（权重来源切换为本地 in-house 路径）
    REPO_ID = "in-house/voxistts-82m"

    def __init__(self, config=DEFAULT_CONFIG, model=DEFAULT_MODEL, device="cpu"):
        # config / model 均接受本地路径字符串
        super().__init__(config=config, model=model)
        self.to(device).eval()
