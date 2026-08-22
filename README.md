# VoxisTTS -- 轻量中英文 TTS (Kokoro-82M 架构，去 misaki/kokoro 依赖)

Python 包：`pip install voxistts`

## 在线试听

**[🎵 点击播放音频](https://sillien-afk.github.io/VoxisTTS/demo.html)**

> 浏览器内直接播放三个 demo 音频（中文 / 英文 / 耳测对比），无需下载

如果链接打不开，本地运行：
```bash
python -m voxistts "你好世界" -v zf_xiaoxiao -o out.wav
```

## 安装

```bash
pip install voxistts
```

依赖：torch、scipy、transformers、huggingface-hub、pypinyin、espeakng-loader、loguru、jieba

## 快速上手

```python
from voxistts import VoxisTTS, VoxisPipeline

model = VoxisTTS(device="cpu")
pipe = VoxisPipeline(lang_code="z")

for graphemes, ps, audio in pipe("你好世界", voice="zf_xiaoxiao"):
    print(ps)  # 音素
    # audio: torch.Tensor, 采样率 24000
```

CLI：
```bash
python -m voxistts "你好世界" -v zf_xiaoxiao -o out.wav
```

## 文件说明

- `acoustic.py` -- vendored 自 Kokoro-82M / StyleTTS2（Apache 2.0 / MIT）
- `g2p_zh.py` -- 中文 G2P，pypinyin→IPA（零 misaki/spacy 依赖）
- `g2p_en.py` -- 英文 G2P，espeak-ng ctypes 直调（零 phonemizer 依赖）
- `pipeline.py` -- VoxisPipeline，替代 kokoro.KPipeline
- `model.py` -- VoxisTTS 模型类，默认加载本地 E:/llm_models/kokoro-82m/
- `vocab.py` -- 178 个 IPA token 契约
- `transcription.py` -- 拼音→Kokoro IPA 映射

## 权重

本包不含模型权重，需自行准备：
- `config.json` + `model.pth`：Kokoro-82M 官方权重（~327MB）
- `voices/`：说话人嵌入 .pt 文件（如 zf_xiaoxiao.pt ~523KB）

## License

声学代码保留上游 MIT / Apache 2.0 署名。本包新增代码 MIT。
