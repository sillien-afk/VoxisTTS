"""
tools/synthesize_demo.py
=======================
端到端合成演示：用 VoxisTTS 自己的 G2P + Kokoro 声学模型合成中文语音。
验证「我们自己的 TTS」完整跑通。
"""

import sys

import soundfile as sf
import torch

sys.path.insert(0, ".")

from voxistts.pipeline import VoxisPipeline, SAMPLE_RATE

TEXT = "你好，世界。这是VoxisTTS，我们自己的中文语音合成系统，完全不依赖 misaki。"


def main():
    pipe = VoxisPipeline(lang_code="z")
    print("model device:", pipe.model.device)

    pack = pipe.load_voice("zf_xiaoxiao")
    print("voice pack shape:", tuple(pack.shape))

    wrote = False
    for graphemes, ps, audio in pipe(TEXT, voice="zf_xiaoxiao", speed=1.0):
        if audio is not None:
            sf.write("demo_zh.wav", audio.cpu().numpy(), SAMPLE_RATE)
            dur = len(audio) / SAMPLE_RATE
            print(f"[OK] 写入 demo_zh.wav（{dur:.1f}s，{len(ps)} 音素）")
            print(f"[OK] 音素序列: {ps}")
            wrote = True
    if not wrote:
        print("[FAIL] 未生成音频", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
