"""
VoxisTTS CLI
===========
用法：
  python -m voxistts "你好，世界" -v zf_xiaoxiao -o out.wav
  python -m voxistts "Hello world" -l a -v af_heart -o out_en.wav
"""

import argparse
import sys

import soundfile as sf

from .pipeline import VoxisPipeline, SAMPLE_RATE


def main():
    ap = argparse.ArgumentParser(prog="voxistts", description="我们自己的轻量中英文 TTS")
    ap.add_argument("text", help="要合成的文本")
    ap.add_argument("-v", "--voice", default="zf_xiaoxiao", help="声库名")
    ap.add_argument("-o", "--out", default="voxistts_out.wav", help="输出 wav 路径")
    ap.add_argument("-s", "--speed", type=float, default=1.0, help="语速")
    ap.add_argument("-l", "--lang", default="z", help="语言代码 z=中文 a=英文")
    args = ap.parse_args()

    pipe = VoxisPipeline(lang_code=args.lang)
    wrote = False
    for graphemes, ps, audio in pipe(args.text, voice=args.voice, speed=args.speed):
        if audio is not None:
            sf.write(args.out, audio.cpu().numpy(), SAMPLE_RATE)
            dur = len(audio) / SAMPLE_RATE
            print(f"[VoxisTTS] 已写入 {args.out}（{dur:.1f}s）")
            print(f"[VoxisTTS] 音素: {ps}")
            wrote = True
    if not wrote:
        print("[VoxisTTS] 未生成音频（可能 G2P 无输出或模型未加载）", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
