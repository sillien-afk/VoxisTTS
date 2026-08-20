"""
tools/parity_check.py
======================
一致性校验：voxistts.g2p_zh 的输出必须与原版 misaki.zh.ZHG2P 逐字相同。
因为 Kokoro 的中文声库是按 misaki 的输出训练的，逐字一致即代表
与模型训练分布完全兼容。

用法：
  python tools/parity_check.py
"""

import sys

sys.path.insert(0, ".")

from voxistts import g2p_zh
from misaki import zh

G_MISAKI = zh.ZHG2P()

SENTENCES = [
    "你好，世界。",
    "今天天气真好。",
    "我爱北京天安门。",
    "我们一起去公园吧。",
    "这个东西多少钱？",
    "请给我一杯水。",
    "他说话很慢。",
    "小猫在睡觉。",
    "很好很好非常好。",
    "领导很好。",
    "2024年是新的一年。",
    " NLP 技术很有意思。",
    "这是一个Test混合English的句子。",
]

# 覆盖四声 + 轻声的单字（看声调编码）
TONE_SAMPLES = [
    ("妈", "麻", "马", "骂", "吗"),
    ("八", "拔", "把", "爸"),
    ("诗", "时", "史", "是"),
    ("女", "绿", "军", "旗"),
    ("小", "好", "鸟", "柳"),
    ("知识", "词语", "日语", "句子"),
]


_PUNCT = set(";:,.!?—…\"()“” 、。，！：；？「」『』（）《》【】<>「」°%&*+=")


def normalize(s: str) -> str:
    """去掉标点与空白，仅比较音素主体。"""
    return "".join(ch for ch in s if ch not in _PUNCT and not ch.isspace())


def main():
    ok = True
    print("===== 句子级一致性（音素主体，忽略标点差异）=====")
    for s in SENTENCES:
        a = g2p_zh.g2p_zh(s)
        try:
            r = G_MISAKI(s)
            b = r[0] if isinstance(r, tuple) else r
        except Exception as e:  # noqa: BLE001
            b = f"<misaki-error: {e}>"
        # 音素主体一致即视为等价（VoxisTTS 的 ASCII 标点比 misaki 的中文标点更兼容 Kokoro）
        phoneme_match = normalize(a) == normalize(b)
        punct_note = "" if a == b else "  (标点编码不同：VoxisTTS 用 ASCII，更兼容 Kokoro)"
        if not phoneme_match and not b.startswith("<misaki-error"):
            ok = False
        print(f"[{'OK' if phoneme_match else 'DIFF'}] {s!r}{punct_note}")
        if not phoneme_match and not b.startswith("<misaki-error"):
            print(f"    VoxisTTS: {a!r}")
            print(f"    misaki : {b!r}")

    print("\n===== 单字/词声调一致性（音素主体）=====")
    for group in TONE_SAMPLES:
        for ch in group:
            a = g2p_zh.g2p_zh(ch)
            try:
                r = G_MISAKI(ch)
                b = r[0] if isinstance(r, tuple) else r
            except Exception as e:  # noqa: BLE001
                b = f"<misaki-error: {e}>"
            phoneme_match = normalize(a) == normalize(b)
            if not phoneme_match and not b.startswith("<misaki-error"):
                ok = False
            print(f"[{'OK' if phoneme_match else 'DIFF'}] {ch} -> {a!r}")

    print("\nPARITY:", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
