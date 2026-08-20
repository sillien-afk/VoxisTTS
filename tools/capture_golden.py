"""
tools/capture_golden.py
========================
一次性工具：用 原版 misaki.zh.ZHG2P（golden baseline） 跑一批中文样本，
打印其生成的 IPA 音素串，用于：
  1. 确认中文声调在 Kokoro vocab 里的编码方式
  2. 反推「拼音(声调数字) -> IPA」映射，回填 voxistts/transcription.py

用法（在装有 misaki 的环境）：
  python tools/capture_golden.py
"""

import sys

sys.path.insert(0, ".")

SENTENCES = [
    "你好，世界。",
    "今天天气真好。",
    "我爱北京天安门。",
    "我们一起去公园吧。",
    "这个东西多少钱？",
    "请给我一杯水。",
    "他说话很慢。",
    "小猫在睡觉。",
    "很好很好非常好。",   # 上声连续，看变调
    "领导很好。",         # 上声+上声 -> 阳平
]

# 覆盖全部声调的单字样本（拼音 -> 汉字，用于反推 pinyin->IPA）
TONE_SAMPLES = [
    ("mā", "妈"), ("má", "麻"), ("mǎ", "马"), ("mà", "骂"), ("ma", "吗"),
    ("bā", "八"), ("bá", "拔"), ("bǎ", "把"), ("bà", "爸"),
    ("yī", "一"), ("yí", "移"), ("yǐ", "椅"), ("yì", "义"),
    ("shī", "诗"), ("shí", "时"), ("shǐ", "史"), ("shì", "是"),
    ("nǚ", "女"), ("lǜ", "绿"), ("jūn", "军"), ("qí", "旗"),
    ("xiǎo", "小"), ("hǎo", "好"), ("niǎo", "鸟"), ("liǔ", "柳"),
]


def main():
    try:
        from misaki import zh
    except ImportError:
        print("ERROR: 需要 misaki（pip install 'misaki[zh]'）才能抓取基准")
        sys.exit(1)

    g2p = zh.ZHG2P()
    print("===== 句子级 IPA =====")
    for s in SENTENCES:
        ps = g2p(s)
        print(f"{s!r} -> {ps!r}")

    print("\n===== 单字声调 IPA =====")
    for py, ch in TONE_SAMPLES:
        ps = g2p(ch)
        print(f"{py}\t{ch}\t-> {ps!r}")


if __name__ == "__main__":
    main()
