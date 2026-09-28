"""Shared constants and fixtures for the Audio8-TTS-Preview-0.6b -> LiteRT lane.

Model facts (config.json of Edge0/Audio8-TTS-Preview-0.6b @ f07040f3):
  slow AR 24L dim 896 / 14 heads / 2 kv heads / head_dim 64 / ffn 4864 / qkv bias
  fast AR  4L dim 896 / 14 heads / 2 kv heads / head_dim 64 / ffn 4864 / no bias
  vocab 155776 (tied), semantic ids 151678..155773 (4096), eos 151645, pad 151643
  10 codebooks x 4096, codec 44.1 kHz, 2048 samples / frame
"""
import os

REPO = os.path.expanduser("~/code/litertlm-convert")
WORK = os.path.join(REPO, "audio8_tts_work")
OUT = os.path.join(WORK, "out")
FIX = os.path.join(WORK, "fixtures")
SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Edge0--Audio8-TTS-Preview-0.6b/snapshots/"
    "f07040f3d151f1ba0253bfb92cb2f5dd38b44594")
HF_REPO = "Edge0/Audio8-TTS-Preview-0.6b"
HF_REV = "f07040f3d151f1ba0253bfb92cb2f5dd38b44594"

VOCAB = 155776
SEM_BEGIN, SEM_END = 151678, 155773
EOS, PAD = 151645, 151643
N_SEM = SEM_END - SEM_BEGIN + 1  # 4096
SLOW_LOGITS = N_SEM + 1          # semantic range then eos (same layout as the publisher's ONNX)
NUM_CB, CB_SIZE = 10, 4096
DIM, HEADS, KV_HEADS, HEAD_DIM, FFN = 896, 14, 2, 64, 4864
N_LAYER, N_FAST = 24, 4
ROPE_BASE, EPS = 1e6, 1e-6
MAX_SEQ = 2048
SR, FRAME = 44100, 2048

# generation defaults (generation_config.json + the publisher's ONNX runtime use the same)
TEMPERATURE, TOP_P, TOP_K = 0.7, 0.9, 50
RAS_TOP_P, RAS_TEMP, RAS_WINDOW = 0.9, 1.0, 10
MAX_NEW_TOKENS = 512

# ---------------- fixtures ----------------
# references: 16 kHz mono clips already in this repo (funasr_nano_work/out/fixtures), transcripts:
#   en_clip13 = LibriSpeech dev-clean 1272-128104-0013 (CC BY 4.0), human transcript, re-cased by hand
#   example_ja = FunAudioLLM/Fun-ASR-Nano-2512 example/ja.mp3 (Apache-2.0), transcript = Fun-ASR-Nano oracle output
REFS = {
    "en": {
        "wav16": os.path.join(REPO, "funasr_nano_work/out/fixtures/en_clip13.wav"),
        "text": "Mister Quilter has missed his chance, for he has failed even to make himself the Tupper of painting.",
    },
    "ja": {
        "wav16": os.path.join(REPO, "funasr_nano_work/out/fixtures/example_ja.wav"),
        "text": "うちの中学は弁当制で、持っていけない場合は、五十円の学校販売のパンを買う。",
    },
}

SENTENCES = {
    "en": [
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Please remember to water the plants before you leave for the weekend.",
        "Speech synthesis on a phone used to sound robotic, but not anymore.",
        "She opened the window and let the cool morning air fill the room.",
        "Our train departs early in the morning, so we should reach the station on time.",
        "Reading a good book by the fire is my favorite way to spend a rainy evening.",
    ],
    "ja": [
        "今日は天気が良いので、公園まで散歩に行きましょう。",
        "この製品は、スマートフォンの上で音声を合成することができます。",
        "明日の会議は午後から始まる予定です。",
        "駅前の新しいパン屋さんは、朝早くから行列ができています。",
        "彼女は静かな図書館で、一冊の本を読み終えました。",
        "雨が降る前に、洗濯物を取り込んでおいてください。",
    ],
}


def cases():
    """(case_id, lang, ref_key or None, text, seed)."""
    out = []
    for lang in ("en", "ja"):
        for i, text in enumerate(SENTENCES[lang]):
            out.append((f"{lang}_ref_{i}", lang, lang, text, 1000 + i))
        # generation without a reference voice (README "Generation without a reference")
        out.append((f"{lang}_noref_0", lang, None, SENTENCES[lang][0], 2000))
    return out
