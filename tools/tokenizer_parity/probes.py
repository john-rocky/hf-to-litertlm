"""Shared probe strings for the tokenizer parity check.

Every side of the audit (upstream HF tokenizer, offline tokenization of the bundle's own
tokenizer section, and the real engine) encodes exactly these strings, so the ids can be
compared position by position.
"""
Q = "What is the capital of France?"          # the default user turn

# Fixed probes. Keys are stable identifiers used in every JSON.
PROBES = {
    # a Latin-1 probe string (accents, symbols, a fraction, Greek)
    "latin1_s24": "café · naïve — 20°C × 3 ½ ñ ü Ω",
    # digit grouping / pre-tokenizer regex (Llama-3 groups 1-3 digits, Qwen splits every digit)
    "digits": "2025 20250830 12345 3.14159 1,000,000 v2.1.0",
    # plain English with contractions and punctuation (pre-tokenizer regex)
    "text_en": "The quick brown fox jumps over the lazy dog. It's 5 o'clock; isn't it? Don't worry—we'll be there by 10:30am.",
    # words whose letters live in the byte-token collision ranges (Latin-1 / Latin Ext-A)
    "text_eu": "Zürich São Paulo Kraków İstanbul Łódź Đà Nẵng Ærø Þórshöfn Ångström Çağlar",
    # scripts outside Latin
    "text_world": "Θεσσαλονίκη Москва 東京 القاهرة 서울 Ελλάδα Київ",
    # emoji / symbols (multi-byte sequences that are usually byte-fallback in BPE)
    "emoji": "hello 😀🎉 ok",
    # code
    "code": "def f(x):\n    return x**2 + 1  # squared\nprint(f(3))",
}

# Every standalone character U+00A1..U+017F (Latin-1 Supplement + Latin Extended-A).
# The GPT-2 byte<->unicode table spells the 256 single-byte tokens with exactly these code
# points (plus ASCII), which is where the collision lives.
LATIN_SWEEP = [chr(c) for c in range(0xA1, 0x180)]

# Special-token strings worth probing on every bundle regardless of vocabulary; the
# per-model added-token inventory (from the upstream tokenizer) is added on top.
GENERIC_SPECIALS = [
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>",
    "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "<|begin_of_text|>",
    "<s>", "</s>", "[INST]", "[/INST]", "<|end|>", "<|user|>", "<|assistant|>", "<|system|>",
    "<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>",
    "<|start_of_role|>", "<|end_of_role|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
    "<image>", "<img>", "</img>", "<IMG_CONTEXT>", "<tool_call>", "</tool_call>",
]
