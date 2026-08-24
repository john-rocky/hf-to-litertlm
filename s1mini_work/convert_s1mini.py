#!/usr/bin/env python3
"""Convert superwhisper/s1-mini (Qwen3-0.6B ASR-transcript normalizer) to .litertlm.

S1-mini is a single-task model with a strict input contract (model card):
an exact system prompt, a `[Styling: ..] [Structure: ..] [Context: ..]` control
line, and Qwen3's `enable_thinking=False` render (the empty `<think>` scaffold
in the generation prompt). A LiteRT-LM runtime never passes `enable_thinking`,
and the vendor template defaults to thinking ON — embedding it verbatim
produces the render the card warns gives "no usable output". So this script
installs `s1mini_bundle.jinja` into the model dir BEFORE export: it bakes the
required system prompt (incoming system messages are dropped — the card
requires this exact wording anyway) and hardcodes the non-thinking scaffold in
the generation prompt and in assistant history renders.

It then strips the exporter's punctuation-prefixed composite token_str stop
entries (".<|im_end|>\n" etc. — a SentencePiece-merge workaround that
misfires on BPE models: the runtime strips the whole matched stop, silently
eating the reply's final "." / "?" / "!"), leaving the two real stop ids.

int8 (dynamic_wi8_afp32) ONLY: the converted bundle reproduces HF fp32 greedy
byte-for-byte (10/10 task cases, CPU and GPU — gate_normalize.py). int4-b32
was measured and rejected: it drops commas and discourse words, and decodes
SLOWER than int8 at this size; punctuation fidelity is this model's job.

Env: litert-torch==0.9.3, litert-lm-builder>=0.15, transformers>=4.51, torch.

    python s1mini_work/convert_s1mini.py [out_dir]
"""
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ID = "superwhisper/s1-mini"
REVISION = "ae016a2cc6e42d0298932503b0874f048a947f40"  # main, 2026-08-25; no v1 tag on the Hub yet
STOP_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|> (generation_config eos list)


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/s1-mini-int8")
    src = HERE / "src_models" / "s1-mini"

    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=str(src))

    # Install the bundle template before export so the stock exporter embeds it.
    tpl = (HERE / "s1mini_bundle.jinja").read_text()
    (src / "chat_template.jinja").write_text(tpl)
    tc = json.loads((src / "tokenizer_config.json").read_text())
    tc["chat_template"] = tpl
    (src / "tokenizer_config.json").write_text(
        json.dumps(tc, indent=2, ensure_ascii=False))

    from litert_torch.generative.export_hf.export import export
    export(
        model=str(src),
        output_dir=str(out_dir),
        cache_length=4096,
        # Full ladder: the card recommends inputs up to ~1000 tokens, so the
        # 1024 signature earns its place. At 0.6B the signature RAM cost is small.
        prefill_lengths=[1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
    )
    bundle = out_dir / "model.litertlm"

    # Drop composite token_str stops; keep/ensure the two real stop ids.
    from litert_lm_builder import pack_litertlm_file, unpack_litertlm_file
    with tempfile.TemporaryDirectory(dir=bundle.parent) as td:
        unpack_litertlm_file(str(bundle), td)
        pb = Path(td) / "LlmMetadataProto.pbtext"
        text = re.sub(r'stop_tokens \{\n  token_str: "(?:[^"\\]|\\.)*"\n\}\n',
                      "", pb.read_text())
        for tid in STOP_IDS:
            if f"ids: {tid}" not in text:
                text += "stop_tokens {\n  token_ids {\n    ids: %d\n  }\n}\n" % tid
        pb.write_text(text)
        fixed = bundle.parent / (bundle.name + ".fix")
        pack_litertlm_file(str(Path(td) / "model.toml"), str(fixed))
        fixed.replace(bundle)

    print(f"DONE {bundle} ({bundle.stat().st_size:,} B)")
    print("Gate it: python s1mini_work/gate_normalize.py", bundle)


if __name__ == "__main__":
    main()
