#!/usr/bin/env python3
"""Measure the fix candidate: convert a public tokenizer with the original converter and with
fix_candidate_lib, encode the probes of probes.py + the 223-character sweep + the specials with
SentencePiece, and count mismatches against the HF tokenizer. No bundle, no LiteRT-LM."""
import os, sys, json, logging
os.environ.setdefault("HF_HUB_DISABLE_XET", "1"); logging.basicConfig(level=logging.ERROR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get("LITERT_TORCH_SRC", os.path.expanduser("~/code/litert-torch")), "litert_torch", "generative", "tools"))
import sentencepiece as spm, transformers
from sentencepiece import sentencepiece_model_pb2 as spb
from transformers import AutoTokenizer
import tokenizer_to_sentencepiece_lib as orig
import fix_candidate_lib as fixed
from probes import PROBES, LATIN_SWEEP
TYPE = spb.ModelProto.SentencePiece.Type
out = {}
for model in sys.argv[1:] or ["HuggingFaceTB/SmolLM3-3B", "Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "mistralai/Ministral-3-3B-Reasoning-2512"]:
    tok = AutoTokenizer.from_pretrained(model); tok.vocab_file = None
    hf = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    specials = [t.content for t in tok.added_tokens_decoder.values()]
    rec = {}
    for name, lib in (("original", orig), ("fixed", fixed)):
        proto = lib.convert(tok); mp = spb.ModelProto(); mp.ParseFromString(proto)
        sp = spm.SentencePieceProcessor()
        try:
            sp.LoadFromSerializedProto(proto); load = "ok"
        except RuntimeError as e:  # sentencepiece >= 0.2.1 rejects a NUL piece
            load = f"load error: {str(e)[:60]}"
            for p in mp.pieces:
                if "\x00" in p.piece: p.piece = "<NUL>"
            sp.LoadFromSerializedProto(mp.SerializeToString())
        enc = lambda s: list(sp.Encode(s))
        unk = [(i, p.piece) for i, p in enumerate(mp.pieces) if p.type == TYPE.UNKNOWN]
        unk_ids = [i for i, _ in unk]
        r = {"load": load, "pieces": len(mp.pieces), "unk": unk,
             "byte_pieces": sum(1 for p in mp.pieces if p.type == TYPE.BYTE), "byte_fallback": mp.trainer_spec.byte_fallback,
             "probes_identical": {k: enc(v) == hf(v) for k, v in PROBES.items()},
             "sweep_mismatch": sum(1 for c in LATIN_SWEEP if enc(c) != hf(c)),
             "sweep_to_unk": sum(1 for c in LATIN_SWEEP if enc(c) == unk_ids),
             "specials_split": [s for s in specials if len(hf(s)) == 1 and enc(s) != hf(s)],
             "emoji": {"hf": hf(PROBES["emoji"]), "sp": enc(PROBES["emoji"])},
             "samples": {s: {"hf": hf(s), "sp": enc(s)} for s in ("<|im_end|>", "<|endoftext|>", "é", "·", "Łódź", "😀", "2025")}}
        rec[name] = r
        print(f"### {model} [{name}] load={load} pieces={r['pieces']} unk={unk} byte_pieces={r['byte_pieces']} byte_fallback={r['byte_fallback']}")
        print(f"    probes identical {sum(r['probes_identical'].values())}/{len(PROBES)} {r['probes_identical']}")
        print(f"    sweep mismatch {r['sweep_mismatch']}/{len(LATIN_SWEEP)}  sweep->UNK {r['sweep_to_unk']}  specials split {r['specials_split'][:6]}{'...' if len(r['specials_split']) > 6 else ''} ({len(r['specials_split'])})")
        for s, v in r["samples"].items(): print(f"      {s!r:16} hf={v['hf']} sp={v['sp']} {'OK' if v['hf'] == v['sp'] else 'MISMATCH'}")
    out[model] = rec
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fix_candidate_results.json"), "w"), ensure_ascii=False, indent=1)
