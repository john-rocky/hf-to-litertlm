#!/usr/bin/env python3
"""gptq_quantize.py OUT_DIR CALIB.jsonl [--calib-only] - the int4 step of convert_g9v3_3b.sh: GPTQ-quantize ai9stars/G9v3-3B
with gptqmodel (bits 4, group 128, sym, desc_act False, damp 0.01, true_sequential) on Apple silicon (MPS), for
scripts/ingest_gptq_dequant.py (fp32clip) -> the blockwise-128 recovery export (recipes/gptqrec_int4_block128.json).

Needs a python 3.12 venv with `gptqmodel==7.3.5 datasets requests torchvision` and the MPS edits of patch_gptqmodel_mps.py.
The checkpoint is read from HF_HUB_CACHE (the pinned snapshot fetch_pinned.py wrote); the run is offline after the calibration
set exists.

Calibration = 256 samples in the bundle's chat form with thinking off (the checkpoint's chat template, enable_thinking=False:
user turn + assistant turn with the empty think block), tokenized with add_special_tokens=False (the template writes the one
<s>), each clipped to 2048 tokens, shorter than 64 tokens skipped, taken in order from three datasets at pinned revisions:
  128 HuggingFaceH4/ultrachat_200k train_sft (first user / assistant exchange)
   64 tatsu-lab/alpaca (instruction [+ input] / output)
   64 openai/gsm8k main TRAIN split (question / worked answer; a GSM8K evaluation uses the test split)
CALIB.jsonl is built once from those revisions (streaming) and must hash to CALIB_SHA256 (the set the published file was
calibrated on); a mismatch stops the run. The file is not distributed here: tatsu-lab/alpaca is CC BY-NC 4.0.
With the same set, the GPTQ step reproduced bit for bit on an M4 Max (two runs, the same model.safetensors sha256)."""
import hashlib
import json
import os
import sys

REPO = "ai9stars/G9v3-3B"
REVISION = "d9553445ff92dbce667381954c9699fbcbc924f9"
DATASETS = [("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a"),
            ("tatsu-lab/alpaca", "dce01c9b08f87459cf36a430d809084718273017"),
            ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866")]
CALIB_SHA256 = "26e9cc95b54543f97ee27033caf732d85cbd377bbacab7fcb27d01a54ab098df"
MAXLEN, MINLEN = 2048, 64


def snapshot():
    cache = os.environ.get("HF_HUB_CACHE")
    if not cache:
        raise SystemExit("set HF_HUB_CACHE to the cache fetch_pinned.py filled")
    return os.path.join(cache, "models--ai9stars--G9v3-3B", "snapshots", REVISION)


def render(tok, user, assistant):
    text = tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
                                   tokenize=False, add_generation_prompt=False, enable_thinking=False)
    ids = tok(text, add_special_tokens=False).input_ids[:MAXLEN]
    return text, ids


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build(tok, path):
    if not os.path.exists(path):
        from datasets import load_dataset
        rows = []

        def add(src, user, assistant):
            if not user or not assistant:
                return False
            text, ids = render(tok, user, assistant)
            if len(ids) < MINLEN:
                return False
            rows.append({"source": src, "text": text, "input_ids": ids})
            return True

        (uc, uc_rev), (al, al_rev), (gs, gs_rev) = DATASETS
        n = 0
        for ex in load_dataset(uc, split="train_sft", streaming=True, revision=uc_rev):
            msgs = ex.get("messages") or []
            if len(msgs) >= 2 and msgs[0].get("role") == "user" and msgs[1].get("role") == "assistant":
                n += add("ultrachat_200k", msgs[0]["content"], msgs[1]["content"])
            if n >= 128:
                break
        n = 0
        for ex in load_dataset(al, split="train", streaming=True, revision=al_rev):
            user = ex["instruction"] + (("\n\n" + ex["input"]) if ex.get("input") else "")
            n += add("alpaca", user, ex.get("output"))
            if n >= 64:
                break
        n = 0
        for ex in load_dataset(gs, "main", split="train", streaming=True, revision=gs_rev):
            n += add("gsm8k_train", ex["question"], ex["answer"])
            if n >= 64:
                break
        assert len(rows) == 256, len(rows)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"calibration: built {len(rows)} samples -> {path}", flush=True)
    got = sha256(path)
    if got != CALIB_SHA256:
        raise SystemExit(f"calibration set {path} has sha256 {got}, expected {CALIB_SHA256}: not the published set")
    rows = [json.loads(l) for l in open(path)]
    print(f"calibration: {len(rows)} samples from {path} (sha256 {got[:16]}... as expected)", flush=True)
    return rows


def main():
    out_dir, calib = sys.argv[1], sys.argv[2]
    from transformers import AutoTokenizer
    snap = snapshot()
    tok = AutoTokenizer.from_pretrained(snap)
    rows = build(tok, calib)
    lens = sorted(len(r["input_ids"]) for r in rows)
    print(f"calibration tokens: total {sum(lens)}, median {lens[len(lens) // 2]}, max {lens[-1]}; first ids "
          f"{rows[0]['input_ids'][:3]} (one <s> = id 0)", flush=True)
    if "--calib-only" in sys.argv:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    from gptqmodel import GPTQModel, QuantizeConfig
    qcfg = QuantizeConfig(bits=4, group_size=128, desc_act=False, sym=True, true_sequential=True, damp_percent=0.01)
    print("QuantizeConfig: bits=4 group_size=128 desc_act=False sym=True true_sequential=True damp_percent=0.01", flush=True)
    model = GPTQModel.load(snap, qcfg)
    calib_rows = [{"input_ids": r["input_ids"], "attention_mask": [1] * len(r["input_ids"])} for r in rows]
    model.quantize(calib_rows, batch_size=1)
    os.makedirs(out_dir, exist_ok=True)
    model.save(out_dir)
    print(f"GPTQ_DONE saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
