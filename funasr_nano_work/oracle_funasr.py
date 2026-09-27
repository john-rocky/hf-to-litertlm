"""funasr oracle for Fun-ASR-Nano-2512: fp32 CPU, dither 0, greedy, itn true, language None, no hotwords.

Run with ~/venvs/funasr_oracle/bin/python (funasr 1.4.16). The official repo is read from the local
dir out/hf_official (model.pt sha256-verified by dl_weights.sh), so AutoModel does not download.

Outputs
  oracle_transcripts.json            25 clips, itn=True (+ run facts, determinism, WER)
  oracle_transcripts_en_noitn.json   20 en clips, itn=False
  out/oracle_dumps/<id>_{lfr,enc,adp,embeds}.npy   funasr-internal tensors (itn=True run)
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "hf"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OFFICIAL = os.path.join(C.OUT, "hf_official")
DUMPS = os.path.join(C.OUT, "oracle_dumps")
LLM_KWARGS = {"do_sample": False, "num_beams": 1}
DET_IDS = ["example_zh", "en_clip04", "en_clip14"]  # run twice (determinism + dither evidence)


def build():
    import funasr
    import funasr.utils.fbank as ffb
    from funasr import AutoModel
    from funasr.models.fun_asr_nano.model import FunASRNano  # noqa: F401  (stop condition: must exist)

    torch.set_num_threads(8)
    m = AutoModel(model=OFFICIAL, hub="hf", device="cpu", disable_update=True, ncpu=8,
                  disable_pbar=True, frontend_conf={"dither": 0.0})
    fe = m.kwargs["frontend"]
    llm = m.model.llm
    facts = {
        "funasr_version": funasr.__version__,
        "torch_version": torch.__version__,
        "model_dir": OFFICIAL,
        "init_param": m.kwargs.get("init_param"),
        "frontend_class": type(fe).__name__,
        "frontend_dither": fe.dither,
        "frontend_conf": {k: getattr(fe, k) for k in ["fs", "window", "n_mels", "frame_length", "frame_shift",
                                                      "lfr_m", "lfr_n", "dither", "snip_edges", "upsacle_samples"]},
        "frontend_cmvn": None if fe.cmvn is None else "present",
        "fbank_backend": "torchaudio.compliance.kaldi.fbank" if ffb._HAS_TORCHAUDIO else "kaldi_native_fbank",
        "torchaudio_version": __import__("torchaudio").__version__ if ffb._HAS_TORCHAUDIO else None,
        "ctc_decoder_is_none": m.model.ctc_decoder is None,
        "llm_class": type(llm).__name__,
        "llm_param_dtype_after_build": str(next(llm.parameters()).dtype),
        "llm_lm_head_tied": llm.lm_head.weight.data_ptr() == llm.model.embed_tokens.weight.data_ptr(),
        "llm_generation_config": llm.generation_config.to_dict(),
        "ncpu": m.kwargs.get("ncpu"),
        "torch_threads": torch.get_num_threads(),
    }
    assert fe.dither == 0.0, fe.dither
    return m, facts


def instrument(m):
    cap = {}
    enc, adp, llm = m.model.audio_encoder, m.model.audio_adaptor, m.model.llm

    def enc_pre(mod, args):
        # SenseVoiceEncoderSmall.forward scales xs_pad IN PLACE (xs_pad *= sqrt(512)) -> clone first
        cap["lfr"] = args[0].detach().clone()
        cap["ilens"] = args[1].detach().clone()

    def enc_post(mod, args, out):
        cap["enc"] = out[0].detach().clone()
        cap["enc_lens"] = out[1].detach().clone()

    def adp_post(mod, args, out):
        cap["adp"] = out[0].detach().clone()
        cap["adp_lens"] = out[1].detach().clone()

    enc.register_forward_pre_hook(enc_pre)
    enc.register_forward_hook(enc_post)
    adp.register_forward_hook(adp_post)

    orig_prep = m.model.inference_prepare

    def prep_wrap(*a, **k):
        r = orig_prep(*a, **k)
        inputs_embeds, _contents, batch, _source_ids, _meta = r
        cap["fake_token_len"] = batch["fake_token_len"].detach().clone()
        cap["speech_lengths"] = batch["speech_lengths"].detach().clone()
        cap["source_ids"] = batch["source_ids"].detach().clone()
        cap["fbank_beg"] = batch["fbank_beg"].detach().clone()
        cap["embeds_dtype_at_prepare"] = str(inputs_embeds.dtype)
        cap["embeds"] = inputs_embeds.detach().float().clone()
        return r

    m.model.inference_prepare = prep_wrap
    orig_gen = llm.generate

    def gen_wrap(*a, **k):
        cap["llm_param_dtype_at_generate"] = str(next(llm.parameters()).dtype)
        cap["inputs_embeds_dtype"] = str(k["inputs_embeds"].dtype)
        cap["gen_call_kwargs"] = {kk: vv for kk, vv in k.items() if isinstance(vv, (int, float, str, bool, type(None)))}
        out = orig_gen(*a, **k)
        cap["gen_ids"] = out[0].tolist()
        return out

    llm.generate = gen_wrap
    return cap


def run_one(m, cap, row, itn):
    cap.clear()
    path = os.path.join(C.WORK, row["file"])
    t0 = time.perf_counter()
    res = m.generate(input=path, llm_kwargs=dict(LLM_KWARGS), itn=itn, hotwords=[], language=None,
                     max_length=512)
    wall = time.perf_counter() - t0
    r = res[0]
    L = int(cap["speech_lengths"].reshape(-1)[0])
    ftl = int(cap["fake_token_len"].reshape(-1)[0])
    src = cap["source_ids"][0].tolist()
    beg = int(cap["fbank_beg"].reshape(-1)[0])
    prefix, suffix = src[:beg], src[beg + ftl:]
    out = {
        "id": row["id"], "file": row["file"], "duration_s": row["duration_s"], "n_samples": row["n_samples"],
        "L": L, "fake_token_len": ftl, "ceil_L_over_8": math.ceil(L / 8),
        "L_expected_from_n": C.lfr_len(row["n_samples"]),
        "enc_rows": int(cap["enc"].shape[1]), "enc_lens": int(cap["enc_lens"].reshape(-1)[0]),
        "adp_rows": int(cap["adp"].shape[1]), "adp_lens": int(cap["adp_lens"].reshape(-1)[0]),
        "prompt_len": len(src), "prefix_len": len(prefix), "prefix_ids": prefix, "suffix_ids": suffix,
        "text": r["text"], "text_tn": r["text_tn"], "gen_ids": cap["gen_ids"], "n_gen": len(cap["gen_ids"]),
        "wall_s": round(wall, 3), "llm_kwargs": LLM_KWARGS, "dither": m.kwargs["frontend"].dither, "itn": itn,
        "llm_param_dtype_at_generate": cap["llm_param_dtype_at_generate"],
        "inputs_embeds_dtype": cap["inputs_embeds_dtype"], "gen_call_kwargs": cap["gen_call_kwargs"],
        "embeds_dtype_at_prepare": cap["embeds_dtype_at_prepare"],
    }
    return out, {k: cap[k] for k in ["lfr", "enc", "adp", "embeds"]}


def check_row(o):
    assert o["fake_token_len"] == o["ceil_L_over_8"], ("STOP: fake_token_len != ceil(L/8)", o["id"], o["L"], o["fake_token_len"])
    assert o["fake_token_len"] == C.fake_token_len(o["L"]), o
    assert o["L"] == o["L_expected_from_n"], ("L != ceil(T_fbank/6)", o)
    assert o["suffix_ids"] == C.SUFFIX_IDS, o["suffix_ids"]
    if o["itn"]:
        assert o["prefix_ids"] == C.PREFIX_IDS, o["prefix_ids"]
    assert o["enc_rows"] == o["L"] and o["enc_lens"] == o["L"] and o["adp_rows"] == o["L"] and o["adp_lens"] == o["L"], o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    meta = C.load_meta()
    if args.limit:
        meta = meta[:args.limit]
    by_id = {r["id"]: r for r in meta}
    t_all = time.time()
    m, facts = build()
    cap = instrument(m)
    print(json.dumps(facts, ensure_ascii=False, indent=1), flush=True)
    os.makedirs(DUMPS, exist_ok=True)

    # Warm-up: FunASRNano builds the LLM in bf16 (llm_conf.llm_dtype) and casts it to fp32 only
    # inside inference_llm, AFTER inference_prepare has built inputs_embeds with the bf16 embedding
    # table. So the first generate() of a process writes the fp32 adaptor rows into a bf16 tensor.
    # One discarded call makes every measured call fp32 end to end; the quirk is recorded here.
    warm_row = by_id.get("example_zh", meta[0])
    ow, tw = run_one(m, cap, warm_row, itn=True)
    warm = {"id": ow["id"], "text": ow["text"], "gen_ids": ow["gen_ids"],
            "embeds_dtype_at_prepare": ow["embeds_dtype_at_prepare"],
            "llm_param_dtype_at_generate": ow["llm_param_dtype_at_generate"]}
    print("warmup", warm, flush=True)

    rows, first_tensors = [], {}
    for row in meta:
        o, ten = run_one(m, cap, row, itn=True)
        check_row(o)
        rows.append(o)
        first_tensors[row["id"]] = ten
        for k, v in ten.items():
            np.save(os.path.join(DUMPS, f"{row['id']}_{k}.npy"), v.numpy())
        print(f"{o['id']:12s} L={o['L']:3d} ftl={o['fake_token_len']:2d} n_gen={o['n_gen']:3d} {o['wall_s']:6.2f}s | {o['text']}", flush=True)

    # determinism + dither evidence: same clip twice -> LFR / enc / adp max|diff| and text / ids equal
    det = []
    for fid in [f for f in DET_IDS if f in by_id]:
        o2, ten2 = run_one(m, cap, by_id[fid], itn=True)
        o1 = next(r for r in rows if r["id"] == fid)
        d = {k: float((ten2[k] - first_tensors[fid][k]).abs().max()) for k in ["lfr", "enc", "adp", "embeds"]}
        det.append({"id": fid, "text_equal": o1["text"] == o2["text"], "gen_ids_equal": o1["gen_ids"] == o2["gen_ids"],
                    "max_abs_diff": d})
        print("determinism", det[-1], flush=True)

    o_main = next((r for r in rows if r["id"] == warm_row["id"]), None)
    if o_main is not None:
        warm["vs_measured"] = {"text_equal": warm["text"] == o_main["text"], "gen_ids_equal": warm["gen_ids"] == o_main["gen_ids"],
                               "embeds_max_abs_diff": float((tw["embeds"] - first_tensors[warm_row["id"]]["embeds"]).abs().max()),
                               "adp_max_abs_diff": float((tw["adp"] - first_tensors[warm_row["id"]]["adp"]).abs().max()),
                               "adp_abs_max": float(first_tensors[warm_row["id"]]["adp"].abs().max())}
        print("warmup vs measured", warm["vs_measured"], flush=True)
    wer = C.corpus_wer(rows, by_id)
    zh = next((r for r in rows if r["id"] == "example_zh"), None)
    doc = {"facts": facts, "warmup_first_call": warm, "determinism": det, "wer_en_itn": {k: wer[k] for k in ["errors", "words", "wer"]},
           "wer_per_clip": wer["per_clip"],
           "example_zh": None if zh is None else {"text": zh["text"], "expected": by_id["example_zh"].get("expected_text"),
                                                  "equal": zh["text"] == by_id["example_zh"].get("expected_text")},
           "rows": rows}
    with open(os.path.join(C.WORK, "oracle_transcripts.json"), "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("WER itn=True", doc["wer_en_itn"], "zh", doc["example_zh"], flush=True)

    # en with itn=False (prompt "语音转写，不进行文本规整：")
    rows_n = []
    for row in [r for r in meta if r["id"].startswith("en_")]:
        o, _ = run_one(m, cap, row, itn=False)
        check_row(o)
        rows_n.append(o)
        print(f"noitn {o['id']:12s} prefix_len={o['prefix_len']} | {o['text']}", flush=True)
    wer_n = C.corpus_wer(rows_n, by_id)
    doc_n = {"facts": facts, "prompt": C.PROMPT_NOITN, "prefix_ids": rows_n[0]["prefix_ids"] if rows_n else None,
             "wer_en_noitn": {k: wer_n[k] for k in ["errors", "words", "wer"]}, "wer_per_clip": wer_n["per_clip"],
             "rows": rows_n}
    with open(os.path.join(C.WORK, "oracle_transcripts_en_noitn.json"), "w") as f:
        json.dump(doc_n, f, ensure_ascii=False, indent=1)
    print("WER itn=False", doc_n["wer_en_noitn"], f"total {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
