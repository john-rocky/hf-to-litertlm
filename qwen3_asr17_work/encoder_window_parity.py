#!/usr/bin/env python3
"""Encoder window check (copied from confucius4_r2t2_work/encoder_window_parity.py, model dir = the Hub snapshot): the
export-side audio encoder (litert-torch Qwen3AsrEncoder, the module that becomes the bundle's
audio_encoder_hw 'encode' signature) against the transformers 5.14.1 audio tower + projector, fp32, eager torch, same
input_features. Two export-side variants: litert-torch 0918 as shipped (13-token chunks) and encoder_window.py
(104-token block-diagonal windows). Compared rows = the audio rows of the export output ([3 : 3 + N], between the baked
prefix and postfix embeddings) vs HF get_audio_features(...).pooler_output ([N, 2048]).

Inputs:
  crop5_en / crop5_ja   two 5 s round-1 crops (79,999 samples -> mel [1, 128, 500], 65 tokens, one HF window)
  full30                30 s of speech (FLEURS en_us_1670 29.3 s + the start of en_us_1671, cut at 480,000 samples)
                        -> mel [1, 128, 3000], 390 tokens, HF windows 104 + 104 + 104 + 78
  pad30_<clip>          (product-path record, not a window-fix criterion) a 11.7 s clip zero-padded to 480,000
                        samples as the engine pads its last window; export side sees 390 tokens, HF side is the
                        unpadded clip; only the clip's own rows are compared. Shows what the padding in the last
                        window costs the valid rows.
Output: out/encoder_window_parity.json

  ~/venvs/ltmain0918/bin/python encoder_window_parity.py
"""
import gc
import json
import os
import sys
import time
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

FIXW = os.path.join(common.FIX, "fleurs")


def inputs(proc):
    def feats(wav):
        x = proc(text=common.LITERT_TORCH_PROMPT, audio=wav, return_tensors="pt")
        return x["input_features"].float(), x["input_features_mask"]
    out = {}
    for name in ("en_us_1660__crop5s", "ja_jp_1680__crop5s"):
        wav = common.read_wav(os.path.join(common.OUT, "crops79999", name + ".wav"))
        out["crop5_" + name[:2]] = (name, wav)
    a = common.read_wav(os.path.join(FIXW, "en_us", "1670.wav"))
    b = common.read_wav(os.path.join(FIXW, "en_us", "1671.wav"))
    out["full30"] = ("en_us_1670 + en_us_1671 cut at 480000", np.concatenate([a, b])[:480000])
    res = {}
    for key, (desc, wav) in out.items():
        f, m = feats(wav)
        res[key] = {"desc": desc, "samples": int(len(wav)), "hf_feats": f, "hf_mask": m, "exp_feats": f}
    clip = common.read_wav(os.path.join(FIXW, "cmn_hans_cn", "1683.wav"))
    f_hf, m_hf = feats(clip)
    padded = np.concatenate([clip, np.zeros(480000 - len(clip), np.float32)])
    f_pad, _ = feats(padded)
    res["pad30_cmn_hans_cn_1683"] = {"desc": "cmn_hans_cn_1683 (11.7 s) zero-padded to 480000 samples on the export "
                                     "side; HF side unpadded", "samples": int(len(clip)), "hf_feats": f_hf,
                                     "hf_mask": m_hf, "exp_feats": f_pad}
    return res


def compare(a, b):
    a = a.double()
    b = b.double()
    d = (a - b).abs()
    cos_rows = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    cos_all = torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=-1)[0]
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "ref_max_abs": float(b.abs().max()),
            "cos_flat": float(cos_all), "cos_row_min": float(cos_rows.min()), "rows": int(a.shape[0])}


def main():
    torch.set_num_threads(8)
    import transformers
    proc = transformers.AutoProcessor.from_pretrained(common.MODEL_DIR)
    data = inputs(proc)
    # 1) transformers 5.14.1 reference (pristine model, before litert-torch patches anything globally)
    t0 = time.time()
    model = transformers.Qwen3ASRForConditionalGeneration.from_pretrained(common.MODEL_DIR, dtype=torch.float32).eval()
    enc_cfg = model.model.audio_tower.config
    with torch.no_grad():
        for key, d in data.items():
            out = model.model.get_audio_features(input_features=d["hf_feats"], input_features_mask=d["hf_mask"])
            d["hf_rows"] = out.pooler_output.reshape(-1, out.pooler_output.shape[-1]).float()
            print(key, "hf rows", tuple(d["hf_rows"].shape), flush=True)
    hf_load = time.time() - t0
    del model
    gc.collect()
    # 2) export-side encoder module exactly as export_lib builds it (Qwen3Asr(..., override_transformers=True))
    from litert_torch.generative.export_hf.model_ext.qwen3 import qwen3_asr
    import encoder_window
    asr = qwen3_asr.Qwen3Asr(common.MODEL_DIR, override_transformers=True)
    enc = asr.get_encoder().eval()
    attn_cls = transformers.models.qwen3_asr.modeling_qwen3_asr.Qwen3ASRAudioAttention
    attn = [m for m in asr._model.modules() if isinstance(m, attn_cls)]
    variants = {
        "litert_torch_0918_chunk13": qwen3_asr._audio_attention_forward,
        "encoder_window_blockdiag104": encoder_window._audio_attention_forward,
    }
    res = {"stack": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "litert_torch": os.path.dirname(qwen3_asr.__file__)},
           "n_window": enc_cfg.n_window, "n_window_infer": enc_cfg.n_window_infer,
           "window_tokens": encoder_window.window_tokens(enc_cfg), "mask_value": encoder_window.MASK_VALUE,
           "compare": "export rows [3:3+N] of Qwen3AsrEncoder(feats) vs HF get_audio_features(...).pooler_output, fp32",
           "pass_rule": "5 s: encoder_window cos_flat >= 0.9999 (max_abs recorded); stop if not",
           "inputs": {}}
    with torch.no_grad():
        for key, d in data.items():
            n_hf = d["hf_rows"].shape[0]
            row = {"desc": d["desc"], "samples": d["samples"], "mel_frames_hf": int(d["hf_feats"].shape[-1]),
                   "mel_frames_export": int(d["exp_feats"].shape[-1]), "hf_rows": n_hf}
            for vname, fn in variants.items():
                for m in attn:
                    m.forward = types.MethodType(fn, m)
                out = enc(d["exp_feats"])[0][0]
                n_exp = out.shape[0] - 8
                row[vname] = {"export_audio_rows": int(n_exp), **compare(out[3:3 + n_hf], d["hf_rows"])}
                if key.startswith("pad30"):
                    row[vname]["windows_export"] = encoder_window.windows(n_exp, res["window_tokens"])
                print(key, vname, json.dumps(row[vname]), flush=True)
            if not key.startswith("pad30"):
                row["windows_hf"] = encoder_window.windows(n_hf, res["window_tokens"])
            res["inputs"][key] = row
    c5 = [res["inputs"][k]["encoder_window_blockdiag104"]["cos_flat"] for k in res["inputs"] if k.startswith("crop5")]
    res["pass_5s"] = all(c >= 0.9999 for c in c5)
    res["hf_load_and_ref_seconds"] = round(hf_load, 1)
    with open(os.path.join(common.OUT, "encoder_window_parity.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("pass_5s", res["pass_5s"], flush=True)


if __name__ == "__main__":
    main()
