#!/usr/bin/env python3
"""Convert the LiquidAI LFM2.5-Encoder-350M task finetunes to LiteRT.

    python convert_lfm25_encoder_finetunes.py router|linter|pii|gec

All four ride the same bidirectional LFM2 trunk as the shipped encoders; each
adds a task head (custom remote code):

  router  LFM2.5-Encoder-350M-Prompt-Router  Lfm2BidirForSequenceRouting
          sigs route_{128,512}: (ids i32 [1,S], mask i32 [1,S],
          text_pool f32 [1,1,S], category_pool f32 [1,R=8,S]) -> logits [1,8].
          Pool matrices are host-built from tokenizer offsets (model card's
          `route()`); rows for unused lanes stay all-zero -> their logit is
          exactly score_bias, host softmaxes only the real lanes.
  linter  LFM2.5-Encoder-350M-Policy-Linter  Lfm2BidirForRuleMatching
          sigs lint_{128,512}: (ids, mask, rule_pool f32 [1,R=8,S])
          -> scores [1,S,8] (sigmoid>0.5 = token flagged by rule).
  pii     LFM2.5-Encoder-350M-PII-Detector   Lfm2BidirP2ForTokenClassification
          sigs pii_{128,512}: (ids, mask) -> BIOES logits [1,S,161].
  gec     LFM2.5-Encoder-350M-Spellchecker   GecTaggerForGEC (tagger only; the
          repo's reranker + iterative decode stay host-side)
          sig gec_128: (ids, mask) -> label_logits [1,128,128802] +
          detect_logits [1,128,2]. tie_replace: label logits = concat(base[2],
          proj(h)@E^T + bias twice) with E = embed[:64400].

Shared traps (same as convert_lfm25_encoder.py): transformers'
apply_mask_to_padding_states is a NO-OP at batch==1 — every one of these repos
inlines its own copy of the bidirectional patches, so the rebind here goes
through `Lfm2ShortConv.slow_forward.__globals__` (the defining module's
namespace, whichever repo file that is). Gate = pad-content invariance
(bitwise 0). Per-token outputs are zeroed at padded positions.

Quant: wi8fc (FC int8 DRQ + embedding int8 cw, convs float). For gec the two
tied vocab matmuls must land as FULLY_CONNECTED to be quantized — the op
histogram printout is the check.

"""
import os
import sys

import types


class _D:
    def __getattr__(self, n):
        return lambda *a, **k: None

    def __call__(self, *a, **k):
        return None


_pp = types.ModuleType("scipy.sparse.linalg._propack")
_pp.__file__ = "<stub>"
_pp.__spec__ = None
_pp.__getattr__ = lambda n: _D()  # noqa: E731
sys.modules.setdefault("scipy.sparse.linalg._propack", _pp)

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rank4_repeat_kv  # rank-4 GQA repeat_kv respelling (GPU delegation)

KIND = sys.argv[1] if len(sys.argv) > 1 else "router"
R = 8  # static lane/rule slots for router/linter

SPEC = {
    "router": ("LiquidAI/LFM2.5-Encoder-350M-Prompt-Router", "out/lfm25_ft_router"),
    "linter": ("LiquidAI/LFM2.5-Encoder-350M-Policy-Linter", "out/lfm25_ft_linter"),
    "pii": ("LiquidAI/LFM2.5-Encoder-350M-PII-Detector", "out/lfm25_ft_pii"),
    "gec": ("LiquidAI/LFM2.5-Encoder-350M-Spellchecker", "out/lfm25_ft_gec"),
}
MODEL, OUT_DIR = SPEC[KIND]
os.makedirs(OUT_DIR, exist_ok=True)
FP32 = os.path.join(OUT_DIR, f"{KIND}_fp32.tflite")
WI8 = os.path.join(OUT_DIR, f"{KIND}_wi8fc.tflite")
SEQ_LENS = (128,) if KIND == "gec" else (512, 128)


def rebind_pad_mask():
    """Make apply_mask_to_padding_states unconditional in whichever remote-code
    module defined the installed ShortConv patch (batch-1 no-op fix)."""
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    g = Lfm2ShortConv.slow_forward.__globals__
    assert "apply_mask_to_padding_states" in g, "unexpected shortconv patch layout"

    def _always(hidden_states, attention_mask):
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None].to(
                hidden_states.dtype
            )
        return hidden_states

    g["apply_mask_to_padding_states"] = _always
    print(f"rebound apply_mask in {g.get('__name__', '?')}")


def load_model():
    from transformers import AutoModel, AutoModelForTokenClassification

    _rank4_repeat_kv.install()
    if KIND == "pii":
        m = AutoModelForTokenClassification.from_pretrained(
            MODEL, trust_remote_code=True, dtype=torch.float32)
    else:
        m = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32)
    m = m.eval()
    rebind_pad_mask()
    _rank4_repeat_kv.rebind_all()
    # tf5.x meta-load trap: rope inv_freq must be real
    trunk = m.encoder if KIND == "gec" else m.lfm2
    assert float(trunk.rotary_emb.inv_freq.min()) > 0, "inv_freq zeroed by meta-load"
    return m


class RouterWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, text_pool, category_pool):
        return self.m(input_ids=input_ids,
                      attention_mask=attention_mask.to(torch.float32),
                      text_pool=text_pool, category_pool=category_pool)["logits"]


class LinterWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, rule_pool):
        mf = attention_mask.to(torch.float32)
        s = self.m(input_ids=input_ids, attention_mask=mf,
                   rule_pool=rule_pool)["logits"]
        return s * mf[:, :, None]  # zero scores at padded positions


class PiiWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        mf = attention_mask.to(torch.float32)
        lg = self.m(input_ids=input_ids, attention_mask=mf).logits
        return lg * mf[:, :, None]


class GecWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        mf = attention_mask.to(torch.float32)
        out = self.m(input_ids=input_ids, attention_mask=mf)
        return (out["label_logits"] * mf[:, :, None],
                out["detect_logits"] * mf[:, :, None])


def sample(S, with_pad=True):
    ids = torch.randint(10, 60000, (1, S), dtype=torch.int32)
    mask = torch.ones(1, S, dtype=torch.int32)
    if with_pad:
        n_pad = S // 4
        ids[0, S - n_pad:] = 0
        mask[0, S - n_pad:] = 0
    return ids, mask


def pools(S, n_valid):
    """Plausible pool matrices over the valid range (values only matter for the
    trace; verification uses real offsets)."""
    text_pool = torch.zeros(1, 1, S)
    text_pool[0, 0, 1:n_valid] = 1.0 / (n_valid - 1)
    category_pool = torch.zeros(1, R, S)
    for r in range(R // 2):  # half the slots active, like real prompts
        a, b = 1 + 3 * r, 1 + 3 * r + 3
        category_pool[0, r, a:b] = 1.0 / 3
    return text_pool, category_pool


def build_inputs(S, with_pad=True):
    ids, mask = sample(S, with_pad)
    n_valid = int(mask.sum())
    if KIND == "router":
        tp, cp = pools(S, n_valid)
        return {"input_ids": ids, "attention_mask": mask,
                "text_pool": tp, "category_pool": cp}
    if KIND == "linter":
        _, cp = pools(S, n_valid)
        return {"input_ids": ids, "attention_mask": mask, "rule_pool": cp}
    return {"input_ids": ids, "attention_mask": mask}


def first_out(y):
    return y[0] if isinstance(y, tuple) else y


def main():
    torch.manual_seed(0)
    print(f"[{KIND}] loading {MODEL} ...")
    m = load_model()
    wrap = {"router": RouterWrap, "linter": LinterWrap,
            "pii": PiiWrap, "gec": GecWrap}[KIND](m).eval()
    print(f"[{KIND}] params {sum(p.numel() for p in m.parameters())/1e6:.1f}M")

    # pad-content invariance gate (valid positions must be bitwise unchanged
    # when pad-region ids change; for router the [1,R] logits must be identical)
    S = SEQ_LENS[-1]
    kw = build_inputs(S)
    n_valid = int(kw["attention_mask"].sum())
    kw2 = {k: v.clone() for k, v in kw.items()}
    kw2["input_ids"][0, n_valid:] = torch.randint(
        10, 60000, (S - n_valid,), dtype=torch.int32)
    with torch.no_grad():
        a = first_out(wrap(**kw)).numpy()
        b = first_out(wrap(**kw2)).numpy()
    if a.ndim == 3:
        a, b = a[:, :n_valid], b[:, :n_valid]
    leak = float(np.abs(a - b).max())
    print(f"[{KIND}] pad-content invariance: max|diff| {leak:.3e} (must be 0)")
    assert leak == 0.0

    print(f"[{KIND}] converting ...")
    import litert_torch

    conv = None
    for S in SEQ_LENS:
        kw = build_inputs(S)
        name = f"{KIND}_{S}" if KIND != "router" else f"route_{S}"
        if KIND == "linter":
            name = f"lint_{S}"
        conv = (litert_torch.signature(name, wrap, sample_kwargs=kw)
                if conv is None else conv.signature(name, wrap, sample_kwargs=kw))
    conv.convert().export(FP32)
    print(f"[{KIND}] fp32: {os.path.getsize(FP32)/1e6:.1f} MB")

    import collections
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=FP32)
    hist = collections.Counter(d["op_name"] for d in it._get_ops_details())
    print(f"[{KIND}] ops:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))
    assert "BROADCAST_TO" not in hist, (
        "rank-5 repeat_kv survived — the rank-4 patch missed the live call site")

    # smoke parity, primary signature
    S = SEQ_LENS[-1]
    kw = build_inputs(S)
    with torch.no_grad():
        ref = wrap(**kw)
    sig = f"route_{S}" if KIND == "router" else (
        f"lint_{S}" if KIND == "linter" else f"{KIND}_{S}")
    rn = Interpreter(model_path=FP32, num_threads=os.cpu_count()).get_signature_runner(sig)
    got = rn(**{k: v.numpy() for k, v in kw.items()})
    refs = ref if isinstance(ref, tuple) else (ref,)
    gots = [got[k] for k in sorted(got)]
    for i, (r0, g0) in enumerate(zip(refs, gots)):
        r0 = r0.numpy()
        print(f"[{KIND}] SMOKE out{i} fp32 vs torch: max|diff| "
              f"{np.abs(r0-g0).max():.3e} corr "
              f"{np.corrcoef(r0.ravel(), g0.ravel())[0,1]:.8f}")

    print(f"[{KIND}] quantizing wi8fc ...")
    from ai_edge_quantizer import quantizer, recipe_manager, qtyping

    rm = recipe_manager.RecipeManager()
    rm.add_dynamic_config(regex=".*",
                          operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
                          num_bits=8)
    rm.add_dynamic_config(regex=".*",
                          operation_name=qtyping.TFLOperationName.EMBEDDING_LOOKUP,
                          num_bits=8,
                          granularity=qtyping.QuantGranularity.CHANNELWISE)
    qt = quantizer.Quantizer(FP32, rm.get_quantization_recipe())
    assert not qt.need_calibration
    qt.quantize().export_model(WI8)
    print(f"[{KIND}] wi8fc: {os.path.getsize(WI8)/1e6:.1f} MB")

    rn8 = Interpreter(model_path=WI8, num_threads=os.cpu_count()).get_signature_runner(sig)
    got8 = rn8(**{k: v.numpy() for k, v in kw.items()})
    g8 = [got8[k] for k in sorted(got8)]
    for i, (r0, g0) in enumerate(zip(refs, g8)):
        r0 = r0.numpy()
        print(f"[{KIND}] SMOKE out{i} wi8fc vs torch: corr "
              f"{np.corrcoef(r0.ravel(), g0.ravel())[0,1]:.6f}")
    print(f"[{KIND}] DONE:", FP32, WI8)


if __name__ == "__main__":
    main()
