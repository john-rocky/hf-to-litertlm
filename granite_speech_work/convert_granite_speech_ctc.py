#!/usr/bin/env python3
"""Convert ibm-granite/granite-speech-5.0-470m-turboctc (CTC conformer ASR) to LiteRT.

    python3 convert_granite_speech_ctc.py

Encoder lane (same lane as convert_granite_embedding_r2.py): no KV cache, so
the eager model is traced directly with litert_torch multi-signature convert.
The model is loaded through load_model.load_eager (the Hub checkpoint carries
stock-5.16 tensor names; the shipped remote code is the older packaged module —
see load_model.py for the rename table and the strict-load proof).

Signatures (batch 1, fixed audio windows, host right-pads audio with zeros):
  transcribe_{5,10,30}s: input_features f32 [1, T', 320] -> (ids i32 [1, T''],
      logits f32 [1, T'', 16384]),  T' = 50*seconds, T'' = T'/4.
  input_features = the model's log-mel(+delta)+stack front-end (host side,
  processing_ctc_conformer.CtcConformerProcessor). ids = argmax over the CTC
  vocab; host does the ~10-line CTC greedy collapse (drop repeats, drop id 0)
  + tokenizer.json decode.

Known semantics of the fixed window (measured, eager_gate.py --pad-seconds 30):
attention is block-local (128 frames), so padding moves block boundaries;
19/20 LibriSpeech fixtures transcribe identically vs unpadded, 1 clip changed
one word. Pick the smallest window that fits the clip.

Export traps handled here:
  * sdpa_kernel([EFFICIENT_ATTENTION, MATH]) context manager inside the remote
    attention forward: torch.export on CPU has no efficient kernel; if the
    trace refuses the context manager we stub it to a no-op (MATH decomposes
    to matmul+softmax either way).
  * BatchNorm num_batches_tracked / attention_dists are int64 buffers
    (c25-audio-codec-seanet-pad class) — eval-mode BN never reads the
    counter and attention_dists is only sliced by static bounds; op_report
    prints any int64 tensor that still reaches the graph.

Quantization: wi8fc = dynamic-range int8 on FULLY_CONNECTED only (pointwise
convs here ARE Linears, so this reaches them; the 7-tap depthwise convs and
the tiny rel_pos_emb table stay float on purpose — small-audio-model caution,
paddleocr-vl-shipped). fp16 = float_casting on everything.
"""
import collections
import json
import os
import sys

WORK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "..", "scripts"))
import _stub  # noqa: F401  (macOS scipy/_propack guard, import FIRST)

import numpy as np
import torch
import torch.nn as nn

from eager_gate import load_wav
from load_model import load_eager

SIG_SECONDS = (5, 10, 30)
OUT_DIR = os.path.join(WORK, "out")


class Transcriber(nn.Module):
    """model.forward + in-graph argmax. Outputs (ids i32, logits f32)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features):
        logits = self.model(input_features)
        ids = logits.argmax(dim=-1).to(torch.int32)
        return ids, logits


def feats_for(processor, audio, seconds):
    n = int(seconds * 16000)
    assert len(audio) <= n, (len(audio), n)
    padded = np.zeros(n, dtype=np.float32)
    padded[: len(audio)] = audio
    return processor([padded])["input_features"]


def op_report(path, tag):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path)
    hist = collections.Counter(d["op_name"] for d in it._get_ops_details())
    print(f"{tag} ops:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))
    i64 = [d["name"] for d in it.get_tensor_details()
           if "int64" in str(d["dtype"]).lower()]
    print(f"{tag} int64 tensors: {len(i64)}", i64[:5])
    for hostile in ("GATHER_ND", "LOGICAL_AND"):
        if hostile in hist:
            print(f"  note: {hostile} x{hist[hostile]} present")


def run_sig(path, name, feats):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=os.cpu_count())
    r = it.get_signature_runner(name)(input_features=feats.numpy())
    ids = next(v for v in r.values() if v.dtype in (np.int32, np.int64))
    logits = next(v for v in r.values() if v.dtype == np.float32)
    return ids, logits


def smoke(path, variant, seconds, feats, ref_ids, ref_logits, results):
    got_ids, got_logits = run_sig(path, f"transcribe_{seconds}s", feats)
    assert np.isfinite(got_logits).all(), f"{variant} {seconds}s logits not finite"
    id_match = float((got_ids == ref_ids).mean())
    row = {"id_match": id_match,
           "logit_maxdiff": float(np.abs(got_logits - ref_logits).max())}
    print(f"SMOKE transcribe_{seconds}s {variant} vs eager: argmax-id match "
          f"{id_match:.4f}  logit max|diff| {row['logit_maxdiff']:.3e}")
    results["sigs"][f"transcribe_{seconds}s"][variant] = row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fp32 = os.path.join(OUT_DIR, "granite_speech_ctc_fp32.tflite")
    wi8 = os.path.join(OUT_DIR, "granite_speech_ctc_wi8fc.tflite")
    fp16 = os.path.join(OUT_DIR, "granite_speech_ctc_fp16.tflite")
    for stale in (wi8, fp16):  # quantizer refuses to overwrite; stale = shipped-by-accident
        if os.path.exists(stale):
            print(f"removing stale {os.path.basename(stale)}")
            os.remove(stale)

    torch.manual_seed(0)
    model, processor = load_eager(dtype=torch.float32)
    trans = Transcriber(model).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params/1e6:.1f}M")

    # Real audio for sample inputs and smokes: clip04 (29.4 s) exercises the
    # 30 s window nearly full; clip00 (5.86 s) fits 10 s; clip14 (2.25 s) fits 5 s.
    meta = json.load(open(os.path.join(WORK, "fixtures", "meta.json")))
    clip_for = {30: "fixtures/clip04.wav", 10: "fixtures/clip00.wav",
                5: "fixtures/clip14.wav"}
    sample_feats, refs = {}, {}
    for s in SIG_SECONDS:
        audio = load_wav(os.path.join(WORK, clip_for[s]))
        feats = feats_for(processor, audio, s)
        sample_feats[s] = feats
        with torch.inference_mode():
            ids, logits = trans(feats)
        refs[s] = (ids.numpy(), logits.numpy())
        print(f"sig transcribe_{s}s: input {tuple(feats.shape)} -> ids "
              f"{tuple(ids.shape)} logits {tuple(logits.shape)}")

    # Shaw-gather rewrite (see shaw_patch.py): kills the ~0.5 GB of per-layer
    # per-block-length folded rel-bias constants. Refs above were computed with
    # the UNPATCHED forward, so this A/B quantifies the re-association noise.
    import shaw_patch

    shaw_patch.apply(model)
    for s in SIG_SECONDS:
        with torch.inference_mode():
            p_ids, p_logits = trans(sample_feats[s])
        m = float((p_ids.numpy() == refs[s][0]).mean())
        md = float(np.abs(p_logits.numpy() - refs[s][1]).max())
        print(f"SHAW eager A/B transcribe_{s}s: id match {m:.4f} "
              f"logit max|diff| {md:.3e}")
        assert m == 1.0, f"Shaw rewrite changed argmax ids at {s}s"

    print("converting (litert_torch multi-signature) ...")
    import litert_torch

    conv = None
    for s in SIG_SECONDS:
        kw = {"input_features": sample_feats[s]}
        sig = (f"transcribe_{s}s", trans)
        conv = (litert_torch.signature(*sig, sample_kwargs=kw) if conv is None
                else conv.signature(*sig, sample_kwargs=kw))
    edge = conv.convert()
    edge.export(fp32)
    print(f"fp32: {os.path.getsize(fp32)/1e6:.1f} MB")
    op_report(fp32, "fp32")

    results = {"model": "ibm-granite/granite-speech-5.0-470m-turboctc",
               "params_m": n_params / 1e6,
               "sigs": {f"transcribe_{s}s": {} for s in SIG_SECONDS}}
    for s in SIG_SECONDS:
        smoke(fp32, "fp32", s, sample_feats[s], *refs[s], results=results)

    print("quantizing wi8fc ...")
    from ai_edge_quantizer import quantizer, recipe_manager, qtyping

    OP = qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=8)
    qt = quantizer.Quantizer(fp32, rm.get_quantization_recipe())
    assert not qt.need_calibration
    qt.quantize().export_model(wi8)
    print(f"wi8fc: {os.path.getsize(wi8)/1e6:.1f} MB")
    for s in SIG_SECONDS:
        smoke(wi8, "wi8fc", s, sample_feats[s], *refs[s], results=results)

    print("quantizing fp16 ...")
    rm16 = recipe_manager.RecipeManager()
    rm16.add_quantization_config(
        regex=".*", operation_name=OP.ALL_SUPPORTED,
        algorithm_key="float_casting",
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT))
    quantizer.Quantizer(fp32, rm16.get_quantization_recipe()) \
        .quantize().export_model(fp16)
    print(f"fp16: {os.path.getsize(fp16)/1e6:.1f} MB")
    for s in SIG_SECONDS:
        smoke(fp16, "fp16", s, sample_feats[s], *refs[s], results=results)

    with open(os.path.join(OUT_DIR, "convert_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE:", fp32, wi8, fp16)


if __name__ == "__main__":
    main()
