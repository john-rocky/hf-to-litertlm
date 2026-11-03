#!/usr/bin/env python3
"""A transformers ParakeetForTDT checkpoint -> one .tflite with the signatures LiteRT-LM's Parakeet TDT runtime runs.

    uv run python asr/export_parakeet_tdt.py <snapshot_dir> <work_dir> [--input-sec 5] [--stateful-after 4]
                                             [--recipe int8|fp32]

No NeMo: the graphs are plain PyTorch modules over the checkpoint's own transformers modules (encoder,
encoder_projector, the prediction network's embedding and LSTM weights, decoder_projector, the joint head). The
signatures and tensor shapes are the ones litert-torch's NeMo-based ParakeetTDT export writes and LiteRT-LM's
omni/speech2text/parakeet_tdt.cc + tdt_decoder.cc read (litert-community/parakeet-tdt-0.6b-v3's
parakeet_tdt_0.6b_v3_5s_i8_stateful.tflite, read back 2026-11-03):

  encode    args_0 [1, n_mels, F] f32 (the runtime's log-mel, mel-major)          -> output_0 [1, D, T] f32
  decode    args_0 [1, D, T] f32, args_1 [1, U] i32, args_2 / args_3 [L, 1, H] f32
            -> output_0 [1, T, U, V + 1 + K] f32 (log-softmax over the last axis), output_1 / output_2 [L, 1, H]
  decode_1  the same with U = 1 (the runtime switches to it once the U slots of `decode` are full)

F = round(input_sec * 100) mel frames (10 ms hop), T = the encoder's frames after 8x subsampling, D = 1024 (the
runtime reads T as numel / 1024), V + 1 = the vocabulary with the blank last, K = len(durations) (the runtime
reads the duration index as the frame step, so durations must be 0..K-1), L x H = the LSTM's layers x width.

The LSTM is written as explicit gates (torch's LSTMCell arithmetic, the token loop unrolled), as in
parakeet_redux_work/scripts/r2_graphs.py; the joint head runs on [1, T * U, H] (litert-torch PR #1426's
_FlattenedJointNet flattens the same way: "Avoids a TFLite MLIR BatchMatMul reshape fusion rank-mismatch failure
when `U == 1` in `decode_1`."; this export did not try the rank-4 form).

Writes <work_dir>/asr_model_fp32.tflite, <work_dir>/asr_model_<recipe>.tflite (recipe int8 = the stock exporter's
dynamic_wi8_afp32 via litert_torch's quantize_model), <work_dir>/speech2text_metadata.json (the fields PR #1426's
ParakeetTDT.get_speech2text_metadata writes for this input_sec / stateful_after) and <work_dir>/export.json
(shapes, the PyTorch checks, timings). Exit 0 = written and every check passed, 1 = a check failed, 2 = the
checkpoint is not one the runtime reads (export.json's "unsupported" says why; nothing converted), 3 = failed.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEATURES_RUNTIME = 1024  # omni/speech2text/tdt_decoder.cc kNumFeatures (v0.19.0)
MEL_FRAMES_PER_SECOND = 100  # 10 ms hop; PR #1426 passes mel_frames_per_second=100 for Parakeet


class Encode(nn.Module):
    """[1, n_mels, F] -> [1, D, T]: the transformers encoder on the runtime's mel-major layout."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, features):
        h = self.encoder(input_features=features.transpose(1, 2)).last_hidden_state
        return h.transpose(1, 2)


class Decode(nn.Module):
    """(enc [1, D, T], ids [1, U] i32, h [L, 1, H], c [L, 1, H]) -> (log-probs [1, T, U, V+1+K], h, c)."""

    def __init__(self, model):
        super().__init__()
        d = model.decoder
        lstm = d.lstm
        if not (lstm.batch_first and lstm.bias and not lstm.bidirectional and lstm.proj_size == 0):
            raise ValueError("decoder LSTM is not a plain batch_first unidirectional LSTM with bias")
        self.emb = d.embedding
        self.nl = lstm.num_layers
        self.w_ih = nn.ParameterList([getattr(lstm, f"weight_ih_l{i}") for i in range(self.nl)])
        self.w_hh = nn.ParameterList([getattr(lstm, f"weight_hh_l{i}") for i in range(self.nl)])
        self.b_ih = nn.ParameterList([getattr(lstm, f"bias_ih_l{i}") for i in range(self.nl)])
        self.b_hh = nn.ParameterList([getattr(lstm, f"bias_hh_l{i}") for i in range(self.nl)])
        self.dec_proj = d.decoder_projector
        self.enc_proj = model.encoder_projector
        self.act = model.joint.activation
        self.head = model.joint.head

    def forward(self, enc, ids, h, c):
        x = self.emb(ids)
        hs = [h[i] for i in range(self.nl)]
        cs = [c[i] for i in range(self.nl)]
        outs = []
        for u in range(ids.shape[1]):
            xi = x[:, u]
            for i in range(self.nl):
                g = F.linear(xi, self.w_ih[i], self.b_ih[i]) + F.linear(hs[i], self.w_hh[i], self.b_hh[i])
                gi, gf, gg, go = g.chunk(4, dim=-1)
                cs[i] = torch.sigmoid(gf) * cs[i] + torch.sigmoid(gi) * torch.tanh(gg)
                hs[i] = torch.sigmoid(go) * torch.tanh(cs[i])
                xi = hs[i]
            outs.append(xi)
        g = self.dec_proj(torch.stack(outs, dim=1))
        f = self.enc_proj(enc.transpose(1, 2))
        z = self.act(f[:, :, None, :] + g[:, None, :, :])
        b, t, u, hh = z.shape
        logits = self.head(z.reshape(b, t * u, hh)).reshape(b, t, u, -1)
        return torch.log_softmax(logits, dim=-1), torch.stack(hs, dim=0), torch.stack(cs, dim=0)


def reference_decode(model, enc, ids, h, c):
    """The same step through the checkpoint's own modules (nn.LSTM, the joint network's forward)."""
    e = model.decoder.embedding(ids.long())
    out, (h2, c2) = model.decoder.lstm(e, (h, c))
    g = model.decoder.decoder_projector(out)
    f = model.encoder_projector(enc.transpose(1, 2))
    logits = model.joint(decoder_hidden_states=g[:, None, :, :], encoder_hidden_states=f[:, :, None, :])
    return torch.log_softmax(logits, dim=-1), h2, c2


def check_config(cfg, fe):
    """Reasons the runtime would read this checkpoint wrong (empty = supported)."""
    why = []
    enc = cfg.encoder_config
    if enc.hidden_size != N_FEATURES_RUNTIME:
        why.append(f"encoder hidden_size {enc.hidden_size} != {N_FEATURES_RUNTIME} (LiteRT-LM v0.19.0 "
                   "tdt_decoder.cc reads the encoder frames as numel / 1024)")
    if list(cfg.durations) != list(range(len(cfg.durations))):
        why.append(f"durations {list(cfg.durations)} are not 0..K-1 (the runtime steps by the duration index)")
    if cfg.blank_token_id != cfg.vocab_size - 1:
        why.append(f"blank_token_id {cfg.blank_token_id} is not the last of vocab_size {cfg.vocab_size}")
    if enc.num_mel_bins != RUNTIME_FRONT_END["feature_size"]:
        why.append(f"encoder num_mel_bins {enc.num_mel_bins}, the runtime's Parakeet front end makes 128")
    for k, v in (fe or {}).items():
        if k in RUNTIME_FRONT_END and abs(float(v) - RUNTIME_FRONT_END[k]) > 1e-6:
            why.append(f"feature extractor {k}={v!r}, the runtime's Parakeet front end is {RUNTIME_FRONT_END[k]}")
    return why


# omni/speech2text/parakeet_tdt.cc ResolveMetadata defaults (v0.19.0) and the 16 kHz / 25 ms / 10 ms framing
RUNTIME_FRONT_END = {"feature_size": 128, "sampling_rate": 16000, "hop_length": 160, "win_length": 400,
                     "n_fft": 512, "preemphasis": 0.97}


def feature_extractor_config(snapshot):
    """The feature extractor's settings from processor_config.json (or preprocessor_config.json), read as JSON:
    the runtime computes the log-mel itself, and transformers' ParakeetFeatureExtractor would need librosa.
    None when the repo has neither file (a finetune that ships config.json, the weights and tokenizer.json only:
    the runtime's front end is then assumed, with the encoder's num_mel_bins checked)."""
    for name in ("processor_config.json", "preprocessor_config.json"):
        p = os.path.join(snapshot, name)
        if os.path.exists(p):
            d = json.load(open(p))
            fe = d.get("feature_extractor", d)
            return {k: fe[k] for k in RUNTIME_FRONT_END if k in fe}
    return None


def tdt_greedy(run_decode, enc, blank, n_dur, max_symbols=10):
    """Plain greedy TDT over one window with the stateful step (transformers' algorithm: token argmax, duration
    argmax, a blank advances at least one frame, at most max_symbols non-blank emissions per frame)."""
    t_max = enc.shape[-1]
    tok = np.array([[blank]], dtype=np.int32)
    h = np.zeros(run_decode.state_shape, np.float32)
    c = np.zeros(run_decode.state_shape, np.float32)
    lp, h1, c1 = run_decode(enc, tok, h, c)
    out, t, sym = [], 0, 0
    while t < t_max:
        row = lp[0, t, 0]
        k = int(np.argmax(row[:-n_dur]))
        dur = int(np.argmax(row[-n_dur:]))
        if k == blank:
            t += max(dur, 1)
            sym = 0
            continue
        out.append(k)
        h, c = h1, c1
        lp, h1, c1 = run_decode(enc, np.array([[k]], dtype=np.int32), h, c)
        sym += 1
        if dur > 0 or sym >= max_symbols:
            t += max(dur, 1) if dur > 0 else 1
            sym = 0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("snapshot")
    ap.add_argument("work_dir")
    ap.add_argument("--input-sec", type=float, default=5.0)
    ap.add_argument("--stateful-after", type=int, default=4,
                    help="token slots of the stateless `decode` (decode_1 is added when > 1)")
    ap.add_argument("--recipe", choices=("int8", "fp32"), default="int8")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.work_dir, exist_ok=True)
    rec = {"snapshot": a.snapshot, "input_sec": a.input_sec, "stateful_after": a.stateful_after,
           "recipe": a.recipe, "checks": {}}
    t0 = time.time()

    from transformers import AutoConfig, ParakeetForTDT
    cfg = AutoConfig.from_pretrained(a.snapshot)
    fe = feature_extractor_config(a.snapshot)
    why = check_config(cfg, fe)
    if why:
        rec["unsupported"] = why
        json.dump(rec, open(os.path.join(a.work_dir, "export.json"), "w"), indent=1)
        print("UNSUPPORTED: " + "; ".join(why), file=sys.stderr)
        return 2
    model = ParakeetForTDT.from_pretrained(a.snapshot, dtype=torch.float32).eval()
    n_frames = round(a.input_sec * MEL_FRAMES_PER_SECOND)
    u_slots = a.stateful_after if a.stateful_after > 0 else math.ceil(a.input_sec * 8 / 32) * 32
    L, H = cfg.num_decoder_layers, cfg.decoder_hidden_size
    n_dur = len(cfg.durations)
    rec["front_end"] = {"processor_config": fe, "assumed": RUNTIME_FRONT_END if fe is None else None}
    rec["shapes"] = {"n_mels": cfg.encoder_config.num_mel_bins, "mel_frames": n_frames, "u_slots": u_slots, "lstm": [L, 1, H],
                     "logits_last": cfg.vocab_size + n_dur, "blank": cfg.blank_token_id, "durations": n_dur}

    enc_m, dec_m = Encode(model).eval(), Decode(model).eval()
    g = torch.Generator().manual_seed(0)
    feats = torch.randn(1, cfg.encoder_config.num_mel_bins, n_frames, generator=g)
    with torch.no_grad():
        enc = enc_m(feats)
        rec["shapes"]["enc"] = list(enc.shape)
        # the explicit-gate decoder against nn.LSTM + the checkpoint's joint, on real encoder output and random ids
        ids = torch.randint(0, cfg.vocab_size, (1, u_slots), generator=g, dtype=torch.int32)
        ids[0, 0] = cfg.blank_token_id
        h0 = torch.randn(L, 1, H, generator=g) * 0.1
        c0 = torch.randn(L, 1, H, generator=g) * 0.1
        mine = dec_m(enc, ids, h0, c0)
        ref = reference_decode(model, enc, ids, h0, c0)
        diffs = [float((x - y).abs().max()) for x, y in zip(mine, ref)]
        rec["checks"]["decode_vs_modules_maxabs"] = diffs
        ok_dec = max(diffs) < 1e-4
    print(f"torch: enc {list(enc.shape)}, decode vs nn.LSTM + joint max |d| {max(diffs):.2e}", flush=True)

    import litert_torch
    from litert_torch.generative.export_hf.core.export_lib import quantize_model
    ids_u = torch.zeros(1, u_slots, dtype=torch.int32)
    ids_1 = torch.zeros(1, 1, dtype=torch.int32)
    hz = torch.zeros(L, 1, H)
    conv = litert_torch.signature("encode", enc_m, (feats,))
    conv = conv.signature("decode", dec_m, (enc, ids_u, hz, hz.clone()))
    if u_slots > 1 and a.stateful_after > 1:
        conv = conv.signature("decode_1", dec_m, (enc, ids_1, hz, hz.clone()))
    t1 = time.time()
    edge = conv.convert()
    fp32_path = os.path.join(a.work_dir, "asr_model_fp32.tflite")
    edge.export(fp32_path)
    rec["convert_s"] = round(time.time() - t1, 1)
    out_path = fp32_path
    if a.recipe == "int8":
        t2 = time.time()
        q = quantize_model(fp32_path, "dynamic_wi8_afp32")
        out_path = os.path.join(a.work_dir, "asr_model_int8.tflite")
        os.replace(q, out_path)
        rec["quantize_s"] = round(time.time() - t2, 1)
    rec["tflite"] = {"fp32": fp32_path, "out": out_path, "bytes": os.path.getsize(out_path)}
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)", flush=True)

    # the converted graphs on the same inputs, and a whole greedy TDT window against the PyTorch modules
    from ai_edge_litert.interpreter import Interpreter
    ok_tfl = True
    for tag, path in (("fp32", fp32_path), (a.recipe, out_path)):
        it = Interpreter(model_path=path, num_threads=a.threads)
        sigs = {k: {"in": {n: d["shape"].tolist() for n, d in it.get_signature_runner(k).get_input_details().items()},
                    "out": {n: d["shape"].tolist() for n, d in it.get_signature_runner(k).get_output_details().items()}}
                for k in it.get_signature_list()}
        rec.setdefault("signatures", {})[tag] = sigs
        er = it.get_signature_runner("encode")
        dr = it.get_signature_runner("decode")
        d1 = it.get_signature_runner("decode_1") if "decode_1" in sigs else None
        e_t = er(args_0=feats.numpy())["output_0"]
        o = dr(args_0=e_t, args_1=ids.numpy(), args_2=h0.numpy(), args_3=c0.numpy())
        enc_d = float(np.abs(e_t - enc.numpy()).max())
        lp_d = float(np.abs(o["output_0"] - mine[0].numpy()).max())
        rec["checks"][f"{tag}_encode_maxabs"] = enc_d
        rec["checks"][f"{tag}_decode_logprob_maxabs"] = lp_d

        def run_t(e, tok, h, c, _d1=d1, _dr=dr):
            r = (_d1 or _dr)(args_0=e, args_1=tok if _d1 else np.pad(tok, ((0, 0), (0, u_slots - 1))),
                             args_2=h, args_3=c)
            return r["output_0"][:, :, :1], r["output_1"], r["output_2"]
        run_t.state_shape = (L, 1, H)

        def run_p(e, tok, h, c):
            with torch.no_grad():
                r = dec_m(torch.from_numpy(e), torch.from_numpy(tok), torch.from_numpy(h), torch.from_numpy(c))
            return r[0].numpy(), r[1].numpy(), r[2].numpy()
        run_p.state_shape = (L, 1, H)
        tok_t = tdt_greedy(run_t, e_t, cfg.blank_token_id, n_dur)
        tok_p = tdt_greedy(run_p, enc.numpy(), cfg.blank_token_id, n_dur)
        rec["checks"][f"{tag}_greedy_tokens_equal_torch"] = tok_t == tok_p
        print(f"{tag}: encode max |d| {enc_d:.2e}, decode log-prob max |d| {lp_d:.2e}, greedy tokens "
              f"{'equal' if tok_t == tok_p else 'DIFFER'} ({len(tok_p)} tokens on random features)", flush=True)
        if tag == "fp32":
            ok_tfl = ok_tfl and enc_d < 1e-2 and lp_d < 1e-2 and tok_t == tok_p
        if len(sigs) < 2:
            ok_tfl = False

    meta = {"asr_model_type": {"parakeet": {}}, "input_milliseconds": round(a.input_sec * 1000),
            "log_mel_spectrogram_config": {"n_frames": n_frames}}
    if a.stateful_after > 0:
        meta["decode_statefully_after"] = a.stateful_after
    if cfg.blank_token_id != 8192:  # the runtime's Parakeet default start (blank) token
        meta["decode_start_token_ids"] = [cfg.blank_token_id]
    json.dump(meta, open(os.path.join(a.work_dir, "speech2text_metadata.json"), "w"), indent=1)
    rec["speech2text_metadata"] = meta
    rec["total_s"] = round(time.time() - t0, 1)
    rec["ok"] = bool(ok_dec and ok_tfl)
    json.dump(rec, open(os.path.join(a.work_dir, "export.json"), "w"), indent=1)
    print(f"export {'OK' if rec['ok'] else 'CHECK FAILED'} in {rec['total_s']} s", flush=True)
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
