"""Torch-level check of the port against the oracle dumps (teacher forcing).

  ~/venvs/lt094dev/bin/python3 audio8_tts_work/verify_port.py [n_cases]
"""
import os, sys, glob, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import arktts_port as P

torch.set_num_threads(8)
NEG = -1e9
w = P.load_main_weights()
slow = P.SlowAR(w).eval()
fast = P.FastAR(w).eval()

# rope tables vs the oracle's buffers
fc = np.load(f"{C.FIX}/oracle_freqs_cis.npy")
cos, sin = P.rope_tables(C.MAX_SEQ, C.HEAD_DIM, C.ROPE_BASE)
print(f"rope table vs oracle freqs_cis: max|d| cos {np.abs(cos.numpy()-fc[...,0]).max():.2e} sin {np.abs(sin.numpy()-fc[...,1]).max():.2e}")
ffc = np.load(f"{C.FIX}/oracle_fast_freqs_cis.npy")
fcos, fsin = P.rope_tables(C.NUM_CB, C.HEAD_DIM, C.ROPE_BASE)
print(f"fast rope vs oracle: max|d| {max(np.abs(fcos.numpy()-ffc[...,0]).max(), np.abs(fsin.numpy()-ffc[...,1]).max()):.2e}")

CACHE = C.MAX_SEQ
def zeros_kv(n):
    return [torch.zeros(1, C.KV_HEADS, CACHE, C.HEAD_DIM) for _ in range(2 * n)]

def causal_mask(T, pos0):
    m = torch.full((1, 1, T, CACHE), NEG)
    for i in range(T):
        m[0, 0, i, : pos0 + i + 1] = 0.0
    return m

n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 3
files = sorted(glob.glob(f"{C.OUT}/oracle/*.npz"))[:n_cases]
for f in files:
    d = np.load(f)
    prompt, codes, sem = d["prompt"], d["codes"], d["semantic"]
    P_, T = prompt.shape[1], codes.shape[1]
    full = np.concatenate([prompt, np.concatenate([sem[None], codes], 0)], 1)  # [11,P+T]
    kv = zeros_kv(C.N_LAYER)
    # prefill the prompt in one shot, then teacher-force each generated column
    t0 = time.time()
    with torch.no_grad():
        out = slow(torch.tensor(prompt[None], dtype=torch.int32), torch.arange(P_, dtype=torch.int32), causal_mask(P_, 0), *kv)
        logits, hidden, kv = out[0], out[1], list(out[2:])
        port_logits, port_hidden = [logits[0].numpy()], [hidden[0, 0].numpy()]
        for t in range(T):
            col = torch.tensor(full[:, P_ + t][None, :, None], dtype=torch.int32)
            out = slow(col, torch.tensor([P_ + t], dtype=torch.int32), causal_mask(1, P_ + t), *kv)
            logits, hidden, kv = out[0], out[1], list(out[2:])
            port_logits.append(logits[0].numpy()); port_hidden.append(hidden[0, 0].numpy())
    dt = time.time() - t0
    pl, ph = np.stack(port_logits), np.stack(port_hidden)
    ol, oh = d["slow_logits"], d["fast_hidden"]
    n = min(len(pl), len(ol))
    dl = np.abs(pl[:n] - ol[:n]); dh = np.abs(ph[:n] - oh[:n])
    argmax_match = (pl[:n].argmax(1) == ol[:n].argmax(1)).mean()
    print(f"{os.path.basename(f)}: P {P_} T {T} slow logits max|d| {dl.max():.3e} (mean {dl.mean():.2e}, |logit| scale {np.abs(ol).max():.1f}) "
          f"hidden max|d| {dh.max():.3e} argmax match {argmax_match:.3f}  [{dt:.1f}s]")
    # fast AR on the first frames with recorded logits: hidden -> pos0, then codes[0..8] -> logits for positions 1..9
    fl = d["fast_logits_first8"]
    if len(fl):
        maxd = 0.0; am = 0; tot = 0
        with torch.no_grad():
            for fi in range(len(fl)):
                k_all = torch.zeros(C.N_FAST, 1, C.KV_HEADS, C.NUM_CB, C.HEAD_DIM); v_all = torch.zeros_like(k_all)
                h = torch.tensor(oh[fi]).view(1, 1, -1)
                def step(hid, tok, use, pos, k_all, v_all):
                    m = torch.where(torch.arange(C.NUM_CB) <= pos, 0.0, NEG).view(1, 1, 1, -1)
                    return fast(hid, torch.tensor([tok], dtype=torch.int32), torch.tensor([use], dtype=torch.float32),
                                torch.tensor([pos], dtype=torch.int32), m, k_all, v_all)
                _, k_all, v_all = step(h, 0, 1.0, 0, k_all, v_all)
                cur = int(codes[0, fi])
                for p in range(1, C.NUM_CB):
                    lg, k_all, v_all = step(h, cur, 0.0, p, k_all, v_all)
                    ref = fl[fi, p - 1]
                    maxd = max(maxd, float(np.abs(lg[0].numpy() - ref).max()))
                    am += int(lg[0].argmax()) == int(ref.argmax()); tot += 1
                    cur = int(codes[p, fi])
        print(f"   fast AR ({len(fl)} frames x 9 steps): logits max|d| {maxd:.3e} argmax match {am}/{tot}")
print("VERIFY_PORT_DONE")
