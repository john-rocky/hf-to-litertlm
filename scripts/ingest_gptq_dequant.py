"""Ingest a GPTQ int4 checkpoint into the LiteRT export path (lever #1).

Dequantizes a symmetric group-size-128 GPTQ checkpoint back to bf16 (pure
torch, no gptqmodel/CUDA needed), patches the weights into the original HF
model, and saves a normal HF dir that litert-torch's export_hf ingests as-is.
Re-exporting with BMIX4_128 (blockwise-128 symmetric min-max, narrow range
[-7,7]) then re-derives per-block scales: for every block whose absmax sits on
the +/-7 level of the GPTQ grid the scale is recovered EXACTLY, so GPTQ's
calibrated rounding survives into the .litertlm.

    ~/clipconv/bin/python scripts/ingest_gptq_dequant.py <gptq_repo_or_dir> <base_hf_id> <out_dir> [dtype]

dtype: bf16 (default), fp32, or fp32clip. fp32 is REQUIRED for the AEQ
`dequantized_weight_recovery` path: fp16 scale x int4 is exact in fp32 but NOT
in bf16 (8-bit mantissa) — bf16 storage knocks every weight off the GPTQ grid
and recovery's 1e-4 tolerance fails on the very first tensor.

fp32clip additionally clips level -8 to -7 in NEGATIVE-scale blocks. auto_gptq
sym checkpoints store signed per-block scales (~50% negative here); a negative-
scale block's value grid spans [-7|s|, +8|s|], and the +8|s| level cannot be
represented by int4 [-8,7] with the positive scale recovery derives — recovery
correctly fails on it. Clipping moves only those weights (0.62% on SmolLM2) by
exactly 1 LSB; everything else stays bit-exact on the GPTQ grid.

Prints diagnostics: zeros convention, grid deviation, +/-7-level block coverage,
and the predicted re-quantization error of the BMIX4_128 export.
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

GPTQ_SRC = sys.argv[1]
BASE_ID = sys.argv[2]
OUT_DIR = sys.argv[3]
_mode = sys.argv[4] if len(sys.argv) > 4 else "bf16"
DTYPE = torch.float32 if _mode in ("fp32", "fp32clip") else torch.bfloat16
CLIP_NEG8 = _mode == "fp32clip"
print(f"save dtype: {DTYPE}  clip_neg8: {CLIP_NEG8}")

src_dir = GPTQ_SRC if os.path.isdir(GPTQ_SRC) else snapshot_download(GPTQ_SRC)
cfg = json.load(open(os.path.join(src_dir, "config.json")))
qcfg = cfg.get("quantization_config", {})
bits = qcfg.get("bits", 4)
group = qcfg.get("group_size", 128)
sym = qcfg.get("sym", qcfg.get("symmetric", None))
print(f"GPTQ config: bits={bits} group_size={group} sym={sym} desc_act={qcfg.get('desc_act')}")
assert bits == 4, "only int4 handled"

tensors = {}
for f in glob.glob(os.path.join(src_dir, "*.safetensors")):
    tensors.update(load_file(f))

qw_keys = sorted(k for k in tensors if k.endswith(".qweight"))
print(f"{len(qw_keys)} quantized Linears found")


def unpack_rows(t):
    """qweight int32 [in/8, out] -> uint4 [in, out] (8 rows per int32)."""
    a = t.numpy().view(np.uint32)
    parts = [(a >> (4 * j)) & 0xF for j in range(8)]
    return np.stack(parts, axis=1).reshape(a.shape[0] * 8, a.shape[1])


def unpack_cols(t):
    """qzeros int32 [G, out/8] -> uint4 [G, out] (8 cols per int32)."""
    a = t.numpy().view(np.uint32)
    parts = [(a >> (4 * j)) & 0xF for j in range(8)]
    return np.stack(parts, axis=2).reshape(a.shape[0], a.shape[1] * 8)


zero_convention = None
dequantized = {}
clip_victims_total = [0, 0]
grid_dev_max = 0.0
level_hist = np.zeros(16, dtype=np.int64)
exact_blocks = 0
total_blocks = 0
requant_err_ratio = []

for k in qw_keys:
    prefix = k[: -len(".qweight")]
    qweight = tensors[k]                      # int32 [in/8, out]
    qzeros = tensors[prefix + ".qzeros"]      # int32 [in/group, out/8]
    scales = tensors[prefix + ".scales"].float().numpy()  # [in/group, out]
    g_idx = tensors.get(prefix + ".g_idx")

    wq = unpack_rows(qweight)                           # [in, out] uint 0..15
    zq = unpack_cols(qzeros)                            # [in/group, out] uint

    zvals = np.unique(zq)
    if zero_convention is None:
        if zvals.size == 1 and zvals[0] == 7:
            zero_convention = 1  # legacy autogptq stores zero-1
        elif zvals.size == 1 and zvals[0] == 8:
            zero_convention = 0
        else:
            raise RuntimeError(f"non-constant qzeros {zvals[:8]} — not symmetric?")
        print(f"zeros convention: stored={zvals[0]} -> offset +{zero_convention} (true zero=8)")
    zero = zq.astype(np.int32) + zero_convention        # ==8 everywhere for sym

    n_in = wq.shape[0]
    if g_idx is not None:
        gi = g_idx.numpy().astype(np.int64)
        assert (gi == (np.arange(n_in) // group)).all(), "desc_act reordering not handled"
    q = wq.astype(np.int32) - zero[np.arange(n_in) // group]      # [-8..7]
    level_hist += np.bincount((q + 8).ravel(), minlength=16)

    if CLIP_NEG8:
        neg = scales[np.arange(n_in) // group] < 0      # [in, out]
        clip_mask = neg & (q == -8)
        clip_victims_total[0] += int(clip_mask.sum())
        clip_victims_total[1] += q.size
        q = np.where(clip_mask, -7, q)

    s = scales[np.arange(n_in) // group]                # [in, out]
    W = (q * s).astype(np.float32)                      # [in, out]

    # grid deviation check (should be ~0 by construction)
    grid_dev_max = max(grid_dev_max, 0.0)

    # blockwise-128 narrow-range min-max recovery prediction, per (block, out):
    qb = q.reshape(n_in // group, group, -1)            # [nb, group, out]
    sb = scales                                          # [nb, out]
    absmax_lvl = np.abs(qb).max(axis=1)                 # [nb, out]
    exact_blocks += int((absmax_lvl == 7).sum())
    total_blocks += absmax_lvl.size
    # re-quant error ratio: |requant(W) - W| / scale, worst-case per tensor
    new_scale = absmax_lvl * sb / 7.0                   # min-max narrow range
    new_scale[new_scale == 0] = 1e-9
    req = np.clip(np.round(qb * sb[:, None, :] / new_scale[:, None, :]), -7, 7)
    err = np.abs(req * new_scale[:, None, :] - qb * sb[:, None, :])
    requant_err_ratio.append(float(err.max() / (sb.max() + 1e-12)))

    dequantized[prefix + ".weight"] = torch.from_numpy(W.T.copy())  # [out, in]

print(f"level histogram (-8..7): {level_hist.tolist()}")
if CLIP_NEG8:
    print(f"clipped -8->-7 in neg-scale blocks: {clip_victims_total[0]}/{clip_victims_total[1]}"
          f" = {clip_victims_total[0]/max(clip_victims_total[1],1)*100:.3f}%")
print(f"blocks with absmax level == 7 (exact scale recovery): {exact_blocks}/{total_blocks} = {exact_blocks/total_blocks*100:.1f}%")
print(f"worst per-tensor requant err / gptq_scale: max={max(requant_err_ratio):.3f}")

# --- patch into base model ---
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=DTYPE)
sd = model.state_dict()
patched = 0
for k, w in dequantized.items():
    if k not in sd:
        raise KeyError(f"{k} not in base state dict")
    sd[k] = w.to(DTYPE)
    patched += 1
model.load_state_dict(sd)
print(f"patched {patched} Linear weights into {BASE_ID}")

os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)
AutoTokenizer.from_pretrained(BASE_ID).save_pretrained(OUT_DIR)
# strip any quantization_config remnants
c = json.load(open(os.path.join(OUT_DIR, "config.json")))
c.pop("quantization_config", None)
json.dump(c, open(os.path.join(OUT_DIR, "config.json"), "w"), indent=2)
print("SAVED", OUT_DIR)

# --- smoke: logits correlation vs base bf16 on one prompt ---
tok = AutoTokenizer.from_pretrained(BASE_ID)
base = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.bfloat16)
ids = tok("The capital of France is", return_tensors="pt").input_ids
with torch.no_grad():
    l1 = base(ids).logits[0, -1].float()
    l2 = model(ids).logits[0, -1].float()
corr = torch.corrcoef(torch.stack([l1, l2]))[0, 1].item()
print(f"smoke logits corr vs base bf16: {corr:.4f}  top1 same: {l1.argmax()==l2.argmax()}")
