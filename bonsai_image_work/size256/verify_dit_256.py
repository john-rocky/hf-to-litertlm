"""Fidelity + timing check of a quantized DiT export against torch with REAL ids.

Like ../verify_dit.py, but (a) parameterized by SIZE and (b) inputs mapped by
ARGUMENT POSITION: at 256x256 img_ids and txt_ids are both (256, 4), so a
shape-keyed map (what verify_dit.py did at 512) would silently collide.

    SIZE=256 Q=~/models/bonsai-image-256/dit_256_int4b32.tflite python verify_dit_256.py
"""
import collections
import glob
import os
import time

import numpy as np
import torch

SNAP = os.environ.get("SNAP") or sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*")))[-1]
WORK = os.path.expanduser(os.environ.get("WORK", "~/models/bonsai-image-256"))
SIZE = int(os.environ.get("SIZE", "256"))
Q = os.path.expanduser(os.environ.get("Q", os.path.join(WORK, f"dit_{SIZE}_int4b32.tflite")))
GRID = SIZE // 16
TXT = 256

# ---------- (a) dtype census ----------
from ai_edge_litert import schema_py_generated as S

m = S.Model.GetRootAs(bytearray(open(Q, "rb").read()), 0)
tn = {v: k for k, v in vars(S.TensorType).items() if isinstance(v, int)}
sg = m.Subgraphs(0)
tot, cnt, seen = collections.Counter(), collections.Counter(), set()
for i in range(sg.TensorsLength()):
    t = sg.Tensors(i)
    b = m.Buffers(t.Buffer())
    n = b.DataLength() or (b.Size() or 0)
    if n > 1024 and t.Buffer() not in seen:
        seen.add(t.Buffer())
        tot[tn.get(t.Type(), t.Type())] += n
        cnt[tn.get(t.Type(), t.Type())] += 1
print(f"{os.path.basename(Q)}: {os.path.getsize(Q)/2**30:.3f} GiB")
for k in sorted(tot, key=lambda k: -tot[k]):
    print(f"   {k:9s} {cnt[k]:4d} tensors {tot[k]/2**20:9.1f} MiB")

# ---------- (b) numerical check with REAL ids ----------
import diffusers.models.transformers.transformer_flux2 as flux2mod
from diffusers import Flux2Transformer2DModel

one = torch.arange(1)
img_ids = torch.cartesian_prod(one, torch.arange(GRID), torch.arange(GRID), one).float()
txt_ids = torch.cartesian_prod(one, one, one, torch.arange(TXT)).float()
print(f"\nreal ids: img {tuple(img_ids.shape)} max={img_ids.max():.0f}, "
      f"txt {tuple(txt_ids.shape)} max={txt_ids.max():.0f}")

mod = Flux2Transformer2DModel.from_pretrained(
    f"{SNAP}/transformer", torch_dtype=torch.float32).eval()
g = torch.Generator().manual_seed(0)
hs = torch.randn(1, GRID * GRID, mod.config.in_channels, generator=g)
eh = torch.randn(1, TXT, mod.config.joint_attention_dim, generator=g)
ts = torch.tensor([1.0])
kw = dict(hidden_states=hs, encoder_hidden_states=eh, timestep=ts,
          img_ids=img_ids, txt_ids=txt_ids, return_dict=False)

orig = flux2mod.maybe_adjust_dtype_for_device
with torch.no_grad():
    ref64 = mod(**kw)[0]
flux2mod.maybe_adjust_dtype_for_device = (
    lambda d, dev: torch.float32 if d == torch.float64 else d)
with torch.no_grad():
    ref32 = mod(**kw)[0]
flux2mod.maybe_adjust_dtype_for_device = orig
del mod

d = (ref32 - ref64).abs()
scale = ref64.abs().max()
print(f"fp32-rope vs fp64-rope : max|d|={d.max():.3e} rel={d.max()/scale:.3e} "
      f"rms={d.pow(2).mean().sqrt():.3e}")

from ai_edge_litert.interpreter import Interpreter

it = Interpreter(model_path=Q, num_threads=8)
it.allocate_tensors()


def argpos(det):
    parts = det["name"].rsplit("args_", 1)
    return int(parts[1].split(":")[0]) if len(parts) == 2 else 0


inputs = sorted(it.get_input_details(), key=argpos)      # by position, never by shape
for det, arr in zip(inputs, (hs, eh, ts, img_ids, txt_ids)):
    assert tuple(det["shape"]) == tuple(arr.shape), (det["name"], det["shape"], arr.shape)
    it.set_tensor(det["index"], arr.numpy().astype(det["dtype"]))
t0 = time.time()
it.invoke()
y = it.get_tensor(it.get_output_details()[0]["index"])
dt = time.time() - t0

e = np.abs(y - ref64.numpy())
print(f"int4 tflite vs fp64 ref: max|d|={e.max():.3e} rel={e.max()/scale.item():.3e} "
      f"rms={np.sqrt((e ** 2).mean()):.3e}")
print(f"cosine similarity      : "
      f"{float((y.ravel() @ ref64.numpy().ravel()) / (np.linalg.norm(y) * np.linalg.norm(ref64.numpy()))):.6f}")
print(f"one DiT step (Mac CPU, 8 threads, {GRID*GRID}+{TXT} tokens): {dt:.2f} s")
