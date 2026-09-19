"""Hadamard activation transform for PrismML Bonsai 2 (rotated-basis ternary Qwen3.5 hybrid).

The pack folds a blockwise (block 1024) normalized Sylvester Walsh-Hadamard rotation with explicit
+-1 signs into every rotated linear's weight (input dim) and into the embedding table. The runtime
applies, per rotated linear input x of width K:  T(x) = reshape(x * s_K, [-1, 1024]) @ H_1024 / 32
and the inverse on the embedding output:        T^-1(e) = (reshape(e, [-1, 1024]) @ H_1024 / 32) * s_K
(H symmetric orthonormal, so the inverse is the same matmul with the sign multiply after).

`install_hadamard(model, ckpt_dir)` wraps the modules named in hadamard.json (HF names) so the
exported graph carries the transform as MUL(const) + RESHAPE + FULLY_CONNECTED(const H/32) + RESHAPE.
The H constant stays float (its +-1/32 entries would be exact in int4/int2, but a quantized FC would
run the DRQ kernel and quantize the activation on the way in; ai-edge-quantizer's own decomposed
Hadamard insertion keeps it float for the same reason). The recipe therefore excludes it by name.
"""
import json
import math
import os

import numpy as np
import torch
from safetensors.torch import load_file

BLOCK = 1024
HADAMARD_FC_NAME = "hadamard_rotation"  # substring in the wrapper's linear attribute path -> recipe exclusion


def sylvester_hadamard(n):
  h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
  m = h
  while m.shape[0] < n:
    m = np.kron(m, h)
  return m


class HadamardRotate(torch.nn.Module):
  """x -> reshape(x * signs, [-1, BLOCK]) @ (H/sqrt(BLOCK)) -> reshape back."""

  def __init__(self, signs: torch.Tensor, hn: torch.Tensor, inverse: bool = False):
    super().__init__()
    self.register_buffer("signs", signs.to(torch.float32), persistent=False)
    self.width = signs.numel()
    assert self.width % BLOCK == 0, self.width
    # A Linear module so the lowering emits FULLY_CONNECTED with a shared constant.
    self.hadamard_rotation = torch.nn.Linear(BLOCK, BLOCK, bias=False)
    self.hadamard_rotation.weight = torch.nn.Parameter(hn.to(torch.float32), requires_grad=False)
    self.inverse = inverse

  def forward(self, x):
    # The transform runs in fp32 (as PrismML's runtime does: fwht casts to float32, transforms, casts back).
    # For the fp32 export these casts are no-ops and emit nothing.
    dtype = x.dtype
    shape = x.shape
    x32 = x if dtype == torch.float32 else x.float()
    if not self.inverse:
      x32 = x32 * self.signs
    y = self.hadamard_rotation(x32.reshape(-1, BLOCK)).reshape(shape)
    if self.inverse:
      y = y * self.signs
    return y if dtype == torch.float32 else y.to(dtype)


class HadamardLinear(torch.nn.Module):
  def __init__(self, linear: torch.nn.Linear, rot: HadamardRotate):
    super().__init__()
    self.rot = rot
    self.linear = linear

  def forward(self, x):
    return self.linear(self.rot(x))


class HadamardEmbedding(torch.nn.Module):
  def __init__(self, emb: torch.nn.Embedding, rot: HadamardRotate):
    super().__init__()
    self.emb = emb
    self.rot = rot
    # keep the attributes exporters read
    self.weight = emb.weight
    self.num_embeddings = emb.num_embeddings
    self.embedding_dim = emb.embedding_dim
    self.padding_idx = emb.padding_idx

  def forward(self, ids):
    return self.rot(self.emb(ids))


def _hf_name(mlx_name: str) -> str:
  if mlx_name.startswith("language_model.lm_head"):
    return mlx_name.replace("language_model.lm_head", "lm_head")
  assert mlx_name.startswith("language_model.model."), mlx_name
  return "model.language_model." + mlx_name[len("language_model.model."):]


def install_hadamard(model: torch.nn.Module, ckpt_dir: str) -> int:
  """Wraps the rotated modules in place. Returns the number of wrapped modules (0 = not a Bonsai pack)."""
  hpath = os.path.join(ckpt_dir, "hadamard.json")
  spath = os.path.join(ckpt_dir, "hadamard_signs.safetensors")
  if not (os.path.exists(hpath) and os.path.exists(spath)):
    return 0
  h = json.load(open(hpath))
  assert h["prism.hadamard.block_size"] == BLOCK and h["prism.hadamard.sign_mode"] == "explicit", h
  assert h["prism.hadamard.axis"] == "input-last-dimension"
  signs = {int(k.split("_")[1]): v for k, v in load_file(spath).items()}
  hn = torch.from_numpy(sylvester_hadamard(BLOCK) / math.sqrt(BLOCK))
  rots = {w: HadamardRotate(s, hn) for w, s in signs.items()}
  rots_inv = {w: HadamardRotate(s, hn, inverse=True) for w, s in signs.items()}
  modules = dict(model.named_modules())

  def resolve(mlx_name):
    name = _hf_name(mlx_name).removesuffix(".weight")
    if name not in modules and name.startswith("model.language_model."):
      name = "model." + name[len("model.language_model."):]  # Qwen3_5ForCausalLM (text-only class) naming
    assert name in modules, (mlx_name, name)
    parent_name, _, attr = name.rpartition(".")
    return modules[name], modules[parent_name] if parent_name else model, attr

  n = 0
  for mlx_name in h["prism.hadamard.weight_names"]:
    lin, parent, attr = resolve(mlx_name)
    assert isinstance(lin, torch.nn.Linear), (mlx_name, type(lin))
    setattr(parent, attr, HadamardLinear(lin, rots[lin.in_features]))
    n += 1
  for mlx_name in h["prism.hadamard.inverse_weight_names"]:
    emb, parent, attr = resolve(mlx_name)
    assert isinstance(emb, torch.nn.Embedding), (mlx_name, type(emb))
    setattr(parent, attr, HadamardEmbedding(emb, rots_inv[emb.embedding_dim]))
    n += 1
  print(f"Hadamard transform installed on {n} modules (widths {sorted(signs)})")
  return n
