"""Fused experts for the litert_moe_sequential route: the per-expert FCs as two large FCs, the same math.

litert-torch's `litert_sequential_experts_forward` (generative/layers/moe.py, git 0d3869c L540-604) runs every expert as
a separate chain (mask, weight, FC gate_up, chunk, act, mul, FC down, mul, add: ~13 ops per expert per layer). Here:

  w[t, e]  = sum_k [top_k_index[t, k] == e] * top_k_weights[t, k]                      [T, E, 1]
  gate_up  = F.linear(x, W_gu, b_gu),  W_gu = gate_up_proj.reshape(E * 2I, H)          [T, E * 2I] -> [T, E, 2I]
  h        = act_fn(gate) * up * w                                                     [T, E, I]
  out      = F.linear(h.reshape(T, E * I), W_d) (+ w[T, E] @ down_bias[E, H]),
             W_d = down_proj.permute(1, 0, 2).reshape(H, E * I)                        [T, H]

Non-selected experts get w = 0, so the sum over all E experts equals the sum over the top-k (as in the sequential
route, which also computes every expert). W_gu / W_d (and the biases) are built once before the trace by
`pre_fuse_model_experts`, which takes the place of export_lib.pre_split_model_experts in the sequential route and
deletes the 3-D tensors as that step does.

Routing variants (how w is computed; the FCs are the same):
  bcast  one broadcast EQUAL of top_k_index [T, 1, K] against the expert ids [1, E, 1], CAST, MUL by the weights,
         SUM over K (keepdim). The recipe uses this one.
  loop   per expert: EQUAL(top_k_index, e) with a rank-0 constant, CAST, MUL, SUM(keepdim) (the four ops the
         sequential route uses for an expert's routing weight), then the 32 [T, 1] columns joined by 4-input
         CONCATENATIONs into [T, E], reshaped to [T, E, 1].

down_groups G > 1: W_d split into G FCs over consecutive expert groups (E / G experts each, per-channel scales per FC),
their outputs added. The recipe uses G = 1.

  install(routing="bcast", down_groups=1, act=None)   registers the forward as "litert_moe_sequential" in
      transformers' experts interface and points export_lib.pre_split_model_experts at pre_fuse_model_experts.
"""
import gc

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN

CONFIG = {"routing": "bcast", "down_groups": 1, "act": None}
RECORD = {"modules": [], "forward_calls": 0, "routing": None, "down_groups": None, "act": None}


class GeluTanh(nn.Module):
  """The `moe` custom op's activation; only for a negative control (act="gelu_tanh"), never in the recipe."""

  def forward(self, x):
    return F.gelu(x, approximate="tanh")


def pre_fuse_model_experts(model: nn.Module) -> nn.Module:
  """Builds the fused weights of every experts module once, before the trace, and deletes the 3-D tensors."""
  targets = [(n, m) for n, m in model.named_modules()
             if hasattr(m, "gate_up_proj") and getattr(m, "num_experts", None) is not None]
  for name, module in targets:  # collected first: a ParameterList child below must not change the walked dicts
    if getattr(module, "is_transposed", False) or not getattr(module, "is_concatenated", True) or not getattr(
        module, "has_gate", True):
      raise NotImplementedError(f"{name}: fused experts cover the concatenated [E, 2I, H] gate_up layout only")
    e = int(module.num_experts)
    gu = module.gate_up_proj.data  # [E, 2I, H]
    dn = module.down_proj.data  # [E, H, I]
    if gu.dim() != 3 or dn.dim() != 3 or gu.shape[0] != e or dn.shape[0] != e or gu.shape[2] != dn.shape[1] or (
        gu.shape[1] != 2 * dn.shape[2]):
      raise NotImplementedError(f"{name}: unexpected expert shapes {tuple(gu.shape)} / {tuple(dn.shape)}")
    two_i, h = int(gu.shape[1]), int(gu.shape[2])
    i = two_i // 2
    g = int(CONFIG["down_groups"])
    if e % g:
      raise ValueError(f"{name}: {e} experts do not split into {g} groups")
    module.fused_gate_up_proj = nn.Parameter(gu.reshape(e * two_i, h).contiguous(), requires_grad=False)
    if g == 1:
      module.fused_down_proj = nn.Parameter(dn.permute(1, 0, 2).reshape(h, e * i).contiguous(), requires_grad=False)
    else:
      n = e // g
      module.fused_down_proj_groups = nn.ParameterList([
          nn.Parameter(dn[j * n:(j + 1) * n].permute(1, 0, 2).reshape(h, n * i).contiguous(), requires_grad=False)
          for j in range(g)
      ])
    gub = getattr(module, "gate_up_bias", None)
    dnb = getattr(module, "down_bias", None)
    module.fused_gate_up_bias = None if gub is None else nn.Parameter(gub.data.reshape(e * two_i).contiguous(),
                                                                       requires_grad=False)
    module.fused_down_bias_t = None if dnb is None else nn.Parameter(dnb.data.t().contiguous(),
                                                                      requires_grad=False)  # [H, E]
    module.register_buffer("fused_expert_ids", torch.arange(e, dtype=torch.int32).reshape(1, e, 1),
                           persistent=False)
    for attr in ("gate_up_proj", "down_proj", "gate_up_bias", "down_bias"):
      if module._parameters.get(attr) is not None:
        delattr(module, attr)  # as pre_split_model_experts: the 3-D tensors leave the module
    if CONFIG["act"] == "gelu_tanh":
      module.act_fn = GeluTanh()
    act = getattr(module, "act_fn", None)
    RECORD["modules"].append({"name": name, "num_experts": e, "hidden": h, "intermediate": i,
                              "gate_up_bias": gub is not None, "down_bias": dnb is not None,
                              "act_fn": type(act).__name__ if act is not None else None})
    gc.collect()
  RECORD.update(routing=CONFIG["routing"], down_groups=CONFIG["down_groups"], act=CONFIG["act"])
  print(f"FUSED_MODULES {len(RECORD['modules'])} routing={CONFIG['routing']} down_groups={CONFIG['down_groups']} "
        f"act={CONFIG['act'] or 'model'}", flush=True)
  return model


def _routing_weights(self, top_k_index, top_k_weights, dtype):
  """-> w [T, E, 1]: the routing weight of every expert (0 for the experts not in the token's top-k)."""
  e = int(self.num_experts)
  if CONFIG["routing"] == "bcast":
    mask = (top_k_index.unsqueeze(1) == self.fused_expert_ids).to(dtype)  # [T, 1, K] vs [1, E, 1] -> [T, E, K]
    return (mask * top_k_weights.unsqueeze(1)).sum(dim=-1, keepdim=True)  # [T, E, 1]
  if CONFIG["routing"] == "loop":
    cols = []
    for idx in range(e):
      expert_idx_tensor = torch.tensor(idx, dtype=torch.int32, device=top_k_index.device)
      expert_mask = top_k_index == expert_idx_tensor
      cols.append((expert_mask.to(dtype) * top_k_weights).sum(dim=-1, keepdim=True))  # [T, 1]
    while len(cols) > 1:
      cols = [torch.cat(cols[j:j + 4], dim=-1) if len(cols[j:j + 4]) > 1 else cols[j] for j in range(0, len(cols), 4)]
    return cols[0].unsqueeze(-1)  # [T, E, 1]
  raise ValueError(f"unknown routing {CONFIG['routing']!r}")


def fused_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
  """Experts forward with the same arguments as litert_sequential_experts_forward; needs pre_fuse_model_experts."""
  if not hasattr(self, "fused_gate_up_proj"):
    raise RuntimeError("fused_experts_forward: the module was not pre-fused (pre_fuse_model_experts)")
  RECORD["forward_calls"] += 1
  shape = hidden_states.shape
  x = hidden_states if hidden_states.dim() == 2 else hidden_states.reshape(-1, shape[-1])
  if top_k_index.dim() != 2:
    top_k_index = top_k_index.reshape(-1, top_k_index.shape[-1])
    top_k_weights = top_k_weights.reshape(-1, top_k_weights.shape[-1])
  t, e = x.shape[0], int(self.num_experts)
  w = _routing_weights(self, top_k_index, top_k_weights, x.dtype)  # [T, E, 1]
  gate_up = F.linear(x, self.fused_gate_up_proj, self.fused_gate_up_bias)  # [T, E * 2I]
  gate, up = gate_up.reshape(t, e, -1).chunk(2, dim=-1)  # [T, E, I] each
  act_fn = getattr(self, "act_fn", None) or ACT2FN[self.config.hidden_activation]
  h = act_fn(gate) * up * w  # [T, E, I]
  if hasattr(self, "fused_down_proj"):
    out = F.linear(h.reshape(t, -1), self.fused_down_proj)  # [T, H]
  else:
    groups = self.fused_down_proj_groups
    n = e // len(groups)
    out = None
    for j, wd in enumerate(groups):
      part = F.linear(h[:, j * n:(j + 1) * n, :].reshape(t, -1), wd)
      out = part if out is None else out + part
  if self.fused_down_bias_t is not None:
    out = out + F.linear(w.reshape(t, e), self.fused_down_bias_t)  # w [T, E] @ down_bias [E, H]
  return out if hidden_states.dim() == 2 else out.reshape(shape)


def install(routing="bcast", down_groups=1, act=None):
  """Points the sequential route of litert-torch at the fused experts (process-local monkeypatch)."""
  CONFIG.update(routing=routing, down_groups=int(down_groups), act=act)
  import litert_torch.generative.export_hf.model_ext  # noqa: F401  registers the original experts functions first
  from litert_torch.generative.export_hf.core import export_lib
  from transformers.integrations import moe as transformers_moe
  before = transformers_moe.ALL_EXPERTS_FUNCTIONS.get("litert_moe_sequential")
  transformers_moe.ALL_EXPERTS_FUNCTIONS.register("litert_moe_sequential", fused_experts_forward)
  export_lib.pre_split_model_experts = pre_fuse_model_experts
  after = transformers_moe.ALL_EXPERTS_FUNCTIONS.get("litert_moe_sequential")
  def qual(fn):
    return f"{getattr(fn, '__module__', None)}.{getattr(fn, '__name__', None)}" if fn is not None else None

  info = {"replaced": qual(before), "now": qual(after), "pre_split_now": qual(export_lib.pre_split_model_experts)}
  print("FUSED_INSTALLED", info, flush=True)
  return info
