#!/usr/bin/env python3
"""export_depth_frame2.py WEIGHTS OUT.tflite - LFM3-Omni-2B's depth decoder for one whole speech frame in one call, in
float32, made for the phone GPU: the depth projection once, the 3 passes (4 codebooks revealed per pass, in the order
the host draws), and the sampler inside the graph, with temperature 0.4 and repetition penalty 1.2 (the source's
generation_config.json) as constants. The 12 codes are the only output.

Signature `depth_frame`:
  hidden    [1, 2048] f32   the decoder's final-norm output of the speech step
  reveal0, reveal1, reveal2 [1, 12] f32   1 on the codebooks revealed at that pass
  u         [1, 12] f32     the uniform drawn for each codebook; the host caps it at 1 - 2^-11 (float16 rounds a larger
                            value to 1.0)
  presence  [12, 2017] f32  1 where a code is in that codebook's penalty window (its last 64 frames)
  -> codes  [1, 12] i32

The sampler: the penalty divides a positive logit by 1.2 and multiplies a negative one by 1.2 on the codes that are
present (written without SELECT, blended by the 0 / 1 presence so a kept logit is exact), divides by the temperature,
takes the softmax, builds the cdf as a blocked prefix sum (2017 codes as 32 x 64, two triangular matmuls, the tail
zero-filled by a concatenation), and draws count(cdf <= u * cdf[-1]): the threshold scales with the cdf's total,
so at any precision it stays under the last value and a draw never counts every code.

Run as the main script: the wrapper class FrameSig is then `__main__.FrameSig`, and that name is in the graph's tensor
names. LFM3-Audio-2B's speech graph code (lfm3audio_work/scripts/r1b_graphs.py) is loaded by path under the module name
`lfm3_peer_r1b_graphs`, which is in the tensor names too. WEIGHTS = the source model.safetensors.
"""
import importlib.util
import pathlib
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F

R = pathlib.Path(__file__).resolve().parents[1]
TEMPERATURE, PENALTY = 0.4, 1.2
NCB, V = 12, 2017


def load_by_path(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


class FrameSig(nn.Module):
  """depth_linear once + 3 passes + the runner's sampler; the draw = count(cdf <= u * cdf[-1]); codes out only."""

  def __init__(self, m):
    super().__init__()
    self.m = m
    self.register_buffer("tri64", torch.triu(torch.ones(64, 64)))             # [j, i] = 1 for j <= i
    self.register_buffer("tri32x", torch.triu(torch.ones(32, 32), diagonal=1))  # [j, i] = 1 for j < i
    self.register_buffer("temp", torch.tensor(TEMPERATURE, dtype=torch.float32))
    self.register_buffer("pen", torch.tensor(PENALTY, dtype=torch.float32))

  def draw(self, logits, u, presence):
    # logits [12, 2017], u [12, 1], presence [12, 2017] -> code per codebook (float)
    pen = F.relu(logits) / self.pen - F.relu(-logits) * self.pen
    s = presence * pen + (1.0 - presence) * logits
    z = s / self.temp
    z = z - z.amax(-1, keepdim=True)
    e = torch.exp(z)
    p = e / e.sum(-1, keepdim=True)
    pb = torch.cat([p, torch.zeros(NCB, 32 * 64 - V, dtype=p.dtype)], -1).reshape(NCB, 32, 64)
    inc = torch.matmul(pb, self.tri64)                       # prefix sums inside each block of 64
    pre = torch.matmul(inc[..., 63], self.tri32x)            # sums of the blocks before
    cdf = (inc + pre.unsqueeze(-1)).reshape(NCB, 32 * 64)[:, :V]
    # the threshold scales with the cdf's total instead of the cdf being divided by it
    idx = (cdf <= u * cdf[:, V - 1:V]).to(torch.float32).sum(-1)
    return torch.clamp(idx, max=float(V - 1))

  def forward(self, hidden, reveal0, reveal1, reveal2, u, presence):
    depth_in = self.m.project(hidden)
    got = torch.zeros(1, NCB, dtype=torch.float32)
    masked = torch.ones(1, NCB, dtype=torch.float32)
    for rv in (reveal0, reveal1, reveal2):
      lg = self.m.passes(depth_in, got.to(torch.int32), masked)[0]
      idx = self.draw(lg, u.reshape(NCB, 1), presence)
      got = got + rv * idx.reshape(1, NCB)
      masked = masked - rv
    return {"codes": got.to(torch.int32)}


def sample_inputs():
  order = torch.arange(NCB)
  reveal = torch.zeros(3, NCB)
  for p in range(3):
    reveal[p, order[4 * p:4 * p + 4]] = 1.0
  return {"hidden": torch.zeros(1, 2048), "reveal0": reveal[0:1], "reveal1": reveal[1:2], "reveal2": reveal[2:3],
          "u": torch.full((1, NCB), 0.5), "presence": torch.zeros(NCB, V)}


def main():
  if len(sys.argv) != 3:
    print(__doc__, file=sys.stderr)
    return 2
  weights, out = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve()
  assert __name__ == "__main__"
  G = load_by_path("lfm3_peer_r1b_graphs", R / "lfm3audio_work" / "scripts" / "r1b_graphs.py")
  step = G.DepthStep(G.load_depth_state_dict(str(weights), "hf"), "hf", split=True).eval()
  frame = FrameSig(step).eval()
  out.parent.mkdir(parents=True, exist_ok=True)
  tmp = out.with_name(out.name + ".part")
  t0 = time.time()
  G._export(frame, "depth_frame", sample_inputs(), tmp)  # pylint: disable=protected-access
  tmp.rename(out)
  print(f"exported {out} ({out.stat().st_size} B) in {time.time() - t0:.0f} s", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
