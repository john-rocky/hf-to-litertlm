#!/usr/bin/env python3
"""export_parts.py --part P --hf HF --out FILE [--out2 FILE2] - one LFM3-Audio part graph as an fp32 .tflite
(litert-torch 0.10.0), one part per process:

  frame       r1b_graphs.FrameEmbed: codes [1,12] int32 -> embeds [1,1,2048] (the sum of the 12 codebook embeddings
              that is the language model's next input during speech)
  audio       r4b_parts.AudioSectionR4b: 16 kHz PCM [1,3000,160] -> features [1,375,2048] + uint8 mask [1,375]
              (the `.litertlm` file's audio section, before its fully connected weights are cast to fp16)
  depth       r4b_parts.DepthProjR4b -> FILE (hidden [1,2048] -> depth_in [1,12,1024], once per speech frame) and
              r4b_parts.DepthPassR4b -> FILE2 (depth_in, codes [1,12], masked [1,12] -> logits [1,12,2017], three
              passes per frame reveal 4 + 4 + 4 codebooks)
  codec12     r4b_parts.CodecChunkR4b with 12 frames per call: codes [1,12,12] + streaming state -> 72 windowed frames
              of 1,280 samples (24 kHz audio after the host's overlap-add) + the next state
  depth_npu   DepthPassR4b traced with the Qualcomm HTP rewrites below (the source graph of the SM8850 NPU file)

The module names `r1b_graphs`, `r4b_parts` and `r1a_audio_section` reach the graphs' tensor names; they are loaded
under those names, as when the published files were exported.

depth_npu: the HTP runs float graphs in fp16, and an elementwise nonlinearity that reads a FULLY_CONNECTED output
directly loses accuracy there. While the graph is traced, F.silu(x) becomes x * s(x) and torch.sigmoid(x) becomes s(x)
with s(x) = exp(min(x, 0)) / (1 + exp(-|x|)) (no LOGISTIC in the graph), and torch.exp / cos / sin / log10 and F.gelu
read their input through a split + CONCATENATION on the last dim (an identity). In float32 the values stay those of
the plain graph within rounding.
"""
import argparse
import contextlib
import importlib.util
import pathlib
import sys
import time

import torch
import torch.nn.functional as F

K = pathlib.Path(__file__).resolve().parents[1]


def load_by_path(name, path, register=True):
  if register and name in sys.modules:
    return sys.modules[name]
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  if register:
    sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


_ORIG = {}
COUNT = {}


def _mat(x):
  """Identity through a split + CONCATENATION on the last dim (a dim of 1 is returned as is)."""
  n = x.shape[-1]
  if n < 2:
    return x
  h = n // 2
  return torch.cat([x[..., :h], x[..., h:]], dim=-1)


def _sig(x):
  e = _ORIG["exp"]
  return e(torch.clamp(x, max=0.0)) / (1.0 + e(-torch.abs(x)))


def _silu(x, inplace=False):
  del inplace
  COUNT["silu"] = COUNT.get("silu", 0) + 1
  return x * _sig(x)


def _sigmoid(x):
  COUNT["sigmoid"] = COUNT.get("sigmoid", 0) + 1
  return _sig(x)


def _gelu(x, approximate="none"):
  COUNT["gelu"] = COUNT.get("gelu", 0) + 1
  return _ORIG["gelu"](_mat(x), approximate=approximate)


def _wrap(name):
  def f(x, *a, **kw):
    COUNT[name] = COUNT.get(name, 0) + 1
    return _ORIG[name](_mat(x), *a, **kw)
  return f


@contextlib.contextmanager
def htp_swaps():
  """The rewrites of the module docstring, for the duration of one trace (restored after)."""
  _ORIG.update({"silu": F.silu, "sigmoid": torch.sigmoid, "gelu": F.gelu, "log10": torch.log10, "exp": torch.exp,
                "cos": torch.cos, "sin": torch.sin})
  COUNT.clear()
  F.silu, torch.sigmoid = _silu, _sigmoid
  F.gelu = _gelu
  torch.log10, torch.exp, torch.cos, torch.sin = _wrap("log10"), _wrap("exp"), _wrap("cos"), _wrap("sin")
  try:
    yield COUNT
  finally:
    F.silu, torch.sigmoid, F.gelu = _ORIG["silu"], _ORIG["sigmoid"], _ORIG["gelu"]
    torch.log10, torch.exp, torch.cos, torch.sin = _ORIG["log10"], _ORIG["exp"], _ORIG["cos"], _ORIG["sin"]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--part", required=True, choices=["frame", "audio", "depth", "codec12", "depth_npu"])
  ap.add_argument("--hf", required=True, help="the source repository folder")
  ap.add_argument("--out", required=True)
  ap.add_argument("--out2", help="depth: the depth_pass file")
  a = ap.parse_args()
  hf = pathlib.Path(a.hf).resolve()
  main_st, codec_st = hf / "model.safetensors", hf / "audio_tokenizer" / "model.safetensors"
  out = pathlib.Path(a.out).resolve()
  out.parent.mkdir(parents=True, exist_ok=True)
  t0 = time.time()
  if a.part == "frame":
    G = load_by_path("r1b_graphs", K / "scripts" / "r1b_graphs.py", register=False)
    G.export_frame_embed(G.load_frame_state_dict(main_st, "audio2b"), out, "audio2b")
  else:
    P = load_by_path("r4b_parts", K / "scripts" / "r4b_parts.py")
    if a.part == "audio":
      sec, _, n = P.build_audio_section(str(hf))
      P.export_audio(sec, out)
      print(f"audio section: {n} tensors loaded", flush=True)
    elif a.part == "depth":
      assert a.out2, "--out2 (the depth_pass file) is required for --part depth"
      sd = P.G.load_depth_state_dict(main_st)
      P.export_depth_split(sd, out, pathlib.Path(a.out2).resolve())
    elif a.part == "codec12":
      sd = P.G.load_codec_state_dict(codec_st)
      P.export_codec(sd, 12, out)
    else:
      import litert_torch  # pylint: disable=g-import-not-at-top
      with htp_swaps() as count:
        sd = P.G.load_depth_state_dict(main_st)
        step = P.DepthStepR4b(sd).eval()
        kw = {"depth_in": torch.zeros(1, P.G.NUM_CB, P.G.DEPTH_HIDDEN),
              "codes": torch.zeros(1, P.G.NUM_CB, dtype=torch.int32), "masked": torch.ones(1, P.G.NUM_CB)}
        litert_torch.signature("depth_pass", P.DepthPassR4b(step).eval(), sample_kwargs=kw).convert().export(str(out))
        calls = dict(count)
      print(f"HTP rewrites traced: {calls}", flush=True)
  for p in [out] + ([pathlib.Path(a.out2).resolve()] if a.part == "depth" else []):
    print(f"{p}: {p.stat().st_size} B ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
  main()
