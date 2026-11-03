"""A drop-in for torch.nn.functional.conv1d on depthwise kernels (groups = channels, one input channel per group,
stride 1, dilation 1), for the upstream reference runs on the Mac CPU.

Why: the GatedDeltaNet layers of the Qwen3.5 text model run a depthwise causal conv1d over 6,144 channels (kernel 4).
On macOS arm64 PyTorch 2.14.1 sends it to aten::_slow_conv2d_forward once per channel: one decode step of
MiniCPM-OCR-1.0 made 110,592 such calls (18 layers x 6,144) and spent 9.2 of its 9.9 s there (a torch
profiler trace). Here y[:, c, t] = bias[c] + sum_k w[c, 0, k] * xpad[:, c, t + k], the same products and the
same left-to-right sum over the K taps, as K shifted multiply-adds over the whole tensor.

install() patches torch.nn.functional.conv1d (nn.Conv1d calls it at run time); every other call goes to the
original. check() compares the two on random inputs of the GatedDeltaNet shapes.
"""
import torch
import torch.nn.functional as F

_orig = F.conv1d


def _is_depthwise(x, w, stride, dilation, groups):
  def one(v):
    return v == 1 or v == (1,) or v == [1]
  return (x.dim() == 3 and w.dim() == 3 and w.shape[1] == 1 and groups == x.shape[1] == w.shape[0]
          and one(stride) and one(dilation))


def dwconv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):  # noqa: A002 (F.conv1d names)
  x, w = input, weight
  if not _is_depthwise(x, w, stride, dilation, groups) or isinstance(padding, str):
    return _orig(x, w, bias, stride, padding, dilation, groups)
  p = padding if isinstance(padding, int) else padding[0]
  if p:
    x = F.pad(x, (p, p))
  K = w.shape[-1]
  L = x.shape[-1] - K + 1
  wk = w[:, 0, :]                                     # [C, K]
  y = x[:, :, 0:L] * wk[None, :, 0:1]
  for k in range(1, K):
    y = y + x[:, :, k:k + L] * wk[None, :, k:k + 1]
  if bias is not None:
    y = y + bias[None, :, None]
  return y


def install():
  F.conv1d = dwconv1d
  torch.nn.functional.conv1d = dwconv1d


def check(seed=0):
  g = torch.Generator().manual_seed(seed)
  out = {}
  for name, L, pad in (("prefill", 300, 3), ("decode_window", 5, 0)):
    x = torch.randn(1, 6144, L, generator=g)
    w = torch.randn(6144, 1, 4, generator=g) * 0.3
    b = torch.randn(6144, generator=g) * 0.1
    for bias in (None, b):
      ref = _orig(x, w, bias, 1, pad, 1, 6144)
      got = dwconv1d(x, w, bias, 1, pad, 1, 6144)
      out[f"{name}{'_bias' if bias is not None else ''}"] = {
          "max_abs": float((ref - got).abs().max()), "ref_absmax": float(ref.abs().max()),
          "bit_equal": bool(torch.equal(ref, got))}
  return out


if __name__ == "__main__":
  import json
  print(json.dumps(check(), indent=1))
