# fp16-safe RMSNorm for raw LiteRT graphs

A graph that is right on the GPU at fp32 precision and collapses at fp16 (the GPU's default precision, `FP16_WITH_FP32_ACCUM`, an NPU's fp16 arithmetic) often overflows inside its norms: an RMSNorm squares every value of the row and sums them, and fp16 stops at 65,504. The case on record is EmbeddingGemma 2's audio encoder (12 conformer layers, hidden 1024), converted to [mlboydaisuke/EmbeddingGemma-2-Transcribe-LiteRT](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-LiteRT) by [`scripts/eg2_transcribe/`](../scripts/eg2_transcribe/) (the REPRODUCE.md entry "EmbeddingGemma 2 Transcribe"). Of its 111 norm sites, 76 have a row sum of squares above 65,504 (up to 436,697, 6.7 times) and 33 have an element whose square alone exceeds it (|x| up to 401.8); the mean of squares fits everywhere (at most 426.5). Metal at its default precision returned soft tokens with a mean cosine of 0.178 to the fp32 output, with no error. These are the steps that made the graph build and run at fp16 without changing a single fp32 bit. For LayerNorm the same idea is step 4 of [vision-tower-reauthor.md](vision-tower-reauthor.md).

1. **Find the site class on the host, before the device.** Round the tensors of one class of site to float16 and back in torch and compare the output with fp32. Here the norms alone reproduced the device's collapse (mean cosine 0.152, minimum −0.239; Metal 0.178 / −0.254), while rounding the 135 fully connected outputs gave 0.999999 / 0.998 and the 84 attention intermediates 0.999999 / 0.9973. Record, per norm site, the largest row sum of x² and the largest |x| over every row the graph computes, padding rows included: they run through the graph too.
2. **Rescale each norm by a static power of two.** Multiply the norm's input by 2^-k and its eps by 4^-k. Powers of two commute with float32 rounding, so the output is x · rsqrt(mean(x²) + eps) · w to the bit, and the graph gains one MUL per site (plus the clamp of step 3) and no reduction. k is per site: the smallest k ≥ 0 that leaves 4× headroom under 65,504 for both the largest row sum of x² and the largest single x² over the whole check set. Both matter: the sum overflowed at 76 sites and x² alone at 33. Of the 109 RMSNorms here, k = 0 at 8, 1 at 25, 2 at 72 and 3 at 4; the two LayerNorms needed none. Dividing each row by its own runtime max also avoids the overflow, but it is not bit-exact in float32 and it puts a reduce-max in the graph.
3. **Clamp the rsqrt for padding rows.** eps·4^-k can drop out of fp16's range: eps 1e-6 at k = 3 is 1.6e-8, below half the smallest fp16 subnormal, so it rounds to 0, an all-zero row becomes rsqrt(0) = inf, and 0 × inf = NaN. Clamp the rsqrt at 2^k/√eps, the zero row's own value (times 1 + 2^-20, so that float32 never reaches it). eps 1e-6 is itself a subnormal in fp16: emulating a device that flushes subnormals to zero gave NaN on padding rows. Keep the norm as plain ops; a fused norm op has no place for the clamp.
4. **Keep the attention mask constant finite.** −1e9 is −inf in fp16, and a mask applied as (1 − valid) × c gives 0 × −inf = NaN on every valid position. −30000 is exact in fp16. With the logits bounded (this model's softcap holds them to ±50) the masked exp is 0 in float32 with either constant, so the float32 output does not move.
5. **No boolean on the way to a graph output.** An output mask computed as `(mask > 0.5)` and cast back to float made the GPU's fp16 kernel build fail on Adreno and Mali (OpenCL, `write_imageh`) and on Metal (`write`), at the same kernel source line, while fp32 precision compiled. The same output computed from the float input mask by float ops (a strided slice twice, for the two stride-2 convolutions) compiled at fp16 on all three and stayed bit-exact in float32. Small graphs with the same GREATER → CAST pair compiled, so bisect on the real graph. Keep masks float (1 = valid) from input to output.
6. **Check in this order.** (a) torch float32 before vs after the rewrite: max |Δ| 0 on every row of every window, padding rows included. (b) The converted file on the host CPU against the unrewritten file: bit-equal (here also on two phones' CPUs). (c) The device at each precision on the whole check set, long inputs included. The rewrite removes the overflow, not fp16's arithmetic error: Metal's default precision went from a mean cosine of 0.178 to 0.99977, but its per-token minimum was 0.9027 against a bar of 0.99. On a Galaxy S26 a 9-row subset passed the NPU (per-token minimum 0.99558), and the whole set (41–90 windows, long recordings included) then failed it (0.773) and the GPU at `FP16_WITH_FP32_ACCUM` (0.884). The published files run the GPU at fp32 precision; fp16 is an option the rewrite opens, decided per form on the whole set.

The module, as converted (`F16SafeRMSNorm` in [`scripts/eg2_transcribe/export_encoder.py`](../scripts/eg2_transcribe/export_encoder.py), whose `kmap` step computes k):

```python
class F16SafeRMSNorm(torch.nn.Module):
    """x * rsqrt(mean(x^2) + eps) * w computed on x * 2^-k with eps * 4^-k; exact in fp32 (powers of two)."""
    def __init__(self, weight, eps, k):
        super().__init__()
        self.weight, self.k = weight, int(k)
        self.scale, self.eps_k = 2.0 ** -self.k, eps * 4.0 ** -self.k
        self.r_max = (2.0 ** self.k) / math.sqrt(eps) * (1.0 + 2.0 ** -20)   # a zero row's own value, never reached in fp32

    def forward(self, x):
        if self.k:
            x = x * self.scale
        r = torch.clamp(torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps_k), max=self.r_max)
        y = x * r
        return y * self.weight if self.weight is not None else y
```
