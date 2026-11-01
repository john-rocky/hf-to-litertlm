# Sampling inside a graph that runs at float16

A speech model that writes several codebooks per frame calls its depth decoder many times per second of audio, and each call ends in a sampling step on the host. Moving the sampling into the graph saves those trips to the host and lets the phone GPU read back only the codes. At float16, which is the GPU's default precision, the obvious way to draw from a categorical distribution inside a graph fails in a way that does not show at float32. The case on record is LFM3-Omni-2B's depth decoder (`speech/LFM3-Omni-2B_depth_frame2_f32.tflite` in [litert-community/LFM3-Omni-2B-LiteRT](https://huggingface.co/litert-community/LFM3-Omni-2B-LiteRT), built by [`lfm3omni_work/export_depth_frame2.py`](../lfm3omni_work/export_depth_frame2.py); the REPRODUCE.md entry "LFM3-Omni-2B"): 12 codebooks of 2,017 codes per 80 ms frame, 3 passes per frame, and code 2016 means the end of the speech.

1. **Take the host's random draws as inputs, in the host's order.** The host sampler draws the reveal order of the codebooks first and then one uniform per codebook in that order (numpy's `Generator.choice(p=p)` is a search for `random()` in the normalised cumulative sum and takes exactly one `random()` per draw). The graph gets the three reveal rows as three `[1, 12]` inputs and the 12 uniforms as one `[1, 12]` input, so the graph and the host sampler give the same codes from the same seed. Slicing one `[3, 12]` input per pass instead gave wrong results in the later passes on the Mac GPU.
2. **Write the sampler from ops every delegate takes.** The repetition penalty (a positive logit divided by 1.2, a negative one multiplied by 1.2, on the codes of that codebook's last 64 frames) is `relu(s) / 1.2 − relu(−s) · 1.2`, blended in by a 0 / 1 presence input, so a logit without a penalty passes through exactly and no SELECT is needed. Temperature, max-subtraction, `exp` and the softmax follow. The cumulative sum is a blocked prefix sum: the 2,017 probabilities padded to 32 × 64 with a CONCATENATION of zeros (Metal refuses a rank-2 PAD: `Invalid paddings tensor shape: expected 4x2 or 3x2, got 2x2`, and the GPU / CPU split it left made the third pass wrong), one matmul with an upper-triangular 64 × 64 matrix for the sums inside each block and one with a strictly upper-triangular 32 × 32 matrix for the sums of the blocks before, instead of CUMSUM. The draw is a count, `sum(cdf ≤ threshold)`, clamped to the last code.
3. **Cap u on the host, and scale the threshold, not the cdf.** At float16 a uniform of 0.999756 or more rounds to 1.0; every code then passes the comparison, and the clamp turns the count into the last code, the end of the speech. The first one-frame graph (the cdf divided by its last value and compared with u, no cap) did exactly that on the Mac GPU at Metal's default float16: with u = 0.9998, 2,246 of 2,412 draws gave the last code (19 at float32, the same count the host sampler gives for that u: end-of-speech draws the model itself makes). The shipped form changes two things, and only the two together were measured on a phone GPU: the host caps each uniform at 1 − 2^-11 (the largest float16 value below 1), and the graph compares `cdf ≤ u × cdf[−1]`, so the threshold scales with the cdf's total instead of the cdf being divided by it.
4. **Read back only the codes.** The graph's only output is the 12 codes as int32; the logits of the three passes stay inside, so each call copies back 12 integers, not also the 3 × 12 × 2,017 logits that the first one-frame graph returned.
5. **Gate it on the task, on the device.** Codes that differ from the host sampler's where a uniform falls on the boundary between two codes are expected (the graph sums the probabilities in blocks at float32 or float16, the host in float64); judge the graph by the speech it makes: transcripts, length and a listening score over many utterances. On the record, the first graph without the cap stopped some speech early at Metal's float16 (word error rate 3.404 %). The shipped form, with both parts of step 3, made speech on a Galaxy S26 GPU at float16 that passed the same test as the first graph, with the cap, on the same phone's CPU (114 answers: word error rate 0.000 % for both, UTMOS22 4.481 against 4.471, 21 of 24 spoken replies right for both). The numbers per phone are on the model card.

The sampler as exported (from `FrameSig.draw` in [`lfm3omni_work/export_depth_frame2.py`](../lfm3omni_work/export_depth_frame2.py); `NCB = 12`, `V = 2017`, `tri64 = triu(ones(64, 64))`, `tri32x = triu(ones(32, 32), diagonal=1)`):

```python
def draw(self, logits, u, presence):
    # logits [12, 2017], u [12, 1], presence [12, 2017] -> code per codebook (float)
    pen = F.relu(logits) / self.pen - F.relu(-logits) * self.pen
    s = presence * pen + (1.0 - presence) * logits
    z = s / self.temp
    z = z - z.amax(-1, keepdim=True)
    e = torch.exp(z)
    p = e / e.sum(-1, keepdim=True)
    pb = torch.cat([p, torch.zeros(NCB, 32 * 64 - V, dtype=p.dtype)], -1).reshape(NCB, 32, 64)
    inc = torch.matmul(pb, self.tri64)                 # prefix sums inside each block of 64
    pre = torch.matmul(inc[..., 63], self.tri32x)      # sums of the blocks before
    cdf = (inc + pre.unsqueeze(-1)).reshape(NCB, 32 * 64)[:, :V]
    idx = (cdf <= u * cdf[:, V - 1:V]).to(torch.float32).sum(-1)
    return torch.clamp(idx, max=float(V - 1))
```
