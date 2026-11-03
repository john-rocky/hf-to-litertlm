"""r4 torch modules: the pre-encode (subsampling) fused into the conformer graph, so the Galaxy S26's NPU runs one graph
per chunk (its just-in-time compile takes one graph per process). graphs.py stays as is (CONTRACT
v1); this file only adds modules on top of it.

PreMasked  graphs.Pre (the steady chunk: mel [1, 32, 128] + the three 1-row caches) with two row masks after the first
           two ReLUs: pre_m0 [1, 1, 16, 1] on conv_in's output rows, pre_m1 [1, 1, 8, 1] on stage 1's output rows.
           Chunk >= 1: both masks 1 -> graphs.Pre exactly (x * 1.0 is exact).
           Chunk 0 through the same graph (the exact first-chunk path):
             mel   = 7 zero rows + mel_first's 25 rows ([1, 32, 128]); cache0 / cache1 / cache2 = 0
             pre_m0 = [0, 0, 0, 1 x 13], pre_m1 = [0, 1 x 7]
           Why: graphs.PreFirst pads 2 zero rows in front of EACH of the three convolutions (stride 2, kernel 3), i.e. in
           the post-ReLU domain at stages 1 and 2. Fed the zero rows through the mel, the steady graph's conv_in turns
           them into ReLU(bias) rows, not zeros; the masks zero exactly the rows PreFirst pads (conv_in rows 0..2 =
           the 3 rows in front of PreFirst's row 0, stage 1 row 0). Row alignment: steady conv_in row p covers
           [cache0, mel] rows 2p..2p+2 -> PreFirst's conv_in row 0 (pad, pad, mel_first row 0) = steady row 3 (zero,
           zero, mel row 7); stage 1 row r covers conv_in rows 2r-1..2r+1 -> PreFirst's stage-1 row 0 = steady row 1;
           stage 2 row k covers stage-1 rows 2k-1..2k+1 -> PreFirst's stage-2 row k = steady row k (4 rows each). The
           cache outputs at chunk 0 are the last rows = PreFirst's c0 / c1 / c2.
ConformerPre  PreMasked -> graphs.Conformer in one graph:
           inputs  mel [1,32,128], cache0 [1,1,1,131], cache1 [1,256,1,68], cache2 [1,256,1,36], pre_m0 [1,1,16,1],
                   pre_m1 [1,1,8,1], neg_mask [1,1,4,60], k_cache / v_cache [24,8,56,128], conv_cache [24,1024,8],
                   one_hot [1,128]
           outputs enc_proj [1,4,640], k_out, v_out, conv_out, cache0_out, cache1_out, cache2_out
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import graphs as G
from contract_r4 import M0_ZERO, M1_ZERO, PAD_ROWS, T_M0, T_M1, first_mel, masks  # noqa: F401  (host rules)


class PreMasked(nn.Module):
    def __init__(self, sub):
        super().__init__()
        self.sub = sub
        self.stats = None          # dict -> per point max|x| (range pass, None at export)
        self.bound = None          # dict -> per op max over outputs of sum |w x| (any summation order's partial sums)

    def _st(self, name, t):
        if self.stats is None:
            return
        a = float(t.detach().abs().max())
        self.stats[name] = max(a, self.stats.get(name, 0.0))

    def _bd(self, name, conv, x, groups=1, linear=False):
        """max over outputs of sum_i |w_i x_i| (+ |bias|): bounds every partial sum of the op, whatever the order."""
        if self.bound is None:
            return
        with torch.no_grad():
            if linear:
                b = F.linear(x.abs(), conv.weight.abs(), conv.bias.abs() if conv.bias is not None else None)
            else:
                b = F.conv2d(x.abs(), conv.weight.abs(), conv.bias.abs() if conv.bias is not None else None,
                             stride=conv.stride, groups=groups)
        a = float(b.max())
        self.bound[name] = max(a, self.bound.get(name, 0.0))

    def forward(self, mel, cache0, cache1, cache2, pre_m0, pre_m1):          # [1, 32, 128] + caches + masks
        st = self._st
        st("mel", mel)
        xf = G._freq_pad(mel.unsqueeze(1))                                     # [1, 1, 32, 131]
        c0 = xf[:, :, -1:, :]
        xi = torch.cat([cache0, xf], dim=2)                                    # [1, 1, 33, 131]
        self._bd("conv_in", self.sub.conv_in, xi)
        a = self.sub.conv_in(xi)
        st("conv_in", a)
        h = torch.relu(a) * pre_m0                                             # [1, 256, 16, 65]
        s1, s2 = self.sub.layers
        h1f = G._freq_pad(h)
        c1 = h1f[:, :, -1:, :]
        x1 = torch.cat([cache1, h1f], dim=2)                                   # [1, 256, 17, 68]
        self._bd("s1_dw", s1.depthwise_conv, x1, groups=G.SUB_CH)
        d1 = s1.depthwise_conv(x1)
        st("s1_dw", d1)
        self._bd("s1_pw", s1.pointwise_conv, d1)
        p1 = s1.pointwise_conv(d1)
        st("s1_pw", p1)
        h = torch.relu(p1) * pre_m1                                            # [1, 256, 8, 33]
        h2f = G._freq_pad(h)
        c2 = h2f[:, :, -1:, :]
        x2 = torch.cat([cache2, h2f], dim=2)                                   # [1, 256, 9, 36]
        self._bd("s2_dw", s2.depthwise_conv, x2, groups=G.SUB_CH)
        d2 = s2.depthwise_conv(x2)
        st("s2_dw", d2)
        self._bd("s2_pw", s2.pointwise_conv, d2)
        p2 = s2.pointwise_conv(d2)
        st("s2_pw", p2)
        h = torch.relu(p2)                                                     # [1, 256, 4, 17]
        h = h.transpose(1, 2).reshape(1, h.shape[2], -1)
        self._bd("linear", self.sub.linear, h, linear=True)
        e = self.sub.linear(h)
        st("embeds", e)
        return e, c0, c1, c2


class ConformerPre(nn.Module):
    def __init__(self, sub, conf):
        super().__init__()
        self.pre = PreMasked(sub)
        self.conf = conf

    def forward(self, mel, cache0, cache1, cache2, pre_m0, pre_m1, neg_mask, k_cache, v_cache, conv_cache, one_hot):
        e, c0, c1, c2 = self.pre(mel, cache0, cache1, cache2, pre_m0, pre_m1)
        enc, k, v, c = self.conf(e, neg_mask, k_cache, v_cache, conv_cache, one_hot)
        return enc, k, v, c, c0, c1, c2


class ExConformerPre(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, mel, cache0, cache1, cache2, pre_m0, pre_m1, neg_mask, k_cache, v_cache, conv_cache, one_hot):
        enc, k, v, c, c0, c1, c2 = self.m(mel, cache0, cache1, cache2, pre_m0, pre_m1, neg_mask, k_cache, v_cache,
                                          conv_cache, one_hot)
        return {"enc_proj": enc, "k_out": k, "v_out": v, "conv_out": c,
                "cache0_out": c0, "cache1_out": c1, "cache2_out": c2}


def load_sub(path=G.WEIGHTS):
    """Subsampling only (the encoder.subsampling.* tensors of the pinned checkpoint, strict)."""
    from safetensors import safe_open
    sub = G.Subsampling()
    sd = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.startswith("encoder.subsampling."):
                sd[k[len("encoder.subsampling."):]] = f.get_tensor(k).to(torch.float32)
    sub.load_state_dict(sd, strict=True)
    return sub.eval()
