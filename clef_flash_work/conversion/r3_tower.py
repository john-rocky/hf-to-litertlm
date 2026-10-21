"""ClefTower: the vision tower and merger of Cloudflare/clef-flash (`model.visual` of Qwen3_5ForConditionalGeneration:
patch embed, learned position embedding, 27 blocks, merger) at ONE fixed grid, written for the LiteRT export. Plain
torch (no transformers), so the same module runs in venv-ref (the torch check against the author's module) and in
lt094dev (the export).

    forward(patches f32 [1, N, 1536]) -> {"rows": f32 [1, N/4, 4096]}

patches = the processor's pixel_values of one still image, merge-block-major (scripts/r3_host.py tile_patches); rows =
the merger output = the author's image_embeds (what masked_scatter writes at the image-pad positions of the row graph's
embeds). g256: N = 256 (16 x 16 patches) -> 64 rows; g448: N = 784 (28 x 28) -> 196 rows.

The author's forward (transformers 5.17.0 Qwen3_5VisionModel, config vision_config: depth 27, hidden 1152, heads 16,
FFN 4304, gelu_pytorch_tanh, out 4096, patch 16, temporal 2, merge 2, no deepstack) and what changes here:
  patch_embed   Conv3d(3, 1152, kernel = stride = (2, 16, 16)) on the [N, 3, 2, 16, 16] view of the patches
                -> one FULLY_CONNECTED [1536 -> 1152] with the conv weight flattened in the same (C, T, 16, 16) order
                   (the same dot products; no temporal summing, no conv op)
  pos_embed     (pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1) from 5.17.0's bilinear taps of the
                48 x 48 table -> a constant [1, N, 1152] computed by the author's own code in venv-ref
                (scripts/r3_tower_torch.py -> cache/r3/consts_<grid>.npz), added to the patch rows
  rotary        Qwen3_5VisionRotaryEmbedding(position_ids of get_vision_position_ids, merge-block-major) -> constants
                cos / sin [1, N, 1, 72] (the same file); q / k rotated in float32 as apply_rotary_pos_emb_vision
  attention     sdpa over one image (cu_seqlens = [0, N]) -> explicit rank-4 matmuls: softmax((q * scale) k^T) v with
                q / k / v [1, 16, N, 72] from three FULLY_CONNECTED (the qkv weight split at its (3, heads, 72) rows)
  blocks        pre-norm LayerNorm(eps 1e-6) -> attention -> residual; LayerNorm -> fc1 -> gelu(tanh) -> fc2 -> residual
  merger        LayerNorm(1152) per patch -> reshape [1, N/4, 4608] (4 consecutive patches = one 2 x 2 block in the
                merge-block-major order: a plain reshape, no strided slice, no gather) -> fc1 -> GELU (erf) -> fc2
Every activation keeps the leading batch dim (rank 3, attention rank 4); no GATHER / SELECT_V2 / BROADCAST_TO / INT64.
The weights come from the checkpoint's `model.visual.*` tensors (bf16) widened to float32 (load_visual_state)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HIDDEN, HEADS, HEAD_DIM, FFN, OUT, DEPTH = 1152, 16, 72, 4304, 4096, 27
PATCH_DIM = 3 * 2 * 16 * 16
MERGE_UNIT = 4
EPS = 1e-6
PREFIX = "model.visual."
GRIDS = {"g256": 16, "g448": 28}          # patches per side
# The graph form: every option computes the same values in torch (bit for bit, scripts/r3_tower_tfl.py checks it);
# they differ only in the ops the converter emits.
#   rope   "bcast"    q / k [1, N, H, d] x cos / sin constants [1, N, 1, d] (broadcast over the heads)
#          "full"     the constants pre-broadcast to [1, N, H, d] (no broadcast in the MUL)
#          "bhnd"     the rotation after the transpose: q / k [1, H, N, d] x constants [1, 1, N, d]
#          "fullbhnd" after the transpose with [1, H, N, d] constants
#          "none"     no rotation (a diagnostic form only: it changes the function)
#   rot    "slice"    rotate_half = cat(-x[..., d/2:], x[..., :d/2])
#          "mm"       rotate_half = x @ R, R the constant [d, d] signed permutation (each output = one exact product)
#   merge  "reshape"  [1, N, C] -> [1, N/4, 4C] (4 consecutive patches = one 2 x 2 block, merge-block-major order)
#          "strided"  cat([x[:, i::4, :] for i in 0..3], -1) (strided slices + concat, no channel-changing reshape)
#          "split4"   [1, N, C] -> [1, N/4, 4, C] (C kept), member i by a contiguous slice, merger fc1 = the sum of
#                     the 4 FCs of its column blocks (the same products, summed in another order: float32
#                     rounding only, not bit for bit)
#   pos    "const"    patch FC output [1, N, C] + the position constant [1, N, C] (rank 3)
#          "pos4"     the add at rank 4: [1, 1, N, C] + a [1, 1, N, C] constant, then back to [1, N, C]
#          "posin"    the position rows as a second graph input `pos` [1, N, C] (the host feeds the grid's table)
#   (Metal 2.2.0 computes the "const" add wrong with every node delegated: results/r3_metal_bisect.json, component
#   c_patch_fc_pos_n256 cos 0.02; the MLDrift 4D-layout class of LiteRT #10445 / #10231, tools/RANK3_PRELINT.md.)
DEFAULT_FORM = {"rope": "bcast", "rot": "slice", "merge": "reshape", "pos": "const"}


def load_visual_state(snapshot: str | Path, dtype=torch.float32) -> dict:
    """{key without 'model.visual.': tensor} of the checkpoint's vision tensors, bf16 widened to `dtype`."""
    from safetensors import safe_open

    snapshot = Path(snapshot)
    wm = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    keys = sorted(k for k in wm if k.startswith(PREFIX))
    out = {}
    for shard in sorted({wm[k] for k in keys}):
        with safe_open(str(snapshot / shard), framework="pt") as f:
            for k in keys:
                if wm[k] == shard:
                    out[k[len(PREFIX):]] = f.get_tensor(k).to(dtype)
    return out


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotate_matrix(d: int, dtype=torch.float32) -> torch.Tensor:
    """R [d, d] with x @ R == rotate_half(x): output j < d/2 takes -x[j + d/2], output j >= d/2 takes x[j - d/2]."""
    r = torch.zeros(d, d, dtype=dtype)
    h = d // 2
    for j in range(h):
        r[j + h, j] = -1.0
        r[j, j + h] = 1.0
    return r


def _linear(w: torch.Tensor, b: torch.Tensor | None) -> nn.Linear:
    m = nn.Linear(w.shape[1], w.shape[0], bias=b is not None, dtype=w.dtype)
    m.weight = nn.Parameter(w.detach().clone(), requires_grad=False)
    if b is not None:
        m.bias = nn.Parameter(b.detach().clone(), requires_grad=False)
    return m


def _norm(w: torch.Tensor, b: torch.Tensor) -> nn.LayerNorm:
    m = nn.LayerNorm(w.shape[0], eps=EPS, dtype=w.dtype)
    m.weight = nn.Parameter(w.detach().clone(), requires_grad=False)
    m.bias = nn.Parameter(b.detach().clone(), requires_grad=False)
    return m


class Block(nn.Module):
    def __init__(self, sd: dict, i: int, hidden: int, heads: int, form: dict | None = None):
        super().__init__()
        p = f"blocks.{i}."
        self.form = dict(DEFAULT_FORM, **(form or {}))
        self.hidden, self.heads, self.head_dim = hidden, heads, hidden // heads
        self.scaling = self.head_dim ** -0.5
        self.norm1 = _norm(sd[p + "norm1.weight"], sd[p + "norm1.bias"])
        self.norm2 = _norm(sd[p + "norm2.weight"], sd[p + "norm2.bias"])
        w, b = sd[p + "attn.qkv.weight"], sd[p + "attn.qkv.bias"]
        self.q = _linear(w[0:hidden], b[0:hidden])
        self.k = _linear(w[hidden:2 * hidden], b[hidden:2 * hidden])
        self.v = _linear(w[2 * hidden:3 * hidden], b[2 * hidden:3 * hidden])
        self.proj = _linear(sd[p + "attn.proj.weight"], sd[p + "attn.proj.bias"])
        self.fc1 = _linear(sd[p + "mlp.linear_fc1.weight"], sd[p + "mlp.linear_fc1.bias"])
        self.fc2 = _linear(sd[p + "mlp.linear_fc2.weight"], sd[p + "mlp.linear_fc2.bias"])

    def rotate(self, x, cos, sin, rmat):
        if self.form["rope"] == "none":
            return x
        r = torch.matmul(x, rmat) if self.form["rot"] == "mm" else rotate_half(x)
        return x * cos + r * sin

    def forward(self, h, cos, sin, rmat=None):              # h [1, N, C]; cos / sin in the form's layout
        n = h.shape[1]
        x = self.norm1(h)
        q = self.q(x).reshape(1, n, self.heads, self.head_dim)
        k = self.k(x).reshape(1, n, self.heads, self.head_dim)
        v = self.v(x).reshape(1, n, self.heads, self.head_dim)
        after = self.form["rope"] in ("bhnd", "fullbhnd")
        if not after:
            q, k = self.rotate(q, cos, sin, rmat), self.rotate(k, cos, sin, rmat)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)        # [1, H, N, d]
        if after:
            q, k = self.rotate(q, cos, sin, rmat), self.rotate(k, cos, sin, rmat)
        s = torch.matmul(q * self.scaling, k.transpose(2, 3))                     # [1, H, N, N]
        o = torch.matmul(torch.softmax(s, dim=-1), v)                             # [1, H, N, d]
        o = o.transpose(1, 2).reshape(1, n, self.hidden)
        h = h + self.proj(o)
        return h + self.fc2(F.gelu(self.fc1(self.norm2(h)), approximate="tanh"))


class ClefTower(nn.Module):
    def __init__(self, sd: dict, pos_embeds: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, heads: int = HEADS,
                 form: dict | None = None):
        """sd = load_visual_state(...) (any float dtype); pos_embeds [N, C], cos / sin [N, d] = the grid's constants;
        form = DEFAULT_FORM with any option replaced (the notes at DEFAULT_FORM)."""
        super().__init__()
        self.form = dict(DEFAULT_FORM, **(form or {}))
        w = sd["patch_embed.proj.weight"]                                   # [C, 3, 2, 16, 16]
        hidden = w.shape[0]
        n = pos_embeds.shape[0]
        assert n % MERGE_UNIT == 0, n
        self.n, self.hidden = n, hidden
        self.patch = _linear(w.reshape(hidden, -1), sd["patch_embed.proj.bias"])
        pe = pos_embeds.to(w.dtype)
        self.pos_feed = pe.reshape(1, n, hidden).clone()       # what the host feeds when pos = "posin"
        if self.form["pos"] == "pos4":
            self.register_buffer("pos", pe.reshape(1, 1, n, hidden).clone(), persistent=False)
        elif self.form["pos"] == "const":
            self.register_buffer("pos", pe.reshape(1, n, hidden).clone(), persistent=False)
        d = cos.shape[-1]
        shape = {"bcast": (1, n, 1, d), "full": (1, n, heads, d), "bhnd": (1, 1, n, d), "fullbhnd": (1, heads, n, d),
                 "none": (1, n, 1, d)}[self.form["rope"]]
        for name, t in (("cos", cos), ("sin", sin)):
            t = t.to(w.dtype)
            t = (t.reshape(1, 1, n, d) if self.form["rope"] in ("bhnd", "fullbhnd") else t.reshape(1, n, 1, d)).expand(*shape)
            self.register_buffer(name, t.contiguous().clone(), persistent=False)
        self.register_buffer("rmat", rotate_matrix(d, w.dtype), persistent=False)
        depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        self.blocks = nn.ModuleList(Block(sd, i, hidden, heads, self.form) for i in range(depth))
        self.norm = _norm(sd["merger.norm.weight"], sd["merger.norm.bias"])
        self.fc1 = _linear(sd["merger.linear_fc1.weight"], sd["merger.linear_fc1.bias"])
        self.fc2 = _linear(sd["merger.linear_fc2.weight"], sd["merger.linear_fc2.bias"])
        if self.form["merge"] == "split4":
            w1 = sd["merger.linear_fc1.weight"]
            self.fc1_parts = nn.ModuleList(_linear(w1[:, i * hidden:(i + 1) * hidden], None) for i in range(MERGE_UNIT))

    def feed(self, patches) -> dict:
        """The graph inputs for one image: {"patches"} (+ {"pos"} when pos = "posin")."""
        f = {"patches": patches}
        if self.form["pos"] == "posin":
            f["pos"] = self.pos_feed
        return f

    def forward(self, patches, pos=None):                                         # [1, N, 1536] (+ [1, N, C])
        if self.form["pos"] == "posin":
            h = self.patch(patches) + pos
        elif self.form["pos"] == "pos4":
            h = (self.patch(patches).reshape(1, 1, self.n, self.hidden) + self.pos).reshape(1, self.n, self.hidden)
        else:
            h = self.patch(patches) + self.pos
        rmat = self.rmat if self.form["rot"] == "mm" else None
        for blk in self.blocks:
            h = blk(h, self.cos, self.sin, rmat)
        x = self.norm(h)
        m = self.n // MERGE_UNIT
        if self.form["merge"] == "split4":
            x4 = x.reshape(1, m, MERGE_UNIT, self.hidden)
            y = None
            for i, fc in enumerate(self.fc1_parts):
                part = fc(x4[:, :, i:i + 1, :].reshape(1, m, self.hidden))
                y = part if y is None else y + part
            return {"rows": self.fc2(F.gelu(y + self.fc1.bias))}
        if self.form["merge"] == "strided":
            x = torch.cat([x[:, i::MERGE_UNIT, :] for i in range(MERGE_UNIT)], dim=-1)
        else:
            x = x.reshape(1, m, MERGE_UNIT * self.hidden)
        return {"rows": self.fc2(F.gelu(self.fc1(x)))}


def tiny_state(hidden=64, heads=4, ffn=96, out=80, depth=2, seed=0) -> dict:
    """A random state dict with the checkpoint's key layout at tiny sizes (the rehearsals; no checkpoint read)."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=0.05: torch.randn(*s, generator=g) * sc   # noqa: E731
    sd = {"patch_embed.proj.weight": r(hidden, 3, 2, 16, 16, sc=0.02), "patch_embed.proj.bias": r(hidden)}
    for i in range(depth):
        p = f"blocks.{i}."
        sd.update({p + "norm1.weight": 1 + r(hidden, sc=0.1), p + "norm1.bias": r(hidden),
                   p + "norm2.weight": 1 + r(hidden, sc=0.1), p + "norm2.bias": r(hidden),
                   p + "attn.qkv.weight": r(3 * hidden, hidden), p + "attn.qkv.bias": r(3 * hidden),
                   p + "attn.proj.weight": r(hidden, hidden), p + "attn.proj.bias": r(hidden),
                   p + "mlp.linear_fc1.weight": r(ffn, hidden), p + "mlp.linear_fc1.bias": r(ffn),
                   p + "mlp.linear_fc2.weight": r(hidden, ffn), p + "mlp.linear_fc2.bias": r(hidden)})
    sd.update({"merger.norm.weight": 1 + r(hidden, sc=0.1), "merger.norm.bias": r(hidden),
               "merger.linear_fc1.weight": r(4 * hidden, 4 * hidden), "merger.linear_fc1.bias": r(4 * hidden),
               "merger.linear_fc2.weight": r(out, 4 * hidden), "merger.linear_fc2.bias": r(out)})
    return sd


def tiny_tower(form: dict | None = None, n: int = 256, hidden: int = 64, heads: int = 4, seed: int = 0):
    """(ClefTower at tiny sizes with random constants, the two random inputs [1, n, 1536])."""
    d = hidden // heads
    g = torch.Generator().manual_seed(seed + 2)
    pos = torch.randn(n, hidden, generator=g) * 0.1
    ang = torch.rand(n, d, generator=g) * 6.0
    tower = ClefTower(tiny_state(hidden=hidden, heads=heads, seed=seed), pos, torch.cos(ang), torch.sin(ang),
                      heads=heads, form=form).eval()
    xs = [torch.randn(1, n, PATCH_DIM, generator=g) for _ in range(2)]
    return tower, xs


def load_consts(path: str | Path) -> dict:
    """cache/r3/consts_<grid>.npz -> {"pos_embeds", "cos", "sin"} torch float32 + the raw npz fields."""
    z = np.load(path)
    return {"pos_embeds": torch.from_numpy(z["pos_embeds"]), "cos": torch.from_numpy(z["cos"]),
            "sin": torch.from_numpy(z["sin"]), "npz": {k: z[k] for k in z.files}}


def contract(grid: str, form: dict | None = None) -> dict:
    n = GRIDS[grid] ** 2
    form = dict(DEFAULT_FORM, **(form or {}))
    inputs = [{"name": "patches", "shape": [1, n, PATCH_DIM], "dtype": "float32",
               "meaning": "the processor's pixel_values of one still image (merge-block-major, vector (C, T=2, "
                          "16, 16); scripts/r3_host.py tile_patches)"}]
    if form["pos"] == "posin":
        inputs.append({"name": "pos", "shape": [1, n, HIDDEN], "dtype": "float32",
                       "meaning": "the grid's position rows (cache/r3/consts_<grid>.npz pos_embeds = transformers "
                                  "5.17.0's bilinear resample of the 48 x 48 table, merge-block-major), the same every call"})
    return {"graph": f"ClefTower {grid} (vision tower + merger, one fixed grid)", "grid": grid, "form": form,
            "patches_per_side": GRIDS[grid], "merged_per_side": GRIDS[grid] // 2,
            "inputs": inputs,
            "outputs": [{"name": "rows", "shape": [1, n // MERGE_UNIT, OUT], "dtype": "float32",
                         "meaning": "the image rows (merger output), row k = merged token k (row-major over the merged "
                                    "grid) = embeds position P + 1 + k of the row graph"}]}
