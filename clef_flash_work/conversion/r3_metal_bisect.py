#!/usr/bin/env python3
"""Round 3: which op of the tower graph Metal (ai-edge-litert 2.2.0, LITERT_METAL) miscomputes, and the graph form to
use instead. Window 1's rehearsal (results/r3_tiny_tfl_form0.json): the tiny random tower in the default form ran with
every node on Metal in one partition and came back wrong (cos 0.11-0.22 against torch; XNNPACK right, 6e-7).
Run 1 (window 2, results/r3_metal_bisect_1.json): of the components only the patch FC + the [1, N, C] position-constant
ADD is wrong on Metal (cos 0.02); every rope / merge / attention / GELU / LayerNorm form is right, so every candidate
of run 1 (they all keep that add) was wrong. Run 2 (this list): the add at rank 4 ("pos4") or the rows as an input
("posin"), with the components that split the FC from the add.

    TMPDIR=<K>/tmp ~/venvs/lt094dev/bin/python scripts/r3_metal_bisect.py
        -> results/r3_metal_bisect.json, cache/r3/tower_form.json (the chosen form; only when one passes)
        (files under out/r3/bisect/ are deleted after each check; logs/r3_bisect_<name>_<accel>.runtime.log)

Every check: a torch module -> litert_torch.convert -> CompiledModel CPU (XNNPACK 8 threads) and Metal fp32
(GpuOptions(enforce_f32=True)) on two random inputs -> rel_rms / cos against the module's own torch output, the runtime's
delegation line. "right" = rel_rms <= 1e-4 on Metal fp32 and <= 1e-5 on the CPU (float32 noise is ~1e-6 here; the
wrong default form sits at ~1.2).
A. components at the real widths (N 256 / 784, C 1152, 16 heads x 72, the checkpoint's shapes; random weights):
   rope with the [1, N, 1, d] constants (broadcast) / with [1, N, H, d] constants / after the transpose; rotate_half as
   slices / as a matmul; the merger's [1, N, C] -> [1, N/4, 4C] reshape / its strided form; the attention core; the
   patch FC + the position add; GELU (tanh); LayerNorm.
B. the tiny tower (hidden 64, 4 heads, depth 2) in diagnostic forms (no rotation; no rotation + strided merge) and in
   the candidate forms, N = 256.
Choice (fixed here, before any number): the first candidate in CANDIDATES that is right at N 256 and N 784 on the tiny
tower and on a real-width one-block tower (C 1152, 16 heads, FFN 4304, out 4096, N 784) is written to
cache/r3/tower_form.json; none -> exit 3 (the round's stop rule: report the op, its rank / dtype and the evidence)."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import r3_tower as RT  # noqa: E402
import r3_lrt as RL  # noqa: E402
from clef_scan import full_scan  # noqa: E402

CANDIDATES = (
    {"rope": "bcast", "rot": "slice", "merge": "reshape", "pos": "pos4"},
    {"rope": "bcast", "rot": "slice", "merge": "reshape", "pos": "posin"},
    {"rope": "full", "rot": "slice", "merge": "reshape", "pos": "pos4"},
    {"rope": "full", "rot": "slice", "merge": "strided", "pos": "pos4"},
    {"rope": "full", "rot": "slice", "merge": "reshape", "pos": "posin"},
    {"rope": "full", "rot": "slice", "merge": "strided", "pos": "posin"},
    {"rope": "fullbhnd", "rot": "mm", "merge": "split4", "pos": "posin"},
)
DIAGNOSTIC = ({"rope": "bcast", "rot": "slice", "merge": "reshape", "pos": "const"},)
TOL_GPU, TOL_CPU = 1e-4, 1e-5
BIS = K / "out/r3/bisect"


def stats(got, want) -> dict:
    g = np.asarray(got, np.float64).ravel()
    w = np.asarray(want, np.float64).ravel()
    nw = np.linalg.norm(w)
    return {"rel_rms": float(np.linalg.norm(g - w) / nw) if nw else float(np.linalg.norm(g - w)),
            "cos": float(g @ w / (np.linalg.norm(g) * nw)) if nw and np.linalg.norm(g) else None,
            "max_abs": float(np.abs(g - w).max()), "finite": bool(np.isfinite(g).all())}


DRY = "--dry" in sys.argv          # torch only: build every module and run it (no file, no runtime)


def check(name: str, module: nn.Module, inputs: list[dict], out_key: str) -> dict:
    """inputs = [{input name: tensor}] (two samples); the module returns {out_key: tensor}."""
    module = module.eval()
    with torch.no_grad():
        want = [module(**x)[out_key].numpy() for x in inputs]
    if DRY:
        print(f"dry {name}: out {want[0].shape} finite {bool(np.isfinite(want[0]).all())}", flush=True)
        return {"name": name, "dry": True, "right": True}
    import litert_torch

    BIS.mkdir(parents=True, exist_ok=True)
    path = BIS / f"{name}.tflite"
    if path.exists():
        path.unlink()
    item = {"name": name}
    t0 = time.perf_counter()
    try:
        litert_torch.convert(module, sample_kwargs=inputs[0]).export(str(path))
    except BaseException as e:  # a failed conversion is a result
        item.update(status="CONVERT_FAIL", error=f"{type(e).__name__}: {str(e)[:400]}")
        return item
    item["convert_seconds"] = round(time.perf_counter() - t0, 2)
    s = full_scan(path)
    item["ops"] = s["op_histogram"]
    item["operator_count"] = s["operator_count"]
    item["forbidden"] = {k: v for k, v in s["forbidden_counts"].items() if v}
    item["rank_gt4"] = s["rank_gt4_tensor_count"]
    for accel in ("cpu", "gpu_f32"):
        rl = RL.RuntimeLog(K / f"logs/r3_bisect_{name}_{accel}.runtime.log")
        r = {}
        try:
            with rl:
                m = RL.Model(path, accel)
                got = [m.run({k: v.numpy() for k, v in x.items()})[out_key] for x in inputs]
                m.close()
            st = [stats(g, w) for g, w in zip(got, want)]
            r.update(status="OK", rel_rms=max(x["rel_rms"] for x in st), cos=min((x["cos"] for x in st if x["cos"] is not None), default=None),
                     max_abs=max(x["max_abs"] for x in st), finite=all(x["finite"] for x in st))
        except BaseException as e:
            r.update(status="FAIL", error=f"{type(e).__name__}: {str(e)[:400]}")
        r["delegation"] = rl.delegation
        item[accel] = r
    path.unlink()
    item["right"] = bool(item.get("cpu", {}).get("status") == "OK" and item.get("gpu_f32", {}).get("status") == "OK"
                         and item["cpu"]["rel_rms"] <= TOL_CPU and item["gpu_f32"]["rel_rms"] <= TOL_GPU)
    g = item.get("gpu_f32", {})
    print(f"{name}: right {item['right']} cpu {item.get('cpu', {}).get('rel_rms')} gpu {g.get('rel_rms')} cos {g.get('cos')} "
          f"deleg {g.get('delegation')}", flush=True)
    return item


def fname(form: dict) -> str:
    return f"{form['rope']}-{form['rot']}-{form['merge']}-{form.get('pos', 'const')}"


class Rope(nn.Module):
    """y = rotate(x) with the tower's own Block.rotate in a given form; x [1, N, H, d] (before the transpose) or
    [1, H, N, d] (bhnd / fullbhnd)."""

    def __init__(self, n, heads, d, form):
        super().__init__()
        g = torch.Generator().manual_seed(5)
        ang = torch.rand(n, d, generator=g) * 6.0
        sd = RT.tiny_state(hidden=heads * d, heads=heads, depth=1)
        self.blk = RT.Block(sd, 0, heads * d, heads, form)
        t = RT.ClefTower(sd, torch.zeros(n, heads * d), torch.cos(ang), torch.sin(ang), heads=heads, form=form)
        self.register_buffer("cos", t.cos, persistent=False)
        self.register_buffer("sin", t.sin, persistent=False)
        self.register_buffer("rmat", t.rmat, persistent=False)

    def forward(self, x):
        return {"y": self.blk.rotate(x, self.cos, self.sin, self.rmat)}


class Merge(nn.Module):
    def __init__(self, n, c, how):
        super().__init__()
        self.n, self.c, self.how = n, c, how
        self.norm = nn.LayerNorm(c, eps=RT.EPS)

    def forward(self, x):
        x = self.norm(x)
        if self.how == "strided":
            return {"y": torch.cat([x[:, i::4, :] for i in range(4)], dim=-1)}
        return {"y": x.reshape(1, self.n // 4, 4 * self.c)}


class MergeFc(nn.Module):
    """The merger's LayerNorm -> 2 x 2 merge -> fc1 at the real width, in each merge form of ClefTower."""

    def __init__(self, n, c, how):
        super().__init__()
        sd = RT.tiny_state(hidden=c, heads=16, depth=1, ffn=8, out=8, seed=7)
        self.t = RT.ClefTower(sd, torch.zeros(n, c), torch.ones(n, c // 16), torch.zeros(n, c // 16), heads=16,
                              form={"merge": how})

    def forward(self, x):
        t, m = self.t, self.t.n // 4
        x = t.norm(x)
        if t.form["merge"] == "split4":
            x4 = x.reshape(1, m, 4, t.hidden)
            y = None
            for i, fc in enumerate(t.fc1_parts):
                part = fc(x4[:, :, i:i + 1, :].reshape(1, m, t.hidden))
                y = part if y is None else y + part
            return {"y": y + t.fc1.bias}
        if t.form["merge"] == "strided":
            x = torch.cat([x[:, i::4, :] for i in range(4)], dim=-1)
        else:
            x = x.reshape(1, m, 4 * t.hidden)
        return {"y": t.fc1(x)}


class Attn(nn.Module):
    """The attention core of Block.forward without the rotation."""

    def __init__(self, c, heads):
        super().__init__()
        sd = RT.tiny_state(hidden=c, heads=heads, depth=1, ffn=8, out=8)
        self.blk = RT.Block(sd, 0, c, heads, {"rope": "none"})

    def forward(self, x):
        b, n = self.blk, x.shape[1]
        q = b.q(x).reshape(1, n, b.heads, b.head_dim).transpose(1, 2)
        k = b.k(x).reshape(1, n, b.heads, b.head_dim).transpose(1, 2)
        v = b.v(x).reshape(1, n, b.heads, b.head_dim).transpose(1, 2)
        s = torch.matmul(q * b.scaling, k.transpose(2, 3))
        o = torch.matmul(torch.softmax(s, dim=-1), v)
        return {"y": b.proj(o.transpose(1, 2).reshape(1, n, b.hidden))}


class Patch(nn.Module):
    """The patch FC and the position add: mode "fc+const" (the tower's default), "fc" (no add), "add" (input + the
    [1, N, C] constant, no FC), "pos4" (the add at rank 4), "posin" (the rows as a second input)."""

    def __init__(self, n, c, mode="fc+const"):
        super().__init__()
        g = torch.Generator().manual_seed(6)
        self.n, self.c, self.mode = n, c, mode
        self.fc = nn.Linear(RT.PATCH_DIM, c)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(c, RT.PATCH_DIM, generator=g) * 0.02)
        p = torch.randn(1, n, c, generator=g) * 0.1
        self.pos_feed = p
        self.register_buffer("pos", p.reshape(1, 1, n, c).clone() if mode == "pos4" else p.clone(), persistent=False)

    def forward(self, x, pos=None):
        if self.mode == "fc":
            return {"y": self.fc(x)}
        if self.mode == "add":
            return {"y": x + self.pos}
        if self.mode == "pos4":
            return {"y": (self.fc(x).reshape(1, 1, self.n, self.c) + self.pos).reshape(1, self.n, self.c)}
        if self.mode == "posin":
            return {"y": self.fc(x) + pos}
        return {"y": self.fc(x) + self.pos}


class Act(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.norm = nn.LayerNorm(1152, eps=RT.EPS)

    def forward(self, x):
        if self.kind == "gelu_tanh":
            return {"y": torch.nn.functional.gelu(x, approximate="tanh")}
        if self.kind == "gelu":
            return {"y": torch.nn.functional.gelu(x)}
        return {"y": self.norm(x)}


def real_width_tower(form, n):
    """One block at the checkpoint's widths (random weights): C 1152, 16 heads x 72, FFN 4304, out 4096."""
    sd = RT.tiny_state(hidden=1152, heads=16, ffn=4304, out=4096, depth=1, seed=3)
    g = torch.Generator().manual_seed(4)
    ang = torch.rand(n, 72, generator=g) * 6.0
    t = RT.ClefTower(sd, torch.randn(n, 1152, generator=g) * 0.1, torch.cos(ang), torch.sin(ang), heads=16, form=form)
    xs = [torch.randn(1, n, RT.PATCH_DIM, generator=g) for _ in range(2)]
    return t, xs


def main() -> int:
    out = K / ("results/r3_metal_bisect_dry.json" if DRY else "results/r3_metal_bisect.json")
    assert DRY or not out.exists(), f"refusing to overwrite {out}"
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    t_all = time.time()
    g = torch.Generator().manual_seed(9)
    res = {"what": "round 3: Metal fp32 vs torch on tower components and forms", "tol": {"gpu_f32": TOL_GPU, "cpu": TOL_CPU},
           "candidates": list(CANDIDATES), "components": [], "tiny_forms": [], "chosen": None}
    # A. components at the real widths
    comps = []
    for n in (256, 784):
        for form in ({"rope": "bcast", "rot": "slice"}, {"rope": "full", "rot": "slice"}, {"rope": "bhnd", "rot": "slice"},
                     {"rope": "bcast", "rot": "mm"}):
            shp = (1, 16, n, 72) if form["rope"] in ("bhnd", "fullbhnd") else (1, n, 16, 72)
            comps.append((f"rope_{form['rope']}_{form['rot']}_n{n}", Rope(n, 16, 72, dict(form, merge="reshape")),
                          [{"x": torch.randn(*shp, generator=g)} for _ in range(2)]))
        for how in ("reshape", "strided"):
            comps.append((f"merge_{how}_n{n}", Merge(n, 1152, how), [{"x": torch.randn(1, n, 1152, generator=g)} for _ in range(2)]))
        for how in ("reshape", "strided", "split4"):
            comps.append((f"mergefc_{how}_n{n}", MergeFc(n, 1152, how), [{"x": torch.randn(1, n, 1152, generator=g)} for _ in range(2)]))
    comps += [("attn_n256", Attn(1152, 16), [{"x": torch.randn(1, 256, 1152, generator=g) * 0.5} for _ in range(2)]),
              ("patch_fc_pos_n256", Patch(256, 1152), [{"x": torch.randn(1, 256, RT.PATCH_DIM, generator=g)} for _ in range(2)]),
              ("patch_fc_n256", Patch(256, 1152, "fc"), [{"x": torch.randn(1, 256, RT.PATCH_DIM, generator=g)} for _ in range(2)]),
              ("pos_add_n256", Patch(256, 1152, "add"), [{"x": torch.randn(1, 256, 1152, generator=g)} for _ in range(2)]),
              ("patch_fc_pos4_n256", Patch(256, 1152, "pos4"), [{"x": torch.randn(1, 256, RT.PATCH_DIM, generator=g)} for _ in range(2)]),
              ("patch_fc_pos4_n784", Patch(784, 1152, "pos4"), [{"x": torch.randn(1, 784, RT.PATCH_DIM, generator=g)} for _ in range(2)]),
              ("patch_fc_posin_n256", Patch(256, 1152, "posin"), None),
              ("patch_fc_posin_n784", Patch(784, 1152, "posin"), None),
              ("gelu_tanh", Act("gelu_tanh"), [{"x": torch.randn(1, 256, 4304, generator=g)} for _ in range(2)]),
              ("gelu_erf", Act("gelu"), [{"x": torch.randn(1, 64, 4608, generator=g)} for _ in range(2)]),
              ("layernorm", Act("ln"), [{"x": torch.randn(1, 256, 1152, generator=g) * 3} for _ in range(2)])]
    for name, mod, ins in comps:
        if ins is None:                       # the posin components: the position rows as the second input
            ins = [{"x": torch.randn(1, mod.n, RT.PATCH_DIM, generator=g), "pos": mod.pos_feed} for _ in range(2)]
        res["components"].append(check(f"c_{name}", mod, ins, "y"))
    # B. the tiny tower: diagnostics, then candidates; a candidate right at N 256 is tried at N 784 and at real width
    for form in DIAGNOSTIC:
        t, xs = RT.tiny_tower(form, n=256)
        res["tiny_forms"].append({"form": form, "role": "diagnostic", **check(f"t_{fname(form)}_n256", t, [t.feed(x) for x in xs], "rows")})
    for form in CANDIDATES:
        t, xs = RT.tiny_tower(form, n=256)
        a = check(f"t_{fname(form)}_n256", t, [t.feed(x) for x in xs], "rows")
        row = {"form": form, "role": "candidate", **a}
        if a["right"]:
            t2, xs2 = RT.tiny_tower(form, n=784)
            row["n784"] = check(f"t_{fname(form)}_n784", t2, [t2.feed(x) for x in xs2], "rows")
            if row["n784"]["right"]:
                t3, xs3 = real_width_tower(form, 784)
                row["real_width_n784"] = check(f"w_{fname(form)}_n784", t3, [t3.feed(x) for x in xs3], "rows")
                if row["real_width_n784"]["right"] and res["chosen"] is None:
                    res["chosen"] = form
        res["tiny_forms"].append(row)
        if res["chosen"] is not None:
            break
    res["seconds"] = round(time.time() - t_all, 1)
    res["tmpdir_after"] = sorted(p.name for p in Path(os.environ.get("TMPDIR", "/nonexistent")).glob("*"))[:20]
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    if res["chosen"] is not None and not DRY:
        (K / "cache/r3/tower_form.json").write_text(json.dumps({"form": res["chosen"], "by": "results/r3_metal_bisect.json",
                                                                 "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
    print(json.dumps({"chosen": res["chosen"], "seconds": res["seconds"],
                      "components_wrong_on_metal": [c["name"] for c in res["components"] if not c.get("right")]}, indent=1))
    return 0 if res["chosen"] is not None else 3


if __name__ == "__main__":
    raise SystemExit(main())
