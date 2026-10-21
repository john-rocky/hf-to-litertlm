#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "torchvision==0.24.0",
#     "transformers==5.17.0",
#     "accelerate",
#     "pillow",
#     "safetensors>=0.8.0",
#     "huggingface_hub>=1.5.0,<2",
#     "numpy",
# ]
# [tool.uv]
# index-url = "https://pypi.org/simple"
# ///
"""fp32 reference for clef-flash, from the checkpoint's own joint_schema_model.py (this conversion's round 1).

Copy of the zoo's conversion/clef_flash/oracle_clef.py (john-rocky/coreai-model-zoo main 5ef2247a, sha256 96a2e28c...)
with these changes and nothing else: the `_paths` import is gone (K = this script's parent's parent; the snapshot is
K/hf/hub/models--Cloudflare--clef-flash/snapshots/<REVISION>, placed by scripts/run_dl.sh and `hf download`);
REVISION = f0b7d3cd (the Hub main of 2026-10-19: the 6 LFS files have the oids of 17f0b0ad, the small files differ only
in README.md); the default --fixtures / --out-dir / --results-dir are K/fixtures, K/ref, K/results; the output names
records_oracle* / oracle_summary* are records_ref* / ref_summary*; and four options: --records-file (another file of
fixture-schema records, e.g. fixtures/red_arms.json, run with its own --tag), --no-independent (skip the independent
forwards), --deadline-s (start no new run after that many seconds; the run stops with the partial file and resumes
with --resume), and the header records which records file it read. The model, the hooks, the checks and the record
format are the zoo script's.

Cloudflare/clef-flash (revision 17f0b0ad, Apache-2.0) answers typed questions about a state in one
prefill: a Qwen3.5-9B backbone (24 Gated DeltaNet + 8 full-attention layers, a 27-block vision
tower) returns its final-norm hidden states at every position, and a joint schema head turns them
into one logit per allowed option. This script imports the checkpoint's `joint_schema_model.py`
unchanged (path import, sha256 asserted), loads it with the author's `load_release_model(...,
device="cpu", dtype=torch.float32)`, and answers every fixture record through the author's own
`systemone()` (the /v1/systemone request -> response function). Forward hooks record what every
later Core AI stage is compared against:

* the backbone output the head receives: `last_hidden_state` [T, 4096] (after the final RMSNorm);
* the M-RoPE planes the text rotary received [3, T];
* the tower output the decoder receives: `image_embeds` [N, 4096] (merger output);
* the head output (per-question option logits) and the batch systemone() built (its ids and the
  EncodedRecord, which must equal a separate `encode_record()` of the same request);
* per decoder layer: absmax of the residual stream (position 0, the rest, overall) and of the
  MLP down-projection input, for the fp16 overflow survey;
* for two runs (one text, one g256 image) the 33 residual-stream states [33, T, 4096] (layer-0
  input after the image rows are scattered in, then each decoder layer's output).

Asserted on every run: systemone()'s answers equal `systemone_answer()` applied to the softmax of
the hooked logits; the head re-run on the hooked hidden states reproduces the logits (allclose
1e-6, bit equality recorded); the ids and spans systemone() used equal a separate encode; every
question span decodes to its rendered instructions and every option span to its rendered option
JSON; the rotary planes equal the closed form below; the tower rows equal the image-pad count.
After the loop, five runs (text, JSON, g256, native 1024x768, SemIf) go again through an independent
`model(collate_records(...))` call, whose logits must be bit-identical to the hooked ones (the first of
them is fixture record 0, the determinism check).

M-RoPE closed form (P = len(prefix ids) = media token_offset; index P = <|vision_start|>; merged
grid H x W, N = H*W image tokens at indices P+1 .. P+N):
  index i <= P -> (i, i, i); image token k -> (P+1, P+1 + k // W, P+1 + k % W);
  index i >= P+N+1 -> i - (N - max(H, W)) on all three planes; text-only records -> i.

Arms: text records run once (`text`); image records run `g256` / `g448` (PIL BICUBIC resize to
256x256 / 448x448 before the processor, grid asserted) and `native` (the image as drawn). `--arms
g672,g896 --tag _large` runs the two larger fixed grids (672x672 = 21x21 = 441 image tokens,
896x896 = 28x28 = 784) on the image records only and writes every output under its own name
(records_oracle_large.json, header_large.json, results/oracle_summary_large.json, ...), so the
default run's files are left as they are.

    HF_HUB_OFFLINE=1 venv-ref/bin/python scripts/clef_ref.py --resume --deadline-s 1500      # (from K) -> K/ref
    HF_HUB_OFFLINE=1 venv-ref/bin/python scripts/clef_ref.py --records-file fixtures/red_arms.json --tag _red \\
        --no-independent
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]

HF_ID = "Cloudflare/clef-flash"
REVISION = "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"


def hf_snapshot(repo_id: str, revision: str) -> str:
    """The pinned snapshot in the work directory's Hugging Face cache (K/hf)."""
    snap = K / "hf/hub" / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
    if not snap.is_dir():
        raise FileNotFoundError(f"no snapshot {snap}: run scripts/run_dl.sh and the `hf download` of the small files")
    return str(snap)
JSM_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
VISION_START, IMAGE_PAD, VISION_END = 248053, 248056, 248054
MERGE = 2
GRIDS = {"g256": 256, "g448": 448, "g672": 672, "g896": 896}
IMAGE_ARMS = ("g256", "g448", "native")
NEAR_TIE = 0.02
HIDDEN_RUNS = {("own_t01", "text"), ("img_01", "g256")}
INDEPENDENT_RUNS = (("own_t01", "text"), ("own_j01", "text"), ("img_01", "g256"), ("img_04", "native"))  # + the first semif record


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_npz(path: Path, arrays: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def expected_rope(T: int, P: int | None, grid_thw) -> np.ndarray:
    """The closed-form M-RoPE planes [3, T] (see the module docstring)."""
    pos = np.broadcast_to(np.arange(T, dtype=np.int64), (3, T)).copy()
    if grid_thw is None:
        return pos
    gt, gh, gw = (int(v) for v in grid_thw)
    assert gt == 1, grid_thw
    H, W = gh // MERGE, gw // MERGE
    N = H * W
    k = np.arange(N)
    pos[0, P + 1:P + 1 + N] = P + 1
    pos[1, P + 1:P + 1 + N] = P + 1 + k // W
    pos[2, P + 1:P + 1 + N] = P + 1 + k % W
    pos[:, P + N + 1:] = np.arange(P + N + 1, T) - (N - max(H, W))
    return pos


def kernel_report(mq) -> dict:
    """Which implementation each decorated GDN / conv function resolved to, read from its closure."""
    import importlib.util as iu
    out = {}
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"):
        fn = getattr(mq, name)
        env = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in (fn.__closure__ or ()))))
        impl = env.get("implementation")
        out[name] = {"implementation": f"{impl.__module__}.{impl.__qualname__}" if impl is not None else None,
                     "is_new_implementation": env.get("is_new_implementation")}
    for pkg in ("fla", "causal_conv1d", "kernels"):
        out[f"{pkg}_importable"] = iu.find_spec(pkg) is not None
    return out


class Taps:
    """Hooks on the author's model; nothing in the model is replaced except wrapping get_image_features."""

    def __init__(self, model):
        self.model = model
        backbone = model.language_model                 # Qwen3_5ForConditionalGeneration
        self.vlm = backbone.model                       # Qwen3_5Model
        self.text = self.vlm.language_model             # Qwen3_5TextModel
        self.keep_hidden = False
        self.reset()
        model.register_forward_pre_hook(self._clef_pre)
        model.head.register_forward_hook(self._head_out)
        self.text.register_forward_hook(self._text_out)
        self.text.rotary_emb.register_forward_pre_hook(self._rope_pre)
        self.text.embed_tokens.register_forward_hook(self._embed_out)
        self.text.layers[0].register_forward_pre_hook(self._layer0_in)
        for i, layer in enumerate(self.text.layers):
            layer.register_forward_hook(self._layer_out(i))
            layer.mlp.down_proj.register_forward_pre_hook(self._down_in(i))
        orig = self.vlm.get_image_features

        def get_image_features(*a, **k):
            out = orig(*a, **k)
            self.image_embeds.append([t.detach().clone() for t in out.pooler_output])
            self.vision_last.append(tuple(out.last_hidden_state.shape))
            return out

        self.vlm.get_image_features = get_image_features

    def reset(self):
        self.batches, self.head, self.last_hidden, self.rope = [], [], [], []
        self.image_embeds, self.vision_last = [], []
        self.embed_absmax = None
        self.layer_stats = {}
        self.down_absmax = {}
        self.hidden_states = {}

    @staticmethod
    def _stats(x):
        a = x[0].abs().float()                          # [T, D]
        flat = int(a.argmax())
        T = a.shape[0]
        return {"pos0": float(a[0].max()), "rest": float(a[1:].max()) if T > 1 else None,
                "overall": float(a.max()), "argmax_pos": flat // a.shape[1], "argmax_ch": flat % a.shape[1]}

    def _clef_pre(self, module, args):
        batch = args[0]
        self.batches.append({"input_ids": batch["input_ids"].detach().clone(), "records": list(batch["records"]),
                             "media_keys": sorted(batch.get("media") or {})})

    def _head_out(self, module, args, output):
        self.head.append([[t.detach().clone() for t in rec] for rec in output])

    def _text_out(self, module, args, output):
        self.last_hidden.append(output.last_hidden_state.detach().clone())

    def _rope_pre(self, module, args):
        self.rope.append(args[1].detach().clone())

    def _embed_out(self, module, args, output):
        self.embed_absmax = self._stats(output)

    def _layer0_in(self, module, args, kwargs=None):
        x = args[0]
        self.layer_stats["input"] = self._stats(x)
        if self.keep_hidden:
            self.hidden_states[0] = x[0].detach().float().clone()

    def _layer_out(self, i):
        def hook(module, args, output):
            self.layer_stats[i] = self._stats(output)
            if self.keep_hidden:
                self.hidden_states[i + 1] = output[0].detach().float().clone()
        return hook

    def _down_in(self, i):
        def hook(module, args):
            self.down_absmax[i] = float(args[0].abs().max())
        return hook


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixtures", default=str(K / "fixtures"))
    ap.add_argument("--out-dir", default=str(K / "ref"))
    ap.add_argument("--results-dir", default=str(K / "results"))
    ap.add_argument("--records-file", default=None,
                    help="fixture-schema records to run (default <fixtures>/records.json); give it its own --tag")
    ap.add_argument("--no-independent", action="store_true", help="skip the independent model(batch) forwards")
    ap.add_argument("--deadline-s", type=float, default=None,
                    help="start no new run after this many seconds (resume the rest with --resume)")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--records", default=None, help="comma-separated record ids (subset)")
    ap.add_argument("--limit", type=int, default=None, help="first N records only (timing probe)")
    ap.add_argument("--resume", action="store_true", help="skip (id, arm) runs already in records_oracle.partial.jsonl")
    ap.add_argument("--finalize-only", action="store_true", help="no model: rebuild the outputs from the partial jsonl")
    ap.add_argument("--arms", default="text," + ",".join(IMAGE_ARMS),
                    help="arms to run: text (text records) and any of " + ", ".join([*GRIDS, "native"]) + " (image records)")
    ap.add_argument("--tag", default="", help="suffix of every output name (records_oracle<tag>.json, header<tag>.json, ...)")
    ap.add_argument("--independent", default=None,
                    help="id:arm list for the independent model(batch) forwards (default: the five of the main run)")
    args = ap.parse_args()
    args.arm_list = args.arms.split(",")
    bad = [a for a in args.arm_list if a not in ("text", "native", *GRIDS)]
    assert not bad, f"unknown arms {bad}"

    fx = Path(args.fixtures).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    res_dir = Path(args.results_dir).expanduser()
    (out_dir / "npz").mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    records_file = Path(args.records_file) if args.records_file else fx / "records.json"
    doc = json.loads(records_file.read_text())
    all_records = doc["records"]
    meta = {im["path"].split("/")[-1]: im for im in json.loads((fx / "images_meta.json").read_text())["images"]}
    partial = out_dir / f"records_ref{args.tag}.partial.jsonl"
    header_path = out_dir / f"header{args.tag}.json"
    if args.finalize_only:
        return finalize(args, doc, partial, out_dir, res_dir, header=json.loads(header_path.read_text()))

    import torch
    import transformers
    import torchvision
    import PIL
    from PIL import Image
    import huggingface_hub

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    t_start = time.monotonic()
    snapshot = Path(hf_snapshot(HF_ID, revision=REVISION))
    jsm_path = snapshot / "joint_schema_model.py"
    assert sha256_file(jsm_path) == JSM_SHA256, "joint_schema_model.py is not the pinned file"
    spec = importlib.util.spec_from_file_location("joint_schema_model", jsm_path)
    jsm = importlib.util.module_from_spec(spec)
    sys.modules["joint_schema_model"] = jsm
    spec.loader.exec_module(jsm)                                     # author's code, unchanged
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq       # noqa: E402

    t0 = time.monotonic()
    model, processor = jsm.load_release_model(str(snapshot), device="cpu", dtype=torch.float32)
    load_s = time.monotonic() - t0
    assert all(p.dtype == torch.float32 for p in model.parameters())
    tok = processor.tokenizer
    pad_id = tok.pad_token_id
    backbone = model.language_model
    lm_head_w = backbone.get_output_embeddings().weight
    taps = Taps(model)
    kernels = kernel_report(mq)
    ip = processor.image_processor
    print(f"loaded {HF_ID}@{REVISION[:7]} fp32 cpu in {load_s:.0f}s, threads {torch.get_num_threads()}, "
          f"kernels {[(k, v['implementation']) for k, v in kernels.items() if isinstance(v, dict)]}", flush=True)

    header = {
        "schema": "clef-flash-oracle/1",
        "source": {"hf_id": HF_ID, "revision": REVISION, "joint_schema_model_sha256": JSM_SHA256,
                   "sha256": {n: sha256_file(snapshot / n) for n in ("config.json", "joint_head_config.json", "processor_config.json",
                                                                     "tokenizer.json", "tokenizer_config.json",
                                                                     "model.safetensors.index.json")}},
        "oracle": "joint_schema_model.load_release_model(device='cpu', dtype=float32) + systemone(), torch.inference_mode",
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                     "transformers": transformers.__version__, "pillow": PIL.__version__, "numpy": np.__version__,
                     "huggingface_hub": huggingface_hub.__version__},
        "device": "cpu", "dtype": "float32", "torch_num_threads": torch.get_num_threads(), "load_seconds": load_s,
        "platform": {"machine": platform.machine(), "macos": platform.mac_ver()[0]},
        "processor": {"processor_class": type(processor).__name__, "image_processor_class": type(ip).__name__,
                      "image_processor_mro": [c.__name__ for c in type(ip).__mro__][:4], "tokenizer_class": type(tok).__name__,
                      "resample": int(ip.resample), "patch_size": ip.patch_size, "merge_size": ip.merge_size,
                      "temporal_patch_size": ip.temporal_patch_size, "image_mean": list(ip.image_mean), "image_std": list(ip.image_std),
                      "size": {"shortest_edge": ip.size.shortest_edge, "longest_edge": ip.size.longest_edge},
                      "pad_token_id": pad_id},
        "kernels": kernels,
        "attn_implementation": {"text": backbone.config.text_config._attn_implementation,
                                "vision": backbone.config.vision_config._attn_implementation},
        "mrope_closed_form": ("P = len(prefix ids) = token_offset; i <= P -> (i,i,i); image token k at P+1+k -> "
                              "(P+1, P+1+k//W, P+1+k%W); i >= P+N+1 -> i-(N-max(H,W)) on all planes; text-only -> i"),
        "arms": {"text": "no image", **{a: f"PIL BICUBIC resize to {s}x{s} before the processor" for a, s in GRIDS.items()},
                 "native": "the image as drawn"},
        "arms_run": args.arm_list, "tag": args.tag,
        "near_tie_threshold": NEAR_TIE,
        "fixture": {"records_json_sha256": sha256_file(records_file),
                    "records_file": str(records_file.resolve().relative_to(K)),
                    "images_meta_json_sha256": sha256_file(fx / "images_meta.json")},
        "hidden_state_runs": sorted([list(x) for x in HIDDEN_RUNS]),
    }
    prev_header = json.loads(header_path.read_text()) if header_path.exists() else {}
    header["invocations"] = prev_header.get("invocations", [])
    prev_fx = prev_header.get("fixture", {}).get("records_json_sha256")
    if args.resume and prev_fx and prev_fx != header["fixture"]["records_json_sha256"]:
        # runs already in the partial file came from an earlier records.json; the caller asserts those records are unchanged
        header["fixture"]["earlier_records_json_sha256"] = sorted(set(prev_header["fixture"].get("earlier_records_json_sha256", []))
                                                                  | {prev_fx})
    write_atomic(header_path, (json.dumps(header, indent=1) + "\n").encode())

    records = all_records
    if args.records:
        keep = set(args.records.split(","))
        records = [r for r in all_records if r["id"] in keep]
        assert len(records) == len(keep), sorted(keep - {r["id"] for r in records})
    if args.limit:
        records = records[:args.limit]
    done = set()
    if args.resume and partial.exists():
        for line in partial.read_text().splitlines():
            x = json.loads(line)
            done.add((x["id"], x["arm"]))
    elif partial.exists() and not args.resume:
        raise SystemExit(f"{partial} exists; pass --resume to extend it or move it away")

    def load_image(name, arm):
        im = Image.open(fx / "images" / name).convert("RGB")
        if arm in GRIDS:
            im = im.resize((GRIDS[arm], GRIDS[arm]), Image.Resampling.BICUBIC)
        return im

    def run(r, arm):
        request = copy.deepcopy(r["request"])
        images = [load_image(f, arm) for f in r.get("image_files", [])] if arm != "text" else []
        if images:
            request["images"] = images
        enc = jsm.encode_record(tok, request, processor=processor)            # separate encode (for the spans check)
        taps.reset()
        taps.keep_hidden = (r["id"], arm) in HIDDEN_RUNS
        t0 = time.perf_counter()
        resp = jsm.systemone(model, processor, request)
        wall = time.perf_counter() - t0
        taps.keep_hidden = False
        assert len(taps.batches) == 1 and len(taps.head) == 1 and len(taps.last_hidden) == 1 and len(taps.rope) == 1
        b = taps.batches[0]
        ids = b["input_ids"][0].numpy().astype(np.int64)
        T = len(ids)
        enc_sys = b["records"][0]
        checks = {}
        checks["encode_equal_systemone_batch"] = bool(enc_sys == enc and tuple(ids.tolist()) == enc.input_ids)
        assert checks["encode_equal_systemone_batch"], (r["id"], arm)
        logits = [t.float() for t in taps.head[0][0]]
        probs = [torch.softmax(t, -1) for t in logits]
        # systemone answers == systemone_answer(softmax(hooked logits))
        qmap = request["questions"]
        recomputed = {q.question_id: jsm.systemone_answer(qmap[q.question_id], dict(zip(q.option_ids, p.tolist())))
                      for q, p in zip(enc.questions, probs)}
        checks["systemone_answers_equal_hook_path"] = recomputed == resp["answers"]
        checks["systemone_model_field"] = resp["model"] == request["model"]
        checks["systemone_usage_input_tokens"] = resp["usage"]["input_tokens"] == T
        assert checks["systemone_answers_equal_hook_path"] and checks["systemone_model_field"] and checks["systemone_usage_input_tokens"], (r["id"], arm)
        # head re-run on the hooked hidden state == hooked logits (proves the capture point)
        hidden = taps.last_hidden[0]                                            # [1, T, 4096]
        assert tuple(hidden.shape) == (1, T, 4096) and hidden.dtype == torch.float32
        th = time.perf_counter()
        with torch.inference_mode():
            relog = model.head(hidden, b["input_ids"], torch.ones_like(b["input_ids"]), [enc], lm_head_w)[0]
        head_wall = time.perf_counter() - th
        diffs = [float((a.float() - c).abs().max()) for a, c in zip(relog, logits)]
        checks["head_rerun_max_abs_diff"] = max(diffs)
        checks["head_rerun_bit_equal"] = all(torch.equal(a.float(), c) for a, c in zip(relog, logits))
        assert checks["head_rerun_max_abs_diff"] <= 1e-6, (r["id"], arm, diffs)
        # spans decode to the rendered pieces
        span_ok = True
        for q in enc.questions:
            qd = qmap[q.question_id]
            instr = qd.get("instructions")
            instr = str(q.question_id) if instr is None or instr == "" else instr
            s, e = q.question_span
            span_ok &= tok.decode(ids[s:e].tolist()) == jsm.render(instr)
            opts = jsm.question_options(qd)
            assert [o for o, _ in opts] == list(q.option_ids)
            for (oid, desc), (s, e) in zip(opts, q.option_spans):
                sem = {"option_id": oid}
                if desc is not None:
                    sem["description"] = desc
                span_ok &= tok.decode(ids[s:e].tolist()) == jsm.render(sem)
        checks["spans_decode_to_rendered_text"] = bool(span_ok)
        assert span_ok, (r["id"], arm)
        # rope planes
        rope = taps.rope[0][:, 0].numpy().astype(np.int64)                     # [3, T]
        media = enc.media or {}
        grid = media["image_grid_thw"][0].tolist() if "image_grid_thw" in media else None
        P = media.get("token_offset")
        exp = expected_rope(T, P, grid)
        checks["rope_closed_form_equal"] = bool(np.array_equal(rope, exp))
        if not checks["rope_closed_form_equal"]:
            bad = np.argwhere(rope != exp)[:8].tolist()
            raise AssertionError(f"M-RoPE planes differ from the closed form on {r['id']}/{arm}: first mismatches {bad}")
        n_img = int((ids == IMAGE_PAD).sum())
        rec = {"id": r["id"], "arm": arm, "source": r["source"], "tokens": T, "n_image_tokens": n_img,
               "state_tokens": r.get("state_tokens"), "ids": ids.tolist(), "wall_s": wall, "head_rerun_s": head_wall,
               "device": "cpu"}
        arrays = {"input_ids": ids, "last_hidden": hidden[0].numpy().astype(np.float32), "rope_pos": rope}
        if images:
            assert len(taps.image_embeds) == 1 and len(taps.image_embeds[0]) == 1
            emb = taps.image_embeds[0][0]
            gt, gh, gw = grid
            assert tuple(emb.shape) == (n_img, 4096) and n_img == gt * gh * gw // MERGE ** 2, (emb.shape, grid)
            assert taps.vision_last[0] == (gt * gh * gw, 1152), taps.vision_last
            assert ids[P] == VISION_START and (ids[P + 1:P + 1 + n_img] == IMAGE_PAD).all() and ids[P + 1 + n_img] == VISION_END
            pv = media["pixel_values"].numpy().astype(np.float32)
            mm = np.asarray(media["mm_token_type_ids"], dtype=np.int64)
            rec.update(grid_thw=grid, token_offset=P, image_files=r["image_files"],
                       image_size_in=meta[r["image_files"][0]]["size"], image_size_fed=list(images[0].size),
                       resized_hw=[gh * 16, gw * 16], merged_hw=[gh // MERGE, gw // MERGE],
                       rope_shift_amount=int(n_img - max(gh, gw) // MERGE),
                       pixel_values_shape=list(pv.shape), pixel_values_sha256=sha256_array(pv),
                       image_embeds_shape=list(emb.shape), mm_token_type_ids_media=mm.tolist(),
                       media_batch_keys=b["media_keys"])
            if arm in GRIDS:
                g = GRIDS[arm] // 16
                assert grid == [1, g, g], ("processor chose its own grid", r["id"], arm, grid)
            arrays.update(pixel_values=pv, image_grid_thw=media["image_grid_thw"].numpy().astype(np.int64),
                          image_embeds=emb.numpy().astype(np.float32), mm_token_type_ids_media=mm)
        qrecs, flat_logits, flat_probs, n_opts, qspans, ospans, qtypes = [], [], [], [], [], [], []
        for q, lg, p in zip(enc.questions, logits, probs):
            pl = p.tolist()
            order = np.argsort(-np.asarray(pl))
            top2 = float(pl[order[0]] - pl[order[1]])
            am = int(order[0])
            g = r["gold"].get(q.question_id)
            qrecs.append({"question_id": q.question_id, "type": ["noul", "choice", "score"][q.question_type],
                          "question_span": list(q.question_span), "option_spans": [list(s) for s in q.option_spans],
                          "option_ids": list(q.option_ids), "logits": lg.tolist(), "probs": pl,
                          "argmax_index": am, "argmax_id": q.option_ids[am], "top2_margin": top2, "near_tie": top2 <= NEAR_TIE,
                          "gold": g, "gold_correct": (q.option_ids[am] == g) if g is not None else None})
            flat_logits += lg.tolist(); flat_probs += pl; n_opts.append(len(q.option_ids))
            qspans.append(list(q.question_span)); ospans += [list(s) for s in q.option_spans]; qtypes.append(q.question_type)
        rec.update(questions=qrecs, systemone_response=resp, checks=checks,
                   absmax={"embed_tokens": taps.embed_absmax, "layer_input": taps.layer_stats["input"],
                           "layers": [taps.layer_stats[i] for i in range(len(taps.text.layers))],
                           "final_norm": Taps._stats(hidden),
                           "mlp_down_in": [taps.down_absmax[i] for i in range(len(taps.text.layers))]})
        arrays.update(logits=np.asarray(flat_logits, np.float32), probs=np.asarray(flat_probs, np.float32),
                      n_options=np.asarray(n_opts, np.int64), question_spans=np.asarray(qspans, np.int64).reshape(-1, 2),
                      option_spans=np.asarray(ospans, np.int64).reshape(-1, 2), question_types=np.asarray(qtypes, np.int64))
        if taps.hidden_states:
            hs = torch.stack([taps.hidden_states[i] for i in range(len(taps.text.layers) + 1)])     # [33, T, 4096]
            arrays["hidden_states"] = hs.numpy().astype(np.float32)
            renorm = taps.text.norm(hs[-1][None])[0]
            rec["hidden_states"] = {"shape": list(hs.shape), "index_0": "layer-0 input (embeddings, image rows scattered in)",
                                    "index_i": "output of decoder layer i-1 (residual stream, before the final norm)",
                                    "final_norm_of_last_equals_last_hidden": bool(torch.equal(renorm, hidden[0])),
                                    "final_norm_of_last_max_abs_diff": float((renorm - hidden[0]).abs().max())}
        return rec, arrays, enc

    n_runs = 0
    stopped_at_deadline = False
    image_arms = [a for a in args.arm_list if a != "text"]
    for r in records:
        arms = image_arms if r.get("image_files") else (["text"] if "text" in args.arm_list else [])
        for arm in arms:
            if (r["id"], arm) in done:
                continue
            if args.deadline_s is not None and time.monotonic() - t_start > args.deadline_s:
                stopped_at_deadline = True
                break
            rec, arrays, _ = run(r, arm)
            save_npz(out_dir / "npz" / f"{r['id']}__{arm}.npz", arrays)
            with open(partial, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_runs += 1
            ans = " | ".join(f"{q['question_id']}={q['argmax_id']} {max(q['probs']):.3f}" for q in rec["questions"][:4])
            print(f"{r['id']:28s} {arm:6s} T={rec['tokens']:5d} img={rec['n_image_tokens']:4d} {ans}  "
                  f"{rec['wall_s']:.1f}s", flush=True)
        if stopped_at_deadline:
            break
    if stopped_at_deadline:
        print(f"DEADLINE: {args.deadline_s:.0f} s reached after {n_runs} runs; resume with --resume", flush=True)

    # Independent forwards: model(collate_records(...)) must reproduce the hooked logits bit for bit.
    indep = []
    if not args.records and not args.limit and not args.no_independent and not stopped_at_deadline:
        oracle_rows = {(x["id"], x["arm"]): x for x in map(json.loads, partial.read_text().splitlines())}
        first_semif = next((x["id"] for x in all_records if x["source"] == "semif_authored144"), None)
        chosen = (tuple(tuple(k.split(":")) for k in args.independent.split(",")) if args.independent
                  else INDEPENDENT_RUNS + (((first_semif, "text"),) if first_semif else ()))
        for rid, arm in chosen:
            r = next(x for x in all_records if x["id"] == rid)
            request = copy.deepcopy(r["request"])
            if arm != "text":
                request["images"] = [load_image(f, arm) for f in r["image_files"]]
            enc = jsm.encode_record(tok, request, processor=processor)
            batch = jsm.collate_records([enc], pad_id, torch.device("cpu"))
            taps.reset()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model(batch)[0]
            wall = time.perf_counter() - t0
            base = oracle_rows[(rid, arm)]
            eq = [list(map(float, t.float().tolist())) == q["logits"] for t, q in zip(out, base["questions"])]
            item = {"id": rid, "arm": arm, "logits_bit_equal_to_systemone_run": all(eq), "per_question": eq, "wall_s": wall,
                    "role": "determinism (fixture record 0)" if (rid, arm) == ("own_t01", "text") else "independent model(batch) call"}
            indep.append(item)
            print(f"independent {rid}/{arm}: bit-equal {all(eq)} ({wall:.1f}s)", flush=True)
            assert all(eq), item
    header["independent_forwards"] = indep
    header["run_seconds_this_invocation"] = time.monotonic() - t_start
    header["runs_this_invocation"] = n_runs
    header["invocations"] = header["invocations"] + [{"runs": n_runs, "seconds": header["run_seconds_this_invocation"],
                                                       "records_arg": args.records, "limit": args.limit, "resume": args.resume,
                                                       "records_json_sha256": header["fixture"]["records_json_sha256"],
                                                       "independent_forwards": len(indep),
                                                       "deadline_s": args.deadline_s,
                                                       "stopped_at_deadline": stopped_at_deadline,
                                                       "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}]
    if not indep and prev_header.get("independent_forwards"):
        header["independent_forwards"] = prev_header["independent_forwards"]
    write_atomic(header_path, (json.dumps(header, indent=1) + "\n").encode())
    return finalize(args, doc, partial, out_dir, res_dir, header)


def finalize(args, doc, partial, out_dir, res_dir, header):
    rows = [json.loads(line) for line in partial.read_text().splitlines()]
    order = {r["id"]: i for i, r in enumerate(doc["records"])}
    arm_rank = {"text": 0, "g256": 1, "g448": 2, "native": 3, "g672": 4, "g896": 5}
    rows.sort(key=lambda x: (order[x["id"]], arm_rank[x["arm"]]))
    seen = set()
    for x in rows:
        assert (x["id"], x["arm"]) not in seen, ("duplicate run", x["id"], x["arm"])
        seen.add((x["id"], x["arm"]))
    image_arms = [a for a in args.arm_list if a != "text"]
    text_arms = ["text"] if "text" in args.arm_list else []
    expected = {(r["id"], a) for r in doc["records"] for a in (image_arms if r.get("image_files") else text_arms)}
    complete = expected <= seen
    slim_keys = ("id", "arm", "source", "tokens", "n_image_tokens", "state_tokens", "ids", "wall_s", "head_rerun_s", "device",
                 "grid_thw", "token_offset", "image_files", "image_size_in", "image_size_fed", "resized_hw", "merged_hw",
                 "rope_shift_amount", "pixel_values_shape", "pixel_values_sha256", "image_embeds_shape",
                 "mm_token_type_ids_media", "questions", "systemone_response", "checks", "hidden_states")
    slim = [{k: x[k] for k in slim_keys if k in x} for x in rows]
    full = dict(header, complete=complete, n_runs=len(rows), n_expected_runs=len(expected),
                missing_runs=sorted(map(list, expected - seen))[:20], rows=slim)
    write_atomic(out_dir / f"records_ref{args.tag}.json", (json.dumps(full, indent=1, ensure_ascii=False) + "\n").encode())

    # summary
    def pct(v, q):
        return float(np.percentile(np.asarray(v, dtype=np.float64), q)) if v else None

    def group(src_pred):
        rs = [x for x in rows if src_pred(x)]
        qs = [q for x in rs for q in x["questions"]]
        gold = [q for q in qs if q["gold"] is not None]
        return {
            "runs": len(rs), "records": len({x["id"] for x in rs}), "questions": len(qs),
            "types": {t: sum(q["type"] == t for q in qs) for t in ("noul", "choice", "score")},
            "tokens_p50": pct([x["tokens"] for x in rs], 50), "tokens_p99": pct([x["tokens"] for x in rs], 99),
            "tokens_max": max((x["tokens"] for x in rs), default=None),
            "max_options": max((len(q["option_ids"]) for q in qs), default=None),
            "near_tie": sum(q["near_tie"] for q in qs),
            "near_tie_ids": [f"{x['id']}/{x['arm']}/{q['question_id']}" for x in rs for q in x["questions"] if q["near_tie"]],
            "gold_questions": len(gold), "gold_correct": sum(bool(q["gold_correct"]) for q in gold),
            "gold_accuracy": (sum(bool(q["gold_correct"]) for q in gold) / len(gold)) if gold else None,
            "wall_s_p50": pct([x["wall_s"] for x in rs], 50), "wall_s_max": max((x["wall_s"] for x in rs), default=None),
            "all_checks_pass": all(all(v for k, v in x["checks"].items() if isinstance(v, bool) and k != "head_rerun_bit_equal")
                                   for x in rs),
            "head_rerun_bit_equal_runs": sum(bool(x["checks"]["head_rerun_bit_equal"]) for x in rs),
            "head_rerun_max_abs_diff": max((x["checks"]["head_rerun_max_abs_diff"] for x in rs), default=None),
        }

    summary = {"complete": complete, "runs": len(rows), "expected_runs": len(expected), "near_tie_threshold": NEAR_TIE,
               "by_source_arm": {}, "own_subset": None, "semif": None}
    for src in sorted({x["source"] for x in rows}):
        for arm in ("text", "g256", "g448", "native", "g672", "g896"):
            g = group(lambda x, s=src, a=arm: x["source"] == s and x["arm"] == a)
            if g["runs"]:
                summary["by_source_arm"][f"{src}/{arm}"] = g
    summary["own_subset"] = group(lambda x: x["source"] != "semif_authored144")
    summary["own_subset_text_and_g256"] = group(lambda x: x["source"] != "semif_authored144" and x["arm"] in ("text", "g256"))
    summary["semif"] = group(lambda x: x["source"] == "semif_authored144")
    summary["all"] = group(lambda x: True)
    semif_fam = {}
    rec_by_id = {r["id"]: r for r in doc["records"]}
    for x in rows:
        if x["source"] == "semif_authored144":
            fam = rec_by_id[x["id"]]["semif"]["family"]
            d = semif_fam.setdefault(fam, {"n": 0, "correct": 0})
            d["n"] += 1
            d["correct"] += bool(x["questions"][0]["gold_correct"])
    summary["semif_by_family"] = semif_fam
    summary["independent_forwards"] = header.get("independent_forwards")
    write_atomic(res_dir / f"ref_summary{args.tag}.json", (json.dumps(summary, indent=1) + "\n").encode())

    # activation absmax survey
    L = len(rows[0]["absmax"]["layers"]) if rows else 0

    def worst(getter):
        best = None
        for x in rows:
            v = getter(x)
            if v is None:
                continue
            if best is None or v > best[0]:
                best = (v, f"{x['id']}/{x['arm']}")
        return {"max": best[0], "run": best[1]} if best else None

    per_layer = []
    for i in range(L):
        per_layer.append({"layer": i,
                          "pos0": worst(lambda x, i=i: x["absmax"]["layers"][i]["pos0"]),
                          "rest": worst(lambda x, i=i: x["absmax"]["layers"][i]["rest"]),
                          "overall": worst(lambda x, i=i: x["absmax"]["layers"][i]["overall"]),
                          "mlp_down_in": worst(lambda x, i=i: x["absmax"]["mlp_down_in"][i]),
                          "argmax_pos_values": sorted({x["absmax"]["layers"][i]["argmax_pos"] for x in rows})[:8],
                          "argmax_ch_values": sorted({x["absmax"]["layers"][i]["argmax_ch"] for x in rows})[:8]})
    absmax = {
        "what": ("absmax over every fixture run (CPU fp32 oracle) of the residual stream: embed_tokens output, layer-0 input "
                 "(image rows scattered in), each decoder layer output; pos0 = position 0, rest = positions >= 1; and of the "
                 "MLP down_proj input per layer; fp16 max = 65504"),
        "runs": len(rows),
        "embed_tokens": {"pos0": worst(lambda x: x["absmax"]["embed_tokens"]["pos0"]),
                         "overall": worst(lambda x: x["absmax"]["embed_tokens"]["overall"])},
        "layer_input": {"pos0": worst(lambda x: x["absmax"]["layer_input"]["pos0"]),
                        "rest": worst(lambda x: x["absmax"]["layer_input"]["rest"]),
                        "overall": worst(lambda x: x["absmax"]["layer_input"]["overall"])},
        "layers": per_layer,
        "final_norm": {"pos0": worst(lambda x: x["absmax"]["final_norm"]["pos0"]),
                       "rest": worst(lambda x: x["absmax"]["final_norm"]["rest"]),
                       "overall": worst(lambda x: x["absmax"]["final_norm"]["overall"])},
        "residual_overall_max": worst(lambda x: max(l["overall"] for l in x["absmax"]["layers"])),
        "residual_rest_max": worst(lambda x: max((l["rest"] or 0.0) for l in x["absmax"]["layers"])),
        "mlp_down_in_max": worst(lambda x: max(x["absmax"]["mlp_down_in"])),
        "fp16_max": 65504.0,
    }
    write_atomic(res_dir / f"activation_absmax{args.tag}.json", (json.dumps(absmax, indent=1) + "\n").encode())
    print(json.dumps({k: summary[k] for k in ("complete", "runs", "expected_runs")}, indent=1))
    print(f"own subset: {summary['own_subset']['questions']} q, gold {summary['own_subset']['gold_correct']}/"
          f"{summary['own_subset']['gold_questions']}, near-tie {summary['own_subset']['near_tie']}; semif: "
          f"{summary['semif']['gold_correct']}/{summary['semif']['gold_questions']}")
    print(f"residual absmax overall {absmax['residual_overall_max']}, rest {absmax['residual_rest_max']}, "
          f"mlp_down_in {absmax['mlp_down_in_max']}")
    print(f"wrote {out_dir / f'records_ref{args.tag}.json'}, {res_dir / f'ref_summary{args.tag}.json'}, "
          f"{res_dir / f'activation_absmax{args.tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
