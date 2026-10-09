"""Round 2's reference: the provider's own code, CPU fp32, on every fixture record -> ref/records_ref.json

    cd K
    HF_HUB_DISABLE_XET=1 HF_MODULES_CACHE=cache/hf_modules_r2 venv-ref/bin/python scripts/ref_probs.py --verify-weights
        -> results/ref_env.json
    HF_HUB_DISABLE_XET=1 HF_MODULES_CACHE=cache/hf_modules_r2 ~/code/standup/tools/quiet/quiet_wait.py -- \
        venv-ref/bin/python scripts/ref_probs.py
        -> ref/npz/<record id>.npz, results/ref_summary.json, then ref/records_ref.json (.tmp -> os.replace, last)
    ... --ids a,b --out X --npz-dir Y --summary Z      a subset written elsewhere (a look run; the defaults refuse --ids)

--verify-weights: the shared blob's sha256 recomputed and asserted equal to its name (= the LFS sha256), one online
snapshot_download(repo, revision=<sha>) with the cache's blob listing (name, size, mtime) compared before and after
(= 0 bytes fetched), the versions, torch threads (set to 12) and the MHA fast path.

Model = AutoModel.from_pretrained(<snapshot dir>, trust_remote_code=True, dtype=torch.float32), eval, CPU,
torch.set_num_threads(12), HF_HUB_OFFLINE=1; HF_MODULES_CACHE (where trust_remote_code copies the snapshot's .py
files) is made absolute against K here, before transformers is imported. The provider's files are not edited: the
numbers are read through instance wrappers on _run / _forward (the rows of each batch and the probabilities it
returns) and forward hooks on encoder (its output = the trunk after the final norm), head (its output = the marker
logits before the temperature, all K slots), vision / audio (the prefix), audio.frontend (mel, frames) and
vision.tower (NaFlex inputs, last_hidden_state), plus an instance wrapper on the tower embeddings'
resize_positional_embeddings (the position table after the resize).

Per record, in fixture order:
  natural    model.probabilities(state, questions, images, audio)    all accepted questions in one call
  system_one model.system_one(state, {name: q}, images, audio)        the response == the host's answer() dict
  single     model.probabilities(state, [q], images, audio)           each question alone (one row, no padding)
Asserted per question: the host readout (host/d1_host.py readout(), the provider's _forward tail in torch f32) on the
captured logits == the returned probabilities bit for bit (natural and single); the host's ids / markers (tokenizers
adapter) == the provider's rows; usage.input_tokens == sum(P + len(ids)); the prefix length == the provider's
layout() / the audio subsampling arithmetic; per batch, the head's input == the trunk output cut per row and the head
re-run on it == its logits (torch.equal). Determinism: record 0's natural call again right after it, and a second
pass over every record (natural call) at the end, both compared bit for bit (logits and probabilities).
Media are read the way the card reads them: images with transformers.image_utils.load_image(path), audio with
soundfile.read(path, dtype="int16"); each file's sha256 is asserted against the fixture.

npz per record (ref/npz/<id>.npz, every record): `qids`; per question `ids.<qid>`, `markers.<qid>`,
`logits_raw.<qid>` / `logits_raw_single.<qid>` [K], `h_markers.<qid>` / `h_markers_single.<qid>` [K, 1024] (trunk
output at P + marker, natural / single call); media records `prefix` [P, 1024] (the vision / audio output as is),
audio `mel` [128, T] + `frames` [1], image `pixel_values` / `spatial_shapes` / `pixel_attention_mask` (NaFlex inputs),
`tower_last_hidden_state`, `pos_resized` [crops, 1024, 768]; three rows `h_full.<qid>` [P + n, 1024] (single call:
card_text/refund, the first semif record, long_3400/component).
"""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
K = Path(__file__).resolve().parents[1]
_mc = os.environ.get("HF_MODULES_CACHE", "cache/hf_modules_r2")
os.environ["HF_MODULES_CACHE"] = str(_mc if os.path.isabs(_mc) else (K / _mc).resolve())

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import d1_src as S  # noqa: E402

THREADS = 12
NEAR_TIE = 0.02
BUCKETS = (128, 256, 512, 1024, 2048, 4096)  # host/d1_host.py BUCKETS
H_FULL = (("card_text", "refund"), ("<first semif>", "answer"), ("long_3400", "component"))
BLOBS = S.SNAP.parents[1] / "blobs"
WEIGHTS_SHA256 = "0713bb05270c2685ad106522f4092bceeeb3a93cf79b401f399a712296c911e1"
WEIGHTS_BYTES = 2348774500
DEFAULT_OUT = K / "ref/records_ref.json"


def versions():
    import importlib.metadata as md

    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in ("torch", "transformers", "numpy", "tokenizers", "safetensors", "huggingface_hub", "soundfile",
              "torchvision", "pillow"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def blob_listing():
    return {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(BLOBS.iterdir())}


def verify_weights():
    t0 = time.time()
    blob = BLOBS / WEIGHTS_SHA256
    assert S.WEIGHTS.resolve() == blob.resolve(), (S.WEIGHTS.resolve(), blob)
    incomplete_before = sorted(p.name for p in BLOBS.iterdir() if p.name.endswith(".incomplete"))
    assert not incomplete_before, incomplete_before
    t1 = time.time()
    sha = S.sha256_file(blob)
    sha_s = time.time() - t1
    assert sha == WEIGHTS_SHA256, sha
    assert blob.stat().st_size == WEIGHTS_BYTES, blob.stat().st_size
    trees = S.SNAP.parents[1] / "trees" / f"{S.REV}.json"
    before = blob_listing()
    trees_before = trees.stat().st_mtime_ns if trees.exists() else None
    from huggingface_hub import constants, snapshot_download

    assert not constants.HF_HUB_OFFLINE, "--verify-weights makes one online call: unset HF_HUB_OFFLINE"
    t2 = time.time()
    path = snapshot_download(S.REPO, revision=S.REV)
    dl_s = time.time() - t2
    after = blob_listing()
    added = sorted(set(after) - set(before))
    changed = sorted(k for k in before if k in after and before[k] != after[k])
    fetched = sum(after[k][0] for k in added) + sum(after[k][0] for k in changed)
    assert Path(path).resolve() == S.SNAP.resolve(), path
    assert not added and not changed, (added, changed)
    incomplete_after = sorted(p.name for p in BLOBS.iterdir() if p.name.endswith(".incomplete"))
    assert not incomplete_after, incomplete_after
    tree_doc = json.loads(trees.read_text())

    import torch

    torch.set_num_threads(THREADS)
    out = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "repo": S.REPO, "revision": S.REV,
        "blob": str(blob), "blob_bytes": blob.stat().st_size, "blob_sha256": sha,
        "blob_sha256_equals_lfs": sha == tree_doc["files"]["model.safetensors"]["lfs_sha256"] == WEIGHTS_SHA256,
        "sha256_seconds": round(sha_s, 1), "snapshot_symlink": str(S.WEIGHTS), "snapshot_symlink_target": str(
            os.readlink(S.WEIGHTS)),
        "snapshot_download": {"call": f"snapshot_download({S.REPO!r}, revision={S.REV!r}) online, HF_HUB_DISABLE_XET="
                                      f"{os.environ.get('HF_HUB_DISABLE_XET')}", "returned": path,
                              "seconds": round(dl_s, 1), "blobs_before": len(before), "blobs_after": len(after),
                              "blobs_added": added, "blobs_changed": changed, "bytes_fetched": fetched,
                              "trees_json": str(trees), "trees_json_existed_before": trees_before is not None,
                              "trees_json_rewritten": trees_before is not None and trees.stat().st_mtime_ns != trees_before,
                              "trees_files": sorted(tree_doc["files"])},
        "incomplete_blobs": {"before": incomplete_before, "after": incomplete_after},
        "versions": versions(),
        "torch_threads": {"set": THREADS, "get_num_threads": torch.get_num_threads(),
                          "get_num_interop_threads": torch.get_num_interop_threads()},
        "mha_fastpath_enabled": torch.backends.mha.get_fastpath_enabled(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "hf_modules_cache": os.environ["HF_MODULES_CACHE"], "dont_write_bytecode": sys.dont_write_bytecode,
        "seconds": round(time.time() - t0, 1),
    }
    (K / "results/ref_env.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


# --------------------------------------------------------------------------- model reading


class Taps:
    """Reads the provider's model without changing its files: instance wrappers on _run / _forward and on the tower
    embeddings' resize_positional_embeddings, forward hooks on the trunk, the head and the media modules."""

    def __init__(self, model):
        import torch

        self.torch = torch
        self.flat, self.cur, self.calls = None, None, []
        self.media = {"vision": [], "tower": [], "audio": [], "frontend": [], "pos": []}
        orig_run, orig_forward = model._run, model._forward

        def run(rows, *args, **kwargs):
            self.flat = rows
            return orig_run(rows, *args, **kwargs)

        def forward(rows):
            self.cur = {"rows": rows}
            out = orig_forward(rows)
            self.cur["probs"] = out
            self.calls.append(self.cur)
            self.cur = None
            return out

        model._run, model._forward = run, forward
        emb = model.vision.tower.embeddings
        resize = type(emb).resize_positional_embeddings

        def resize_and_keep(positional_embeddings, spatial_shapes, max_length):
            out = resize(positional_embeddings, spatial_shapes, max_length)
            self.media["pos"].append((spatial_shapes.detach().clone(), out.detach().clone()))
            return out

        emb.resize_positional_embeddings = resize_and_keep
        model.encoder.register_forward_hook(self._trunk)
        model.head.register_forward_hook(self._head)
        model.vision.register_forward_hook(lambda m, a, o: self.media["vision"].append(o.detach().clone()))
        model.vision.tower.register_forward_hook(self._tower, with_kwargs=True)
        model.audio.register_forward_hook(lambda m, a, o: self.media["audio"].append(o.detach().clone()))
        model.audio.frontend.register_forward_hook(
            lambda m, a, o: self.media["frontend"].append((o[0].detach().clone(), o[1].detach().clone())))

    def _trunk(self, module, args, output):
        if self.cur is not None:
            h, pad, prefix = args
            self.cur["trunk_in"] = h.detach().clone()
            self.cur["trunk_pad"], self.cur["trunk_prefix"] = pad.detach().clone(), prefix.detach().clone()
            self.cur["trunk_out"] = output.detach().clone()

    def _head(self, module, args, output):
        if self.cur is not None:
            self.cur["head_in"] = tuple(a.detach().clone() for a in args)
            self.cur["head_out"] = output.detach().clone()

    def _tower(self, module, args, kwargs, output):
        self.media["tower"].append({k: v.detach().clone() for k, v in kwargs.items()}
                                   | {"last_hidden_state": output.last_hidden_state.detach().clone()})

    def reset(self):
        self.flat, self.cur, self.calls = None, None, []
        for v in self.media.values():
            v.clear()


def load_media(record):
    """-> (images or None, audio or None, info). The card's readers; the file's sha256 asserted."""
    media = record.get("media")
    if media is None:
        return None, None, None
    path = K / media["ref"]
    sha = S.sha256_file(path)
    assert sha == media["sha256"], (record["id"], sha, media["sha256"])
    info = {"kind": media["kind"], "file": media["ref"], "sha256": sha, "bytes": path.stat().st_size}
    if media["kind"] == "image":
        from PIL import Image
        from transformers.image_utils import load_image

        image = load_image(str(path))
        with Image.open(path) as raw:
            info["pil_mode_file"] = raw.mode
            info["same_pixels_as_plain_open_rgb"] = bool(np.array_equal(np.asarray(image),
                                                                        np.asarray(raw.convert("RGB"))))
        info["px"] = list(image.size)
        info["pil_mode"] = image.mode
        return [image], None, info
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="int16")
    assert rate == 16000 and audio.ndim == 1, (record["id"], rate, audio.shape)
    info.update({"sample_rate": rate, "samples": int(audio.shape[0]), "seconds": round(audio.shape[0] / rate, 4),
                 "read_dtype": str(audio.dtype)})
    return None, audio, info


def expected_prefix(vision_mod, images, audio):
    """The provider's own arithmetic: layout() crops for an image, n // 160 then three stride-2 steps for audio."""
    if images is not None:
        total = 0
        for im in images:
            plan = vision_mod.layout(*im.convert("RGB").size)
            if plan["tiled"]:
                gw, gh = plan["grid"]
                total += gw * gh * (vision_mod.TILE // 16) ** 2 // 4
            th, tw = plan["thumbnail"]
            total += (th // 16) * (tw // 16) // 4
        return total
    if audio is not None:
        n = min(len(audio), 30 * 16000)
        n = max(n, 8000)
        frames = (n + 512 // 2 * 2 - 512) // 160
        for _ in range(3):
            frames = (frames + 2 - 3) // 2 + 1
        return frames
    return 0


def argmax_key(q, probs):
    best = max(range(len(probs)), key=probs.__getitem__)
    if q.type == "noul":
        return best, ("true" if best == 0 else "false")  # probabilities are [yes, no]
    if q.type == "choice":
        return best, list(q.criteria)[best]
    return best, str(best)


def bucket_for(n):
    return next((b for b in BUCKETS if n <= b), None)


def floats(t):
    return [float(v) for v in t.reshape(-1).tolist()]


class Oracle:
    def __init__(self, model, taps, host_mod, temps):
        self.model, self.taps, self.H, self.temps = model, taps, host_mod, temps
        self.host = host_mod.D1Host(S.SNAP, runner=None)
        self.torch = taps.torch

    def per_question(self, calls, names):
        """The captured batches -> one dict per question (question order), via the identity of the row objects."""
        flat = self.taps.flat
        index = {id(row): j for j, row in enumerate(flat)}
        out = [None] * len(names)
        for c, call in enumerate(calls):
            offs = call["trunk_prefix"].tolist()
            for b, row in enumerate(call["rows"]):
                j = index[id(row)]
                prefix, ids, markers, q, calibrate = row
                p = 0 if prefix is None else int(prefix.shape[1])
                assert offs[b] == p
                out[j] = {"call": c, "b": b, "B": len(call["rows"]), "P": p, "ids": list(ids),
                          "markers": list(markers), "q": q, "calibrate": calibrate,
                          "logits_full": call["head_out"][b].clone(), "trunk": call["trunk_out"][b],
                          "probs_returned": list(call["probs"][b])}
        assert all(x is not None for x in out)
        return out

    def hook_proof(self, calls):
        torch = self.torch
        proof = []
        for call in calls:
            rows, out = call["rows"], call["trunk_out"]
            offs = [0 if r[0] is None else int(r[0].shape[1]) for r in rows]
            lens = [len(r[1]) for r in rows]
            text = torch.nn.utils.rnn.pad_sequence([out[i, o:o + n] for i, (o, n) in enumerate(zip(offs, lens))],
                                                   batch_first=True)
            head_in = call["head_in"]
            same = torch.equal(text, head_in[0])
            with torch.no_grad():
                again = self.model.head(text, *head_in[1:])
            proof.append({"B": len(rows), "head_input_is_trunk_rows": same,
                          "head_rerun_bit_equal": torch.equal(again, call["head_out"])})
        return proof

    def host_readout(self, x):
        """The host's readout() on a score vector holding the captured logits at P + marker (zeros elsewhere)."""
        q = self.H.Pm.as_question(x["qd"])
        K_ = q.options
        scores = np.zeros((1, x["P"] + len(x["ids"])), np.float32)
        for k, m in enumerate(x["markers"][:K_]):
            scores[0, x["P"] + m] = float(x["logits_full"][k])
        p_t = self.H.readout(scores, x["P"], x["markers"], q, x["calibrate"], self.temps)
        p_64 = self.H.readout_f64(scores, x["P"], x["markers"], q, x["calibrate"], self.temps)
        return p_t, p_64

    def natural(self, state, qds, images, audio):
        self.taps.reset()
        t0 = time.perf_counter()
        probs = self.model.probabilities(state, qds, images, audio)
        sec = time.perf_counter() - t0
        calls = list(self.taps.calls)
        media = {k: list(v) for k, v in self.taps.media.items()}
        return probs, calls, media, sec


def run_record(o, record, names, images, audio, vision_mod, keep_full):
    torch = o.torch
    qdict = record["request"]["questions"]
    qds = [qdict[n] for n in names]
    state = record["request"]["state"]
    kind = "text" if record["media"] is None else record["media"]["kind"]

    # natural call
    probs, calls, media, sec = o.natural(state, qds, images, audio)
    per_q = o.per_question(calls, names)
    for j, x in enumerate(per_q):
        x["qd"] = qds[j]
        assert x["probs_returned"] == list(probs[j]), (record["id"], j)
    proof = o.hook_proof(calls)
    assert all(p["head_input_is_trunk_rows"] and p["head_rerun_bit_equal"] for p in proof), (record["id"], proof)
    P = per_q[0]["P"]
    assert all(x["P"] == P for x in per_q)
    p_expected = expected_prefix(vision_mod, images, audio)
    assert P == p_expected, (record["id"], P, p_expected)
    prefix_t = None
    if kind == "image":
        assert len(media["vision"]) == 1
        prefix_t = media["vision"][0]
    elif kind == "audio":
        assert len(media["audio"]) == 1
        prefix_t = media["audio"][0]
    if prefix_t is not None:
        assert prefix_t.shape == (1, P, 1024), prefix_t.shape

    # host rows: tokenizers adapter ids / markers == the provider's rows
    stub = None if P == 0 else np.zeros((P, 1), np.float32)
    hrows = o.host.rows(state, qds, stub, kind)
    ids_equal = all(h["ids"] == x["ids"] and h["markers"] == x["markers"] and h["P"] == x["P"]
                    and h["calibrate"] == x["calibrate"] for h, x in zip(hrows, per_q))
    assert ids_equal, record["id"]

    # host readout == returned probabilities, bit for bit
    host_bits, host64 = [], []
    for j, x in enumerate(per_q):
        p_t, p_64 = o.host_readout(x)
        host_bits.append(p_t == list(probs[j]))
        host64.append(max(abs(a - b) for a, b in zip(p_64, probs[j])))
    assert all(host_bits), (record["id"], host_bits)

    # system_one: the response == host answer() dict (on that run's probabilities; and on the natural run's)
    o.taps.reset()
    t1 = time.perf_counter()
    resp = o.model.system_one(state, {n: qdict[n] for n in names}, images, audio)
    sec_so = time.perf_counter() - t1
    calls_so = list(o.taps.calls)
    per_so = o.per_question(calls_so, names)
    probs_so = [x["probs_returned"] for x in per_so]
    hqs = [o.H.Pm.as_question(qd) for qd in qds]
    usage = sum(P + len(x["ids"]) for x in per_q)
    mine_so = {"answers": {n: o.H.Pm.answer(hq, p) for n, hq, p in zip(names, hqs, probs_so)},
               "usage": {"input_tokens": usage, "output_tokens": 0}}
    mine_nat = {"answers": {n: o.H.Pm.answer(hq, list(p)) for n, hq, p in zip(names, hqs, probs)},
                "usage": {"input_tokens": usage, "output_tokens": 0}}
    assert mine_so == resp, (record["id"], mine_so, resp)
    so_logits_equal = all(torch.equal(a["logits_full"], b["logits_full"]) for a, b in zip(per_so, per_q))

    # single calls
    singles, sec_single = [], 0.0
    for j, n in enumerate(names):
        p1, calls1, _, s1 = o.natural(state, [qdict[n]], images, audio)
        sec_single += s1
        x1 = o.per_question(calls1, [n])[0]
        x1["qd"] = qdict[n]
        assert x1["B"] == 1 and x1["ids"] == per_q[j]["ids"] and x1["P"] == P
        p_t, _ = o.host_readout(x1)
        assert p_t == list(p1[0]), (record["id"], n)
        x1["probs"] = list(p1[0])
        singles.append(x1)

    questions, npz = [], {"qids": np.array(names)}
    gold_map = record.get("gold") or {}
    for j, (n, x, hq) in enumerate(zip(names, per_q, hqs)):
        q = x["q"]
        k = q.options
        z = x["logits_full"][:k]
        assert torch.all(x["logits_full"][k:] == -1e4)
        T = (o.temps.get(o.H.Pm.temperature_key(hq), o.temps.get(q.type, 1.0)) if x["calibrate"] else None)
        p_raw = torch.softmax(z, -1)
        p_pub = [float(v) for v in probs[j]]
        best, key = argmax_key(q, p_pub)
        order = sorted(p_pub, reverse=True)
        margin = float(order[0] - order[1])
        gold = gold_map.get(n)
        positions = P + len(x["ids"])
        s = singles[j]
        z1 = s["logits_full"][:k]
        dps = max(abs(a - b) for a, b in zip(p_pub, s["probs"]))
        mk = [P + m for m in x["markers"]]
        questions.append({
            "qid": n, "type": q.type, "K": k, "calibrate": x["calibrate"],
            "temperature_key": o.H.Pm.temperature_key(hq) if x["calibrate"] else None, "T": T,
            "prefix": P, "n_ids": len(x["ids"]), "positions": positions, "bucket": bucket_for(positions),
            "ids": x["ids"], "markers": x["markers"], "logits_raw": floats(z), "probs_raw": floats(p_raw),
            "probs": p_pub, "argmax_index": best, "argmax": key, "top2_margin": margin,
            "near_tie": margin <= NEAR_TIE, "gold": gold, "correct": None if gold is None else key == gold,
            "host_softmax_max_abs_dp": 0.0 if host_bits[j] else None, "host_math_bit_equal": host_bits[j],
            "host_numpy_f64_max_abs_dp": host64[j],
            "batch": {"call": x["call"], "index": x["b"], "B": x["B"]},
            "probs_single": s["probs"], "logits_raw_single": floats(z1), "max_abs_dp_batch_vs_single": dps,
            "max_abs_dlogit_batch_vs_single": float((z - z1).abs().max()),
        })
        npz[f"ids.{n}"] = np.asarray(x["ids"], np.int32)
        npz[f"markers.{n}"] = np.asarray(x["markers"], np.int32)
        npz[f"logits_raw.{n}"] = z.numpy().astype(np.float32)
        npz[f"logits_raw_single.{n}"] = z1.numpy().astype(np.float32)
        npz[f"h_markers.{n}"] = x["trunk"][mk].numpy().astype(np.float32)
        npz[f"h_markers_single.{n}"] = s["trunk"][mk].numpy().astype(np.float32)
        if n == keep_full:
            npz[f"h_full.{n}"] = s["trunk"][:positions].numpy().astype(np.float32)
    if prefix_t is not None:
        npz["prefix"] = prefix_t[0].numpy().astype(np.float32)
    if kind == "audio":
        assert len(media["frontend"]) == 1
        mel, frames = media["frontend"][0]
        npz["mel"] = mel[0].numpy().astype(np.float32)
        npz["frames"] = frames.numpy().astype(np.int64)
    if kind == "image":
        assert len(media["tower"]) == 1 and len(media["pos"]) == 1
        tw = media["tower"][0]
        for kk in ("pixel_values", "spatial_shapes", "pixel_attention_mask"):
            npz[kk] = tw[kk].numpy()
        npz["tower_last_hidden_state"] = tw["last_hidden_state"].numpy().astype(np.float32)
        npz["pos_resized"] = media["pos"][0][1].numpy().astype(np.float32)
    entry = {"id": record["id"], "source": record["source"], "public": record["publishable"], "mode": kind,
             "native": True, "prefix": P, "prefix_expected": p_expected, "seconds": round(sec, 4),
             "seconds_system_one": round(sec_so, 4), "seconds_single": round(sec_single, 4),
             "batches": [len(c["rows"]) for c in calls], "questions": questions, "response": resp,
             "response_equal_host": True, "response_equal_host_on_natural_probs": mine_nat == resp,
             "system_one_logits_bit_equal_natural": so_logits_equal,
             "usage_input_tokens": resp["usage"]["input_tokens"],
             "usage_equal_host": resp["usage"]["input_tokens"] == usage, "hook_proof": proof,
             "max_abs_dp_batch_vs_single": max(q_["max_abs_dp_batch_vs_single"] for q_ in questions)}
    return entry, npz, [x["logits_full"] for x in per_q], [list(p) for p in probs]


def load_model():
    """The provider's model, fp32 CPU eval, its state asserted equal to model.safetensors; -> (model, load record,
    prompt module, vision module, temperatures)."""
    import torch
    from transformers import AutoModel

    t0 = time.time()
    model, info = AutoModel.from_pretrained(str(S.SNAP), trust_remote_code=True, dtype=torch.float32,
                                            output_loading_info=True)
    model.eval()
    load_s = time.time() - t0
    pkg = type(model).__module__.rsplit(".", 1)[0]
    prompt_mod, vision_mod = sys.modules[pkg + ".prompt"], sys.modules[pkg + ".vision"]
    cfg = model.config
    assert (cfg.max_length, cfg.image_text_length, cfg.audio_text_length) == (
        S.MAX_LENGTH, S.IMAGE_TEXT_LENGTH, S.AUDIO_TEXT_LENGTH)
    temps = dict(cfg.temperatures)
    assert temps == S.config()["temperatures"]

    # the loaded state == model.safetensors, tensor for tensor (transformers renames the tower's vision_model level)
    from safetensors import safe_open

    sd = model.state_dict()
    renamed, n_eq, mismatched = 0, 0, []
    with safe_open(str(S.WEIGHTS), framework="pt") as f:
        keys = list(f.keys())
        for kk in keys:
            ours = kk
            if ours not in sd and kk.startswith("vision.tower.vision_model."):
                ours = "vision.tower." + kk[len("vision.tower.vision_model."):]
                renamed += 1
            if ours in sd and torch.equal(sd[ours], f.get_tensor(kk)):
                n_eq += 1
            else:
                mismatched.append(kk)
    load = {"class": type(model).__name__, "module": type(model).__module__, "model_dtype": str(model.dtype),
            "state_dtypes": dict(sorted({str(v.dtype): 0 for v in sd.values()}.items())),
            "tensors": len(sd), "checkpoint_tensors": len(keys), "state_equals_safetensors": not mismatched
            and n_eq == len(keys) == len(sd), "state_mismatched": mismatched[:20],
            "checkpoint_keys_renamed_on_load": {"from": "vision.tower.vision_model.", "to": "vision.tower.",
                                                "tensors": renamed},
            "loading_info": {k: list(v) for k, v in info.items()}, "load_seconds": round(load_s, 2),
            "transformers": versions()["transformers"], "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "mha_fastpath_enabled": torch.backends.mha.get_fastpath_enabled(),
            "training": model.training, "device": str(model.device), "hf_modules_cache": os.environ["HF_MODULES_CACHE"],
            "tokenizer_class": type(model.tokenizer).__name__}
    for v in sd.values():
        load["state_dtypes"][str(v.dtype)] += 1
    assert load["state_equals_safetensors"], mismatched[:5]
    assert not any(info.values()), info

    return model, load, prompt_mod, vision_mod, temps


def build_checks(entries, det0, second, rejected):
    """The file's `checks` from the per-record entries (every per-row equality was asserted in run_record)."""
    # red arm
    by_id = {e["id"]: e for e in entries}
    red = None
    if "red_arm_000" in by_id and "tv4_000" in by_id:
        pa, pb = by_id["red_arm_000"]["questions"][0], by_id["tv4_000"]["questions"][0]
        red = {"pair": ["red_arm_000", "tv4_000"], "max_abs_dp": max(abs(x - y) for x, y in zip(pa["probs"], pb["probs"])),
               "probs_red": pa["probs"], "probs_tv4_000": pb["probs"], "argmax": [pa["argmax"], pb["argmax"]],
               "expected_gt": 0.02}
        red["pass"] = red["max_abs_dp"] > 0.02
    rows = [q for e in entries for q in e["questions"]]
    bvs = sorted(({"id": e["id"], "qid": q["qid"], "mode": e["mode"], "B_in_request": q["batch"]["B"],
                   "positions": q["positions"], "max_abs_dp": q["max_abs_dp_batch_vs_single"],
                   "max_abs_dlogit": q["max_abs_dlogit_batch_vs_single"],
                   "argmax_equal": int(np.argmax(q["probs_single"])) == q["argmax_index"]}
                  for e in entries for q in e["questions"]), key=lambda x: -x["max_abs_dp"])
    checks = {
        "ids_markers_equal_host_request_rows": sum(len(e["questions"]) for e in entries),
        "response_equal_host_response": sum(e["response_equal_host"] for e in entries),
        "response_equal_host_on_natural_probs": sum(e["response_equal_host_on_natural_probs"] for e in entries),
        "usage_input_tokens_equal_host": sum(e["usage_equal_host"] for e in entries),
        "prefix_length_equal_host": [f"{e['id']}/{e['mode']}: {e['prefix']}" for e in entries if e["mode"] != "text"],
        "hook_point": "every batch: head input == trunk output cut per row (torch.equal) and head re-run on it == "
                      "the run's logits (torch.equal)",
        "hook_point_batches": sum(len(e["hook_proof"]) for e in entries),
        "host_math": {"rows": len(rows), "natural_bit_equal": sum(q["host_math_bit_equal"] for q in rows),
                      "single_bit_equal": len(rows), "numpy_f64_max_abs_dp": max(q["host_numpy_f64_max_abs_dp"]
                                                                                 for q in rows)},
        "determinism": det0, "second_run": second,
        "batch_vs_single": {"rows": len(bvs), "rows_in_multi_question_requests": sum(1 for x in bvs
                                                                                     if x["B_in_request"] > 1),
                            "max_abs_dp": bvs[0]["max_abs_dp"], "max_abs_dlogit": max(x["max_abs_dlogit"] for x in bvs),
                            "argmax_equal": sum(x["argmax_equal"] for x in bvs), "top": bvs[:10]},
        "red_arm": red, "rejected_questions": rejected,
        "system_one_logits_bit_equal_natural": sum(e["system_one_logits_bit_equal_natural"] for e in entries),
    }
    return checks


def write_npz(path, arrays):
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return {"bytes": path.stat().st_size, "sha256": S.sha256_file(path),
            "arrays": {k: list(v.shape) for k, v in arrays.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-weights", action="store_true")
    ap.add_argument("--fixtures", default=str(K / "fixtures/requests.json"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--npz-dir", default=str(K / "ref/npz"))
    ap.add_argument("--summary", default=str(K / "results/ref_summary.json"))
    ap.add_argument("--ids", default="", help="comma-separated record ids (a look run: --out elsewhere)")
    ap.add_argument("--no-second-pass", action="store_true")
    ap.add_argument("--rerun-images", action="store_true", help="v1 -> v2: image records with the float resize path")
    a = ap.parse_args()
    if a.verify_weights:
        return verify_weights()
    if a.rerun_images:
        return rerun_images(a)
    if a.ids and Path(a.out).resolve() == DEFAULT_OUT.resolve():
        sys.exit("--ids writes a subset: give --out / --npz-dir / --summary outside ref/ and results/")
    os.environ["HF_HUB_OFFLINE"] = "1"
    t_start = time.time()

    import torch
    from transformers import AutoModel

    torch.set_num_threads(THREADS)
    sys.path.insert(0, str(K / "host"))
    import d1_host as H

    doc = json.loads(Path(a.fixtures).read_text())
    records = doc["records"]
    if a.ids:
        want = a.ids.split(",")
        records = [r for r in records if r["id"] in want]
        assert len(records) == len(want), sorted(set(want) - {r["id"] for r in records})
    semif_first = next(r["id"] for r in doc["records"] if r["source"] == "semif")
    keep_full = {(semif_first if rid == "<first semif>" else rid): qid for rid, qid in H_FULL}

    model, load, prompt_mod, vision_mod, temps = load_model()

    taps = Taps(model)
    o = Oracle(model, taps, H, temps)
    entries, logits1, probs1, npz_meta, rejected = [], {}, {}, {}, []
    det0 = None
    npz_dir = Path(a.npz_dir)
    npz_dir.mkdir(parents=True, exist_ok=True)
    loop0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(records):
            names = []
            for qid, qd in r["request"]["questions"].items():
                try:
                    prompt_mod.as_question(qd)
                    names.append(qid)
                except ValueError as e:
                    rejected.append({"id": r["id"], "qid": qid, "error": str(e)})
            images, audio, minfo = load_media(r)
            entry, arrays, lg, pr = run_record(o, r, names, images, audio, vision_mod, keep_full.get(r["id"]))
            entry["media"] = minfo
            if any(x["id"] == r["id"] for x in rejected):
                entry["rejected_questions"] = [x["qid"] for x in rejected if x["id"] == r["id"]]
            if i == 0:  # determinism: the same call again right away
                p2, calls2, _, _ = o.natural(r["request"]["state"], [r["request"]["questions"][n] for n in names],
                                             images, audio)
                per2 = o.per_question(calls2, names)
                det0 = {"record": r["id"], "logits_bit_equal": all(torch.equal(a_["logits_full"], b_) for a_, b_ in
                                                                   zip(per2, lg)),
                        "probs_equal": [list(p) for p in p2] == pr}
            entries.append(entry)
            logits1[r["id"]], probs1[r["id"]] = lg, pr
            npz_meta[r["id"]] = write_npz(npz_dir / f"{r['id']}.npz", arrays)
            if (i + 1) % 50 == 0 or i + 1 == len(records):
                print(f"[pass 1] {i + 1}/{len(records)} records, {time.time() - loop0:.1f} s", flush=True)
    loop1_s = time.time() - loop0

    # second pass: every record's natural call again, bit for bit
    second = {"requests": 0, "logits_bit_equal": 0, "probs_equal": 0, "differing": []}
    loop2_s = None
    if not a.no_second_pass:
        t2 = time.time()
        with torch.no_grad():
            for r in records:
                names = [q["qid"] for q in next(e for e in entries if e["id"] == r["id"])["questions"]]
                images, audio, _ = load_media(r)
                p2, calls2, _, _ = o.natural(r["request"]["state"], [r["request"]["questions"][n] for n in names],
                                             images, audio)
                per2 = o.per_question(calls2, names)
                le = all(torch.equal(a_["logits_full"], b_) for a_, b_ in zip(per2, logits1[r["id"]]))
                pe = [list(p) for p in p2] == probs1[r["id"]]
                second["requests"] += 1
                second["logits_bit_equal"] += le
                second["probs_equal"] += pe
                if not (le and pe):
                    second["differing"].append(r["id"])
        loop2_s = time.time() - t2
        print(f"[pass 2] {second['requests']} records, {loop2_s:.1f} s, differing {len(second['differing'])}",
              flush=True)
        for e in entries:
            e["second_run_logits_bit_equal"] = e["id"] not in second["differing"]
            e["second_run_probs_equal"] = e["id"] not in second["differing"]

    checks = build_checks(entries, det0, second, rejected)
    rows = [q for e in entries for q in e["questions"]]
    snap = {p: S.sha256_file(S.SNAP / p) for p in ("encoder.py", "modeling_d1.py", "prompt.py", "vision.py",
                                                   "audio.py", "config.json")}
    out = {"schema": "d1-omni-reference/1", "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "reference": "AutoModel.from_pretrained(snapshot, trust_remote_code=True, dtype=torch.float32) "
                        "(modeling_d1.D1OmniModel), CPU fp32, 12 threads, probabilities(state, questions, images, audio) "
                        "per request (natural) + per question (single)",
           "model": {"hf_id": S.REPO, "revision": S.REV, "snapshot": str(S.SNAP), "source_sha256": snap,
                     "weights_sha256": WEIGHTS_SHA256},
           "load": load, "fixtures": {"path": str(Path(a.fixtures)), "sha256": S.sha256_file(a.fixtures),
                                      "records": len(records)},
           "versions": versions(), "near_tie_threshold_top2": NEAR_TIE, "buckets": list(BUCKETS),
           "checks": checks, "npz": {"dir": str(npz_dir), "files": npz_meta, "h_full": keep_full},
           "loop_seconds": round(loop1_s, 1), "second_pass_seconds": None if loop2_s is None else round(loop2_s, 1),
           "records": entries}
    # summary first, npz already written; the reference file appears last (round 3 waits for it)
    summ = summary(out)
    Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(a.summary).write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n")
    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, dst)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("by_source_mode", "near_ties", "seconds_by_record")},
                     indent=1))
    print(f"done: {len(entries)} records / {len(rows)} rows -> {dst} ({time.time() - t_start:.1f} s)")


RESIZE_PATH = {
    "path": "float",
    "what": "torchvision.transforms.v2.functional.resize replaced in this process (the provider's files unchanged) by "
            "torchvision 0.24.0's resize_image steps for a uint8 CPU tensor when the native uint8 kernel is off: "
            "reshape, channels_last restride, to(float32), interpolate(bilinear, align_corners False, antialias True), "
            "round_(), clamp(0, 255) (a no-op for bilinear, checked per call), to(uint8)",
    "why": "decision 2026-10-08 11:3x: the float path is what torchvision 0.24 on arm64 and every GPU run "
           "do, the Core AI lane d1d's fixture table uses it, and a host (numpy / Kotlin) can copy it bit for bit; "
           "torchvision 0.29.1's native uint8 kernel on arm64 moves 17 % of card_cats' pixels by one level "
           "(results/cross_lane_resize_path.json)",
    "native_path_values": "results/ref_native_resize_images.json, ref/npz_native_resize/<id>.npz (v1 of this file)",
}


def resize_tv024_float(torch, F, image, size):
    """torchvision 0.24.0 resize_image (transforms/v2/functional/_geometry.py) on a uint8 CPU tensor with
    need_cast = True (arm64 there, or any GPU): the copy that matched d1d's npz pixels bit for bit."""
    shape = image.shape
    numel = image.numel()
    c, oh, ow = shape[-3:]
    nh, nw = size
    info = {"in": [int(oh), int(ow)], "out": [int(nh), int(nw)], "identity": (nh, nw) == (oh, ow)}
    if (nh, nw) == (oh, ow):
        return image, info
    dtype = image.dtype
    image = image.reshape(-1, c, oh, ow)
    strides = image.stride()
    if image.is_contiguous(memory_format=torch.channels_last) and image.shape[0] == 1 and numel != strides[0]:
        new_strides = list(strides)
        new_strides[0] = numel
        image = image.as_strided((1, c, oh, ow), new_strides)
    image = image.to(dtype=torch.float32)
    image = F.interpolate(image, size=[nh, nw], mode="bilinear", align_corners=False, antialias=True)
    image = image.round_()
    clamped = image.clamp(0, 255)
    info["clamp_noop"] = bool(torch.equal(clamped, image))
    info["min"], info["max"] = float(image.min()), float(image.max())
    image = clamped.to(dtype=dtype)
    return image.reshape(shape[:-3] + (c, nh, nw)), info


def rerun_images(a):
    """--rerun-images: every image record again with the float resize path (RESIZE_PATH), in place of its v1 entry
    and npz; the v1 file / npz / summary kept under other names; checks recomputed; `version: 2`; the file replaced
    atomically. Text and audio entries are not touched."""
    import shutil

    os.environ["HF_HUB_OFFLINE"] = "1"
    t_start = time.time()
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.v2.functional as tvf

    torch.set_num_threads(THREADS)
    sys.path.insert(0, str(K / "host"))
    import d1_host as H

    dst = DEFAULT_OUT
    v1 = json.loads(dst.read_text())
    assert v1.get("version", 1) == 1, "already rewritten"
    fixtures = {r["id"]: r for r in json.loads((K / "fixtures/requests.json").read_text())["records"]}
    assert v1["fixtures"]["sha256"] == S.sha256_file(K / "fixtures/requests.json")
    image_ids = [e["id"] for e in v1["records"] if e["mode"] == "image"]
    model, load, prompt_mod, vision_mod, temps = load_model()
    orig_resize = tvf.resize
    calls = []

    def resize(inpt, size, interpolation=tvf.InterpolationMode.BILINEAR, max_size=None, antialias=True):
        ok = (isinstance(inpt, torch.Tensor) and inpt.dtype == torch.uint8 and inpt.device.type == "cpu"
              and interpolation == tvf.InterpolationMode.BILINEAR and antialias is True and max_size is None
              and len(size) == 2)
        assert ok, (type(inpt), getattr(inpt, "dtype", None), interpolation, antialias, max_size, size)
        out, info = resize_tv024_float(torch, F, inpt, list(size))
        calls.append(info)
        return out

    tvf.resize = resize
    taps = Taps(model)
    o = Oracle(model, taps, H, temps)
    new, arrays_by, second = {}, {}, {"requests": 0, "logits_bit_equal": 0, "probs_equal": 0, "differing": []}
    d1d_npz = Path.home() / "code/coreai/_d1_omni/ref/npz"
    pixels_vs_d1d = {}
    with torch.no_grad():
        for rid in image_ids:
            r = fixtures[rid]
            old = next(e for e in v1["records"] if e["id"] == rid)
            names = [q["qid"] for q in old["questions"]]
            images, audio, minfo = load_media(r)
            n_calls = len(calls)
            entry, arrays, lg, pr = run_record(o, r, names, images, audio, vision_mod, None)
            minfo["resize_path"] = "float"
            uniq = {json.dumps(c, sort_keys=True) for c in calls[n_calls:]}
            minfo["resize_calls"] = {"count_in_this_record": len(calls) - n_calls,
                                     "distinct": [json.loads(u) for u in sorted(uniq)]}
            entry["media"] = minfo
            p2, calls2, _, _ = o.natural(r["request"]["state"], [r["request"]["questions"][n] for n in names],
                                         images, audio)
            per2 = o.per_question(calls2, names)
            le = all(torch.equal(x["logits_full"], y) for x, y in zip(per2, lg))
            pe = [list(p) for p in p2] == pr
            second["requests"] += 1
            second["logits_bit_equal"] += le
            second["probs_equal"] += pe
            if not (le and pe):
                second["differing"].append(rid)
            entry["second_run_logits_bit_equal"] = entry["second_run_probs_equal"] = bool(le and pe)
            assert [q["ids"] for q in entry["questions"]] == [q["ids"] for q in old["questions"]], rid
            new[rid], arrays_by[rid] = entry, arrays
            if (d1d_npz / f"{rid}.npz").exists():
                with np.load(d1d_npz / f"{rid}.npz") as z:
                    pixels_vs_d1d[rid] = bool(np.array_equal(z["pixel_values"], arrays["pixel_values"]))
            print(rid, f"{entry['seconds']:.2f}s", [round(q["max_abs_dp_batch_vs_single"], 9) for q in entry["questions"]],
                  flush=True)
    tvf.resize = orig_resize
    assert all(c.get("clamp_noop", True) for c in calls), [c for c in calls if not c.get("clamp_noop", True)]
    # keep v1 (native resize) under other names, then write v2
    keep_dir = K / "ref/npz_native_resize"
    keep_dir.mkdir(parents=True, exist_ok=True)
    for rid in image_ids:
        if not (keep_dir / f"{rid}.npz").exists():
            shutil.copy2(K / "ref/npz" / f"{rid}.npz", keep_dir / f"{rid}.npz")
    v1_copy = K / "ref/records_ref.v1_native_resize.json"
    if not v1_copy.exists():
        shutil.copy2(dst, v1_copy)
    sum_copy = K / "results/ref_summary.v1_native_resize.json"
    if not sum_copy.exists():
        shutil.copy2(K / "results/ref_summary.json", sum_copy)
    native = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "what": "the image records of ref/records_ref.json v1 (torchvision 0.29.1's native uint8 resize on arm64), "
                      "kept when v2 switched the image oracle to the float resize path",
              "v1_file": str(v1_copy), "v1_sha256": S.sha256_file(v1_copy),
              "npz": {rid: {"path": str(keep_dir / f"{rid}.npz"), "sha256": S.sha256_file(keep_dir / f"{rid}.npz")}
                      for rid in image_ids},
              "records": [e for e in v1["records"] if e["id"] in image_ids]}
    (K / "results/ref_native_resize_images.json").write_text(json.dumps(native, indent=1, ensure_ascii=False) + "\n")
    npz_meta = dict(v1["npz"]["files"])
    for rid in image_ids:
        npz_meta[rid] = write_npz(K / "ref/npz" / f"{rid}.npz", arrays_by[rid])
    entries = [new.get(e["id"], e) for e in v1["records"]]
    v1_second = v1["checks"]["second_run"]
    merged = {"requests": v1_second["requests"],
              "logits_bit_equal": v1_second["logits_bit_equal"] - len(image_ids) + second["logits_bit_equal"],
              "probs_equal": v1_second["probs_equal"] - len(image_ids) + second["probs_equal"],
              "differing": sorted(set(x for x in v1_second["differing"] if x not in image_ids) | set(second["differing"])),
              "note": "text / audio records from v1's second pass; image records from v2's re-run"}
    checks = build_checks(entries, v1["checks"]["determinism"], merged, v1["checks"]["rejected_questions"])
    out = dict(v1)
    out.update({"version": 2, "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "records": entries, "checks": checks})
    out["load"] = {**v1["load"], "resize_path": RESIZE_PATH}
    out["npz"] = {**v1["npz"], "files": npz_meta, "native_resize_dir": str(keep_dir)}
    out["rerun_v2"] = {"records": image_ids, "resize_calls": len(calls), "resize_identity_calls": sum(c["identity"]
                                                                                                   for c in calls),
                       "clamp_noop_all": True, "pixel_values_equal_d1d_npz": pixels_vs_d1d,
                       "second_pass": second, "seconds": round(time.time() - t_start, 1),
                       "v1_kept": {"records_ref": str(v1_copy), "summary": str(sum_copy),
                                   "native_entries": "results/ref_native_resize_images.json"}}
    summ = summary(out)
    (K / "results/ref_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n")
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, dst)
    print(json.dumps(out["rerun_v2"], indent=1))
    print(json.dumps({k: checks[k] for k in ("second_run", "host_math", "batch_vs_single", "red_arm",
                                             "response_equal_host_response")}, indent=1, default=str)[:3000])
    print(f"done: v2 -> {dst} ({time.time() - t_start:.1f} s)")


def summary(out):
    from collections import defaultdict

    entries = out["records"]
    by = defaultdict(lambda: {"requests": 0, "rows": 0, "types": defaultdict(int), "gold_rows": 0, "correct": 0,
                              "near_tie": 0, "seconds": []})
    ties = []
    for e in entries:
        b = by[f"{e['source']}/{e['mode']}"]
        b["requests"] += 1
        b["seconds"].append(e["seconds"])
        for q in e["questions"]:
            b["rows"] += 1
            b["types"][q["type"]] += 1
            if q["gold"] is not None:
                b["gold_rows"] += 1
                b["correct"] += bool(q["correct"])
            if q["near_tie"]:
                b["near_tie"] += 1
                ties.append(f"{e['id']}/{q['qid']} ({q['top2_margin']:.4f})")
    table = {}
    for k, b in sorted(by.items()):
        s = sorted(b["seconds"])
        table[k] = {"requests": b["requests"], "rows": b["rows"], "types": dict(b["types"]),
                    "gold_rows": b["gold_rows"], "gold_correct": b["correct"], "near_tie": b["near_tie"],
                    "seconds_p50": s[len(s) // 2], "seconds_max": s[-1]}
    c = out["checks"]
    return {"written": out["written"], "fixtures": out["fixtures"], "versions": out["versions"],
            "torch_threads": out["load"]["torch_threads"], "mha_fastpath_enabled": out["load"]["mha_fastpath_enabled"],
            "records": len(entries), "rows": sum(len(e["questions"]) for e in entries),
            "by_source_mode": table, "near_ties": {"threshold_top2": NEAR_TIE, "count": len(ties), "ids": ties},
            "rejected_questions": c["rejected_questions"], "determinism": c["determinism"],
            "second_run": c["second_run"], "host_math": c["host_math"],
            "ids_markers_equal_host_request_rows": c["ids_markers_equal_host_request_rows"],
            "response_equal_host_response": c["response_equal_host_response"],
            "usage_input_tokens_equal_host": c["usage_input_tokens_equal_host"],
            "hook_point_batches": c["hook_point_batches"], "prefix_length_equal_host": c["prefix_length_equal_host"],
            "batch_vs_single": {k: v for k, v in c["batch_vs_single"].items() if k != "top"}, "red_arm": c["red_arm"],
            "load_seconds": out["load"]["load_seconds"], "state_equals_safetensors": out["load"]["state_equals_safetensors"],
            "loop_seconds": out["loop_seconds"], "second_pass_seconds": out["second_pass_seconds"],
            "seconds_by_record": {e["id"]: [e["seconds"], e["seconds_system_one"], e["seconds_single"]] for e in entries},
            "seconds_total_natural": round(sum(e["seconds"] for e in entries), 2),
            "npz_files": len(out["npz"]["files"]), "npz_bytes": sum(v["bytes"] for v in out["npz"]["files"].values())}


if __name__ == "__main__":
    main()
