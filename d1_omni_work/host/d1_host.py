"""d1-omni host for the D1Decision graph: requests in, the graph's six inputs out, answers back.

    from d1_host import D1Host, TorchRunner
    host = D1Host(snapshot_dir, runner)          # runner(inputs: dict of numpy arrays, L) -> scores [1, L]
    host.system_one(state, {name: question}, prefix_embeds=None, kind="text")   # the provider's system_one() shape

    python host/d1_host.py --selftest   # development check against the provider's own PyTorch code (work tree only)

The request side is the provider's own prompt.py (host/d1_prompt.py, verbatim): encode() gives ids and marker
positions. build_inputs() lays a row into a bucket of L positions: [prefix rows | text ids | pad], media = 1 on the
prefix, pad = 1 on real positions, keep_right = 0 only on the last prefix position. The graph returns a score for every
position; readout() takes them at P + marker in option order, divides text answers by the checkpoint's temperature,
softmaxes, and reverses a noul to [yes, no]: the provider's D1OmniModel._forward tail, in the same float32 torch ops.
readout_f64() is the same in float64 numpy (a reference for hosts without torch).
Tokenizer: tokenizers.Tokenizer on the repo's tokenizer.json behind the three calls encode() makes (no transformers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1_prompt as Pm  # noqa: E402  (the provider's prompt.py under a notice header)

BUCKETS = (128, 256, 512, 1024, 2048, 4096)
D = 1024
YES_NO = {"false": "no", "true": "yes"}  # modeling_d1.YES_NO: how image and audio questions were trained


class Tokenizer:
    """tokenizers.Tokenizer behind the three calls prompt.encode() makes (__call__, convert_tokens_to_ids, bos)."""

    def __init__(self, tokenizer_json):
        from tokenizers import Tokenizer as _T

        self._t = _T.from_file(str(tokenizer_json))
        self.bos_token_id = self._t.token_to_id("<|startoftext|>")

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": self._t.encode(text, add_special_tokens=add_special_tokens).ids}

    def convert_tokens_to_ids(self, token):
        i = self._t.token_to_id(token)
        if i is None:
            raise KeyError(token)
        return i


def bucket_for(n, buckets=BUCKETS):
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"{n} positions do not fit the largest bucket ({buckets[-1]})")


def build_inputs(ids, prefix_embeds, L):
    """One row -> ids int32 [1, L], prefix f32 [1, L, 1024], media / pad / keep_right f32 [1, L]."""
    P = 0 if prefix_embeds is None else int(prefix_embeds.shape[0])
    n = len(ids)
    if P + n > L:
        raise ValueError(f"row of {P} + {n} positions does not fit L = {L}")
    t = np.arange(L)
    x_ids = np.zeros((1, L), np.int32)
    x_ids[0, P:P + n] = np.asarray(ids, np.int32)
    prefix = np.zeros((1, L, D), np.float32)
    if P:
        prefix[0, :P] = np.asarray(prefix_embeds, np.float32)
    return {"ids": x_ids, "prefix": prefix, "media": (t < P).astype(np.float32)[None],
            "pad": (t < P + n).astype(np.float32)[None], "keep_right": (t != P - 1).astype(np.float32)[None]}


def qtype_onehot(q):
    v = np.zeros((1, 3), np.float32)
    v[0, Pm.QTYPES[q.type]] = 1.0
    return v


def temperature(q, temperatures):
    return temperatures.get(Pm.temperature_key(q), temperatures.get(q.type, 1.0))


def readout(scores, P, markers, q, calibrate, temperatures):
    """The provider's _forward tail: z[:K] (/ T for text) -> softmax -> a noul reversed. float32 torch ops."""
    import torch

    z = torch.as_tensor(np.asarray(scores, np.float32).reshape(-1)[[P + m for m in markers]])
    z = z[:q.options]
    if calibrate:
        z = z / temperature(q, temperatures)
    p = z.softmax(-1).tolist()
    return p[::-1] if q.type == "noul" else p


def readout_f64(scores, P, markers, q, calibrate, temperatures):
    z = np.asarray(scores, np.float64).reshape(-1)[[P + m for m in markers]][:q.options]
    if calibrate:
        z = z / temperature(q, temperatures)
    e = np.exp(z - z.max())
    p = (e / e.sum()).tolist()
    return p[::-1] if q.type == "noul" else p


class D1Host:
    def __init__(self, snapshot_dir, runner, buckets=BUCKETS, tokenizer=None):
        snapshot_dir = Path(snapshot_dir)
        self.tok = tokenizer or Tokenizer(snapshot_dir / "tokenizer.json")
        cfg = json.loads((snapshot_dir / "config.json").read_text())
        self.temperatures = cfg["temperatures"]
        self.max_length, self.image_text_length = cfg["max_length"], cfg["image_text_length"]
        self.audio_text_length = cfg["audio_text_length"]
        self.runner, self.buckets = runner, buckets

    def rows(self, state, questions, prefix_embeds=None, kind="text"):
        """modeling_d1.probabilities_batch for one request: per-kind settings, then encode() per question."""
        if kind == "image":
            max_len, noul, calibrate, spoken = self.image_text_length, YES_NO, False, False
        elif kind == "audio":
            max_len, noul, calibrate, spoken = self.audio_text_length, YES_NO, False, True
            state = {} if state is None else state
        elif kind == "text":
            max_len, noul, calibrate, spoken = self.max_length, None, True, False
        else:
            raise ValueError(kind)
        P = 0 if prefix_embeds is None else int(prefix_embeds.shape[0])
        if (kind == "text") != (P == 0):
            raise ValueError("a text request has no prefix; an image or audio request has one")
        max_len = min(max_len, self.max_length - P)
        if max_len < 64:
            raise ValueError(f"the media take {P} of the {self.max_length} positions")
        state = "" if state is None else state
        out = []
        for qd in questions:
            q = Pm.as_question(qd)
            ids, markers = Pm.encode(self.tok, state, q, max_len, noul, spoken)
            out.append({"q": q, "ids": ids, "markers": markers, "P": P, "calibrate": calibrate})
        return out

    def probabilities(self, state, questions, prefix_embeds=None, kind="text"):
        res = []
        for r in self.rows(state, questions, prefix_embeds, kind):
            L = bucket_for(r["P"] + len(r["ids"]), self.buckets)
            x = build_inputs(r["ids"], prefix_embeds, L)
            x["qtype_onehot"] = qtype_onehot(r["q"])
            res.append(readout(self.runner(x, L), r["P"], r["markers"], r["q"], r["calibrate"], self.temperatures))
        return res

    def system_one(self, state, questions, prefix_embeds=None, kind="text"):
        named = {n: Pm.as_question(q) for n, q in questions.items()}
        rows = self.rows(state, list(named.values()), prefix_embeds, kind)
        probs = self.probabilities(state, list(named.values()), prefix_embeds, kind)
        return {"answers": {n: Pm.answer(q, p) for (n, q), p in zip(named.items(), probs)},
                "usage": {"input_tokens": sum(r["P"] + len(r["ids"]) for r in rows), "output_tokens": 0}}


class TorchRunner:
    """Eager D1Decision per bucket (scripts/d1_graph.py) with weights from a provider state dict (tests, oracle A/B)."""

    def __init__(self, provider_state, text_config, head_layers):
        import torch  # noqa: F401

        sys.path.insert(0, str(HERE.parent / "scripts"))
        import d1_graph

        self.G, self.sd, self.cfg, self.layers, self.models = d1_graph, provider_state, text_config, head_layers, {}

    def __call__(self, x, L):
        import torch

        if L not in self.models:
            m = self.G.D1Decision(self.cfg, self.layers, L).eval()
            self.G.load_state_dict_from_provider(m, self.sd)
            self.models[L] = m
        with torch.no_grad():
            out = self.models[L](**{k: torch.from_numpy(v) for k, v in x.items()})
        return out["scores"].numpy()


class LiteRTRunner:
    """One single-signature .tflite per bucket (`decide_<L>`), CPU interpreter."""

    def __init__(self, path_for_L, num_threads=4):
        from ai_edge_litert.interpreter import Interpreter

        self.Interpreter, self.path_for_L, self.threads, self.sigs = Interpreter, path_for_L, num_threads, {}

    def __call__(self, x, L):
        if L not in self.sigs:
            it = self.Interpreter(model_path=str(self.path_for_L(L)), num_threads=self.threads)
            self.sigs[L] = it.get_signature_runner(f"decide_{L}")
        return self.sigs[L](**x)["scores"]


INPUT_DTYPES = {"ids": np.int32, "prefix": np.float32, "media": np.float32, "pad": np.float32,
                "keep_right": np.float32, "qtype_onehot": np.float32}   # the decision graph's six inputs (bind by name)


class CompiledModelRunner:
    """One single-signature .tflite per bucket (`decide_<L>`) through ai_edge_litert's CompiledModel.

    accelerator "cpu" = XNNPACK with `threads`; "gpu" = the GPU accelerator (Metal on a Mac) with precision "fp32" =
    GpuOptions(enforce_f32=True), the GPU precision that keeps the answers ("default" = the delegate's default, fp16
    activations: measured to move them past the bar, kept only for A/B runs). A bucket's file is compiled the first
    time a row needs it; `max_loaded` (None = no limit) closes the least recently used bucket beyond that count.
    One call = write the six inputs by name + run + read `scores` back -> float32 [1, L]."""

    def __init__(self, path_for_L, accelerator="cpu", threads=4, precision="fp32", max_loaded=None):
        if accelerator not in ("cpu", "gpu"):
            raise ValueError("accelerator must be 'cpu' or 'gpu'")
        if precision not in ("fp32", "default"):
            raise ValueError("precision must be 'fp32' or 'default'")
        self.path_for_L, self.accelerator, self.threads, self.precision = path_for_L, accelerator, threads, precision
        self.max_loaded, self.loaded = max_loaded, {}     # L -> (model, signature, input buffers, output buffer)

    def _options(self):
        from ai_edge_litert.compiled_model import CpuOptions, GpuOptions, HardwareAccelerator, Options

        if self.accelerator == "gpu":
            return Options(hardware_accelerators=HardwareAccelerator.GPU,
                           gpu_options=GpuOptions(enforce_f32=self.precision == "fp32"))
        return Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=self.threads))

    def load(self, L):
        if L in self.loaded:
            self.loaded[L] = self.loaded.pop(L)          # most recently used last
            return self.loaded[L]
        from ai_edge_litert.compiled_model import CompiledModel

        path = Path(self.path_for_L(L))
        if not path.is_file():
            raise FileNotFoundError(f"no decision graph for L = {L} at {path}")
        model = CompiledModel.from_file(str(path), options=self._options())
        sig = f"decide_{L}"
        if sig not in model.get_signature_list():
            model.close()
            raise ValueError(f"{path.name} has no signature {sig}")
        ins, outs = model.get_input_tensor_details(sig), model.get_output_tensor_details(sig)
        if sorted(ins) != sorted(INPUT_DTYPES) or list(outs) != ["scores"]:
            model.close()
            raise ValueError(f"{path.name}: inputs {sorted(ins)}, outputs {list(outs)}")
        entry = (model, sig, {n: model.create_input_buffer_by_name(sig, n) for n in INPUT_DTYPES},
                 model.create_output_buffer_by_name(sig, "scores"))
        self.loaded[L] = entry
        while self.max_loaded is not None and len(self.loaded) > self.max_loaded:
            self.release(next(iter(self.loaded)))
        return entry

    def __call__(self, x, L):
        model, sig, ins, out = self.load(L)
        for n, dt in INPUT_DTYPES.items():
            v = np.ascontiguousarray(x[n], dtype=dt)
            if n == "qtype_onehot":
                if v.shape != (1, 3):
                    raise ValueError(f"qtype_onehot must be (1, 3), got {v.shape}")
            elif v.shape[:2] != (1, L):
                raise ValueError(f"{n} must start with (1, {L}), got {v.shape}")
            ins[n].write(v)
        model.run_by_name(sig, ins, {"scores": out})
        return np.asarray(out.read(L, np.float32), np.float32).reshape(1, L)

    def release(self, L):
        model, _, ins, out = self.loaded.pop(L)
        for b in list(ins.values()) + [out]:
            try:
                b.destroy()
            except Exception:
                pass
        model.close()

    def close(self):
        for L in list(self.loaded):
            self.release(L)


# ---------------------------------------------------------------- self-test

def _selftest():
    """1. The tokenizers adapter gives encode() the same ids as transformers' AutoTokenizer (results/encoded_rows.json).
    2. readout() against the provider's own D1OmniModel._forward tail: _forward runs unchanged on a stub whose
       encoder passes embeddings through and whose head returns preset random logits; the host gets a score vector
       holding the same logits at P + marker and random values elsewhere. All encoded fixture rows, batches of 8."""
    import time

    import torch
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    sys.path.insert(0, str(HERE.parent / "scripts"))
    import d1_src as S

    t0 = time.time()
    K = S.K
    enc = json.loads((K / "results/encoded_rows.json").read_text())
    doc = json.loads((K / "fixtures/requests.json").read_text())
    recs = {r["id"]: r for r in doc["records"]}
    tok = Tokenizer(S.SNAP / "tokenizer.json")
    host = D1Host(S.SNAP, runner=None, tokenizer=tok)

    # 1. tokenizer adapter
    mism = []
    for row in enc["rows"]:
        r = recs[row["id"]]
        kind = "text" if r["media"] is None else r["media"]["kind"]
        P = {"text": 0, "image": 144, "audio": 125}[kind]
        prefix = None if P == 0 else np.zeros((P, D), np.float32)
        hr = host.rows(r["request"]["state"], [r["request"]["questions"][row["qid"]]], prefix, kind)[0]
        if hr["ids"] != row["ids"] or hr["markers"] != row["markers"]:
            mism.append(f"{row['id']}/{row['qid']}")
    tok_check = {"rows": len(enc["rows"]), "mismatch": mism, "pass": not mism}

    # 2. host readout vs the provider's _forward tail
    cfg = AutoConfig.from_pretrained(S.REPO, revision=S.REV, trust_remote_code=True)
    model_cls = get_class_from_dynamic_module("modeling_d1.D1OmniModel", S.REPO, revision=S.REV)
    provider_prompt = sys.modules[model_cls.__module__.rsplit(".", 1)[0] + ".prompt"]
    g = torch.Generator().manual_seed(3)

    class StubEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(cfg.text_config["vocab_size"], 2)

        def forward(self, h, pad, offsets):
            return h

    class StubHead:
        logits = None

        def __call__(self, text, text_pad, marker_pos, marker_mask, qtype):
            assert self.logits.shape == marker_pos.shape, (self.logits.shape, marker_pos.shape)
            return self.logits

    stub = type("Stub", (), {})()
    stub.device, stub.encoder, stub.head, stub.config = torch.device("cpu"), StubEncoder(), StubHead(), cfg
    rows = enc["rows"]
    worst_t, worst_64, answer_mismatch, n = 0.0, 0.0, [], 0
    per_type = {}
    for i in range(0, len(rows), 8):
        batch = rows[i:i + 8]
        kmax = max(len(r["markers"]) for r in batch)
        logits = torch.randn(len(batch), kmax, generator=g) * 3.0
        stub.head.logits = logits
        prow, hosts = [], []
        for j, row in enumerate(batch):
            r = recs[row["id"]]
            kind = "text" if r["media"] is None else r["media"]["kind"]
            P = {"text": 0, "image": 144, "audio": 125}[kind]
            q = provider_prompt.as_question(r["request"]["questions"][row["qid"]])
            prefix = None if P == 0 else torch.zeros(1, P, 2)
            prow.append((prefix, row["ids"], row["markers"], q, kind == "text"))
            L = bucket_for(P + len(row["ids"]))
            scores = (torch.randn(1, L, generator=g) * 5.0).numpy()
            for k, m in enumerate(row["markers"]):
                scores[0, P + m] = float(logits[j, k])
            hq = Pm.as_question(r["request"]["questions"][row["qid"]])
            hosts.append((readout(scores, P, row["markers"], hq, kind == "text", cfg.temperatures),
                          readout_f64(scores, P, row["markers"], hq, kind == "text", cfg.temperatures), hq, q))
        ref = model_cls._forward(stub, prow)
        for (p_t, p_64, hq, q), p_ref, row in zip(hosts, ref, batch):
            n += 1
            d_t = max(abs(a - b) for a, b in zip(p_t, p_ref))
            d_64 = max(abs(a - b) for a, b in zip(p_64, p_ref))
            worst_t, worst_64 = max(worst_t, d_t), max(worst_64, d_64)
            per_type[q.type] = max(per_type.get(q.type, 0.0), d_t)
            if Pm.answer(hq, p_t) != provider_prompt.answer(q, p_ref):
                answer_mismatch.append(f"{row['id']}/{row['qid']}")
    math = {"rows": n, "batches_of": 8, "max_abs_dp_torch_f32": worst_t, "max_abs_dp_numpy_f64": worst_64,
            "max_abs_dp_torch_f32_by_type": per_type, "answer_dict_mismatch": answer_mismatch,
            "pass": worst_t <= 1e-7 and not answer_mismatch}
    out = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "provider_modeling_d1": str(Path(sys.modules[model_cls.__module__].__file__)),
           "host_prompt_sha256": S.sha256_file(HERE / "d1_prompt.py"),
           "tokenizer_adapter": tok_check, "readout_vs_provider_forward_tail": math,
           "seconds": round(time.time() - t0, 1), "PASS": tok_check["pass"] and math["pass"]}
    (K / "results/host_math_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    print(f"selftest: {'PASS' if out['PASS'] else 'FAIL'} (tokenizer {len(mism)} mismatches / {len(enc['rows'])} rows, "
          f"max|dp| {worst_t:.3g} torch-f32, {worst_64:.3g} numpy-f64, {len(answer_mismatch)} answer mismatches)")
    return 0 if out["PASS"] else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        sys.exit(_selftest())
    print(__doc__)
