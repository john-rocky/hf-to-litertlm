"""d1-omni-600M on LiteRT: the provider's System One API over this repository's graphs.

numpy, tokenizers, Pillow, soundfile and ai-edge-litert only (no PyTorch, no transformers). Everything the host takes
from the repository is read from contract.json: the file of each bucket, the token ids, the temperatures, the settings
of each mode, the position limit and the GPU precision of each graph.

    import sys; sys.path.insert(0, "<repo>/host")
    from d1_omni import D1Omni
    model = D1Omni("<repo>")       # CPU (XNNPACK, 4 threads); accelerator="gpu" = the GPU, each graph at its precision
                                   # from contract.json (precision.mac_metal.graphs)
    model.system_one(state, {name: question})                      # text
    model.system_one(None, {name: question}, images=[image])       # PIL images or paths, in order
    model.system_one(state, {name: question}, audio=samples)       # 16 kHz mono int16 / float samples, or a file path
    model.probabilities(state, [question, ...], images=None, audio=None)    # the distributions, in option order
    model.decide(state, questions)   model.decide_image(image, questions, state=None)
    model.decide_audio(audio, questions, state=None)

A call returns the provider's response: {"answers": {name: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}
with prompt.answer()'s answer (noul: P(yes); choice: choice, confidence, probabilities; score: the expected level,
confidence, probabilities, legend). input_tokens counts every position the trunk reads (prefix + text, per question).

Command line (from the repository directory):
    python -m host.d1_omni --repo . --questions questions.json [--state state.json | --state-text "..."]
        [--image x.jpg ... | --audio x.wav] [--gpu] [--threads N] [--probabilities]

Per question (contract.json host_steps): prompt.encode() -> ids and marker positions; P media rows (vision: tower and
projector per crop; audio: host mel -> the audio graph of the clip's bucket) go first; the smallest decision bucket
L >= P + n; build_inputs() -> six inputs; scores = decide_<L>; readout: the scores at P + marker, divided by the
checkpoint's temperature for a text request, softmax, a noul reversed to [yes, no].
The provider reads up to 16,384 positions; this repository's largest bucket is 4,096, so a longer row raises
ValueError (truncate_state=True cuts the state with encode()'s own rule to fit L4096 instead).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1_audio_host as A  # noqa: E402
import d1_host as H  # noqa: E402
import d1_prompt as Pm  # noqa: E402  (the provider's prompt.py, unchanged)
import d1_vision_host as V  # noqa: E402

KINDS = ("text", "image", "audio")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class D1Omni:
    """repo_dir = a directory holding contract.json, tokenizer.json, the .tflite files and host/.

    accelerator "cpu" (XNNPACK, `threads`, float32) or "gpu". On the GPU, precision None (the default) opens each
    graph at its precision from contract.json `precision.mac_metal.graphs`: "fp32" = GpuOptions(enforce_f32=True) for
    the decision graphs, the vision tower and the projector (the default precision moves their answers past the bar
    on Metal), "default" = the delegate's default precision (fp16 activations) for the audio graph (it passes the bar
    there and is faster); precision "fp32" or "default" sets every graph to that one precision instead. max_loaded =
    how many decision buckets stay compiled at once (None = every bucket used so far). Graphs are compiled the first
    time a request needs them; `graph_precision` = the GPU precision of each graph (None on the CPU)."""

    GRAPHS = ("decide", "vision_tower", "projector", "audio")

    def __init__(self, repo_dir, accelerator="cpu", threads=4, precision=None, max_loaded=None,
                 truncate_state=False):
        self.repo = Path(repo_dir)
        c = json.loads((self.repo / "contract.json").read_text())
        self.contract = c
        tok = c["tokenizer"]
        tok_path = self.repo / tok["file"]
        if sha256_file(tok_path) != tok["sha256"]:
            raise ValueError(f"{tok_path} is not the tokenizer of contract.json (sha256 differs)")
        self.tok = H.Tokenizer(tok_path)
        for token, i in c["token_ids"].items():
            if self.tok.convert_tokens_to_ids(token) != i:
                raise ValueError(f"tokenizer.json gives {token} id {self.tok.convert_tokens_to_ids(token)}, contract {i}")
        if self.tok.bos_token_id != c["token_ids"]["<|startoftext|>"]:
            raise ValueError("the tokenizer's bos is not <|startoftext|>")
        self.temperatures, self.modes, self.max_length = c["temperatures"], c["modes"], int(c["max_length"])
        g = c["graphs"]
        self.text_files = {int(L): self.repo / f for L, f in g["decide"]["by_L"].items()}
        self.buckets = tuple(sorted(self.text_files))
        self.audio_files = {int(T): self.repo / f for T, f in g["audio"]["by_T"].items()}
        self.audio_buckets = tuple(sorted(self.audio_files))
        self.tower_file, self.projector_file = self.repo / g["vision_tower"]["file"], self.repo / g["projector"]["file"]
        self.table_file, self.table_sha256 = (self.repo / c["vision_position_table"]["file"],
                                              c["vision_position_table"]["sha256"])
        if precision not in (None, "fp32", "default"):
            raise ValueError("precision must be None (each graph's from contract.json), 'fp32' or 'default'")
        self.accelerator, self.threads, self.precision = accelerator, threads, precision
        if accelerator == "gpu":
            table = (c.get("precision") or {}).get("mac_metal", {}).get("graphs") if precision is None else \
                {g: precision for g in self.GRAPHS}
            if not isinstance(table, dict) or sorted(table) != sorted(self.GRAPHS) or \
                    not set(table.values()) <= {"fp32", "default"}:
                raise ValueError(f"contract.json precision.mac_metal.graphs must give fp32 / default for {self.GRAPHS}")
            self.graph_precision = {g: table[g] for g in self.GRAPHS}
        else:
            self.graph_precision = {g: None for g in self.GRAPHS}
        self.truncate_state = truncate_state
        self.runner = H.CompiledModelRunner(lambda L: self.text_files[L], accelerator, threads,
                                            self.graph_precision["decide"] or "fp32", max_loaded)
        self._tower = self._projector = self._table = None
        self._audio = {}

    # ------------------------------------------------------------------ media

    def _graph(self, path, graph):
        return V.LiteRTGraph(path, accelerator=self.accelerator, precision=self.graph_precision[graph] or "fp32",
                             threads=self.threads)

    def _vision(self):
        if self._tower is None:
            if sha256_file(self.table_file) != self.table_sha256:
                raise ValueError(f"{self.table_file} is not the position table of contract.json (sha256 differs)")
            self._table = V.load_position_table(self.table_file)
            self._tower = self._graph(self.tower_file, "vision_tower")
            self._projector = self._graph(self.projector_file, "projector")
        return self._tower, self._projector, self._table

    def image_prefix(self, images):
        """A PIL image, a path, or a list of them -> prefix rows [P, 1024]: every image's tiles then its thumbnail,
        images in order (the provider's Vision.forward)."""
        if hasattr(images, "convert") or isinstance(images, (str, Path)):
            images = [images]
        tower, projector, table = self._vision()
        out = []
        for im in images:
            im = V.load_image(im) if isinstance(im, (str, Path)) else im
            out.append(V.image_prefix(im, tower, projector, table))
        return np.concatenate(out).astype(np.float32)

    def audio_prefix(self, audio):
        """16 kHz mono int16 / float samples (or a file path) -> prefix rows [P, 1024]: host mel, the audio graph of the
        smallest bucket T_b >= the clip's STFT frames, its first P rows."""
        if isinstance(audio, (str, Path)):
            audio = A.read_audio(audio)
        x, info = A.prepare(audio, bucket=None)
        T_b = A.bucket_for(info["T"], self.audio_buckets)
        if T_b != info["T_b"]:
            x, info = A.prepare(audio, bucket=T_b)
        if T_b not in self._audio:
            self._audio[T_b] = self._graph(self.audio_files[T_b], "audio")
        return A.prefix_rows(self._audio[T_b](**x), info)

    def _media(self, images, audio):
        if isinstance(images, (list, tuple)) and not images:
            images = None
        if images is not None and audio is not None:
            raise ValueError("a request carries images or audio, not both")
        if images is not None:
            return "image", self.image_prefix(images)
        if audio is not None:
            return "audio", self.audio_prefix(audio)
        return "text", None

    # ------------------------------------------------------------------ rows and scores

    def rows(self, state, questions, prefix=None, kind="text"):
        """modeling_d1.probabilities_batch for one request: the mode's settings, then prompt.encode() per question."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        m = self.modes[kind]
        P = 0 if prefix is None else int(prefix.shape[0])
        if (kind == "text") != (P == 0):
            raise ValueError("a text request has no prefix; an image or audio request has one")
        if kind == "audio" and state is None:
            state = dict(m["state_none_becomes"])           # how the audio questions were trained
        max_len = min(int(m["max_len"]), self.max_length - P)
        if self.truncate_state:
            max_len = min(max_len, self.buckets[-1] - P)
        if max_len < 64:
            raise ValueError(f"the media take {P} of the {self.max_length} positions; send fewer images")
        state = "" if state is None else state
        out = []
        for qd in questions:
            q = Pm.as_question(qd)
            ids, markers = Pm.encode(self.tok, state, q, max_len, m["noul_default"], bool(m["audio"]))
            out.append({"q": q, "ids": ids, "markers": markers, "P": P, "calibrate": bool(m["calibrate"])})
        return out

    def score(self, row, prefix=None):
        """One encoded row -> the question's distribution (option order; [yes, no] for a noul)."""
        n = row["P"] + len(row["ids"])
        if n > self.buckets[-1]:
            raise ValueError(f"{n} positions (prefix {row['P']} + text {len(row['ids'])}) exceed the largest bucket "
                             f"L{self.buckets[-1]} (the provider reads up to {self.max_length}); "
                             "D1Omni(truncate_state=True) cuts the state to fit")
        L = H.bucket_for(n, self.buckets)
        x = H.build_inputs(row["ids"], prefix, L)
        x["qtype_onehot"] = H.qtype_onehot(row["q"])
        return H.readout_f64(self.runner(x, L), row["P"], row["markers"], row["q"], row["calibrate"],
                             self.temperatures)

    # ------------------------------------------------------------------ the provider's API

    def probabilities(self, state, questions, images=None, audio=None):
        """Each question's distribution over its options, in option order (yes, no for a noul)."""
        kind, prefix = self._media(images, audio)
        return [self.score(r, prefix) for r in self.rows(state, list(questions), prefix, kind)]

    def system_one(self, state, questions, images=None, audio=None):
        """Named questions over one state, and its images or audio if any:
        {"answers": {name: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}."""
        named = {n: Pm.as_question(q) for n, q in questions.items()}
        kind, prefix = self._media(images, audio)
        rows = self.rows(state, list(named.values()), prefix, kind)
        probs = [self.score(r, prefix) for r in rows]
        return {"answers": {n: Pm.answer(q, p) for (n, q), p in zip(named.items(), probs)},
                "usage": {"input_tokens": sum(r["P"] + len(r["ids"]) for r in rows), "output_tokens": 0}}

    def decide(self, state, questions):
        return self.system_one(state, questions)

    def decide_image(self, image, questions, state=None):
        return self.system_one(state, questions, images=image)

    def decide_audio(self, audio, questions, state=None):
        return self.system_one(state, questions, audio=audio)

    def check_files(self):
        """sha256 and bytes of every file contract.json lists -> {name: True / reason}."""
        out = {}
        for f in self.contract["files"]:
            p = self.repo / f["name"]
            if not p.is_file():
                out[f["name"]] = "missing"
            elif p.stat().st_size != f["bytes"]:
                out[f["name"]] = f"bytes {p.stat().st_size} != {f['bytes']}"
            else:
                out[f["name"]] = True if sha256_file(p) == f["sha256"] else "sha256 differs"
        return out

    def close(self):
        self.runner.close()
        for g in [self._tower, self._projector] + list(self._audio.values()):
            if g is not None:
                g.close()
        self._tower = self._projector = None
        self._audio = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m host.d1_omni", description="d1-omni-600M on LiteRT: one request")
    ap.add_argument("--repo", default=".", help="the repository directory (contract.json, the .tflite files)")
    ap.add_argument("--questions", required=True, help="JSON file: {name: question}")
    st = ap.add_mutually_exclusive_group()
    st.add_argument("--state", help="JSON file holding the state (a string, any JSON value, or null)")
    st.add_argument("--state-text", help="a text state given inline")
    md = ap.add_mutually_exclusive_group()
    md.add_argument("--image", action="append", help="an image file (repeat for several images, in order)")
    md.add_argument("--audio", help="a 16 kHz mono audio file (WAV / FLAC)")
    ap.add_argument("--gpu", action="store_true",
                    help="the GPU accelerator, each graph at its precision from contract.json (default: CPU)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (default 4)")
    ap.add_argument("--probabilities", action="store_true", help="print the raw distributions instead")
    ap.add_argument("--truncate-state", action="store_true", help="cut a long state to fit the L4096 bucket")
    a = ap.parse_args(argv)
    questions = json.loads(Path(a.questions).read_text())
    state = json.loads(Path(a.state).read_text()) if a.state else a.state_text
    with D1Omni(a.repo, accelerator="gpu" if a.gpu else "cpu", threads=a.threads,
                truncate_state=a.truncate_state) as model:
        if a.probabilities:
            probs = model.probabilities(state, list(questions.values()), images=a.image, audio=a.audio)
            out = dict(zip(questions, probs))
        else:
            out = model.system_one(state, questions, images=a.image, audio=a.audio)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
