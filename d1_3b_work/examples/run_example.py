"""Example: the two example requests of the source model card, answered on the CPU or the GPU, and one refused request.

Run from the repository root:
    pip install -r host/requirements-host.txt
    python examples/run_example.py --check                # CPU (XNNPACK, 8 threads)
    python examples/run_example.py --check --accel gpu    # GPU at float32 precision (Metal on a Mac)

1. Text: fixtures/requests_public.json `card_text_001`, one state and three questions (noul, choice, score).
2. Picture: `card_cats_001`, the card's photo as the whole state and one choice question. The photo (COCO val2017
   000000039769.jpg, two cats) is not in this repository: the script downloads it from cocodataset.org into memory and
   checks its SHA-256 before use. `--image <file>` reads a local copy instead, checked the same way.
3. A state of more than 4,096 tokens: the host refuses its row (RowTooLong) instead of cutting it.

--check compares the answers with examples/run_example.expected.json, which holds the provider's code in float32 on the
CPU (one row per question). Every probability must be within 1e-5 of it, and the most likely option, the input token
count, the route and the refusal message must be equal.

load_host() builds the host of this repository from contract.json: host/d1_litert.py (D1Host: text and pictures on the
row graphs) under host/d1_shared_state.py (D1SharedHost: the shared-state pairs, picked per request from the measured
call times in the contract). A graph file is compiled when a request needs it. At most `keep` text graphs
(row graphs and pairs) stay compiled; the least recently used one is closed to make room.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))

import d1_litert as H  # noqa: E402
import d1_shared_state as S  # noqa: E402
import d1_vision as V  # noqa: E402

# A cold-room log of 900 readings: its row is longer than the largest graph (4,096 tokens).
REFUSED_REQUEST = {
    "state": {"room": "cold-room-2", "unit": "celsius",
              "readings": [{"minute": i, "value": round(3.0 + (i % 9) * 0.1, 1)} for i in range(900)]},
    "questions": {"breach": {"type": "noul", "instructions": "Did the temperature go above 3.5 degrees?"}},
}


# ----------------------------------------------------------------------------------------------- the host


class GraphCache:
    """Compiled graphs by key; at most `keep` stay compiled, the least recently used is closed to make room."""

    def __init__(self, keep: int):
        self.keep, self.items = max(1, int(keep)), OrderedDict()

    def get(self, key, make):
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        while len(self.items) >= self.keep:
            _, old = self.items.popitem(last=False)
            old.close()
        self.items[key] = make()
        return self.items[key]

    def close(self):
        while self.items:
            self.items.popitem(last=False)[1].close()


class LazyRowGraphs(dict):
    """{L: embeds row graph} for D1Host: the keys are the files present; a graph is compiled when it is used."""

    def __init__(self, files: dict, cache: GraphCache, args: tuple):
        super().__init__({L: None for L in files})
        self.files, self.cache, self.args = files, cache, args

    def __getitem__(self, L):
        if L not in self.files:
            raise KeyError(L)
        return self.cache.get(("row", L), lambda: H.LiteRTEmbedsGraph(self.files[L], *self.args))


class LazyPair:
    """A shared-state pair with the attributes D1SharedHost reads; the file is compiled when it is used."""

    def __init__(self, path: Path, Ls: int, Lq: int, hidden: int, cache: GraphCache, args: tuple, embed_table, pad_id):
        self.path, self.Ls, self.Lq, self.hidden_size = Path(path), Ls, Lq, hidden
        self.handover, self.pad_id = "direct", pad_id
        self.cache, self.args, self.embed_table = cache, args, embed_table

    def _pair(self) -> S.SharedStatePair:
        return self.cache.get(("pair", self.Ls, self.Lq), lambda: S.SharedStatePair(
            self.path, *self.args, pad_id=self.pad_id, embed_table=self.embed_table))

    def run_state(self, state_ids):
        return self._pair().run_state(state_ids)

    def run_question(self, own_ids):
        return self._pair().run_question(own_ids)

    def close(self):
        pass   # the cache closes the compiled file


class LazyGraph:
    """One picture graph (tower or projector), compiled when it is called."""

    def __init__(self, path: Path, args: tuple):
        self.path, self.args, self.graph = Path(path), args, None

    def __call__(self, **feeds):
        if self.graph is None:
            self.graph = V.LiteRTGraph(self.path, *self.args)
        return self.graph(**feeds)

    def close(self):
        if self.graph is not None:
            self.graph.close()
            self.graph = None


class LoadedHost:
    """decide(request) -> the response; route(request) -> where its questions run; close()."""

    def __init__(self, shared: S.D1SharedHost, cache: GraphCache):
        self.shared, self.cache = shared, cache

    def decide(self, request: dict) -> dict:
        return self.shared.decide(request)

    def route(self, request: dict) -> dict:
        r = self.shared.route(request)
        if r["route"] == "pair":
            return {"route": "pair", "Ls": r["Ls"], "Lq": r["Lq"]}
        rows = self.shared.host.rows(request)
        return {"route": "row", "L": max(H.pick_L(len(x.ids), self.shared.host.row_buckets()) for x in rows)}

    def close(self):
        self.cache.close()
        if self.shared.host.vision is not None:
            self.shared.host.vision.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_host(root=ROOT, accelerator: str = "cpu", threads: int = 8, keep: int = 1) -> LoadedHost:
    """The host on the files of `root` that are present. accelerator "cpu" (XNNPACK, `threads`) or "gpu" (float32
    precision: GpuOptions(enforce_f32=True))."""
    root = Path(root)
    contract = json.loads((root / "contract.json").read_text())
    args = (accelerator, "fp32", threads)
    pad = int(contract["token_ids"]["pad"])
    embed_table = H.EmbedTable(root / contract["embeds_graph"]["table"])
    host = H.D1Host(H.D1Tokenizer(root / contract["tokenizer"]["file"]),
                    table=H.ReadoutTable.from_file(root / contract["readout_table"]["file"]), pad_id=pad,
                    embed_table=embed_table)
    cache = GraphCache(keep)
    rows = {int(g["L"]): root / g["file"] for g in contract["embeds_graph"]["buckets"] if (root / g["file"]).is_file()}
    host.embeds_graphs = LazyRowGraphs(rows, cache, args)   # after __init__, which copies a plain dict
    vision = contract["vision"]["real_files"]
    tower, projector = (root / vision[g]["v2_fp16fc"]["file"] for g in ("tower", "projector"))
    if tower.is_file() and projector.is_file():
        host.vision = V.VisionPath(LazyGraph(tower, args), LazyGraph(projector, args),
                                   V.load_position_table(root / contract["vision"]["position_table"]["file"]),
                                   V.token_ids(contract))
    hidden = int(contract["model"]["hidden_size"])
    pairs = [LazyPair(root / f["file"], int(f["Ls"]), int(f["Lq"]), hidden, cache, args, embed_table, pad)
             for f in contract["shared_state"]["files"] if (root / f["file"]).is_file()]
    pick = contract["shared_state"]["pick"]["call_ms"]
    call_ms = {"row": {int(L): float(ms) for L, ms in pick["row"].items()},
               "pair": {(int(p["Ls"]), int(p["Lq"])): (float(p["state_call_ms"]), float(p["question_call_ms"]))
                        for p in pick["pair"]}}
    return LoadedHost(S.D1SharedHost(host, pairs, call_ms=call_ms), cache)


# ----------------------------------------------------------------------------------------------- the requests


def record(root: Path, rid: str) -> dict:
    doc = json.loads((Path(root) / "fixtures/requests_public.json").read_text())
    return next(r for r in doc["records"] if r["id"] == rid)


def text_request(root: Path = ROOT) -> dict:
    return record(root, "card_text_001")["request"]


def picture_request(root: Path = ROOT, image: str | None = None) -> dict:
    """card_cats_001 with its photo's bytes; the photo's SHA-256 must match the fixture's."""
    req = dict(record(root, "card_cats_001")["request"])
    meta = req["images"][0]
    if image:
        data = Path(image).read_bytes()
    else:
        with urllib.request.urlopen(meta["url"], timeout=60) as resp:
            data = resp.read()
    got = hashlib.sha256(data).hexdigest()
    if got != meta["sha256"]:
        raise SystemExit(f"the photo's SHA-256 is {got}, expected {meta['sha256']} ({meta['url']}): not used")
    req["images"] = [data]
    return req


# ----------------------------------------------------------------------------------------------- the check


def probabilities(answer: dict, keys: list) -> list:
    if answer["type"] == "noul":
        return [answer["noul"], 1.0 - answer["noul"]]
    return [answer["probabilities"][k] for k in keys]


def check(results: dict, expected: dict) -> list:
    problems, tol = [], float(expected["tolerance_abs_dp"])
    for name in ("text", "picture"):
        exp, got = expected["requests"][name], results[name]
        if got["response"]["usage"]["input_tokens"] != exp["input_tokens"]:
            problems.append(f"{name}: input_tokens {got['response']['usage']['input_tokens']} != {exp['input_tokens']}")
        if got["route"] != exp["route"]:
            problems.append(f"{name}: route {got['route']} != {exp['route']}")
        for q, e in exp["questions"].items():
            p = probabilities(got["response"]["answers"][q], e["keys"])
            dp = max(abs(a - b) for a, b in zip(p, e["probabilities"]))
            top = e["keys"][max(range(len(p)), key=p.__getitem__)]
            got["max_abs_dp"] = max(got.get("max_abs_dp", 0.0), dp)
            if dp > tol or top != e["argmax_key"]:
                problems.append(f"{name}/{q}: max |dp| {dp:.2e} (tolerance {tol:.0e}), most likely {top} vs "
                                f"{e['argmax_key']}")
    exp, got = expected["requests"]["refused"], results["refused"]
    if (got.get("error"), got.get("message")) != (exp["error"], exp["message"]):
        problems.append(f"refused: {got} != {exp}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--accel", choices=["cpu", "gpu"], default="cpu", help="cpu (XNNPACK) or gpu (float32 precision)")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (default 8)")
    ap.add_argument("--image", help="a local copy of the photo (checked against the fixture's SHA-256)")
    ap.add_argument("--check", action="store_true", help="compare with examples/run_example.expected.json")
    a = ap.parse_args()
    results = {}
    with load_host(ROOT, a.accel, a.threads) as host:
        for name, request in (("text", text_request()), ("picture", picture_request(ROOT, a.image))):
            route = host.route(request)
            t = time.perf_counter()
            response = host.decide(request)
            seconds = time.perf_counter() - t
            results[name] = {"route": route, "response": response}
            print(json.dumps({"request": name, "route": route, "response": response}, indent=1))
            print(f"# {name}: {seconds:.1f} s, including the compile of its graphs", flush=True)
        try:
            host.decide(REFUSED_REQUEST)
            results["refused"] = {"error": None, "message": "answered"}
        except H.RequestError as e:
            results["refused"] = {"error": type(e).__name__, "message": str(e)}
        print(json.dumps({"request": "refused", **results["refused"]}))
    if not a.check:
        return 0
    expected = json.loads((ROOT / "examples/run_example.expected.json").read_text())
    problems = check(results, expected)
    for name in ("text", "picture"):
        print(f"# {name}: max |dp| {results[name]['max_abs_dp']:.2e} against the provider's float32 CPU reference")
    if problems:
        print("DIFFERS from run_example.expected.json:\n  " + "\n  ".join(problems))
        return 1
    print(f"matches run_example.expected.json ({a.accel})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
