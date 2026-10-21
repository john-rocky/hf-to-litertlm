"""Answer a request of the SystemOne-compatible request shape with the clef-flash LiteRT graphs of this folder, or check
this folder against the provider's float32 reference.

Run from the repository root, in a Python with host/requirements-host.txt installed (examples/README.md):
    python examples/run_example.py --check                                # the GPU at float32 precision (Metal on a Mac)
    python examples/run_example.py --check --accelerator cpu              # the CPU (XNNPACK, 8 threads)
    python examples/run_example.py --request examples/request.json --image fixtures/images/clock_3oclock_256x240.png --tile 256
    python examples/run_example.py --request my_request.json [--image photo.jpg] [--accelerator cpu]

A request is {"model": str, "state": text or any JSON value, "questions": {id: {"type": "noul" | "choice" | "score",
"instructions": str, "criteria": ...}}}: choice criteria {option id: description}, score criteria a list of levels
(lowest first), noul criteria optional {"true": ..., "false": ...}. --image adds one picture (or put a local path in
"images": [...]), read at a square tile of 448 px (196 picture tokens, the default) or 256 px (64). The response body
is printed as JSON on stdout. A request the host refuses prints `refused: <error>: <message>` on stderr: RequestTooLong
(more than 4,096 tokens, the largest row graph), UnsupportedMedia (more than one picture, videos, media_kwargs, a
picture that cannot be opened, a URL, a tile other than 256 / 448) and the provider's own checks (MissingModelOrState,
NoQuestions, BadQuestionType, EmptyCriteria, SchemaTooLong).

--check runs the four public fixture requests named in examples/run_example.expected.json (a SemIf record, a support
ticket with noul / choice / score questions, a JSON order, and a drawn picture at the 256 px tile; the row graphs of
lengths 256 and 512) and compares them with the provider's own code in float32 on the CPU: on every question whose two
most likely options are more than 0.02 apart the most likely option is the same, every probability is within 0.02, the
mean over the requests of each request's mean |difference| is at most 0.002 (the acceptance line of this conversion),
and the input tokens, row graph length and answer fields are equal; then it sends the refusal requests of REFUSALS and
compares each error's name (and message, where it names no path). It prints one line, PASS or FAIL.

Exit codes: 0 answered / PASS, 1 check FAIL, 2 bad arguments, 3 refused request, 4 a file of the folder is missing."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))

import clef_litert as C  # noqa: E402

EXPECTED = ROOT / "examples/run_example.expected.json"
PICTURE = "fixtures/images/shapes_224.png"


def _refusals() -> dict:
    """name -> (request, tile): one request for each refusal the host names (the --check part)."""
    base = {"model": "clef-flash", "state": "The kettle stopped heating after a week.",
            "questions": {"faulty": {"type": "noul", "instructions": "The product is faulty."}}}
    long_state = {"room": "cold-room-2", "unit": "celsius",
                  "readings": [{"minute": i, "value": round(3.0 + (i % 9) * 0.1, 1)} for i in range(400)]}
    many = {f"option_{i:04d}": f"Description number {i} of a very long list of options" for i in range(1500)}

    def edit(**kw):
        r = copy.deepcopy(base)
        for k, v in kw.items():
            if v is None:
                r.pop(k, None)
            else:
                r[k] = v
        return r

    pic = str(ROOT / PICTURE)
    return {
        "too_long": (edit(state=long_state), 448),
        "two_pictures": (edit(images=[pic, pic]), 448),
        "videos": (edit(videos=["clip.mp4"]), 448),
        "media_kwargs": (edit(media_kwargs={"max_pixels": 1000}), 448),
        "url_picture": (edit(images=["https://example.com/picture.png"]), 448),
        "unreadable_picture": (edit(images=[str(ROOT / "examples/request.json")]), 448),
        "tile_300": (edit(images=[pic]), 300),
        "no_model": (edit(model=None), 448),
        "no_questions": (edit(questions={}), 448),
        "bad_type": (edit(questions={"faulty": {"type": "boolean"}}), 448),
        "empty_criteria": (edit(questions={"team": {"type": "choice", "criteria": {}}}), 448),
        "schema_too_long": (edit(questions={"team": {"type": "choice", "criteria": many}}), 448),
    }


def load_folder(accelerator: str, threads: int) -> C.ClefLiteRT:
    missing = [p for p in (C.FOLDER["tokenizer"], C.FOLDER["embed_tokens"], C.FOLDER["lm_head"],
                           C.FOLDER["joint_head"], C.FOLDER["joint_head_config"]) if not (ROOT / p).is_file()]
    if missing:
        print(f"missing in {ROOT}: {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(4)
    try:
        return C.ClefLiteRT.from_dir(ROOT, accelerator=accelerator, threads=threads)
    except ValueError as e:                      # not one row graph file in the folder
        print(f"{e} in {ROOT}", file=sys.stderr)
        raise SystemExit(4)


def answer(args) -> int:
    request = json.loads(Path(args.request).read_text())
    if args.image:
        request["images"] = list(request.get("images") or []) + [args.image]
    with load_folder(args.accelerator, args.threads) as clef:
        t = time.perf_counter()
        try:
            body = clef.decide(request, tile=args.tile)
        except C.RequestError as e:
            print(f"refused: {type(e).__name__}: {e}", file=sys.stderr)
            return 3
        seconds = time.perf_counter() - t
    print(json.dumps(body, indent=1, ensure_ascii=False))
    print(f"# {seconds:.1f} s on the {args.accelerator}, with the compile of the graphs this request needed",
          file=sys.stderr)
    return 0


def check(args) -> int:
    expected = json.loads(EXPECTED.read_text())
    records = {r["id"]: r for r in json.loads((ROOT / "fixtures/requests_public.json").read_text())["records"]}
    problems, worst, means = [], (0.0, ""), []
    with load_folder(args.accelerator, args.threads) as clef:
        kept = {}
        row_graph = clef.graphs.row_graph

        def keep_hidden(length, inputs):         # the hidden rows decide() reads, kept for the unrounded check
            kept["hidden"] = row_graph(length, inputs)
            return kept["hidden"]

        clef.graphs.row_graph = keep_hidden
        for item in sorted(expected["requests"], key=lambda x: x["length"]):
            name = item["record"]
            request = copy.deepcopy(records[name]["request"])
            if item.get("image"):
                request["images"] = [str(ROOT / item["image"])]
            body = clef.decide(request, tile=item["tile"])
            enc = clef.host.encode(request, tile=item["tile"])
            logits = clef.host.scores(enc, kept["hidden"])
            if clef.host.respond(request, enc, logits) != body:
                problems.append(f"{name}: the response is not the one of its own probabilities")
            if (enc.record.tokens, enc.length) != (item["input_tokens"], item["length"]):
                problems.append(f"{name}: {enc.record.tokens} tokens on L{enc.length}, expected {item['input_tokens']} "
                                f"on L{item['length']}")
            if body["usage"] != {"input_tokens": item["input_tokens"], "output_tokens": 0}:
                problems.append(f"{name}: usage {body['usage']}")
            diffs = []
            for q, z in zip(enc.record.questions, logits):
                e = item["questions"][q.question_id]
                got = dict(zip(q.option_ids, (float(x) for x in C.question_probs(z))))
                if list(got) != e["option_ids"]:
                    problems.append(f"{name}/{q.question_id}: options {list(got)}")
                    continue
                d = [abs(got[o] - r) for o, r in zip(e["option_ids"], e["probabilities"])]
                diffs += d
                if max(d) > worst[0]:
                    worst = (max(d), f"{name}/{q.question_id}")
                top = max(got, key=got.__getitem__)
                if not e["near_tie"] and top != e["argmax"]:
                    problems.append(f"{name}/{q.question_id}: most likely {top}, expected {e['argmax']}")
                if max(d) > expected["line"]["max_abs_dp"]:
                    problems.append(f"{name}/{q.question_id}: |dp| {max(d):.2e}")
                a = body["answers"][q.question_id]
                if [a["type"], sorted(a)] != [e["type"], e["answer_fields"]]:
                    problems.append(f"{name}/{q.question_id}: answer {a}")
            means.append(sum(diffs) / max(len(diffs), 1))
            print(f"# {name}: {enc.record.tokens} tokens, L{enc.length}" + (f", {enc.tile} px tile" if enc.tile else "")
                  + f", max |dp| {max(diffs):.2e}", file=sys.stderr)
        mean = sum(means) / len(means)
        if mean > expected["line"]["mean_of_request_means"]:
            problems.append(f"mean of the requests' mean |dp| {mean:.2e}")
        cases = _refusals()
        for name, want in expected["refusals"].items():
            request, tile = cases[name]
            try:
                clef.decide(request, tile=tile)
                problems.append(f"refusal {name}: answered")
            except C.RequestError as e:
                if type(e).__name__ != want["error"] or (want["message"] is not None and str(e) != want["message"]):
                    problems.append(f"refusal {name}: {type(e).__name__}: {e}")
    verdict = "PASS" if not problems else "FAIL"
    print(f"{verdict} ({args.accelerator}): {len(expected['requests'])} requests, max |dp| {worst[0]:.2e} ({worst[1]}), "
          f"mean of the requests' mean |dp| {mean:.2e}, {len(expected['refusals'])} refusals named"
          + ("" if not problems else "; " + "; ".join(problems)))
    return 0 if not problems else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--request", help="a request JSON file")
    ap.add_argument("--image", help="one picture for the request (a local file)")
    ap.add_argument("--tile", type=int, default=448, help="the picture's square tile: 448 (default) or 256")
    ap.add_argument("--accelerator", choices=["gpu", "cpu"], default="gpu",
                    help="gpu = float32 precision (Metal on a Mac, the default), cpu = XNNPACK")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (default 8)")
    ap.add_argument("--check", action="store_true", help="compare with examples/run_example.expected.json")
    a = ap.parse_args()
    if a.check == bool(a.request):
        ap.error("give --request or --check")
    return check(a) if a.check else answer(a)


if __name__ == "__main__":
    sys.exit(main())
