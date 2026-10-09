"""Host tests for d1-omni-600M on LiteRT (unittest, CPU). Run from anywhere:

    python host/tests.py --repo .                      # the repository's own files: fixtures/public_*.json
    python host/tests.py --repo . --reference records_ref.json --requests requests.json --out tests.json
        # + every row of a full reference file (the provider's float32 run over the whole check set)
    python host/tests.py --repo . --quick              # skip the sha256 of the graph files

What is checked:
  files       every file of contract.json: bytes and sha256 (--quick: bytes only for the .tflite files)
  token ids   contract.json's nine token ids = tokenizer.json's
  encode      prompt.encode() through the host's tokenizer = the provider's ids and marker positions, every question
              (public fixtures; with --reference, every row of the reference)
  buckets     bucket_for() of the decision graph and of the audio graph at their edges; a 30 s clip = T 3001, P 375
  onehot      qtype_onehot() per question type
  readout     the scores at P + marker -> temperature (text) or none (image / audio) -> softmax -> a noul reversed:
              synthetic rows against a direct float64 softmax; with --reference, the reference's own logits ->
              its probabilities within 1e-6 on every row
  media rule  max_len = min(mode max_len, 16384 - P), refused under 64; the L4096 limit and truncate_state
  audio state an audio request's state None is encoded as {} (the provider's rule), an image request's as ""
  smoke       D1Omni on one public record per mode (text / image / audio), CPU: argmax = the reference on every
              question, max |dp| <= 0.02; system_one()'s response = prompt.answer() of the probabilities
  precision   D1Omni(accelerator="gpu") takes each graph's GPU precision from contract.json (nothing is compiled);
              precision="fp32" sets every graph to fp32; the CPU model has none
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1_audio_host as A  # noqa: E402
import d1_host as H  # noqa: E402
import d1_omni as O  # noqa: E402
import d1_prompt as Pm  # noqa: E402

CFG = {"repo": None, "reference": None, "requests": None, "quick": False, "threads": 4}
DETAILS = {}
_MODEL = {}


def model():
    if "m" not in _MODEL:
        _MODEL["m"] = O.D1Omni(CFG["repo"], accelerator="cpu", threads=CFG["threads"])
    return _MODEL["m"]


def public(mode):
    return json.loads((CFG["repo"] / "fixtures" / f"public_{mode}.json").read_text())


def fake_prefix(P):
    return None if P == 0 else np.zeros((P, 1), np.float32)     # rows() reads only the row count


def kind_of(mode):
    return {"text": "text", "image": "image", "audio": "audio"}[mode]


class Files(unittest.TestCase):
    def test_files(self):
        c = model().contract
        bad, checked = {}, 0
        for f in c["files"]:
            p = CFG["repo"] / f["name"]
            checked += 1
            if not p.is_file():
                bad[f["name"]] = "missing"
                continue
            if p.stat().st_size != f["bytes"]:
                bad[f["name"]] = f"bytes {p.stat().st_size} != {f['bytes']}"
                continue
            if CFG["quick"] and f["name"].endswith(".tflite"):
                continue
            if O.sha256_file(p) != f["sha256"]:
                bad[f["name"]] = "sha256 differs"
        DETAILS["files"] = {"files": checked, "sha256_of_tflite": not CFG["quick"], "bad": bad}
        self.assertEqual(bad, {})

    def test_token_ids(self):
        c, tok = model().contract, model().tok
        got = {t: tok.convert_tokens_to_ids(t) for t in c["token_ids"]}
        DETAILS["token_ids"] = {"checked": len(got)}
        self.assertEqual(got, c["token_ids"])
        self.assertEqual(tok.bos_token_id, c["token_ids"]["<|startoftext|>"])


class Encode(unittest.TestCase):
    def check_rows(self, items):
        """items: (key, state, question dict, kind, P, ids, markers) -> mismatching keys."""
        m, bad = model(), []
        for key, state, qd, kind, P, ids, markers in items:
            r = m.rows(state, [qd], fake_prefix(P), kind)[0]
            if r["ids"] != ids or r["markers"] != markers:
                bad.append(key)
        return bad

    def test_encode_public(self):
        items = []
        for mode in ("text", "image", "audio"):
            for rec in public(mode)["records"]:
                for e in rec["expected"]:
                    items.append((f"{rec['id']}/{e['name']}", rec["state"], rec["questions"][e["name"]], kind_of(mode),
                                  e["P"], e["ids"], e["markers"]))
        bad = self.check_rows(items)
        DETAILS["encode_public"] = {"rows": len(items), "equal": len(items) - len(bad), "mismatch": bad[:20]}
        self.assertEqual(bad, [])

    def test_encode_reference_all_rows(self):
        if not CFG["reference"]:
            self.skipTest("no --reference")
        ref = json.loads(CFG["reference"].read_text())
        req = {r["id"]: r for r in json.loads(CFG["requests"].read_text())["records"]}
        items = []
        for rec in ref["records"]:
            request = req[rec["id"]]["request"]
            for q in rec["questions"]:
                items.append((f"{rec['id']}/{q['qid']}", request["state"], request["questions"][q["qid"]],
                              kind_of(rec["mode"]), int(q["prefix"] or 0), q["ids"], q["markers"]))
        bad = self.check_rows(items)
        DETAILS["encode_reference"] = {"records": len(ref["records"]), "rows": len(items),
                                       "equal": len(items) - len(bad), "mismatch": bad[:20]}
        self.assertEqual(bad, [])


class Buckets(unittest.TestCase):
    def test_decide_buckets(self):
        b = model().buckets
        self.assertEqual(b, (128, 256, 512, 1024, 2048, 4096))
        cases = {1: 128, 128: 128, 129: 256, 256: 256, 257: 512, 1024: 1024, 1025: 2048, 4096: 4096}
        for n, L in cases.items():
            self.assertEqual(H.bucket_for(n, b), L, n)
        with self.assertRaises(ValueError):
            H.bucket_for(4097, b)
        DETAILS["decide_buckets"] = {"cases": len(cases) + 1}

    def test_audio_buckets(self):
        b = model().audio_buckets
        self.assertEqual(b, (501, 1001, 2001, 3001))
        cases = {1: 501, 501: 501, 502: 1001, 1001: 1001, 2002: 3001, 3001: 3001}
        for T, Tb in cases.items():
            self.assertEqual(A.bucket_for(T, b), Tb, T)
        with self.assertRaises(ValueError):
            A.bucket_for(3002, b)
        self.assertEqual(A.frame_count(30 * 16000), (3000, 3001))
        self.assertEqual(A.lengths(3000)[2], 375)
        self.assertEqual(A.dims(1001), (501, 251, 126))
        DETAILS["audio_buckets"] = {"cases": len(cases) + 4}


class OneHot(unittest.TestCase):
    def test_qtype_onehot(self):
        cases = {"choice": [1, 0, 0], "score": [0, 1, 0], "noul": [0, 0, 1]}
        for t, v in cases.items():
            q = Pm.as_question({"type": t, "instructions": "x",
                                "criteria": {"a": "", "b": ""} if t == "choice" else (["lo", "hi"] if t == "score"
                                                                                     else None)})
            x = H.qtype_onehot(q)
            self.assertEqual(x.dtype, np.float32)
            self.assertEqual(x.shape, (1, 3))
            self.assertEqual(x[0].tolist(), v)
        DETAILS["qtype_onehot"] = {"types": 3}


class Readout(unittest.TestCase):
    def test_readout_synthetic(self):
        """Rules against a direct float64 softmax: text / temperature, media / none, noul reversed, K slots only."""
        temps, g, worst, n = model().temperatures, np.random.default_rng(11), 0.0, 0
        for t, k in (("noul", 2), ("choice", 2), ("choice", 4), ("choice", 7), ("choice", 12), ("score", 3),
                     ("score", 8)):
            for calibrate in (True, False):
                crit = ({f"o{i}": "" for i in range(k)} if t == "choice" else [f"l{i}" for i in range(k)]
                        if t == "score" else None)
                q = Pm.as_question({"type": t, "instructions": "x", "criteria": crit})
                P, markers = 5, [3 + 4 * i for i in range(k)]
                z = g.normal(0, 3, k)
                scores = g.normal(0, 5, (1, P + markers[-1] + 9)).astype(np.float32)
                for m, v in zip(markers, z.astype(np.float32)):
                    scores[0, P + m] = v
                zz = z.astype(np.float32).astype(np.float64)
                T = temps.get(Pm.temperature_key(q), temps.get(t, 1.0)) if calibrate else 1.0
                e = np.exp(zz / T - (zz / T).max())
                want = (e / e.sum()).tolist()
                want = want[::-1] if t == "noul" else want
                got = H.readout_f64(scores, P, markers, q, calibrate, temps)
                worst = max(worst, max(abs(a - b) for a, b in zip(got, want)))
                n += 1
        DETAILS["readout_synthetic"] = {"rows": n, "max_abs_dp": worst}
        self.assertLessEqual(worst, 1e-12)

    def test_readout_reference_all_rows(self):
        if not CFG["reference"]:
            self.skipTest("no --reference")
        ref, temps = json.loads(CFG["reference"].read_text()), model().temperatures
        worst, n, bad_T, bad_cal, by_mode = 0.0, 0, [], [], {}
        for rec in ref["records"]:
            want_cal = rec["mode"] == "text"
            for q in rec["questions"]:
                qq = Pm.as_question({"type": q["type"], "instructions": "x", "criteria": _criteria(q)})
                P, markers = int(q["prefix"] or 0), q["markers"]
                scores = np.zeros((1, P + max(markers) + 1), np.float32)
                for m, v in zip(markers, q["logits_raw"]):
                    scores[0, P + m] = np.float32(v)
                got = H.readout_f64(scores, P, markers, qq, bool(q["calibrate"]), temps)
                d = max(abs(a - b) for a, b in zip(got, q["probs"]))
                worst, n = max(worst, d), n + 1
                by_mode[rec["mode"]] = max(by_mode.get(rec["mode"], 0.0), d)
                if bool(q["calibrate"]) != want_cal:
                    bad_cal.append(f"{rec['id']}/{q['qid']}")
                if q["calibrate"] and abs(H.temperature(qq, temps) - q["T"]) > 1e-12:
                    bad_T.append(f"{rec['id']}/{q['qid']}")
        DETAILS["readout_reference"] = {"rows": n, "max_abs_dp": worst, "max_abs_dp_by_mode": by_mode,
                                        "temperature_mismatch": bad_T, "calibrate_flag_mismatch": bad_cal}
        self.assertEqual(bad_T, [])
        self.assertEqual(bad_cal, [])
        self.assertLessEqual(worst, 1e-6)


def _criteria(q):
    """A stand-in criteria of the right size for a reference row (readout reads only type and option count)."""
    k = int(q["K"])
    return ({f"o{i}": "" for i in range(k)} if q["type"] == "choice" else [f"l{i}" for i in range(k)]
            if q["type"] == "score" else None)


class MediaRule(unittest.TestCase):
    Q = {"type": "noul", "instructions": "Is anything wrong?"}
    LONG = " ".join(f"word{i}" for i in range(9000))

    def test_media_max_len(self):
        m, out = model(), {}
        for kind, mode_len in (("image", 896), ("audio", 15360)):
            for P in (100, 16000, 16320):
                expect = min(mode_len, 16384 - P)
                r = m.rows(self.LONG, [self.Q], fake_prefix(P), kind)[0]
                self.assertEqual(len(r["ids"]), expect, (kind, P))
                out[f"{kind}_P{P}"] = len(r["ids"])
            with self.assertRaises(ValueError):
                m.rows(self.LONG, [self.Q], fake_prefix(16384 - 63), kind)
            out[f"{kind}_refused_P{16384 - 63}"] = True
        DETAILS["media_max_len"] = out

    def test_largest_bucket(self):
        m = model()
        r = m.rows(self.LONG, [self.Q], None, "text")[0]
        self.assertGreater(len(r["ids"]), 4096)
        with self.assertRaises(ValueError):
            m.score(r)
        t = O.D1Omni(CFG["repo"], truncate_state=True)
        r2 = t.rows(self.LONG, [self.Q], None, "text")[0]
        self.assertEqual(len(r2["ids"]), 4096)
        r3 = t.rows(self.LONG, [self.Q], fake_prefix(300), "image")[0]
        self.assertEqual(len(r3["ids"]), 896)
        DETAILS["largest_bucket"] = {"text_ids_untruncated": len(r["ids"]), "truncated": len(r2["ids"])}

    def test_state_none(self):
        m, qd = model(), {"type": "choice", "instructions": "What is it about?", "criteria": {"a": "A", "b": "B"}}
        a_none = m.rows(None, [qd], fake_prefix(63), "audio")[0]
        a_obj = m.rows({}, [qd], fake_prefix(63), "audio")[0]
        a_str = m.rows("", [qd], fake_prefix(63), "audio")[0]
        i_none = m.rows(None, [qd], fake_prefix(144), "image")[0]
        i_str = m.rows("", [qd], fake_prefix(144), "image")[0]
        self.assertEqual(a_none["ids"], a_obj["ids"])
        self.assertNotEqual(a_none["ids"], a_str["ids"])
        self.assertEqual(i_none["ids"], i_str["ids"])
        DETAILS["state_none"] = {"audio_none_equals_{}": True, "image_none_equals_empty": True}


class Precision(unittest.TestCase):
    def test_gpu_precision_from_contract(self):
        want = model().contract["precision"]["mac_metal"]["graphs"]
        g = O.D1Omni(CFG["repo"], accelerator="gpu")          # no graph is compiled until a request needs it
        f = O.D1Omni(CFG["repo"], accelerator="gpu", precision="fp32")
        try:
            self.assertEqual(g.graph_precision, want)
            self.assertEqual(g.runner.precision, want["decide"])
            self.assertEqual(set(f.graph_precision.values()), {"fp32"})
        finally:
            g.close()
            f.close()
        self.assertEqual(set(model().graph_precision.values()), {None})
        with self.assertRaises(ValueError):
            O.D1Omni(CFG["repo"], accelerator="gpu", precision="fp16")
        DETAILS["gpu_precision"] = {"contract": want}


class Smoke(unittest.TestCase):
    PICK = {"text": "card_text", "image": "img_dogs_01", "audio": "aud_01"}

    def run_mode(self, mode):
        doc = public(mode)
        rec = next(r for r in doc["records"] if r["id"] == self.PICK[mode])
        media = rec["media"]
        kw = {}
        if media:
            path = CFG["repo"] / "fixtures" / media["file"]
            self.assertEqual(O.sha256_file(path), media["sha256"])
            kw = {"images": [path]} if mode == "image" else {"audio": path}
        names = [e["name"] for e in rec["expected"]]
        t0 = time.time()
        probs = model().probabilities(rec["state"], [rec["questions"][n] for n in names], **kw)
        resp = model().system_one(rec["state"], {n: rec["questions"][n] for n in names}, **kw)
        secs = time.time() - t0
        worst, agree = 0.0, 0
        for e, p in zip(rec["expected"], probs):
            worst = max(worst, max(abs(a - b) for a, b in zip(p, e["probs"])))
            agree += int(np.argmax(p) == np.argmax(e["probs"]))
        want = {n: Pm.answer(Pm.as_question(rec["questions"][n]), p) for n, p in zip(names, probs)}
        DETAILS[f"smoke_{mode}"] = {"record": rec["id"], "questions": len(names), "argmax_equal": agree,
                                    "max_abs_dp": worst, "usage_input_tokens": resp["usage"]["input_tokens"],
                                    "reference_usage_input_tokens": rec["usage_input_tokens"],
                                    "seconds": round(secs, 2)}
        self.assertEqual(agree, len(names))
        self.assertLessEqual(worst, 0.02)
        self.assertEqual(resp["answers"], want)
        self.assertEqual(resp["usage"], {"input_tokens": rec["usage_input_tokens"], "output_tokens": 0})

    def test_text(self):
        self.run_mode("text")

    def test_image(self):
        self.run_mode("image")

    def test_audio(self):
        self.run_mode("audio")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(HERE.parent))
    ap.add_argument("--reference", help="a full reference file (records with ids / markers / logits_raw / probs)")
    ap.add_argument("--requests", help="the requests of that reference (records with request.state / questions)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="bytes only for the .tflite files (no sha256)")
    ap.add_argument("--out", help="write the results as JSON")
    a = ap.parse_args()
    if bool(a.reference) != bool(a.requests):
        ap.error("--reference and --requests go together")
    CFG.update(repo=Path(a.repo), reference=Path(a.reference) if a.reference else None,
               requests=Path(a.requests) if a.requests else None, quick=a.quick, threads=a.threads)
    t0 = time.time()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    status = {t.id().split(".", 1)[1]: "pass" for t in suite_tests(suite)}   # before the run: the suite drops its tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    for t, tb in result.failures:
        status[t.id().split(".", 1)[1]] = "fail: " + tb.strip().splitlines()[-1]
    for t, tb in result.errors:
        status[t.id().split(".", 1)[1]] = "error: " + tb.strip().splitlines()[-1]
    for t, why in result.skipped:
        status[t.id().split(".", 1)[1]] = "skip: " + why
    passed = sum(v == "pass" for v in status.values())
    doc = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t0, 1),
           "python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
           "packages": _versions(), "repo": str(CFG["repo"]), "reference": a.reference, "requests": a.requests,
           "quick": a.quick, "counts": {"tests": result.testsRun, "passed": passed, "failed": len(result.failures),
                                        "errors": len(result.errors), "skipped": len(result.skipped)},
           "tests": status, "details": DETAILS, "PASS": result.wasSuccessful()}
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(f"tests: {passed}/{result.testsRun} pass, {len(result.failures)} fail, {len(result.errors)} error, "
          f"{len(result.skipped)} skipped -> {'PASS' if doc['PASS'] else 'FAIL'}")
    if "m" in _MODEL:
        _MODEL["m"].close()
    return 0 if doc["PASS"] else 1


def suite_tests(s):
    for t in s:
        if isinstance(t, unittest.TestSuite):
            yield from suite_tests(t)
        else:
            yield t


def _versions():
    import importlib.metadata as md

    out = {}
    for p in ("ai-edge-litert", "numpy", "tokenizers", "Pillow", "soundfile"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


if __name__ == "__main__":
    sys.exit(main())
