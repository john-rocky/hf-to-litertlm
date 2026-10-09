"""Round 13 (step A-7): the provider's float32 answers for the two audio buckets that hold no reference row — T501
(aud_01 cut to 5 s = its first 80,000 samples) and T3001 (aud_01 + aud_02 + aud_03 concatenated, 434,400 samples =
27.15 s), the clips round 7 built (scripts/audio_graph.py clip_samples / long_samples) and the phone ran in round 12b
(legs A5 / A8). Reference venv, from K:

    cd d1_omni_work
    HF_HUB_OFFLINE=1 HF_MODULES_CACHE=cache/hf_modules_r13 ~/code/standup/tools/quiet/quiet_wait.py -- \\
        venv-ref/bin/python scripts/r13_audio_e2e_ref.py
    -> results/audio_e2e_ref_T501_T3001.json and cache/r13/audio_e2e_ref_prefix.npz (the provider's prefixes)

Requests (state "Voice note from a user.", as both records in fixtures/requests.json have it):
  aud_01_cut5          the 5 s clip with aud_01's three questions (topic / wants / urgency)
  long_card_topic      the 27.15 s clip with card_topic's question (topic: food / travel / weather)
  long_aud_01          the 27.15 s clip with aud_01's three questions (a second set on the same clip)
The provider's code runs as round 2's reference does (scripts/ref_probs.py: load_model() asserts the loaded state
equals model.safetensors; run_record() = the natural call, system_one and one call per question, with every assertion
of that script: host ids / markers = the provider's rows, host readout = the returned probabilities bit for bit, the
prefix length = the provider's arithmetic): AutoModel.from_pretrained(snapshot, trust_remote_code=True,
dtype=float32), CPU, 12 threads. Records are not added to ref/records_ref.json. The provider's prefix of each clip is
compared bit for bit with round 7's provider prefix of the same clip (out/r7_eager_prefix_T501_aud01cut5.npz
`provider`, out/r7_eager_prefix_T3001_long.npz `provider_prefix`), which ties the clips to the ones the phone ran.
"""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
K = Path(__file__).resolve().parents[1]
_mc = os.environ.get("HF_MODULES_CACHE", "cache/hf_modules_r13")
os.environ["HF_MODULES_CACHE"] = str(_mc if os.path.isabs(_mc) else (K / _mc).resolve())
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(K / "scripts"))

import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import wave  # noqa: E402

import numpy as np  # noqa: E402

import d1_src as S  # noqa: E402
import ref_probs as RP  # noqa: E402

OUT = K / "results/audio_e2e_ref_T501_T3001.json"
NPZ = K / "cache/r13/audio_e2e_ref_prefix.npz"
R7 = {"aud_01_cut5": (K / "out/r7_eager_prefix_T501_aud01cut5.npz", "provider"),
      "long": (K / "out/r7_eager_prefix_T3001_long.npz", "provider_prefix")}


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), path
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.int16)


def clips(records):
    """The two clips exactly as scripts/audio_graph.py builds them (clip_samples(..., seconds=5.0), long_samples())."""
    def samples(rid):
        r = records[rid]
        path = K / r["media"]["ref"]
        assert S.sha256_file(path) == r["media"]["sha256"], rid
        return read_wav(path), str(path.relative_to(K))

    a1, p1 = samples("aud_01")
    cut = a1[: int(round(5.0 * 16000))]
    parts = [samples(r) for r in ("aud_01", "aud_02", "aud_03")]
    long = np.concatenate([x for x, _ in parts])
    return {"aud_01_cut5": {"samples": cut, "from": f"{p1}, the first 80,000 samples"},
            "long": {"samples": long, "from": " + ".join(p for _, p in parts)}}


def main():
    import torch

    torch.set_num_threads(RP.THREADS)
    sys.path.insert(0, str(K / "host"))
    import d1_host as H

    t0 = time.time()
    reqs = {r["id"]: r for r in json.loads((K / "fixtures/requests.json").read_text())["records"]}
    cl = clips(reqs)
    a1q, ctq = reqs["aud_01"]["request"], reqs["card_topic"]["request"]
    assert a1q["state"] == ctq["state"] == "Voice note from a user."
    plan = [("aud_01_cut5", "aud_01_cut5", a1q), ("long_card_topic", "long", ctq), ("long_aud_01", "long", a1q)]
    model, load, prompt_mod, vision_mod, temps = RP.load_model()
    taps = RP.Taps(model)
    o = RP.Oracle(model, taps, H, temps)
    entries, prefixes = [], {}
    with torch.no_grad():
        for rid, clip, req in plan:
            x16 = cl[clip]["samples"]
            names = list(req["questions"])
            record = {"id": rid, "source": "own_audio", "publishable": False, "media": {"kind": "audio"},
                      "request": {"state": req["state"], "questions": req["questions"]}, "gold": {}}
            entry, arrays, _, _ = RP.run_record(o, record, names, None, x16, vision_mod, None)
            pre = arrays["prefix"]
            prefixes[f"{rid}.prefix"] = pre
            p7, key7 = R7[clip]
            with np.load(p7) as z:
                r7 = np.asarray(z[key7], np.float32)
            entry["clip"] = {"name": clip, "from": cl[clip]["from"], "samples": int(x16.shape[0]),
                             "seconds": round(x16.shape[0] / 16000, 4),
                             "int16_sha256": hashlib.sha256(x16.astype("<i2").tobytes()).hexdigest()}
            entry["prefix_vs_round7_provider"] = {"file": str(p7.relative_to(K)), "key": key7,
                                                  "bit_equal": bool(r7.shape == pre.shape and np.array_equal(r7, pre)),
                                                  "max_abs": float(np.abs(r7.astype(np.float64) - pre).max())
                                                  if r7.shape == pre.shape else None,
                                                  "shape": list(pre.shape)}
            entry["prefix_sha256"] = hashlib.sha256(np.ascontiguousarray(pre, np.float32).tobytes()).hexdigest()
            entries.append(entry)
            print(f"{rid}: P {entry['prefix']}, {len(names)} questions, prefix vs round 7 provider "
                  f"bit_equal {entry['prefix_vs_round7_provider']['bit_equal']}, "
                  f"{entry['seconds']} s", flush=True)
    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ, **prefixes)
    doc = {"step": "round 13 step A-7: the provider's float32 answers for an audio clip in T501 and one in T3001 "
                   "(no reference row of their own in ref/records_ref.json)",
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t0, 1),
           "reference": "AutoModel.from_pretrained(snapshot, trust_remote_code=True, dtype=torch.float32) "
                        "(modeling_d1.D1OmniModel), CPU fp32, 12 threads, probabilities(state, questions, audio=int16 "
                        "samples) per request (natural) + per question (single); scripts/ref_probs.py run_record",
           "model": {"hf_id": S.REPO, "revision": S.REV, "weights_sha256": RP.WEIGHTS_SHA256},
           "load": {k: load[k] for k in ("torch", "transformers", "torch_threads", "mha_fastpath_enabled",
                                         "state_equals_safetensors", "model_dtype")},
           "requests_sha256": S.sha256_file(K / "fixtures/requests.json"),
           "prefix_npz": {"file": str(NPZ.relative_to(K)), "sha256": S.sha256_file(NPZ)},
           "records": entries}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"out": str(OUT.relative_to(K)), "records": [
        {"id": e["id"], "P": e["prefix"], "questions": len(e["questions"]),
         "prefix_vs_round7": e["prefix_vs_round7_provider"]["bit_equal"],
         "probs": {q["qid"]: [round(v, 4) for v in q["probs"]] for q in e["questions"]}} for e in entries],
        "seconds": doc["seconds"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
