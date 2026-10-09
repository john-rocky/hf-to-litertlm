"""Round 11 (round 13: the ship form): assemble ship/ = the repository d1-omni-600M-LiteRT as it would be uploaded
(no README.md: the card is written elsewhere), and results/ship_manifest.json. Exporter venv, from K:

    cd d1_omni_work
    ~/venvs/lt094dev/bin/python scripts/ship_build.py            # build / refresh ship/ (idempotent)

Every number in contract.json and the manifest is read from a file (the graphs themselves, results/*.json, the
reference); nothing is typed in.
 1. position table: host/vision_position_table.npy = d1_vision_host.read_position_table(checkpoint), float32
    [16, 16, 768], written once (a later run asserts the file equals a fresh read). positions_padded() of it is
    compared bit for bit with the reference run's tower input (ref/npz/<id>.npz `pos_resized`, transformers'
    resize_positional_embeddings) for every crop of the 7 image records, and with round 9's tower input file
    device/g9_vt_pos.f32 (13 crops) when present.
 2. graphs: the 12 ship files in out/ = the fp16-safe fp16 files of every graph (text: 36 of 50 norms rewritten,
    round 8 / 11; vision tower: 13 of 25 LayerNorms, audio: 31 of 87 LayerNorms, round 12) and the projector's fp16
    file; each sha256 is recomputed and asserted equal to its earlier record (results/f16safe_handoff.json,
    quant_L*_f16safe.json, vision_quant_f16safe.json, audio_quant_f16safe.json, contract_draft.json); bytes,
    operators and the signature's inputs (tensor-index order) / outputs come from a flatbuffer scan of the file
    itself (scripts/litert_run.scan).
 3. ship/<name> = os.link(out/<source>) (same volume, no copy); an existing name must already be that inode, or the
    inode of the source the previous manifest records for that name (round 13: the vision tower and the four audio
    files moved to their fp16-safe files) — that old link is removed (the out/ file stays) and the new one made.
 4. tokenizer.json, LICENSE: byte copies of hf_small/ (sha256 asserted equal); NOTICE written here.
 5. host/: byte copies of host/{d1_prompt,d1_host,d1_vision_host,d1_audio_host,d1_omni,tests,verify}.py,
    requirements.txt and vision_position_table.npy; a __pycache__/ left in ship/host by a run from ship/ is removed.
 6. fixtures/public_{text,image,audio}.json: the records of fixtures/requests.json with `publishable: true` whose
    provenance and media do not name COCO / LibriSpeech / FLUX (both conditions are checked and must agree), their
    accepted questions, and per question the reference's ids, markers, P and probabilities (ref/records_ref.json
    version 2: the provider's code, float32, CPU); no logits; `licences` per source. Media files copied to
    fixtures/media/ (sha256 kept); the upstream licences of the text records (SemIf MIT, MMLU MIT, the kev
    repository's Apache-2.0) written to fixtures/LICENSE-*.txt, each with a first line naming where it came from.
 7. contract.json (the host reads it: files, graphs, tokenizer, token ids, temperatures, modes, position table, the
    Mac GPU precision per graph) + the informative sections (bucket rules, host steps, precision, fixtures, limits,
    gates = measured values only).
 8. results/ship_manifest.json: every ship file's bytes / sha256 / source / evidence paths, du, a curation scan of
    every text file for words that must not ship (absolute paths, internal process words, superlatives).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
K = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(K / "host"))
import d1_src as S  # noqa: E402
import d1_vision_host as V  # noqa: E402
import litert_run as R  # noqa: E402

SHIP = K / "ship"
MODEL = "d1-omni-600M"
TEXT_L = (128, 256, 512, 1024, 2048, 4096)
AUDIO_T = (501, 1001, 2001, 3001)
HOST_FILES = ("d1_prompt.py", "d1_host.py", "d1_vision_host.py", "d1_audio_host.py", "d1_omni.py", "tests.py",
              "verify.py", "requirements.txt", "vision_position_table.npy")
EXCLUDE = re.compile(r"(?i)coco|librispeech|asr_dummy|flux")
CURATION = re.compile(os.environ.get("CURATION_REGEX", r"(?!)"))   # words that must not ship: the publisher's list (default: none)
# Upstream licences of the public text records (round 13). cais/mmlu on the Hub holds no LICENSE file: its card
# declares `license: mit` and its Licensing Information links the original repository's LICENSE, read here at a pinned
# commit (cached under cache/, fetched once if missing) and checked by its git blob id. The kev repository's LICENSE
# is the clone at tag kev-1.0 (same blob id as the tag's tree).
MMLU_HUB = {"repo": "cais/mmlu", "revision": "c30699e8356da336a370243923dbaf21066bb9fe"}
MMLU_LICENSE = {"url": "https://raw.githubusercontent.com/hendrycks/test/4450500f923c49f1fb1dd3d99108a0bd9717b660/LICENSE",
                "repo": "github.com/hendrycks/test", "commit": "4450500f923c49f1fb1dd3d99108a0bd9717b660",
                "git_blob": "fe20e951e94daaae0c02c070bad000ed85d1ad19",
                "cache": "cache/r13/licences/hendrycks_test_LICENSE"}
KEV_LICENSE = {"path": "../kev_work/kev/LICENSE", "repo": "github.com/jaredpalmer/kev", "tag": "kev-1.0",
               "commit": "6b719c3c3f367295f6ef336f4f751cf5ff970abc",
               "git_blob": "e1ece45b775283628888a36bc75f56078cee9de0"}
LICENCE_FILES = {"semif": "LICENSE-SemIf-MIT.txt", "mmlu": "LICENSE-MMLU-MIT.txt", "kev": "LICENSE-kev-Apache-2.0.txt"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p):
    return str(Path(p).relative_to(K))


def jread(p):
    return json.loads((K / p).read_text())


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ------------------------------------------------------------------ 1. position table

def position_table():
    npy = K / "host/vision_position_table.npy"
    fresh = V.read_position_table(S.WEIGHTS)
    if npy.exists():
        t = V.load_position_table(npy)
        assert np.array_equal(t, fresh), "host/vision_position_table.npy differs from the checkpoint's table"
        wrote = False
    else:
        np.save(npy, fresh, allow_pickle=False)
        t, wrote = V.load_position_table(npy), True
    ref = jread("ref/records_ref.json")
    crops = bit = 0
    per = []
    for rec in ref["records"]:
        if rec["mode"] != "image":
            continue
        with np.load(K / f"ref/npz/{rec['id']}.npz") as z:
            shapes, pos = z["spatial_shapes"], z["pos_resized"]
        for i, (h, w) in enumerate(shapes.tolist()):
            ours = V.positions_padded(t, (h, w))
            eq = bool(np.array_equal(ours, pos[i]))
            crops, bit = crops + 1, bit + eq
            per.append({"record": rec["id"], "crop": i, "grid": [h, w], "bit_equal": eq,
                        "max_abs": float(np.abs(ours.astype(np.float64) - pos[i]).max())})
    r9 = {"available": False}
    f9, j9 = K / "device/g9_vt_pos.f32", K / "device/g9_vt.json"
    if f9.exists() and j9.exists():
        rows = json.loads(j9.read_text())["rows"]
        arr = np.fromfile(f9, np.float32).reshape(len(rows), 1024, 768)
        eqs = [bool(np.array_equal(V.positions_padded(t, tuple(r["grid"])), arr[r["index"]])) for r in rows]
        r9 = {"available": True, "file": rel(f9), "crops": len(rows), "bit_equal": sum(eqs)}
    doc = {"file": "host/vision_position_table.npy", "written_now": wrote, "dtype": str(t.dtype),
           "shape": list(t.shape), "sha256": sha(npy), "bytes": npy.stat().st_size,
           "equals_checkpoint": True, "checkpoint_tensor": V.POSITION_KEY, "weights": str(S.WEIGHTS.name),
           "vs_reference_pos_resized": {"crops": crops, "bit_equal": bit, "per_crop": per},
           "vs_round9_tower_input": r9}
    assert bit == crops and crops > 0, doc["vs_reference_pos_resized"]
    assert not r9["available"] or r9["bit_equal"] == r9["crops"], r9
    return doc


# ------------------------------------------------------------------ 2-3. graphs and links

def graph_sources():
    """-> [(ship name, out/ source, recorded sha256, record path + key, kind, bucket)]."""
    hand = jread("results/f16safe_handoff.json")["files"]
    draft = jread("results/contract_draft.json")
    out = []
    for L in TEXT_L:
        src = f"out/d1omni_decide_L{L}_f16safe_fp16.tflite"
        if f"L{L}_f16safe_fp16" in hand:
            rec, key = hand[f"L{L}_f16safe_fp16"]["sha256"], f"results/f16safe_handoff.json files.L{L}_f16safe_fp16"
        else:
            rec, key = jread(f"results/quant_L{L}_f16safe.json")["forms"]["fp16"]["sha256"], \
                f"results/quant_L{L}_f16safe.json forms.fp16"
        out.append((f"{MODEL}_decide_L{L}_fp16.tflite", src, rec, key, "decide", L))
    vt = jread("results/vision_quant_f16safe.json")["forms"]["tower"]
    assert vt["output"] == "out/d1omni_vision_tower_f16safe_fp16.tflite", vt["output"]
    out.append((f"{MODEL}_vision_tower_fp16.tflite", vt["output"], vt["sha256"],
                "results/vision_quant_f16safe.json forms.tower", "vision_tower", None))
    vg = draft["vision"]["graphs"]
    out.append((f"{MODEL}_projector_fp16.tflite", "out/d1omni_projector_fp16.tflite",
                vg["projector"]["files"]["fp16"]["sha256"],
                "results/contract_draft.json vision.graphs.projector.files.fp16", "projector", None))
    aq = jread("results/audio_quant_f16safe.json")["by_T"]
    for T in AUDIO_T:
        assert aq[str(T)]["output"] == f"out/d1omni_audio_T{T}_f16safe_fp16.tflite", aq[str(T)]["output"]
        out.append((f"{MODEL}_audio_T{T}_fp16.tflite", aq[str(T)]["output"], aq[str(T)]["sha256"],
                    f"results/audio_quant_f16safe.json by_T.{T}", "audio", T))
    return out


def previous_sources():
    """The source each ship name had in the previous manifest (results/ship_manifest.json), if any."""
    p = K / "results/ship_manifest.json"
    if not p.exists():
        return {}
    return {f["name"]: f.get("source") for f in json.loads(p.read_text())["files"] if f.get("source")}


def link_graphs():
    files, replaced = [], []
    prev = previous_sources()
    for name, src, recorded, key, kind, bucket in graph_sources():
        s = K / src
        sc = R.scan(s, with_sha=True)
        assert sc["sha256"] == recorded, (src, sc["sha256"], recorded)
        dst = SHIP / name
        if dst.exists() and not os.path.samefile(s, dst):
            old = prev.get(name)
            assert old and old != src and (K / old).exists() and os.path.samefile(K / old, dst), \
                f"{dst} exists and is neither a hard link of {src} nor of its previous source {old}"
            replaced.append({"name": name, "old_source": old, "old_sha256": sha(K / old), "new_source": src,
                             "new_sha256": sc["sha256"]})
            dst.unlink()                                  # the ship/ name only: the out/ file keeps its other link
        if not dst.exists():
            os.link(s, dst)
        sig = sc["signatures"]
        assert len(sig) == 1, (src, [x["key"] for x in sig])
        sig = sig[0]
        files.append({"name": name, "bytes": sc["bytes"], "sha256": sc["sha256"], "graph": kind, "bucket": bucket,
                      "signature": sig["key"], "operators": sc["operator_count"],
                      "inputs": [{"name": i["name"], "dtype": i["dtype"], "shape": i["shape"],
                                  "tensor_index": i["tensor_index"]}
                                 for i in sorted(sig["inputs"], key=lambda i: i["tensor_index"])],
                      "outputs": [{"name": o["name"], "dtype": o["dtype"], "shape": o["shape"]} for o in sig["outputs"]],
                      "_source": src, "_recorded_sha256_at": key, "_inode": os.stat(dst).st_ino,
                      "_links": os.stat(dst).st_nlink})
    return files, replaced


# ------------------------------------------------------------------ 4-5. small files

NOTICE = ("d1-omni-600M-LiteRT\n\n"
          "This repository holds a converted and modified Derivative Work of LiquidAI/d1-omni-600M (Hugging Face "
          "revision {rev}; weights model.safetensors sha256 {wsha}), licensed by Liquid AI, Inc. (the Licensor) under the "
          "LFM Open License v1.0, whose full text is in LICENSE (unchanged). Changes: the model was converted to LiteRT "
          "graphs (.tflite): a decision graph per input-length bucket, the vision tower and projector, and an audio "
          "encoder per clip-length bucket; fully connected weights are stored in fp16; {scaled} of the decision graph's "
          "{sites} normalisation layers, {v_scaled} of the vision tower's {v_sites} and {a_scaled} of the audio encoder's "
          "{a_sites} are rewritten in an fp16-safe form (input scaled by a power of two, epsilon scaled to "
          "match), which leaves float32 results unchanged; the image preprocessing, the position-embedding resize, the "
          "pixel unshuffle and the audio mel front end were re-implemented in host code (host/). host/d1_prompt.py is "
          "the provider's prompt.py, unchanged apart from a header comment; tokenizer.json is the provider's file, "
          "unchanged. The check set in fixtures/ carries its own sources' licences (fixtures/LICENSE-*.txt).\n")


def small_files(pos):
    out = []
    for name in ("tokenizer.json", "LICENSE"):
        src, dst = K / "hf_small" / name, SHIP / name
        shutil.copyfile(src, dst)
        assert sha(src) == sha(dst)
        out.append({"name": name, "bytes": dst.stat().st_size, "sha256": sha(dst), "_source": rel(src)})
    ref = jread("ref/records_ref.json")["model"]
    kt = jread("results/f16safe_k_table.json")["summary"]
    vk = jread("results/vision_f16safe_k_table.json")["summary"]
    ak = jread("results/audio_f16safe_k_table.json")["summary"]
    (SHIP / "NOTICE").write_text(NOTICE.format(rev=ref["revision"], wsha=ref["weights_sha256"], sites=kt["sites"],
                                               scaled=kt["sites_k_gt_0"], v_sites=vk["sites"],
                                               v_scaled=vk["sites_k_gt_0"], a_sites=ak["sites"],
                                               a_scaled=ak["sites_k_gt_0"]))
    out.append({"name": "NOTICE", "bytes": (SHIP / "NOTICE").stat().st_size, "sha256": sha(SHIP / "NOTICE"),
                "_source": "written by scripts/ship_build.py"})
    (SHIP / "host").mkdir(exist_ok=True)
    cache = SHIP / "host" / "__pycache__"           # bytecode a run from ship/ leaves behind: never shipped
    if cache.is_dir():
        shutil.rmtree(cache)
    for name in HOST_FILES:
        src, dst = K / "host" / name, SHIP / "host" / name
        shutil.copyfile(src, dst)
        out.append({"name": f"host/{name}", "bytes": dst.stat().st_size, "sha256": sha(dst), "_source": rel(src)})
    return out


# ------------------------------------------------------------------ 6. public fixtures

def provenance_line(r):
    """One public line per record, from its source and provenance (no local paths, no internal names)."""
    src, prov, media = r["source"], r.get("provenance") or {}, r.get("media") or {}
    if r["id"] == "card_text":
        return f"LiquidAI/d1-omni-600M model card (README.md lines {prov['lines'][0]}-{prov['lines'][1]}, revision " \
               f"{prov['revision'][:8]}): the text example, verbatim"
    if r["id"].startswith("card_batch_"):
        return f"LiquidAI/d1-omni-600M model card (README.md, How to use, revision {S.REV[:8]}): the system_one_batch " \
               f"example, ticket {int(r['id'][-2:])}"
    if src == "tv4":
        m = prov["_meta"]
        return f"github.com/jaredpalmer/kev tag {prov['tag']}, {prov['file']} line {prov['line']} ({m['source']}: " \
               f"{m['repo']} {m['split']} row {m['row']})"
    if src == "tv4s":
        return f"github.com/jaredpalmer/kev tag {prov['tag']}, {prov['file']} line {prov['line']} " \
               f"({prov['_meta']['source']}, a score question)"
    if src == "red_arm":
        return "tv4_000 with 'correctly' changed to 'incorrectly' in the instructions (the red arm: its answers must " \
               "move away from tv4_000's)"
    if src == "semif":
        return f"github.com/{prov['repo']} at {prov['pin']}, {prov['file']} line {prov['line']} " \
               f"({prov['family']}), MIT (fixtures/LICENSE-SemIf-MIT.txt), as a 3-way choice"
    if src == "own":
        return "written for this check set; invented names only"
    if src == "own_long_extended":
        return "written for this check set: a sorter log of about 3,400 state tokens; invented names only"
    if src == "img":
        return f"CC0 1.0 photograph from Wikimedia Commons ({media['page']}, by {media['author']}), long side " \
               f"resized to 384 px"
    if src == "own_image":
        s = media["source"]
        return f"CC0 1.0 photograph from Wikimedia Commons ({s['commons_page']}, by {s['artist']}), resized to " \
               f"{media['px'][0]}x{media['px'][1]} px"
    if src in ("aud", "own_audio"):
        voice = media.get("voice") or (media.get("synth") or {}).get("voice") or ""
        return f"speech synthesised with Kokoro-82M (Apache-2.0{', voice ' + voice if voice else ''}) from an " \
               f"English script written for this check set"
    raise ValueError(f"no provenance line for source {src} ({r['id']})")


SOURCE_LICENCES = {     # source -> (licence of the records, licence file in fixtures/ or None)
    "card": ("the model card's own examples (LiquidAI/d1-omni-600M README.md), under the model's LFM Open License "
             "v1.0 (LICENSE)", None),
    "tv4": ("MIT: MMLU (cais/mmlu, test split) as the kev repository's transfer-v4 suite quotes it", LICENCE_FILES["mmlu"]),
    "red_arm": ("MIT: tv4_000, an MMLU record, with one word changed", LICENCE_FILES["mmlu"]),
    "tv4s": ("Apache-2.0: generated by the transfer-v4 suite of github.com/jaredpalmer/kev (legacy_holdout)",
             LICENCE_FILES["kev"]),
    "semif": ("MIT: github.com/TheoLeeCJ/SemIf", LICENCE_FILES["semif"]),
    "own": ("written for this check set (invented names only)", None),
    "own_long_extended": ("written for this check set (invented names only)", None),
    "img": ("CC0 1.0: photographs from Wikimedia Commons (page and author in each record's provenance)", None),
    "own_image": ("CC0 1.0: photographs from Wikimedia Commons (page and author in each record's provenance)", None),
    "aud": ("speech synthesised with Kokoro-82M (Apache-2.0) from scripts written for this check set", None),
    "own_audio": ("speech synthesised with Kokoro-82M (Apache-2.0) from scripts written for this check set", None),
}


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fixture_licences(sources):
    """The upstream licence files of the public text records -> fixtures/LICENSE-*.txt (+ file entries)."""
    out = []
    if "semif" in sources:
        shutil.copyfile(K / "fixtures" / LICENCE_FILES["semif"], SHIP / "fixtures" / LICENCE_FILES["semif"])
        out.append((LICENCE_FILES["semif"], f"fixtures/{LICENCE_FILES['semif']}"))
    if sources & {"tv4", "red_arm"}:
        cache = K / MMLU_LICENSE["cache"]
        if not cache.exists():                         # read-only fetch of a content-addressed file, once
            import urllib.request

            cache.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(MMLU_LICENSE["url"], timeout=60) as r:
                cache.write_bytes(r.read())
        text = cache.read_bytes()
        assert git_blob(text) == MMLU_LICENSE["git_blob"], "the cached MMLU licence is not the pinned blob"
        head = (f"# MMLU (cais/mmlu): the MIT License of {MMLU_LICENSE['repo']} at commit {MMLU_LICENSE['commit']} "
                f"(https://github.com/hendrycks/test/blob/{MMLU_LICENSE['commit']}/LICENSE), which the Hugging Face "
                f"dataset {MMLU_HUB['repo']} (revision {MMLU_HUB['revision']}) declares (license: mit) and links in "
                f"its Licensing Information; it covers the tv4 records and the red arm.\n\n")
        (SHIP / "fixtures" / LICENCE_FILES["mmlu"]).write_bytes(head.encode() + text)
        out.append((LICENCE_FILES["mmlu"], f"{MMLU_LICENSE['cache']} (fetched from {MMLU_LICENSE['url']})"))
    if "tv4s" in sources:
        text = (K / KEV_LICENSE["path"]).resolve().read_bytes()
        assert git_blob(text) == KEV_LICENSE["git_blob"], "kev_work/kev/LICENSE is not the kev-1.0 blob"
        head = (f"# {KEV_LICENSE['repo']} at tag {KEV_LICENSE['tag']} (commit {KEV_LICENSE['commit']}), file "
                f"LICENSE: the licence of the repository whose transfer-v4 suite generated the tv4s records.\n")
        (SHIP / "fixtures" / LICENCE_FILES["kev"]).write_bytes(head.encode() + text)
        out.append((LICENCE_FILES["kev"], f"kev_work/kev/LICENSE (tag {KEV_LICENSE['tag']})"))
    return [{"name": f"fixtures/{n}", "bytes": (SHIP / "fixtures" / n).stat().st_size,
             "sha256": sha(SHIP / "fixtures" / n), "_source": s} for n, s in out]


def public_fixtures():
    reqs = jread("fixtures/requests.json")
    ref = jread("ref/records_ref.json")
    refs = {r["id"]: r for r in ref["records"]}
    out_docs, excluded, flagged = {}, {}, []
    (SHIP / "fixtures/media").mkdir(parents=True, exist_ok=True)
    for mode in ("text", "image", "audio"):
        out_docs[mode] = []
    media_files, semif = [], False
    for r in reqs["records"]:
        text = json.dumps({"media": r.get("media"), "provenance": r.get("provenance"), "note": r.get("note")})
        hit = sorted({m.group(0).lower() for m in EXCLUDE.finditer(text)})
        if hit and r["publishable"]:
            flagged.append({"id": r["id"], "words": hit})
        if not r["publishable"] or hit:
            why = "provenance: " + ", ".join(hit) if hit else f"publishable false ({r['source']})"
            excluded.setdefault(why, []).append(r["id"])
            continue
        rr = refs[r["id"]]
        mode = rr["mode"]
        names = [q["qid"] for q in rr["questions"]]
        dropped = [q for q in r["request"]["questions"] if q not in names]
        media = None
        if r["media"]:
            src = K / r["media"]["ref"]
            assert sha(src) == r["media"]["sha256"]
            dst = SHIP / "fixtures/media" / src.name
            shutil.copyfile(src, dst)
            media = {"kind": r["media"]["kind"], "file": f"media/{src.name}", "sha256": sha(dst),
                     "bytes": dst.stat().st_size}
            if r["media"].get("transcript"):
                media["transcript"] = r["media"]["transcript"]
            media_files.append({"name": f"fixtures/media/{src.name}", "bytes": dst.stat().st_size,
                                "sha256": sha(dst), "_source": rel(src), "_record": r["id"]})
        semif |= r["source"] == "semif"
        out_docs[mode].append({
            "id": r["id"], "source": r["source"], "provenance": provenance_line(r),
            "state": r["request"]["state"],
            "questions": {n: r["request"]["questions"][n] for n in names},
            **({"dropped_questions": {q: "no instructions: the provider's as_question() refuses it" for q in dropped}}
               if dropped else {}),
            "media": media,
            "expected": [{"name": q["qid"], "type": q["type"], "K": q["K"], "P": int(q["prefix"] or 0),
                          "ids": q["ids"], "markers": q["markers"], "probs": q["probs"]} for q in rr["questions"]],
            "usage_input_tokens": rr["usage_input_tokens"]})
    assert not flagged, f"publishable records whose provenance names an excluded source: {flagged}"
    sources = {r["source"] for recs in out_docs.values() for r in recs}
    assert "tv4x" not in sources and sources <= set(SOURCE_LICENCES), sorted(sources - set(SOURCE_LICENCES))
    licence_files = fixture_licences(sources)
    ref_desc = {"model": ref["model"]["hf_id"], "revision": ref["model"]["revision"],
                "how": "the provider's own code: AutoModel.from_pretrained(<snapshot>, trust_remote_code=True, "
                       "dtype=torch.float32), CPU, 12 threads; probabilities(state, questions, images, audio) once "
                       "per record",
                "versions": {"transformers": ref["model"].get("transformers") or ref["load"]["transformers"],
                             "torch": ref["load"]["torch"]},
                "image_reader": "transformers.image_utils.load_image (EXIF orientation applied, RGB); resize on the "
                                "float path (uint8 -> float32 -> bilinear antialias -> round -> clamp), as torchvision "
                                "0.24 does it",
                "audio_reader": "soundfile.read(path, dtype='int16'), 16 kHz mono",
                "values": "expected[].probs = each question's distribution in option order ([yes, no] for a noul); "
                          "ids / markers / P = the provider's encoded row (P = media prefix rows)"}
    counts, files = {}, []
    for mode, recs in out_docs.items():
        srcs = sorted({r["source"] for r in recs})
        doc = {"what": f"d1-omni-600M public check set, {mode} requests, with the provider's own float32 answers "
                       f"(host/verify.py reads it); `licences` gives each source's licence and its file",
               "mode": mode, "reference": ref_desc, "near_tie_gap": 0.02,
               "counts": {"records": len(recs), "questions": sum(len(r["expected"]) for r in recs),
                          "by_source": {s: sum(r["source"] == s for r in recs) for s in srcs}},
               "licences": [{"source": s, "records": sum(r["source"] == s for r in recs),
                             "licence": SOURCE_LICENCES[s][0],
                             "file": None if SOURCE_LICENCES[s][1] is None else f"fixtures/{SOURCE_LICENCES[s][1]}"}
                            for s in srcs],
               "records": recs}
        p = SHIP / "fixtures" / f"public_{mode}.json"
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
        counts[mode] = doc["counts"]
        files.append({"name": f"fixtures/public_{mode}.json", "bytes": p.stat().st_size, "sha256": sha(p),
                      "_source": "fixtures/requests.json + ref/records_ref.json"})
    provenance = "\n".join(f"{r['id']} {r['source']} {r['provenance']} {json.dumps(r['media'])}"
                           for recs in out_docs.values() for r in recs)      # what the public records say of their source
    return files + licence_files + media_files, {
                                 "counts": counts, "excluded": {k: {"records": len(v), "ids": v}
                                                                 for k, v in sorted(excluded.items())},
                                 "excluded_total": sum(len(v) for v in excluded.values()),
                                 "publishable_but_named_excluded_source": flagged,
                                 "public_sources": sorted(sources),
                                 "public_provenance_hits": {w: len(re.findall(w, provenance, re.I))
                                                            for w in ("coco", "librispeech", "flux", "tv4x")},
                                 "licences": [{"source": s, "licence": SOURCE_LICENCES[s][0],
                                               "file": SOURCE_LICENCES[s][1]} for s in sorted(sources)],
                                 "requests_sha256": sha(K / "fixtures/requests.json"),
                                 "reference_sha256": sha(K / "ref/records_ref.json")}


# ------------------------------------------------------------------ 7. contract

def summary_of(path, key="summary"):
    d = jread(path)
    s = d
    for k in key.split("."):
        s = s[int(k)] if isinstance(s, list) else s[k]
    return s


def parity_entry(path, file_label, backend, key="summary"):
    s = summary_of(path, key)
    return {"file": file_label, "backend": backend, "rows": s.get("rows", s.get("rows_compared")),
            "max_abs_dp": s["max_abs_dp"] if "max_abs_dp" in s else s.get("max_abs_dp_vs_oracle"),
            "mean_abs_dp": s["mean_abs_dp"] if "mean_abs_dp" in s else s.get("mean_abs_dp_vs_oracle"),
            "argmax_outside_near_tie": s.get("argmax_outside_near_tie") or
            (f"{s['argmax_equal_outside_near_tie']}/{s['rows_outside_near_tie']}"
             if "argmax_equal_outside_near_tie" in s else
             f"{s['argmax']['equal_outside_near_tie']}/{s['argmax']['rows_outside_near_tie']}"),
            "near_tie": s.get("near_tie") or (f"{s['argmax']['near_tie_equal']}/{s['argmax']['near_tie_rows']}"
                                              if "argmax" in s else f"-/{s.get('near_tie_rows', 0)}"),
            "nonfinite_rows": s["nonfinite_rows"] if isinstance(s["nonfinite_rows"], int) else len(s["nonfinite_rows"]),
            "bar_pass": s["bar_pass"], "_evidence": f"{path} {key}"}


def newest(*paths):
    """The first of these result files that exists (the newest round first)."""
    return next((p for p in paths if (K / p).exists()), None)


def gates():
    """Measured values only, per ship file x backend; the paths go to the manifest (`_evidence`). `file_measured`
    says which file a phone or desktop run used when it was not this repository's file itself."""
    out = {"bar": {"argmax": "equal on every row outside near-ties (reference top-2 gap <= 0.02)",
                   "max_abs_dp": 0.02, "mean_abs_dp": 0.002, "nonfinite": 0,
                   "reference": "the provider's code, float32, CPU (fixtures' expected probabilities)"},
           "host_verify": {}, "decide_mac": [], "decide_galaxy_s26": [], "vision": [], "audio": []}
    for acc, f in (("Mac CPU XNNPACK 4 threads", newest("results/verify_r13_cpu.json", "results/verify_r11_cpu.json")),
                   ("Mac Metal, each graph at its precision of `precision.mac_metal`",
                    newest("results/verify_r13_gpu.json")),
                   ("Mac Metal, fp32 precision", None if (K / "results/verify_r13_gpu.json").exists()
                    else newest("results/verify_r11_gpu.json"))):
        if f:
            d = jread(f)
            out["host_verify"][acc] = {m: {k: v[k] for k in ("rows", "max_abs_dp", "mean_abs_dp",
                                                            "argmax_outside_near_tie", "near_tie", "bar_pass")}
                                       | ({"red_arm_max_abs_dp": v["red_arm"]["max_abs_dp_vs_tv4_000_reference"]}
                                          if (v.get("red_arm") or {}).get("available") else {})
                                       | {"_evidence": f"{f} modes.{m}"}
                                       for m, v in d["modes"].items()}
    for L in TEXT_L:
        name = f"{MODEL}_decide_L{L}_fp16.tflite"
        for lab, back in (("cpu", "Mac CPU XNNPACK 8 threads"), ("gpu_fp32", "Mac Metal, fp32 precision"),
                          ("gpu_default", "Mac Metal, default precision (fp16 activations)")):
            p = f"results/litert_{lab}_parity_L{L}_f16safe_fp16.json"
            if (K / p).exists():
                out["decide_mac"].append({"L": L, **parity_entry(p, name, back)})
    pre = "the pre-rewrite fp16 file of the same bucket (bit-identical scores to this file on the Mac CPU and Metal at fp32)"
    cl32, cl16, npu = ("OpenCL GPU, FP32 precision", "OpenCL GPU, FP16_WITH_FP32_ACCUM",
                       "NPU (Qualcomm HTP, LiteRT JIT, NPU + CPU)")
    s26 = [
        (128, "results/s26_npu_parity_G1_gpu_fp32_f16s_L128_r10.json", "this", cl32),
        (128, "results/s26_npu_parity_G2_gpu_fp16acc32_f16s_L128_r10.json", "this", cl16),
        (128, "results/s26_npu_parity_N1_npu_f16s_L128_r10.json", "this", npu),
        (256, "results/s26_npu_parity_G3_gpu_fp32_f16s_L256_r10.json", "this", cl32),
        (256, "results/s26_npu_parity_G4_gpu_fp16acc32_f16s_L256_r10.json", "this", cl16),
        (256, "results/s26_npu_parity_N2_npu_f16s_L256_r10.json", "this", npu),
        (256, "results/s26_parity_G_cpu4_L256.json", pre, "CPU XNNPACK 4 threads"),
        (512, "results/s26_npu_parity_G1_gpu_fp32_f16s_L512_r12.json", "this", cl32),
        (512, "results/s26_npu_parity_G2_gpu_fp16acc32_f16s_L512_r12.json", "this", cl16),
        (1024, "results/s26_npu_parity_G5_gpu_fp32_f16s_L1024_r12.json", "this", cl32),
        (1024, "results/s26_npu_parity_G6_gpu_fp16acc32_f16s_L1024_r12.json", "this", cl16),
        (2048, "results/s26_npu_parity_G3_gpu_fp32_f16s_L2048_r12.json", "this", cl32),
        (2048, "results/s26_npu_parity_G4_gpu_fp16acc32_f16s_L2048_r12.json", "this", cl16),
        (4096, "results/s26_parity_G9_cpu4_L4096_r9.json", pre, "CPU XNNPACK 4 threads"),
    ]
    for L, p, which, back in s26:
        if (K / p).exists():
            e = parity_entry(p, f"{MODEL}_decide_L{L}_fp16.tflite" if which == "this" else pre, back)
            out["decide_galaxy_s26"].append({"L": L, "file_measured": "this file" if which == "this" else which,
                                             **{k: v for k, v in e.items() if k != "file"},
                                             **({"rows_note": "the 8 rows of L512 laid into L1024 (no fixture row "
                                                              "has 513-1,024 positions)"} if L == 1024 else {})})
    pre_tower = "the pre-rewrite vision tower fp16 file (bit-identical features to this file on the Mac CPU and Metal at fp32)"
    for p, back in (("results/vision_parity_cpu_f16safe_fp16.json", "Mac CPU XNNPACK 8 threads"),
                    ("results/vision_parity_gpu_fp32_f16safe_fp16.json", "Mac Metal, fp32 precision"),
                    ("results/vision_parity_gpu_default_f16safe_fp16.json",
                     "Mac Metal, default precision (fp16 activations), tower and projector")):
        out["vision"].append({**parity_entry(p, "tower + projector, then the decision graph (Mac CPU)", back,
                                             "e2e.summary"), "what": "end to end, 16 image rows",
                              "file_measured": "this file"})
    for p, back, what in (
            ("results/s26_npu_parity_V1_gpu_fp16acc32_f16s_vt_e2e_r12.json", f"Galaxy S26 {cl16}",
             "the tower on the phone, the projector and the decision graph on a Mac CPU"),
            ("results/s26_npu_parity_V2_gpu_fp32_f16s_vt_e2e_r12.json", f"Galaxy S26 {cl32}",
             "the tower on the phone, the projector and the decision graph on a Mac CPU"),
            ("results/s26_npu_parity_NV_npu_f16s_vt_e2e_r12.json", f"Galaxy S26 {npu}",
             "the tower on the phone, the projector and the decision graph on a Mac CPU"),
            ("results/s26_npu_parity_P1_gpu_fp16acc32_pj_e2e_r12.json", f"Galaxy S26 {cl16}",
             "the projector on the phone (the Mac CPU tower's input), the decision graph on a Mac CPU"),
            ("results/s26_npu_parity_NP_npu_pj_e2e_r12.json", f"Galaxy S26 {npu}",
             "the projector on the phone (the Mac CPU tower's input), the decision graph on a Mac CPU")):
        out["vision"].append({**parity_entry(p, what, back), "what": "end to end, 16 image rows",
                              "file_measured": "this file"})
    for p, back in (("results/s26_parity_V9_gpu_pj_e2e_r9.json", f"Galaxy S26 {cl32} (Kotlin CompiledModel)"),
                    ("results/s26_parity_V9_cpu4_pj_e2e_r9.json", "Galaxy S26 CPU XNNPACK 4 threads (Kotlin CompiledModel)")):
        out["vision"].append({**parity_entry(p, "tower + projector on the phone, the decision graph on a Mac CPU",
                                             back), "what": "end to end, 16 image rows", "file_measured": pre_tower})
    pre_audio = "the pre-rewrite audio fp16 files (bit-identical prefixes to these files on the Mac CPU and Metal at fp32)"
    for p, back in (("results/audio_parity_cpu_f16safe_fp16.json", "Mac CPU XNNPACK 8 threads"),
                    ("results/audio_parity_gpu_fp32_f16safe_fp16.json", "Mac Metal, fp32 precision"),
                    ("results/audio_parity_gpu_default_f16safe_fp16.json", "Mac Metal, default precision (fp16 activations)")):
        out["audio"].append({**parity_entry(p, "audio T1001 / T2001, then the decision graph (Mac CPU)", back,
                                            "e2e.host_mel"), "what": "end to end, 19 audio rows",
                             "file_measured": "this file"})
    for p, back, what in (
            ("results/s26_npu_parity_A1A6_gpu_fp16acc32_f16s_audio19_e2e_r12.json", f"Galaxy S26 {cl16}",
             "audio T1001 (6 clips) + T2001 (1 clip) on the phone, the decision graph on a Mac CPU"),
            ("results/s26_npu_parity_A2_gpu_fp32_f16s_T1001_e2e_r12.json", f"Galaxy S26 {cl32}",
             "audio T1001 (6 clips) on the phone, the decision graph on a Mac CPU"),
            ("results/s26_npu_parity_NA_npu_f16s_T1001_e2e_r12.json", f"Galaxy S26 {npu}",
             "audio T1001 (6 clips) on the phone, the decision graph on a Mac CPU")):
        out["audio"].append({**parity_entry(p, what, back), "what": f"end to end, {summary_of(p)['rows']} audio rows",
                             "file_measured": "this file"})
    for p, back in (("results/s26_parity_A9_gpu_au_e2e_r9.json", f"Galaxy S26 {cl32} (Kotlin CompiledModel)"),
                    ("results/s26_parity_A9_cpu4_au_e2e_r9.json", "Galaxy S26 CPU XNNPACK 4 threads (Kotlin CompiledModel)")):
        out["audio"].append({**parity_entry(p, "audio fp16 on the phone, the decision graph on a Mac CPU", back),
                             "what": "end to end, 19 audio rows", "file_measured": pre_audio})
    e13 = K / "results/audio_e2e_T501_T3001_r13.json"
    if e13.exists():
        d = json.loads(e13.read_text())
        for run in d["runs"]:
            for T, sec in run["buckets"].items():
                out["audio"].append({**parity_entry("results/audio_e2e_T501_T3001_r13.json",
                                                    f"audio T{T} ({sec['clip']}), then the decision graph (Mac CPU)",
                                                    run["backend"], f"runs.{run['index']}.buckets.{T}.summary"),
                                     "what": f"end to end, {sec['summary']['rows']} rows on a {sec['clip_seconds']} s "
                                             f"clip against the provider's float32 run of that clip",
                                     "file_measured": "this file"})
    return out


def ms_of(path, key="latency_phase.per_call_ms_write_run_read"):
    s = summary_of(path, key)
    return {"median": s["median"], "min": s["min"], "max": s["max"], "n": s["n"], "_evidence": f"{path} {key}"}


def pair_ratio(summary_path, a, b):
    """The same-hold FP16_WITH_FP32_ACCUM / FP32 time ratio of a pair of legs (None when the legs were not a pair)."""
    for p in jread(summary_path)["gpu_precision_pairs"]:
        if set(p["pair"]) == {a, b} and p.get("paired"):
            return p["ratio_fp16acc32_over_fp32"]
    return None


def precision_section(gs):
    """contract.json `precision`: the Mac GPU precision the host takes per graph, and the phone measurements behind
    the Android recommendation (each value read from its result file; `_evidence` stays in the manifest)."""
    by = lambda lst, **kw: next(e for e in lst if all(e.get(k) == v for k, v in kw.items()))   # noqa: E731
    decide_default = {f"L{e['L']}": e["max_abs_dp"] for e in gs["decide_mac"]
                      if e["backend"].startswith("Mac Metal, default")}
    decide_fp32 = {f"L{e['L']}": e["max_abs_dp"] for e in gs["decide_mac"] if e["backend"] == "Mac Metal, fp32 precision"}
    vis = {e["backend"]: e for e in gs["vision"] if e["backend"].startswith("Mac")}
    aud = {e["backend"]: e for e in gs["audio"] if e["backend"].startswith("Mac") and "19 audio rows" in e["what"]}
    t12 = jread("results/timing_mac_r12.json")["sets"]
    t12b = jread("results/timing_mac_r12b.json")["sets"]
    audio_ms = {f"T{T}": {"default": t12[f"audio_T{T}_f16safe_fp16_gpu_default"]["result"]["ms_write_run_read"]["median"],
                          "fp32": t12[f"audio_T{T}_f16safe_fp16_gpu_fp32"]["result"]["ms_write_run_read"]["median"]}
                for T in AUDIO_T}
    mac = {
        "graphs": {"decide": "fp32", "vision_tower": "fp32", "projector": "fp32", "audio": "default"},
        "meaning": "fp32 = GpuOptions(enforce_f32=True); default = the GPU delegate's default precision (fp16 "
                   "activations). host/d1_omni.py D1Omni(accelerator='gpu') opens each graph at this precision; the CPU "
                   "(XNNPACK) runs float32 throughout",
        "measured": {
            "decide": {"default_precision_max_abs_dp": decide_default, "fp32_precision_max_abs_dp": decide_fp32,
                       "verdict": "the default precision misses the bar at every bucket measured; fp32 precision passes"},
            "vision_tower": {"default_precision_e2e_max_abs_dp": vis["Mac Metal, default precision (fp16 activations), "
                                                                     "tower and projector"]["max_abs_dp"],
                             "fp32_precision_e2e_max_abs_dp": vis["Mac Metal, fp32 precision"]["max_abs_dp"],
                             "verdict": "the default precision misses the bar with the fp16-safe LayerNorms (16 image "
                                        "rows end to end); fp32 precision passes"},
            "projector": {"verdict": "runs with the tower at fp32 precision; not measured alone at the default precision "
                                     "on Metal"},
            "audio": {"default_precision_e2e_max_abs_dp": aud["Mac Metal, default precision (fp16 activations)"]["max_abs_dp"],
                      "default_precision_e2e_mean_abs_dp": aud["Mac Metal, default precision (fp16 activations)"]["mean_abs_dp"],
                      "fp32_precision_e2e_max_abs_dp": aud["Mac Metal, fp32 precision"]["max_abs_dp"],
                      "one_call_ms": audio_ms,
                      "ten_second_question_end_to_end_ms": {
                          "audio_default_text_fp32": t12b["pipeline_T1001_f16safe_default_text_fp32"]["result"]["ms_write_run_read"]["median"],
                          "audio_fp32_text_fp32": t12b["pipeline_T1001_current_fp32_text_fp32"]["result"]["ms_write_run_read"]["median"]},
                      "verdict": "the default precision passes the bar (19 audio rows end to end) and is faster, so the "
                                 "audio graph runs at it"}},
        "_evidence": "results/litert_gpu_{default,fp32}_parity_L*_f16safe_fp16.json summary; "
                     "results/vision_parity_gpu_{default,fp32}_f16safe_fp16.json e2e.summary; "
                     "results/audio_parity_gpu_{default,fp32}_f16safe_fp16.json e2e.host_mel; "
                     "results/timing_mac_r12.json sets.audio_T*_f16safe_fp16_gpu_{default,fp32}.result.ms_write_run_read; "
                     "results/timing_mac_r12b.json sets.pipeline_T1001_*.result.ms_write_run_read"}
    s10, s12, s12b = "results/s26_npu_summary_r10.json", "results/s26_npu_summary_r12.json", "results/s26_npu_summary_r12b.json"

    def leg(path_parity, path_timing, key="summary"):
        s = summary_of(path_parity, key)
        out = {"max_abs_dp": s["max_abs_dp"], "mean_abs_dp": s["mean_abs_dp"], "rows": s["rows"],
               "bar_pass": s["bar_pass"], "_evidence": f"{path_parity} {key}"}
        if path_timing:
            out["one_call_ms"] = ms_of(path_timing)
        return out

    R = "results/s26_npu_{}_{}.json".format
    decide = {
        "L128": {"FP16_WITH_FP32_ACCUM": leg(R("parity", "G2_gpu_fp16acc32_f16s_L128_r10"), R("timing", "G2_gpu_fp16acc32_f16s_L128_r10")),
                 "FP32": leg(R("parity", "G1_gpu_fp32_f16s_L128_r10"), R("timing", "G1_gpu_fp32_f16s_L128_r10")),
                 "time_ratio_fp16acc32_over_fp32": pair_ratio(s10, "G1_gpu_fp32_f16s_L128", "G2_gpu_fp16acc32_f16s_L128")},
        "L256": {"FP16_WITH_FP32_ACCUM": leg(R("parity", "G4_gpu_fp16acc32_f16s_L256_r10"), R("timing", "G4_gpu_fp16acc32_f16s_L256_r10")),
                 "FP32": leg(R("parity", "G3_gpu_fp32_f16s_L256_r10"), R("timing", "G3_gpu_fp32_f16s_L256_r10")),
                 "time_ratio_fp16acc32_over_fp32": pair_ratio(s10, "G3_gpu_fp32_f16s_L256", "G4_gpu_fp16acc32_f16s_L256")},
        "L512": {"FP16_WITH_FP32_ACCUM": leg(R("parity", "G2_gpu_fp16acc32_f16s_L512_r12"), R("timing", "G2_gpu_fp16acc32_f16s_L512_r12")),
                 "FP32": leg(R("parity", "G1_gpu_fp32_f16s_L512_r12"), R("timing", "G1_gpu_fp32_f16s_L512_r12")),
                 "time_ratio_fp16acc32_over_fp32": pair_ratio(s12, "G1_gpu_fp32_f16s_L512", "G2_gpu_fp16acc32_f16s_L512")},
        "L1024": {"FP16_WITH_FP32_ACCUM": leg(R("parity", "G6_gpu_fp16acc32_f16s_L1024_r12"), R("timing", "G6_gpu_fp16acc32_f16s_L1024_r12")),
                  "FP32": leg(R("parity", "G5_gpu_fp32_f16s_L1024_r12"), R("timing", "G5_gpu_fp32_f16s_L1024_r12")),
                  "time_ratio_fp16acc32_over_fp32": pair_ratio(s12, "G5_gpu_fp32_f16s_L1024", "G6_gpu_fp16acc32_f16s_L1024"),
                  "rows_note": "the 8 rows of L512 laid into L1024 (no fixture row has 513-1,024 positions)"},
        "L2048": {"FP16_WITH_FP32_ACCUM": leg(R("parity", "G4_gpu_fp16acc32_f16s_L2048_r12"), R("timing", "G4_gpu_fp16acc32_f16s_L2048_r12")),
                  "FP32": leg(R("parity", "G3_gpu_fp32_f16s_L2048_r12"), R("timing", "G3_gpu_fp32_f16s_L2048_r12")),
                  "time_ratio_fp16acc32_over_fp32": pair_ratio(s12, "G3_gpu_fp32_f16s_L2048", "G4_gpu_fp16acc32_f16s_L2048")},
        "L4096": {"GPU": "not used: compiling the L4096 graph for the GPU took the 12 GB phone's free memory under 2 GB "
                         "(see limits)",
                  "CPU_4_threads": {**leg("results/s26_parity_G9_cpu4_L4096_r9.json", None),
                                    "one_call_ms": ms_of("results/s26_timing_T9_cpu4_L4096_r9.json",
                                                         "sets.d_state_3k4.per_call_ms_write_run_read")}}}
    audio = {
        "T1001": {"FP16_WITH_FP32_ACCUM": {**leg(R("parity", "A1_gpu_fp16acc32_f16s_T1001_e2e_r12"), R("timing", "A1_gpu_fp16acc32_f16s_T1001_r12")),
                                           "what": "end to end, 18 audio rows"},
                  "FP32": {**leg(R("parity", "A2_gpu_fp32_f16s_T1001_e2e_r12"), R("timing", "A2_gpu_fp32_f16s_T1001_r12")),
                           "what": "end to end, 18 audio rows"},
                  "time_ratio_fp16acc32_over_fp32": pair_ratio(s12, "A1_gpu_fp16acc32_f16s_T1001", "A2_gpu_fp32_f16s_T1001")},
        "T2001": {"FP16_WITH_FP32_ACCUM": {**leg(R("parity", "A6_gpu_fp16acc32_f16s_T2001_e2e_r12b"), R("timing", "A6_gpu_fp16acc32_f16s_T2001_r12b")),
                                           "what": "end to end, 1 audio row"},
                  "FP32": {**leg(R("parity", "A7_gpu_fp32_T2001_e2e_r12b"), R("timing", "A7_gpu_fp32_T2001_r12b")),
                           "what": "end to end, 1 audio row", "file_measured": "the pre-rewrite T2001 file"},
                  "time_ratio_fp16acc32_over_fp32": pair_ratio(s12b, "A6_gpu_fp16acc32_f16s_T2001", "A7_gpu_fp32_T2001")},
        "T501": {"FP16_WITH_FP32_ACCUM": {"one_call_ms": ms_of(R("timing", "A5_gpu_fp16acc32_f16s_T501_r12b"))},
                 "FP32": {"one_call_ms": ms_of(R("timing", "A3_gpu_fp32_T501_r12")), "file_measured": "the pre-rewrite T501 file"},
                 "time_ratio_note": "two holds, not a same-hold pair"},
        "T3001": {"FP16_WITH_FP32_ACCUM": {"one_call_ms": ms_of(R("timing", "A8_gpu_fp16acc32_f16s_T3001_r12b"))},
                  "FP32": {"one_call_ms": ms_of(R("timing", "A4_gpu_fp32_T3001_r12")), "file_measured": "the pre-rewrite T3001 file"},
                  "time_ratio_note": "two holds, not a same-hold pair"},
        "19_audio_rows_FP16_WITH_FP32_ACCUM": leg(R("parity", "A1A6_gpu_fp16acc32_f16s_audio19_e2e_r12"), None)}
    e13 = K / "results/audio_e2e_T501_T3001_r13.json"
    if e13.exists():
        d = json.loads(e13.read_text())
        ph = next(r for r in d["runs"] if r["source"] == "phone")
        for T, sec in ph["buckets"].items():
            audio[f"T{T}"]["FP16_WITH_FP32_ACCUM"].update(
                {k: sec["summary"][k] for k in ("max_abs_dp", "mean_abs_dp", "rows", "bar_pass")}
                | {"what": f"end to end, {sec['summary']['rows']} rows on a {sec['clip_seconds']} s clip against the "
                           f"provider's float32 run of that clip",
                   "_evidence_e2e": f"results/audio_e2e_T501_T3001_r13.json runs.{ph['index']}.buckets.{T}.summary"})
    vision = {
        "vision_tower": {"FP16_WITH_FP32_ACCUM": {**leg(R("parity", "V1_gpu_fp16acc32_f16s_vt_e2e_r12"), R("timing", "V1_gpu_fp16acc32_f16s_vt_r12")),
                                                  "what": "end to end, 16 image rows"},
                         "FP32": {**leg(R("parity", "V2_gpu_fp32_f16s_vt_e2e_r12"), R("timing", "V2_gpu_fp32_f16s_vt_r12")),
                                  "what": "end to end, 16 image rows"},
                         "time_ratio_note": "same hold, not a pair: the FP32 leg began under a CPU clock cap"},
        "projector": {"FP16_WITH_FP32_ACCUM": {**leg(R("parity", "P1_gpu_fp16acc32_pj_e2e_r12"), R("timing", "P1_gpu_fp16acc32_pj_r12")),
                                               "what": "end to end, 16 image rows"}}}
    android_cl = {
        "recommended": "FP16_WITH_FP32_ACCUM for every graph and decision bucket up to L2048 (each passes the bar; "
                       "0.59-0.94 x the FP32 time in same-hold pairs); FP32 precision also passes everywhere it was run; "
                       "L4096 on the CPU (the GPU compile does not fit a 12 GB phone)",
        "measured_on": "Galaxy S26 (SM-S942Q, 12 GB, Android 16), LiteRT 2.2.0, OpenCL; one call = write the inputs + "
                       "run + read the output, median of 20 calls (10 at L2048 and T3001) after 5 warm-up calls",
        "decide": decide, "audio": audio, "vision": vision}
    npu_legs = {
        "decide_L128": (R("parity", "N1_npu_f16s_L128_r10"), R("timing", "N1_npu_f16s_L128_r10"), "summary"),
        "decide_L256": (R("parity", "N2_npu_f16s_L256_r10"), R("timing", "N2_npu_f16s_L256_r10"), "summary"),
        "vision_tower": (R("parity", "NV_npu_f16s_vt_e2e_r12"), R("timing", "NV_npu_f16s_vt_r12"), "summary"),
        "audio_T1001": (R("parity", "NA_npu_f16s_T1001_e2e_r12"), R("timing", "NA_npu_f16s_T1001_r12"), "summary"),
        "projector": (R("parity", "NP_npu_pj_e2e_r12"), R("timing", "NP_npu_pj_r12"), "summary")}
    npu_out = {}
    for k, (pp, pt, key) in npu_legs.items():
        e = leg(pp, pt, key)
        comp = jread(pt)["compile"]
        e["jit_compile_s"] = round(comp["runner_compile_ms"] / 1000.0, 1)
        e["verdict"] = "PASS" if e["bar_pass"] else "FAIL"
        npu_out[k] = e
    npu_out["decide_L128"]["cached_start_one_call_ms"] = ms_of(R("timing", "N4_npu_f16s_L128_cached_r10"))
    android_npu = {
        "what": "Qualcomm HTP through LiteRT 2.2.0's JIT compiler plugin (accelerators NPU + CPU), measured with "
                "LiteRT's C API on the Galaxy S26; every operator of each graph ran on the NPU (one partition). The "
                "NPU runtime libraries are not part of this repository",
        "measured_on_galaxy_s26": npu_out,
        "summary": "the decision graph passes at L128 only (L256 misses the mean bound); the projector passes; the "
                   "vision tower and the audio graph miss the bar on the NPU's fp16 arithmetic"}
    return {"mac_metal": mac, "android_opencl": android_cl, "android_npu": android_npu,
            "cpu": "XNNPACK (any thread count), float32 throughout",
            "constant_tensor_sharing": "do not set GpuOptions(constant_tensor_sharing=True): with the float32 "
                                       "embedding table the Metal delegate aborts the process at compile "
                                       "(EMBEDDING_LOOKUP: Unsupported external weights type: 1)"}


def strip_private(o):
    if isinstance(o, dict):
        return {k: strip_private(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, list):
        return [strip_private(v) for v in o]
    return o


def contract(graph_files, small, fixture_files, pos, fx):
    draft = jread("results/contract_draft.json")
    tok = next(f for f in small if f["name"] == "tokenizer.json")
    g = {"decide": {"by_L": {str(f["bucket"]): f["name"] for f in graph_files if f["graph"] == "decide"},
                    "signature": "decide_<L>", "inputs": draft["graph"]["inputs"], "output": draft["graph"]["output"],
                    "bind": "by name (the runtime lists the inputs by name; the tensor order is ids, prefix, media, pad, "
                            "keep_right, qtype_onehot)"},
         "vision_tower": {"file": next(f["name"] for f in graph_files if f["graph"] == "vision_tower"),
                          "signature": "vision_tower", "inputs": {k: draft["vision"]["inputs"][k]
                                                                  for k in ("pixels", "pos", "mask")},
                          "output": {"features": draft["vision"]["outputs"]["features"]}},
         "projector": {"file": next(f["name"] for f in graph_files if f["graph"] == "projector"),
                       "signature": "projector", "inputs": {"soft": draft["vision"]["inputs"]["soft"]},
                       "output": {"prefix": draft["vision"]["outputs"]["prefix"]}},
         "audio": {"by_T": {str(f["bucket"]): f["name"] for f in graph_files if f["graph"] == "audio"},
                   "signature": "audio_<T>", "inputs": draft["audio"]["graph"]["inputs"],
                   "output": draft["audio"]["graph"]["output"]}}
    cfg = json.loads((K / "hf_small/config.json").read_text())        # the provider's config.json
    modes = json.loads(json.dumps(draft["modes"]))
    assert (modes["text"]["max_len"], modes["image"]["max_len"], modes["audio"]["max_len"]) == \
        (cfg["max_length"], cfg["image_text_length"], cfg["audio_text_length"]), (modes, cfg)
    assert draft["temperatures"] == cfg["temperatures"]
    assert modes["image"]["noul_default"] == modes["audio"]["noul_default"] == S.YES_NO
    modes["media_max_len_rule"] = ("max_len = min(mode max_len, max_length - P); a request is refused when that is "
                                   "under 64 (P = the media prefix rows)")
    modes["state_rule"] = "a state None is '' for text and image requests and {} for an audio request"
    gs = gates()
    prec = precision_section(gs)
    samples = (K / "device/r5/G_gpu_fp32_L4096.samples.txt").read_text().splitlines()
    mem = [int(x.split()[1]) for x in samples if x.startswith("MemAvailable:")]
    guard = (K / "device/r5/G_gpu_fp32_L4096.MEMGUARD").read_text()
    l4096 = {"first_kB": mem[0], "min_kB": int(re.search(r"MemAvailable (\d+) kB", guard).group(1))}
    import d1_audio_host as A

    buckets = []
    for T in AUDIO_T:
        sec = (T - 1) / 100                     # T = n // 160 + 1 frames at a 10 ms hop (16 kHz): n = 16,000 x seconds
        frames, t_clip = A.frame_count(int(round(sec * A.SAMPLE_RATE)))
        assert t_clip == T and A.bucket_for(t_clip) == T, (T, frames, t_clip)
        buckets.append({"T": T, "clip_seconds_max": sec, "clip_samples_max": 160 * (T - 1) + 159,
                        "P_max": A.lengths(frames)[2], "T1_T2_T3": list(A.dims(T)), "file": g["audio"]["by_T"][str(T)]})
    g["audio"]["buckets"] = buckets
    g["audio"]["bucket_seconds_rule"] = ("a clip of n samples at 16 kHz has T = n // 160 + 1 frames (10 ms hop); it runs "
                                         "on the smallest T_b >= T: T501 holds clips up to 5 s, T1001 up to 10 s, "
                                         "T2001 up to 20 s, T3001 up to 30 s (clip_samples_max = 160 (T_b - 1) + 159); "
                                         "P_max = the prefix rows of a clip that fills the bucket")
    by_src = {}
    for m in fx["counts"].values():
        for src, n in m["by_source"].items():
            by_src[src] = by_src.get(src, 0) + n
    licences = [{"source": e["source"], "records": by_src.get(e["source"], 0), "licence": e["licence"],
                 "file": None if e["file"] is None else f"fixtures/{e['file']}"} for e in fx["licences"]]
    doc = {
        "what": "d1-omni-600M on LiteRT: the files of this repository and the contract between them and a host "
                "(host/d1_omni.py reads this file)",
        "contract_version": 2, "written": now(),
        "source_model": {"repo": "LiquidAI/d1-omni-600M", "revision": S.REV,
                         "weights_sha256": jread("ref/records_ref.json")["model"]["weights_sha256"],
                         "license": "LFM Open License v1.0 (LICENSE, unchanged); changes listed in NOTICE"},
        "files": [{k: v for k, v in f.items() if not k.startswith("_")} for f in graph_files]
        + [{"name": f["name"], "bytes": f["bytes"], "sha256": f["sha256"]} for f in small
           if f["name"] in ("tokenizer.json", "host/vision_position_table.npy")],
        "graphs": g,
        "tokenizer": {"file": "tokenizer.json", "sha256": tok["sha256"],
                      "library": "Hugging Face tokenizers (byte-level BPE), Tokenizer.from_file",
                      "add_special_tokens": False,
                      "bos": "prompt.encode() adds one <|startoftext|> (id 1) itself; the tokenizer's own post-processor "
                             "is not used"},
        "token_ids": draft["token_ids"], "token_roles": draft["token_roles"],
        "max_length": cfg["max_length"],
        "temperatures": draft["temperatures"],
        "temperature_rule": "text requests only: z / temperatures[temperature_key(q)] (else temperatures[q.type], else "
                            "1.0); temperature_key = '<type>:2' / ':3-5' / ':6-10' / ':11+' by option count",
        "modes": modes,
        "bucket_rules": {
            "text": "L = the smallest of 128 / 256 / 512 / 1024 / 2048 / 4096 with L >= P + n (n = encoded ids); "
                    "P + n > 4096 cannot run (the provider reads up to 16,384)",
            "audio": "T = n // 160 + 1 STFT frames of the clip after waveform() (cut to 30 s, padded to 0.5 s); the "
                     "smallest T_b of 501 / 1001 / 2001 / 3001 with T <= T_b; T1, T2, T3 = the graph's three "
                     "subsampled lengths ((t + 2 - 3) // 2 + 1 per stage from T_b); P = the valid rows "
                     "(the same rule from n // 160 frames)",
            "vision": "every crop is one tower call with 1024 patch slots (real patches first, mask 1); tiling (up to "
                      "10 tiles of 512 px + a thumbnail) and resizing are host steps (layout()); the projector takes "
                      "256 rows; P = sum over crops of (ph / 2)(pw / 2)"},
        "host_steps": {
            "text": [
                "q = as_question(question dict) (d1_prompt.py)",
                "ids, markers = encode(tok, state, q, max_len=16384, noul_default=None, audio=False) (d1_prompt.py)",
                "L = bucket_for(len(ids)) (d1_host.py)",
                "x = build_inputs(ids, None, L) + qtype_onehot(q) (d1_host.py)",
                "scores = decide_<L>(x)['scores'] (CompiledModelRunner)",
                "p = readout_f64(scores, 0, markers, q, calibrate=True, temperatures) (d1_host.py): scores at the "
                "markers, / temperature, softmax, a noul reversed to [yes, no]",
                "answer(q, p) (d1_prompt.py) -> the response entry"],
            "image": [
                "image = load_image(path) (d1_vision_host.py): Pillow, EXIF orientation, RGB",
                "crops = crops_of(image) (d1_vision_host.py): layout(), float-path resize, tiles then thumbnail",
                "per crop: to_patches(crop) -> pixels, mask, grid (d1_vision_host.py)",
                "per crop: pos = positions_padded(load_position_table(npy), grid) (d1_vision_host.py)",
                "per crop: features = vision_tower(pixels, pos, mask)['features'] (LiteRTGraph)",
                "per crop: soft = projector_input(pixel_unshuffle(features[:ph*pw], grid)) (d1_vision_host.py)",
                "per crop: rows = projector(soft)['prefix'][:(ph/2)(pw/2)] (LiteRTGraph)",
                "prefix = the crops' rows concatenated, images in order (image_prefix / D1Omni.image_prefix)",
                "ids, markers = encode(tok, state or '', q, max_len=min(896, 16384 - P), noul_default=YES_NO, "
                "audio=False)",
                "L = bucket_for(P + len(ids)); x = build_inputs(ids, prefix, L) + qtype_onehot(q)",
                "p = readout_f64(scores, P, markers, q, calibrate=False, ...) (no temperature); answer(q, p)"],
            "audio": [
                "samples = read_audio(path) (d1_audio_host.py): soundfile int16, 16 kHz mono",
                "x, info = prepare(samples) (d1_audio_host.py): waveform(), mel() (numpy float32), bucket_for(T), "
                "build_inputs() -> mel, mel_valid, v1, v2, v3",
                "prefix = prefix_rows(audio_<T_b>(x)['prefix'], info) (d1_audio_host.py): the first P rows",
                "ids, markers = encode(tok, state if state is not None else {}, q, max_len=min(15360, 16384 - P), "
                "noul_default=YES_NO, audio=True)",
                "L = bucket_for(P + len(ids)); x = build_inputs(ids, prefix, L) + qtype_onehot(q)",
                "p = readout_f64(scores, P, markers, q, calibrate=False, ...) (no temperature); answer(q, p)"]},
        "precision": strip_private(prec),
        "limits": [
            "the decision graphs stop at L4096: a row with P + n > 4,096 raises in the host (truncate_state=True cuts "
            "the state with encode()'s own rule instead); the provider reads up to 16,384 positions",
            f"Galaxy S26 (12 GB): compiling the L4096 decision graph for the OpenCL GPU took MemAvailable from "
            f"{l4096['first_kB']:,} KiB ({l4096['first_kB'] * 1024 / 1e9:.2f} GB) to {l4096['min_kB']:,} KiB "
            f"({l4096['min_kB'] * 1024 / 1e9:.2f} GB) and the run was stopped under a 2 GB guard; use the CPU for "
            "L4096 there",
            "on Mac Metal the decision graphs, the vision tower and the projector run at fp32 precision (the default "
            "precision moves their answers past the bar); the audio graph passes at the default precision and runs at "
            "it (precision.mac_metal)",
            "on the Galaxy S26 NPU (Qualcomm HTP) the decision graph passes at L128 only and the projector passes; the "
            "vision tower and the audio graph miss the bar there (precision.android_npu)",
            "one 16 kHz mono clip of up to 30 s per request (longer clips are cut, as the provider does); a request "
            "carries images or audio, not both"],
        "gates": strip_private(gs),
        "vision_position_table": {"file": pos["file"], "dtype": pos["dtype"], "shape": pos["shape"],
                                  "sha256": pos["sha256"],
                                  "source": f"checkpoint tensor {pos['checkpoint_tensor']} [256, 768] as [16, 16, 768]",
                                  "resize": "per crop grid (ph, pw): bilinear antialias, align_corners False, width "
                                            "then height in float32 with float64 multiply-add sums "
                                            "(d1_vision_host.resize_positions), rows in raster order; rows past ph * pw "
                                            "repeat row 0 (positions_padded)"},
        "check_set": {"files": [f["name"] for f in fixture_files if f["name"].endswith(".json")],
                      "run": "python host/verify.py --repo . [--gpu]"},
        "fixtures": {"what": "the public check set's sources and their licences (fixtures/public_*.json carry the "
                             "same per file)",
                     "licence_files": sorted(f["name"] for f in fixture_files if Path(f["name"]).name.startswith("LICENSE")),
                     "licences": licences,
                     "not_included": "the records from COCO, LibriSpeech and a FLUX-generated image, and the 140 "
                                     "further transfer-v4 records (emotion, tweet_eval, QNLI, PAWS and SciQ, whose "
                                     "licences are other / unknown / CC BY-NC, with the suite's own holdout records "
                                     "kept in the same set)"}}
    return doc


# ------------------------------------------------------------------ 8. manifest and curation

def curation_scan():
    hits = []
    for p in sorted(SHIP.rglob("*")):
        if p.is_dir() or p.suffix in (".tflite", ".npy", ".png", ".wav", ".flac") or p.name == "tokenizer.json":
            continue
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            for m in CURATION.finditer(line):
                hits.append({"file": str(p.relative_to(SHIP)), "line": i, "word": m.group(0),
                             "text": line.strip()[:160]})
    return hits


def main():
    t0 = time.time()
    SHIP.mkdir(exist_ok=True)
    pos = position_table()
    graph_files, replaced = link_graphs()
    small = small_files(pos)
    fixture_files, fx = public_fixtures()
    c = contract(graph_files, small, fixture_files, pos, fx)
    (SHIP / "contract.json").write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n")
    cfile = {"name": "contract.json", "bytes": (SHIP / "contract.json").stat().st_size,
             "sha256": sha(SHIP / "contract.json"), "_source": "written by scripts/ship_build.py"}
    every = graph_files + small + fixture_files + [cfile]
    on_disk = sorted(str(p.relative_to(SHIP)) for p in SHIP.rglob("*") if p.is_file())
    listed = sorted(f["name"] for f in every)
    du = subprocess.run(["du", "-sh", str(SHIP)], capture_output=True, text=True).stdout.split()[0]
    gate_files = {
        "decide_f16safe_mac": [f"results/litert_{lab}_parity_L{L}_f16safe_fp16.json" for L in TEXT_L
                               for lab in ("cpu", "gpu_fp32")],
        "decide_f16safe_vs_pre_rewrite_fp32": [f"results/litert_{lab}_parity_L{L}_f16safe_fp32.json"
                                               for L in (128, 256, 1024, 4096) for lab in ("cpu", "gpu_fp32")],
        "f16safe_build": ["results/f16safe_handoff.json"] + [f"results/{k}_L{L}_f16safe{s}.json" for L in (1024, 4096)
                                                             for k, s in (("opscan", "_fp32"), ("quant", ""))]
        + ["results/vision_quant_f16safe.json", "results/audio_quant_f16safe.json", "results/vision_f16safe_k_table.json",
           "results/audio_f16safe_k_table.json"],
        "s26": sorted({e["_evidence"].split()[0] for e in gates()["decide_galaxy_s26"]}),
        "vision_audio": sorted({e["_evidence"].split()[0] for k in ("vision", "audio") for e in gates()[k]}),
        "precision": c["precision"] and [x.strip() for x in precision_section(gates())["mac_metal"]["_evidence"].split(";")],
        "host": [f for f in ("results/host_tests_r13.json", "results/verify_r13_cpu.json", "results/verify_r13_gpu.json")
                 if (K / f).exists()]}
    manifest = {"step": "round 13: ship/ = the ship form (fp16-safe files of every graph), assembled by "
                        "scripts/ship_build.py", "written": now(),
                "replaced_links": replaced,
                "precision_with_evidence": precision_section(gates()),
                "seconds": round(time.time() - t0, 1), "ship_dir": rel(SHIP), "du": du,
                "files": [{"name": f["name"], "bytes": f["bytes"], "sha256": f["sha256"],
                           **{k[1:]: v for k, v in f.items() if k.startswith("_")}} for f in every],
                "files_on_disk_not_listed": sorted(set(on_disk) - set(listed)),
                "files_listed_not_on_disk": sorted(set(listed) - set(on_disk)),
                "hard_links": {f["name"]: {"source": f["_source"], "inode": f["_inode"], "links": f["_links"]}
                               for f in graph_files},
                "bytes_total": sum(f["bytes"] for f in every),
                "position_table": pos, "fixtures": fx, "gate_evidence": gate_files,
                "gates_with_evidence": gates(),
                "readme_present": (SHIP / "README.md").exists(),
                "npu_libs_present": any(p.suffix == ".so" or "qnn" in p.name.lower() or "htp" in p.name.lower()
                                        for p in SHIP.rglob("*")),
                "curation_hits": curation_scan()}
    (K / "results/ship_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"files": len(every), "bytes_total": manifest["bytes_total"], "du": du,
                      "not_listed": manifest["files_on_disk_not_listed"],
                      "missing": manifest["files_listed_not_on_disk"], "fixtures": fx["counts"],
                      "excluded_total": fx["excluded_total"], "excluded": {k: v["records"] for k, v in fx["excluded"].items()},
                      "position_table": {"sha256": pos["sha256"], "vs_reference": pos["vs_reference_pos_resized"]["bit_equal"],
                                         "crops": pos["vs_reference_pos_resized"]["crops"], "r9": pos["vs_round9_tower_input"]},
                      "readme_present": manifest["readme_present"], "npu_libs_present": manifest["npu_libs_present"],
                      "curation_hits": len(manifest["curation_hits"]), "seconds": manifest["seconds"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
