"""Step 8: the Hub staging directory (publish/<repo>/): graphs, act head, token table, tokenizer, config, hfmodels.json
(the hfmodels-android descriptor, family laya), conversion scripts, SHA256SUMS. The card (README.md) is written by hand
into card/README.md and copied here. Nothing is uploaded by this script.

    python stage.py [--repo litert-community/Phocinae-Largha-150M-LiteRT] [--default L384_int8emb]
"""
import argparse
import json
import shutil
from pathlib import Path

from common import HF, W, dump_json, sha256_file

WINDOWS = [256, 384, 512]
SHIP = {  # published file name -> source
    "phocinae_act_head_fp32.tflite": "exports/phocinae_act_head_fp32.tflite",
    "tables/tok_embeddings_fp16.bin": "exports/tables/tok_embeddings_fp16.bin",
    "tables/tok_embeddings.json": "exports/tables/tok_embeddings.json",
    "tokenizer/tokenizer.json": "hf/tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json": "hf/tokenizer/tokenizer_config.json",
    "rl_agent_config.json": "hf/rl_agent_config.json",
    "encoder/config.json": "hf/encoder/config.json",
    "LICENSE": "hf/LICENSE",
    "NOTICE": "hf/NOTICE",
}
for L in WINDOWS:
    SHIP[f"phocinae_largha_150m_L{L}_fp16fc_int8emb.tflite"] = f"exports/phocinae_L{L}_ids_v2_fp16fc_i8emb_shift.tflite"
    SHIP[f"phocinae_largha_150m_L{L}_fp16fc_embeds.tflite"] = f"exports/phocinae_L{L}_embeds_wfp16_shift.tflite"
CONVERSION = ["common.py", "make_fixtures.py", "oracle.py", "graph.py", "torch_parity.py", "export.py", "tables.py", "quantize.py",
              "quant_all.sh", "tflite_scan.py", "litert_parity.py", "server_replay.py", "mac_gpu_runs.sh", "mac_timing.sh",
              "device_rows.py", "device_compare.py", "device_run.sh", "hold_keeper.sh", "s26_batch.sh", "s26_timing.sh", "stage.py"]


def device_note(tag):
    """One verification note from a device parity file (the S26 gate), or None."""
    p = W / "device" / f"device_parity_{tag}.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())["summary"]
    t, n, o = s["test_non_near_tie"], s["test_near_tie"], s["overall"]
    tim = W / "device" / f"{tag}_n5_timing.json"          # the timing leg's raw app report (5 gate rows + the timing rows)
    ms = ""
    if tim.exists():
        rows = json.loads(tim.read_text()).get("timing") or []
        med = next((r for r in rows if r["row_len"] >= 160), rows[0] if rows else None)
        if med:
            ms = f"; {med['total_ms_median']:.1f} ms per question (timing leg on a rested phone: median of {med['n']} after {med.get('warmups', 5)} warm-ups, a {med['row_len']}-token row)"
    return dict(level="PUBLISHER_TESTED", result="PASS" if s["bar_pass"] else "FAIL",
                note=f"{t['argmax_equal']}/{t['rows']} non-near-tie test decisions identical to the publisher's fp32 PyTorch, {n['argmax_equal']}/{n['rows']} near ties, "
                     f"max |dp| {o['max_abs_dp']:.4f} over {o['rows']} rows (test + reversed-order + control){ms}; compile {s['compile_ms']/1000:.1f} s",
                device=f"{s['device']} (Galaxy S26)", os_build=s["build"], runtime=f"litert {s['litert']}", date="2026-10-10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="litert-community/Phocinae-Largha-150M-LiteRT")
    ap.add_argument("--default", default="L384_int8emb")
    a = ap.parse_args()
    out = W / "publish" / a.repo.split("/")[-1]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files = {}
    for name, src in SHIP.items():
        s = W / src
        d = out / name
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, d)
        files[name] = dict(bytes=d.stat().st_size, sha256=sha256_file(d))
    (out / "conversion").mkdir()
    for f in CONVERSION:
        shutil.copyfile(W / "scripts" / f, out / "conversion" / f)
    for f in ("REPRODUCE.md",):
        if (W / "publish_src" / f).exists():
            shutil.copyfile(W / "publish_src" / f, out / "conversion" / f)
    if (W / "card" / "README.md").exists():
        shutil.copyfile(W / "card" / "README.md", out / "README.md")

    def fid(name, id_, role):
        return dict(id=id_, role=role, path=name, bytes=files[name]["bytes"], sha256=files[name]["sha256"])

    common_files = [fid("phocinae_act_head_fp32.tflite", "act_head", "model"), fid("tokenizer/tokenizer.json", "tokenizer", "tokenizer"),
                    fid("tokenizer/tokenizer_config.json", "tokenizer_config", "tokenizer"), fid("rl_agent_config.json", "config", "processor_config")]
    variants = []
    for L in WINDOWS:
        for form in ("int8emb", "embeds"):
            main = f"phocinae_largha_150m_L{L}_fp16fc_{form}.tflite"
            vfiles = [fid(main, "main", "model")] + common_files + ([fid("tables/tok_embeddings_fp16.bin", "table", "model")] if form == "embeds" else [])
            hc = dict(family="laya", window=L, head_tokens=192, hidden=384, gpu_precision="fp16", cpu_threads=4, languages=["en", "zh"],
                      files=dict(main="main", act_head="act_head", tokenizer="tokenizer", tokenizer_config="tokenizer_config", config="config"))
            if form == "embeds":
                hc["files"]["table"] = "table"
                hc["table_dtype"] = "float16"
            file_ids = [f["id"] for f in vfiles]
            profiles = []
            src_stem = f"L{L}_ids_v2_fp16fc_i8emb_shift" if form == "int8emb" else f"L{L}_embeds_wfp16_shift_hostlookup"
            # the embeds (host-lookup) form runs whole on the HTP and is the fastest at L384 / L512: npu first; the int8-table
            # form leaves its EMBEDDING_LOOKUP on the CPU (2 partitions): gpu first. GPU precision = fp16 (the fastest that passes the bar).
            order = (("npu", 100, ["gpu", "cpu"]), ("gpu", 90, ["cpu"]), ("cpu", 50, [])) if form == "embeds" else (("gpu", 100, ["cpu"]), ("npu", 80, ["gpu", "cpu"]), ("cpu", 50, []))
            for backend, prio, fallback in order:
                note = device_note(f"phocinae_s26_{backend}_{'fp16' if backend == 'gpu' else 'fp32'}_{src_stem}")
                prof = dict(id=backend, priority=prio, files=file_ids, enabled_inputs=["text"], components=dict(inference=backend),
                            requirements=dict(min_android_api=31 if backend != "cpu" else 26, abis=["arm64-v8a"]), fallback_profiles=fallback,
                            default_selectable=True, verification=[note] if note else [])
                profiles.append(prof)
            variants.append(dict(id=f"L{L}_{form}", runtime="litert", handler=dict(id="litert.typed_decisions", abi=1),
                                 runtime_range=dict(min_inclusive="2.2.0", max_exclusive="3.0.0"), inputs=["text"], files=vfiles,
                                 default_profile=order[0][0], profiles=profiles, handler_config=hc))
    desc = dict(schema_version=1, model_id=a.repo, tasks=["decide"], default_variant=a.default,
                license=dict(id="apache-2.0", url=f"https://huggingface.co/{a.repo}/blob/main/LICENSE"), variants=variants)
    dump_json(out / "hfmodels.json", desc)
    files["hfmodels.json"] = dict(bytes=(out / "hfmodels.json").stat().st_size, sha256=sha256_file(out / "hfmodels.json"))
    sums = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            sums.append(f"{sha256_file(p)}  {p.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    manifest = dict(repo=a.repo, dir=str(out.relative_to(W)), files={str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=sha256_file(p))
                                                                   for p in sorted(out.rglob("*")) if p.is_file()})
    dump_json(W / "results" / "staging_manifest.json", manifest)
    total = sum(v["bytes"] for v in manifest["files"].values())
    print(json.dumps(dict(dir=manifest["dir"], files=len(manifest["files"]), bytes=total, variants=[v["id"] for v in variants]), indent=1))


if __name__ == "__main__":
    main()
