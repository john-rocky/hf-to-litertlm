"""Rewrites the `graph` entry of a repository's head contract (head/kev_<model>_pointer_head.json) from the graph files
that are in the repository: r16_head_json.py with one more entry, graph.npu (--npu-L). Every other key keeps its value
and its place; the head safetensors is not read or touched.

    python r18_head_json.py --repo <repository root> --model 0.8b --npu-L 64,128,256 --out <path of the new json>

graph.inputs / graph.output   the serving_default signature of the row-prefill files (*_rowprefill_L<L>_fp16fc_i8emb.tflite),
                              read from the files (every file must agree)
graph.L                       the row lengths of those files
graph.shared_state            the shared-state pairs present (*_sharedstate_Ls<Ls>_Lq<Lq>_fp16fc_i8emb.tflite): their
                              (Ls, Lq) and the two signatures, read from the files
graph.gpu_precision           the GPU precision that keeps the parity bar for the model (GPU_PRECISION below)
graph.npu                     (--npu-L) the row lengths whose files passed on the NPU, and how they ran there (NPU below)
The signatures are read from the flatbuffer (SignatureDefs and tensor shapes) without running the model. The output is
written as json.dumps(..., indent=1, ensure_ascii=False), the layout of the earlier file. Never overwrites."""
import argparse
import json
import mmap
import re
from pathlib import Path

GPU_PRECISION = {
    "0.8b": "float32 activations, or float16 storage with float32 accumulation (FP16_WITH_FP32_ACCUM in the Kotlin / C "
            "API); plain float16 activations miss the parity bar",
    "4b": "float32 activations; plain float16 activations miss the parity bar",
}
HEAD_JSON = {"0.8b": "head/kev_0.8b_pointer_head.json", "4b": "head/kev_4b_pointer_head.json"}
NPU = {"accelerators": "NPU + CPU (LiteRT kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu): the int8 embedding lookup "
                       "runs on the CPU, the other operators on the NPU",
       "checked_on": "the Qualcomm HTP of a Galaxy S26 (SM-S942Q) through the LiteRT 2.2.0 C API, compiled on the phone by "
                     "the Qualcomm JIT compiler plugin (QAIRT 2.47)"}


def signatures(path):
    """{signature key: {"inputs": {name: (dtype, shape)}, "outputs": {name: (dtype, shape)}}} of a .tflite file."""
    from ai_edge_litert import schema_py_generated as schema
    types = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    out = {}
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        model = schema.Model.GetRootAs(mm, 0)
        sd = g = None
        for i in range(model.SignatureDefsLength()):
            sd = model.SignatureDefs(i)
            g = model.Subgraphs(sd.SubgraphIndex())

            def side(n, get):
                res = {}
                for j in range(n):
                    t = get(j)
                    tensor = g.Tensors(t.TensorIndex())
                    res[t.Name().decode()] = (types[tensor.Type()].lower(), tensor.ShapeAsNumpy().tolist())
                return res

            out[sd.SignatureKey().decode()] = {"inputs": side(sd.InputsLength(), sd.Inputs),
                                               "outputs": side(sd.OutputsLength(), sd.Outputs)}
        del model, sd, g
        mm.close()
    return out


def fmt(dtype, shape, name, lengths):
    """'float32 [1,L,1024]': the sequence axis of the tensor `name` (axis 1 of ids / valid / state_valid / hidden, axis 2
    of the attention k_<l> / v_<l>) is written as its length's name; lengths = {"ids": "L"} style map of the axis
    names to use for that tensor (the Gated DeltaNet states have no sequence axis)."""
    dims = [str(d) for d in shape]
    axis = 2 if re.fullmatch(r"[kv]_\d+", name) else (1 if name in ("ids", "valid", "state_valid", "hidden") else None)
    if axis is not None:
        dims[axis] = lengths[name if name in lengths else "state"]
    return f"{dtype} [{','.join(dims)}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="repository root: the graph files and head/")
    ap.add_argument("--model", choices=sorted(HEAD_JSON), default="0.8b")
    ap.add_argument("--src", default="", help="the earlier head json (default: <repo>/head/kev_<model>_pointer_head.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--npu-L", default="", help="comma list of the row lengths whose files passed on the NPU (graph.npu)")
    a = ap.parse_args()
    repo, out = Path(a.repo), Path(a.out)
    assert not out.exists(), f"refusing to overwrite {out}"
    src = Path(a.src) if a.src else repo / HEAD_JSON[a.model]
    doc = json.loads(src.read_text(encoding="utf-8"))
    hidden = doc["base"]["hidden_size"]
    rows = {}
    for p in sorted(repo.glob("*_rowprefill_L*_fp16fc_i8emb.tflite")):
        L = int(re.search(r"_L(\d+)_", p.name).group(1))
        sig = signatures(p)
        assert list(sig) == ["serving_default"], (p.name, list(sig))
        s = sig["serving_default"]
        names = {"ids": "L", "valid": "L", "hidden": "L"}
        assert all(v[1][1] == L for v in s["inputs"].values()), (p.name, s)
        rows[L] = {"inputs": {k: fmt(*v, k, names) for k, v in s["inputs"].items()},
                   "output": {k: fmt(*v, k, names) for k, v in s["outputs"].items()}}
        assert list(s["outputs"]) == ["hidden"] and s["outputs"]["hidden"][1] == [1, L, hidden], (p.name, s)
    assert rows, f"no row-prefill file in {repo}"
    first = rows[min(rows)]
    assert all(r == first for r in rows.values()), "the row-prefill files disagree on their signature"
    pairs, pair_sigs = [], []
    for p in sorted(repo.glob("*_sharedstate_Ls*_Lq*_fp16fc_i8emb.tflite"),
                    key=lambda p: tuple(int(x) for x in re.search(r"_Ls(\d+)_Lq(\d+)_", p.name).groups())):
        Ls, Lq = (int(x) for x in re.search(r"_Ls(\d+)_Lq(\d+)_", p.name).groups())
        sig = signatures(p)
        state_key, question_key = f"state_prefill_{Ls}", f"question_step_{Ls}_{Lq}"
        assert sorted(sig) == sorted([state_key, question_key]), (p.name, list(sig))
        st, qs = sig[state_key], sig[question_key]
        s_names = {"ids": "Ls", "valid": "Ls", "state": "Ls"}
        q_names = {"ids": "Lq", "valid": "Lq", "hidden": "Lq", "state_valid": "Ls", "state": "Ls"}
        assert all(v[1][1] == Ls for v in st["inputs"].values()), p.name
        assert qs["inputs"]["ids"][1] == [1, Lq] and qs["inputs"]["state_valid"][1] == [1, Ls], p.name
        assert all(v[1][2] == Ls for k, v in st["outputs"].items() if re.fullmatch(r"[kv]_\d+", k)), p.name
        state_out = {k: fmt(*v, k, s_names) for k, v in st["outputs"].items()}
        assert set(qs["inputs"]) == {"ids", "valid", "state_valid"} | set(st["outputs"]), p.name
        assert all(qs["inputs"][k] == st["outputs"][k] for k in st["outputs"]), p.name
        assert list(qs["outputs"]) == ["hidden"] and qs["outputs"]["hidden"][1] == [1, Lq, hidden], p.name
        pairs.append({"Ls": Ls, "Lq": Lq})
        pair_sigs.append({"state_prefill_<Ls>": {"inputs": {k: fmt(*v, k, s_names) for k, v in st["inputs"].items()},
                                                 "outputs": state_out},
                          "question_step_<Ls>_<Lq>": {"inputs": {k: fmt(*v, k, q_names) for k, v in qs["inputs"].items()
                                                                 if k in ("ids", "valid", "state_valid")},
                                                      "output": {k: fmt(*v, k, q_names) for k, v in qs["outputs"].items()}}})
    if pair_sigs:
        assert all(s == pair_sigs[0] for s in pair_sigs), "the pairs disagree on their signatures"
    graph = {"inputs": first["inputs"], "output": {k: v + " (after final RMSNorm)" for k, v in first["output"].items()},
             "L": sorted(rows)}
    if pairs:
        sp = pair_sigs[0]
        groups = {}
        for name, shape in sp["state_prefill_<Ls>"]["outputs"].items():
            kind, layer = name.rsplit("_", 1)
            groups.setdefault((kind, shape), []).append(int(layer))
        graph["shared_state"] = {
            "pairs": pairs,
            "state_prefill_<Ls>": {"inputs": sp["state_prefill_<Ls>"]["inputs"],
                                   "outputs": {f"{kind}_<l>": f"{shape} for l in {sorted(layers)}"
                                               for (kind, shape), layers in groups.items()}},
            "question_step_<Ls>_<Lq>": {"inputs": {**sp["question_step_<Ls>_<Lq>"]["inputs"],
                                                   "<state>": "the state_prefill_<Ls> outputs of the same file, unchanged"},
                                        "output": {k: v + " (after final RMSNorm)"
                                                   for k, v in sp["question_step_<Ls>_<Lq>"]["output"].items()}},
            "positions": "the question's tokens continue after the state: n .. n+Lq-1 with n = sum(state_valid)",
            "readout": "decide = the question's last real token, each option's 248050; indices relative to the question start"}
    graph["gpu_precision"] = GPU_PRECISION[a.model]
    if a.npu_L:
        nL = sorted(int(x) for x in a.npu_L.split(","))
        assert all(L in rows for L in nL), (nL, sorted(rows))
        graph["npu"] = {"L": nL, **NPU}
    old = doc["graph"]
    doc["graph"] = graph
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"src": str(src), "out": str(out), "graph_before": old, "graph_after": graph}, indent=1,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
