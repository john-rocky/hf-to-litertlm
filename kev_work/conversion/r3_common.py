"""Runtime helpers: process memory (macOS proc_pid_rusage v4 = phys_footprint, the number that counts GPU buffers on
unified memory), the dtype of a file's EMBEDDING_LOOKUP table (the GPU delegate takes an in-graph lookup only with an
int8 table), and the GPU-vs-CPU comparison of one file."""
import ctypes
import json
import mmap
import os
from pathlib import Path

import numpy as np

_RU_FIELDS = (
    "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
    "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
    "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
    "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten", "ri_cpu_time_qos_default",
    "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
    "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive", "ri_billed_system_time",
    "ri_serviced_system_time", "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
    "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time")


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in _RU_FIELDS]


def memory():
    """{phys_footprint, lifetime_max_phys_footprint, resident_size} in bytes for this process, plus ru_maxrss."""
    import resource
    doc = {"ru_maxrss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    try:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        info = _RUsageInfoV4()
        rc = lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info))
        if rc == 0:
            doc.update(phys_footprint=int(info.ri_phys_footprint),
                       lifetime_max_phys_footprint=int(info.ri_lifetime_max_phys_footprint),
                       resident_size=int(info.ri_resident_size))
        else:
            doc["proc_pid_rusage_rc"] = rc
    except Exception as e:  # informational
        doc["proc_pid_rusage_error"] = repr(e)
    return doc


def embedding_table_dtypes(path):
    """dtype of input 1 (the table) of every EMBEDDING_LOOKUP in the file (flatbuffers only, mmapped)."""
    from ai_edge_litert import schema_py_generated as schema
    type_names = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        model = schema.Model.GetRootAs(mm, 0)
        emb_codes = {i for i in range(model.OperatorCodesLength())
                     if max(model.OperatorCodes(i).BuiltinCode(), model.OperatorCodes(i).DeprecatedBuiltinCode())
                     == schema.BuiltinOperator.EMBEDDING_LOOKUP}
        out = []
        for gi in range(model.SubgraphsLength()):
            g = model.Subgraphs(gi)
            for oi in range(g.OperatorsLength()):
                op = g.Operators(oi)
                if op.OpcodeIndex() in emb_codes:
                    out.append(type_names.get(g.Tensors(int(op.InputsAsNumpy()[1])).Type()))
        mm.close()
    return out


def gpu_vs_cpu(gpu_rows, cpu_rows_path, gpu_hsel, cpu_hsel_path):
    """Same file, GPU vs CPU: |dp| over all options (finite questions only), h_sel max |diff|, argmax agreement.
    gpu_rows = per-row dicts with key / probs / argmax_key / finite_h_sel; gpu_hsel = {key: h_sel}."""
    cpu_rows_path, cpu_hsel_path = Path(cpu_rows_path), Path(cpu_hsel_path)
    if not cpu_rows_path.exists():
        return {"available": False, "reason": f"missing {cpu_rows_path.name}"}
    cpu = {r["key"]: r for r in json.loads(cpu_rows_path.read_text())["rows"]}
    hc = np.load(cpu_hsel_path) if cpu_hsel_path.exists() else None
    dps, qmax, hmax, agree, n, missing, nonfinite = [], [], [], 0, 0, [], 0
    per_q = []
    for r in gpu_rows:
        c = cpu.get(r["key"])
        if c is None:
            missing.append(r["key"])
            continue
        if not r.get("finite_h_sel", True):
            nonfinite += 1
            continue
        n += 1
        dp = np.abs(np.asarray(r["probs"], np.float64) - np.asarray(c["probs"], np.float64))
        dps.append(dp)
        qmax.append(float(dp.max()))
        agree += int(r["argmax_key"] == c["argmax_key"])
        h = None
        if hc is not None and r["key"] in hc.files and r["key"] in gpu_hsel:
            h = float(np.abs(gpu_hsel[r["key"]].astype(np.float64) - hc[r["key"]].astype(np.float64)).max())
            hmax.append(h)
        per_q.append({"key": r["key"], "max_abs_dp_vs_cpu": float(dp.max()), "h_sel_max_abs_vs_cpu": h})
    if not dps:
        return {"available": False, "reason": "no comparable questions", "nonfinite_gpu_questions": nonfinite}
    allp = np.concatenate(dps)
    top = sorted(per_q, key=lambda x: -x["max_abs_dp_vs_cpu"])[:10]
    return {"available": True, "cpu_rows_file": cpu_rows_path.name,
            "cpu_hsel_file": cpu_hsel_path.name if hc is not None else None, "questions": n,
            "argmax_agree": agree, "max_abs_dp": float(max(qmax)), "mean_abs_dp_all_options": float(allp.mean()),
            "p95_abs_dp_all_options": float(np.percentile(allp, 95)), "options": int(allp.size),
            "h_sel_max_abs": float(max(hmax)) if hmax else None, "h_sel_questions": len(hmax),
            "nonfinite_gpu_questions_excluded": nonfinite, "missing_in_cpu": missing, "top10_by_dp": top}
