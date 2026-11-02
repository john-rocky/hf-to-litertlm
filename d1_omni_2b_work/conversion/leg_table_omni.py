"""Every leg of a phone run's pulled folder as one table row: leg_table.py's row (status, backend, rows, engine build,
call times, the sampler's memory, delegation, the leg check's verdict and its distance from the Mac Metal fp32 run)
plus what the test app's later version adds, and the leg's end in words:
- the end: "DONE" | "app's memory guard" (the app's own guard stopped it: the report's status APPGUARD, with the
  reason, the stage, MemAvailable and the phone's uptime at the stop) | "MemoryLimiter -> kill" (the app cgroup's
  memory.high: `MemoryLimiter: onLimitExceeded: pid=<the leg's app pid>` and a kill line for that pid in the leg's
  logcat of all buffers; also when the leg check said KILLED after a DONE report) | "chain's memory guard"
  (MEMGUARD_STOPPED) | "lmkd" (LMKD_KILLED) | anything else as the run wrote it;
- the memory the app cgroup counts: the peak of RssAnon + VmSwap (the app's guard summary, else the app's MEM lines in
  the leg's logcat: a leg the MemoryLimiter killed has no report but has its MEM lines), the VmHWM peak, the lowest
  MemAvailable (the app's 0.25 s reads, else the run's 2 s sampler), the app cgroup's memory.high as the app read it;
- the app's options and their readings: constant tensor sharing, the GPU program cache dir (files before / after the
  compile), the touch calls (each signature's first call and the memory after it).
The leg checks it reads (--results-dir) are leg_check_omni.py's; the reports and logs come from the phones' test app,
which is not published.

  $EXPORT conversion/leg_table_omni.py <round dir> [--prefix ...] --results-dir results/device [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from leg_table import fmt, leg_row, read  # noqa: E402

MEM_RE = re.compile(r"D1OMNI_GATE MEM t_ms=(\d+) stage=(\S+) avail_kb=(-?\d+) rss_kb=(-?\d+) anon_kb=(-?\d+) "
                    r"file_kb=(-?\d+) shmem_kb=(-?\d+) swap_kb=(-?\d+) hwm_kb=(-?\d+) cg_current_b=(-?\d+)")


def app_pid(R, tag):
    m = re.search(r"result=\S+ status=\S+.*? pid=(\d+)", read(os.path.join(R, tag + ".run.txt")))
    if m:
        return m.group(1)
    m = re.search(r"^pid: (\d+)", read(os.path.join(R, tag + ".running.txt")), re.M)
    return m.group(1) if m else None


def table_row(R, tag, results_dir):
    row = leg_row(R, tag, results_dir)
    rep = {}
    try:
        rep = json.load(open(os.path.join(R, tag + ".json")))
    except Exception:
        pass
    pid = app_pid(R, tag)
    allb = read(os.path.join(R, tag + ".logcat_pkg_all_buffers.txt"))
    limit = [ln.strip() for ln in allb.splitlines() if pid and f"MemoryLimiter: onLimitExceeded: pid={pid} " in ln]
    kill = [ln.strip() for ln in allb.splitlines() if pid and (re.search(rf"killinfo: \[{pid},", ln)
                                                                or re.search(rf"Killing {pid}:", ln)
                                                                or re.search(rf"\(pid {pid}\) has died", ln))]
    mem = [m.groups() for m in (MEM_RE.search(ln) for ln in (read(os.path.join(R, tag + ".logcat.txt")) + allb).splitlines()) if m]
    seen, mem_u = set(), []
    for g in mem:
        if g[0] not in seen:
            seen.add(g[0])
            mem_u.append(g)
    anon_log = max((int(g[4]) + int(g[7]) for g in mem_u if int(g[4]) >= 0 and int(g[7]) >= 0), default=None)
    anon_log_stage = next((g[1] for g in mem_u if int(g[4]) >= 0 and int(g[7]) >= 0 and int(g[4]) + int(g[7]) == anon_log), None)
    hwm_log = max((int(g[8]) for g in mem_u), default=None)
    avail_log = min((int(g[2]) for g in mem_u if int(g[2]) >= 0), default=None)
    ag = rep.get("app_guard") or {}
    stop = rep.get("app_guard_stop") or {}
    cg = (ag.get("cgroup") or {})
    st = row["status"]
    if (rep.get("status") == "APPGUARD") or st == "APPGUARD":
        end = (f"app's memory guard: {stop.get('reason')} (stage {stop.get('stage')}, "
               f"{fmt((stop.get('t_ms_since_start') or 0) / 1000)} s after the start, uptime {stop.get('uptime_s')} s)")
    elif (limit or kill) and (st != "DONE" or row.get("verdict") == "KILLED"):
        end = f"MemoryLimiter → kill ({len(limit)} memory.high lines, {len(kill)} kill lines for pid {pid}; status {st})"
    elif st == "MEMGUARD_STOPPED":
        end = "chain's memory guard (MemAvailable below guard=, force-stop)"
    elif st == "LMKD_KILLED":
        end = "lmkd (a lowmemorykiller line naming the package)"
    else:
        end = st
    touch = [{"signature": t.get("signature"), "ms": t.get("write_run_read_ms"), "nonfinite": t.get("nonfinite"),
              "VmHWM_kb": (t.get("memory_after") or {}).get("VmHWM_kb"),
              "RssAnon_kb": (t.get("memory_after") or {}).get("RssAnon_kb"),
              "VmSwap_kb": (t.get("memory_after") or {}).get("VmSwap_kb"),
              "mem_available_kb_after": t.get("mem_available_kb_after")} for t in (rep.get("touch") or [])]
    # the GPU program cache per process: a load = "Initialized InferenceContext from serialized data", a refused
    # one = DATA_LOSS; "Initializing ... from serialized data" prints even with an empty cache, so it is not a hit
    lcp = read(os.path.join(R, tag + ".logcat.txt"))
    cache_hits = len(re.findall(r"Initialized InferenceContext from serialized data", lcp))
    cache_data_loss = len(re.findall(r"DATA_LOSS", lcp))
    row.update({
        "gpu_cache_loads": cache_hits, "gpu_cache_data_loss": cache_data_loss,
        "end": end, "app_pid": pid, "memory_high_lines": len(limit), "kill_lines": len(kill),
        "app_version": rep.get("app_version"), "gpu_constant_tensor_sharing": rep.get("gpu_constant_tensor_sharing"),
        "gpu_cache_dir": rep.get("gpu_cache_dir"), "gpu_cache_before_compile": rep.get("gpu_cache_before_compile"),
        "gpu_cache_after_compile": rep.get("gpu_cache_after_compile"), "mem_guard_kb": rep.get("mem_guard_kb"),
        "app_guard_stop": stop or None,
        "anon_plus_swap_max_kb": ag.get("max_anon_plus_swap_kb") if ag.get("max_anon_plus_swap_kb") not in (None, -1) else anon_log,
        "anon_plus_swap_max_stage": ag.get("max_anon_plus_swap_stage") if ag.get("max_anon_plus_swap_kb") not in (None, -1) else anon_log_stage,
        "anon_from": "app_guard" if ag.get("max_anon_plus_swap_kb") not in (None, -1) else ("MEM lines" if anon_log is not None else None),
        "mem_lines": len(mem_u), "VmHWM_max_log_kb": hwm_log, "MemAvailable_min_app_kb": ag.get("min_mem_available_kb", avail_log),
        "MemAvailable_min_app_stage": ag.get("min_mem_available_stage"),
        "cgroup_memory_high": (cg.get("files") or {}).get("memory.high") if isinstance(cg.get("files"), dict) else cg.get("memory.high"),
        "cgroup": cg or None, "touch": touch})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round_dir")
    ap.add_argument("--prefix", action="append", default=[])
    ap.add_argument("--results-dir", default=os.path.join(WORK, "results/device"))
    ap.add_argument("--json")
    a = ap.parse_args()
    R = os.path.abspath(a.round_dir)
    tags = sorted({os.path.basename(p)[:-len(".leg_status")] for p in glob.glob(os.path.join(R, "*.leg_status"))}
                  | {os.path.basename(p)[:-len(".attempted")] for p in glob.glob(os.path.join(R, "*.attempted"))})
    if a.prefix:
        tags = [t for t in tags if any(t.startswith(p) for p in a.prefix)]
    att = lambda t: os.path.getmtime(os.path.join(R, t + ".attempted")) if os.path.exists(os.path.join(R, t + ".attempted")) else 0
    tags.sort(key=att)
    rows = [table_row(R, t, a.results_dir) for t in tags]
    mb = lambda kb: fmt(kb / 1024 if isinstance(kb, (int, float)) and kb >= 0 else None, 0)
    # /proc's kB are KiB: kB / 1024 = MiB
    print("| leg | end | backend | sharing | cache | rows | build s | 1st call ms | rows' ms min–max | anon+swap max MiB (stage) |"
          " VmHWM MiB | MemAvail min MiB | delegation | verdict | vs Mac max|d| |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        dl = ";".join(f"{d['replaced']}/{d['of']} {d['delegate']} {d['partitions']}p" for d in r["delegation"]) or "-"
        vm = r.get("vs_mac") or {}
        hw = max([x for x in (r["VmHWM_end_kb"], r["sampler_VmHWM_max_kB"], r["VmHWM_max_log_kb"]) if x], default=None)
        av = min([x for x in (r["MemAvailable_min_kB"], r["MemAvailable_min_app_kb"]) if isinstance(x, int) and x >= 0], default=None)
        cache = r["gpu_cache_dir"] or "-"
        if r["gpu_cache_dir"]:
            cache += f" (loads {r['gpu_cache_loads']}, DATA_LOSS {r['gpu_cache_data_loss']})"
        print(f"| {r['tag']} | {r['end']} | {r['accel']} {r['precision']} | {fmt(r['gpu_constant_tensor_sharing'])} |"
              f" {cache} | {fmt(r.get('rows_ran'))} | {fmt(r['compile_ms'] / 1000 if r['compile_ms'] else None)} |"
              f" {fmt(r.get('first_call_ms'))} | {fmt(r.get('row_ms_min'))}–{fmt(r.get('row_ms_max'))} |"
              f" {mb(r['anon_plus_swap_max_kb'])} ({r['anon_plus_swap_max_stage'] or '-'}; {r['anon_from'] or '-'}) |"
              f" {mb(hw)} | {mb(av)} | {dl} | {r['verdict'] or '-'} | {fmt(vm.get('max_abs_same_input'), 6) if vm else '-'} |")
        for t in r["touch"]:
            print(f"|  ↳ touch {t['signature']} | | | | | | | {fmt(t['ms'])} | nonfinite {t['nonfinite']} |"
                  f" {mb((t['RssAnon_kb'] or 0) + (t['VmSwap_kb'] or 0) if t['RssAnon_kb'] is not None else None)} (after) |"
                  f" {mb(t['VmHWM_kb'])} | {mb(t['mem_available_kb_after'])} | | | |")
    if a.json:
        json.dump({"round_dir": os.path.relpath(R, WORK), "legs": rows}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
