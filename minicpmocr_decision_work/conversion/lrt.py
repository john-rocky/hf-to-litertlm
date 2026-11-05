"""Helpers around the host's Graph: memory of this process, comparison numbers. The delegation verdict of a GPU file = a
CompiledModel built with the GPU alone (is_fully_accelerated says True under GPU + CPU)."""
from __future__ import annotations

import ctypes
import os
import resource
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parents[1] / "host"))
from decide_host.litert_backends import Graph, options  # noqa: E402,F401

_RU_FIELDS = (
    "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
    "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
    "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
    "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten", "ri_cpu_time_qos_default",
    "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
    "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive", "ri_billed_system_time",
    "ri_serviced_system_time", "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
    "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time")


class _RUsageInfoV4(ctypes.Structure):     # every field of v4 (a shorter struct lets the kernel write past it)
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in _RU_FIELDS]


def memory() -> dict:
    """ru_maxrss, phys_footprint, its lifetime max, resident size (bytes)."""
    doc = {"ru_maxrss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    info = _RUsageInfoV4()
    if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) == 0:
        doc.update(phys_footprint=int(info.ri_phys_footprint),
                   lifetime_max_phys_footprint=int(info.ri_lifetime_max_phys_footprint),
                   resident_size=int(info.ri_resident_size))
    return doc


def stats(got, want) -> dict:
    g = np.asarray(got, np.float64).ravel()
    w = np.asarray(want, np.float64).ravel()
    return {"rel_rms": float(np.linalg.norm(g - w) / max(np.linalg.norm(w), 1e-300)), "max_abs": float(np.abs(g - w).max()),
            "cos": float(g @ w / max(np.linalg.norm(g) * np.linalg.norm(w), 1e-300)), "finite": bool(np.isfinite(g).all()),
            "bit_equal": bool(np.array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32)))}
