"""phys_footprint of this process (now and lifetime max) from proc_pid_rusage: bench_host.footprint, copied (importing
bench_host would change sys.path).
"""
from __future__ import annotations

import ctypes
import os


class RusageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
        "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime",
        "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
        "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten",
        "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background",
        "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy", "ri_cpu_time_qos_user_initiated",
        "ri_cpu_time_qos_user_interactive", "ri_billed_system_time", "ri_serviced_system_time",
        "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles", "ri_billed_energy",
        "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time")]


def footprint() -> dict:
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    info = RusageInfoV4()
    rc = lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info))
    if rc != 0:
        return {"error": rc}
    return {"phys_footprint_gb": round(info.ri_phys_footprint / 2 ** 30, 2),
            "lifetime_max_phys_footprint_gb": round(info.ri_lifetime_max_phys_footprint / 2 ** 30, 2),
            "resident_gb": round(info.ri_resident_size / 2 ** 30, 2)}
