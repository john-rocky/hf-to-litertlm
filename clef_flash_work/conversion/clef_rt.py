"""LiteRT runtime helpers, copied unchanged from the d1-3B LiteRT conversion's d1_check.py (sha256 62431726...,
2026-10-20): runtime_log_verbose() switches the runtime's VERBOSE lines on (the TFLite `Replacing N out of M node(s)
with delegate` line, LiteRT's `Partitioned subgraph` line); REPLACING / PARTITIONED / LINE_KEYS parse them."""
import ctypes
import re
from pathlib import Path

LINE_KEYS = re.compile(r"(?i)replacing|partition|delegat|fail|error|unsupported|not supported|abort|fallback|reject")
REPLACING = re.compile(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \(([^)]*)\) node, yielding (\d+) partitions")
PARTITIONED = re.compile(r"Partitioned subgraph<(\d+)>, selected (\d+) ops, from a total of (\d+) ops\. resulted in (\d+) partitions")


def runtime_log_verbose() -> dict:
    """Let the runtime print its VERBOSE lines (the TFLite `Replacing N out of M node(s) with delegate` line, LiteRT's
    `Partitioned subgraph` line). The CompiledModel wrapper runs the runtime linked into libpywrap_litert_common.dylib
    (libLiteRt.dylib is not loaded on this path, and the common library does not export its logger API), so the local
    symbols are used: their `nm` values plus the image's slide from dyld. Before any use, each address is checked
    against the file (code: the first 16 bytes; data: the 4-byte initial value, which must be INFO = 1); on any mismatch
    nothing is touched. Then both MinimalLogger severity words are set to VERBOSE (0) and the LiteRT default logger's
    minimum severity to VERBOSE (LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(), 0))."""
    import subprocess

    import ai_edge_litert.compiled_model  # noqa: F401  (loads the library)

    lib = (Path(ai_edge_litert.compiled_model.__file__).parent / "libpywrap_litert_common.dylib").resolve()
    want = {"get_logger": "_LiteRtGetDefaultLogger", "set_min": "_LiteRtSetMinLoggerSeverity",
            "get_min": "_LiteRtGetMinLoggerSeverity",
            "tflite_min": "__ZN6tflite16logging_internal13MinimalLogger21minimum_log_severity_E",
            "anon_min": "__ZN12_GLOBAL__N_113MinimalLogger21minimum_log_severity_E"}
    syms = {}
    for line in subprocess.run(["nm", str(lib)], capture_output=True, text=True, check=True).stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] in want.values():
            syms[parts[2]] = int(parts[0], 16)
    sysdl = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    sysdl._dyld_image_count.restype = ctypes.c_uint32
    sysdl._dyld_get_image_name.restype = ctypes.c_char_p
    sysdl._dyld_get_image_vmaddr_slide.restype = ctypes.c_long
    idx = [i for i in range(sysdl._dyld_image_count())
           if Path(sysdl._dyld_get_image_name(i).decode()).resolve() == lib]
    doc = {"library": str(lib), "symbols_found": sorted(k for k, v in want.items() if v in syms), "image_matches": len(idx)}
    if len(idx) != 1 or len(syms) != len(want):
        doc["applied"] = False
        return doc
    slide = sysdl._dyld_get_image_vmaddr_slide(idx[0])
    blob = lib.read_bytes()
    addr = {k: syms[v] + slide for k, v in want.items()}
    code_ok = {k: ctypes.string_at(addr[k], 16) == blob[syms[want[k]]: syms[want[k]] + 16]
               for k in ("get_logger", "set_min", "get_min")}
    data_init = {k: int.from_bytes(blob[syms[want[k]]: syms[want[k]] + 4], "little", signed=True)
                 for k in ("tflite_min", "anon_min")}
    data_now = {k: ctypes.c_int.from_address(addr[k]).value for k in ("tflite_min", "anon_min")}
    doc.update(code_bytes_match=code_ok, data_initial=data_init, data_before=data_now)
    if not all(code_ok.values()) or any(v != 1 for v in data_init.values()) or any(v != 1 for v in data_now.values()):
        doc["applied"] = False
        return doc
    for k in ("tflite_min", "anon_min"):
        ctypes.c_int.from_address(addr[k]).value = 0
    get_logger = ctypes.CFUNCTYPE(ctypes.c_void_p)(addr["get_logger"])
    set_min = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int)(addr["set_min"])
    get_min = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))(addr["get_min"])
    logger = get_logger()
    before, after = ctypes.c_int(0), ctypes.c_int(0)
    st_before = get_min(logger, ctypes.byref(before))
    st_set = set_min(logger, 0)
    st_after = get_min(logger, ctypes.byref(after))
    doc.update(applied=True, data_after={k: ctypes.c_int.from_address(addr[k]).value for k in ("tflite_min", "anon_min")},
               litert_logger={"status_get_before": st_before, "before_low_byte": before.value & 0xFF, "status_set": st_set,
                              "status_get_after": st_after, "after_low_byte": after.value & 0xFF})
    return doc
