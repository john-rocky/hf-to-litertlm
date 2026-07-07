"""NanoDet-Plus-m (anchor-free detector, ShuffleNetV2) -> LiteRT.

Like the pose sample, we export ONLY the raw detection head output (per-cell class scores +
box distributions) and do the decode (DFL integral + distance2bbox + NMS) in Kotlin, so the
graph stays pure-conv and GPU-clean -- avoiding the GATHER_ND/NMS ops that block detectors.

    ~/clipconv/bin/python scripts/convert_nanodet.py [out_dir] [size]
"""
import _stub  # noqa: F401
import os, sys, json, traceback

import torch
import torch.nn as nn
REPO = os.path.join(os.path.dirname(__file__), "..", "nanodet_work", "repo")
sys.path.insert(0, REPO)
from probe_convert_module import read_op_histogram

# nanodet.util re-exports its logger/checkpoint helpers (which import pytorch_lightning) but
# build_model never uses them. Stub pytorch_lightning so the imports succeed without installing it.
import types as _types


class _DummyMeta(type):
    def __call__(cls, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]  # used as a decorator -> return the function unchanged
        return super().__call__(*args, **kwargs)

    def __getattr__(cls, name):  # any missing class attr -> the dummy itself (but never dunders)
        if name.startswith("__"):
            raise AttributeError(name)
        return cls


class _Dummy(metaclass=_DummyMeta):  # usable as a base class AND as a decorator
    pass


class _FakeModule(_types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy


# Auto-stub the whole training/eval import tree (pytorch_lightning etc.) — needed both to build
# (logger/checkpoint re-exports) and to unpickle the Lightning .ckpt — without installing them.
import importlib.abc as _iabc
import importlib.machinery as _imach

_STUB_PREFIXES = ("pytorch_lightning", "lightning_fabric", "lightning_utilities",
                  "pycocotools", "tensorboard")


class _StubLoader(_iabc.Loader):
    def create_module(self, spec):
        m = _FakeModule(spec.name)
        m.__path__ = []  # mark as a package so submodule imports route back through the finder
        return m

    def exec_module(self, module):
        pass


class _StubFinder(_iabc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if any(name == p or name.startswith(p + ".") for p in _STUB_PREFIXES):
            return _imach.ModuleSpec(name, _StubLoader())
        return None


sys.meta_path.insert(0, _StubFinder())


# Bypass nanodet.util.__init__ (which drags in logger/checkpoint/coco/lightning). Register a
# minimal nanodet.util loaded from only the light, torch-only submodules the model needs.
import importlib.util as _ilu

_NU = os.path.join(REPO, "nanodet", "util")
_nanodet = _types.ModuleType("nanodet")
_nanodet.__path__ = [os.path.join(REPO, "nanodet")]
sys.modules.setdefault("nanodet", _nanodet)
_util = _types.ModuleType("nanodet.util")
_util.__path__ = [_NU]
sys.modules["nanodet.util"] = _util


def _load_into(modname, path):
    spec = _ilu.spec_from_file_location(modname, path)
    m = _ilu.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m


for _sub in ("box_transform", "misc", "config", "util_mixins"):
    _m = _load_into(f"nanodet.util.{_sub}", os.path.join(_NU, f"{_sub}.py"))
    for _a in dir(_m):
        if not _a.startswith("_"):
            setattr(_util, _a, getattr(_m, _a))
_util.overlay_bbox_cv = lambda *a, **k: None


def _util_getattr(name):
    # Don't fake dunders (e.g. __file__) — inspect walks sys.modules at torchvision import
    # and abspath(__file__) must not get a dummy. Only fake real symbols.
    if name.startswith("__"):
        raise AttributeError(name)
    return _Dummy


_util.__getattr__ = _util_getattr

BANNED = {"GATHER_ND", "SELECT_V2", "BROADCAST_TO", "WHERE", "POW"}
CKPT = os.path.join(os.path.dirname(__file__), "..", "nanodet_work", "nanodet-plus-m_320.ckpt")
CFG = os.path.join(REPO, "config", "nanodet-plus-m_320.yml")


class RawHead(nn.Module):
    """Return the raw head tensor (pre-decode)."""

    def __init__(self, det):
        super().__init__()
        self.det = det

    def forward(self, x):
        return self.det(x)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out/nanodet"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 320
    os.makedirs(out_dir, exist_ok=True)
    res = {"model": "nanodet-plus-m", "size": size, "ok": False, "stage": None,
           "error_type": None, "error_head": None, "failing_op": None}
    try:
        res["stage"] = "build"
        from nanodet.util import cfg, load_config
        from nanodet.model.arch import build_model
        load_config(cfg, CFG)
        det = build_model(cfg.model).eval()

        res["stage"] = "load-weights"
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        avg = {k[len("avg_model."):]: v for k, v in sd.items() if k.startswith("avg_model.")}
        base = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        use = avg if avg else base
        det.load_state_dict(use, strict=False)
        model = RawHead(det).eval()
        print("WEIGHTS OK (ema)" if avg else "WEIGHTS OK")

        res["stage"] = "eager-forward"
        nchw = torch.rand(1, 3, size, size)
        with torch.no_grad():
            out = model(nchw)
        res["eager_out_shape"] = list(out.shape) if torch.is_tensor(out) else str(type(out))
        print("EAGER OK", res["eager_out_shape"])

        res["stage"] = "convert"
        import litert_torch
        clio = litert_torch.to_channel_last_io(model, args=[0])
        nhwc = nchw.permute(0, 2, 3, 1).contiguous()
        edge = litert_torch.convert(clio, (nhwc,))
        tfl = os.path.join(out_dir, f"nanodet_{size}.tflite")
        edge.export(tfl)
        res["tflite"] = tfl
        res["tflite_mb"] = round(os.path.getsize(tfl) / 1e6, 2)
        print("CONVERT OK", tfl, res["tflite_mb"], "MB")

        res["stage"] = "read-ops"
        ops = read_op_histogram(tfl)
        res["ops"] = ops
        res["banned_ops"] = sorted(k for k in ops.get("hist", {}) if k.upper() in BANNED)
        res["flex"] = ops.get("flex")
        res["custom"] = ops.get("custom")
        res["ok"] = True
        res["stage"] = "done"
    except BaseException as e:  # noqa: BLE001
        tb = traceback.format_exc()
        res["error_type"] = type(e).__name__
        msg = str(e).strip()
        res["error_head"] = (msg.splitlines()[0] if msg else tb.strip().splitlines()[-1])[:400]
        for ln in tb.splitlines():
            s = ln.strip()
            if ("aten." in s) or ("Lowering not found" in s) or ("No module" in s) or ("While executing" in s):
                res["failing_op"] = s[:400]
                break

    print("NANODET_RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
