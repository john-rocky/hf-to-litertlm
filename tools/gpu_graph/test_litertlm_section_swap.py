#!/usr/bin/env python3
"""Checks of litertlm_section_swap.py on small files written here with the installed litert_lm_builder (0.19.0 when
written; no download). OLD has five sections: LlmMetadata, a prefill/decode graph and a TFLiteWeights section with
the same model_type (tf_lite_prefill_decode), and the two vision graphs. `--section tf_lite_prefill_decode=FILE`
replaces only the graph: read back from NEW's own header, the graph section holds FILE's bytes, and the TFLiteWeights
section and every other section hold OLD's bytes, with OLD's types and model_type items in OLD's order. Naming the two
vision sections writes the same file as litertlm_vision_swap.py (both with the same --ids-from). OLD rebuilt from its
own graph with OLD's uuid and creation time (--ids-from) is OLD byte for byte, and so is the --clone rebuild of OLD
from its own vision adapter (--clone needs APFS, so that case runs on macOS only). Refused, with no NEW written: a
model_type OLD does not have, alone or next to one it has; a model_type given twice; --clone when a kept section comes
after a replaced one.

    python tools/gpu_graph/test_litertlm_section_swap.py
"""
import hashlib
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import unittest

import litert_lm_builder.litertlm_builder as lbm
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder.runtime.proto import llm_metadata_pb2

HERE = pathlib.Path(__file__).resolve().parent
APFS = platform.system() == "Darwin"
# litert_lm_builder 0.19.0's object API calls VDataCreator, its schema defines VdataCreator (as in the tools)
if not hasattr(schema, "VDataCreator"):
    schema.VDataCreator = schema.VdataCreator
T = schema.AnySectionDataType
PD, VENC, VADP = "tf_lite_prefill_decode", "tf_lite_vision_encoder", "tf_lite_vision_adapter"


def tool(name, *args):
    return subprocess.run([sys.executable, str(HERE / name), *map(str, args)], capture_output=True, text=True)


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def write_old(path, parts):
    """parts = [(AnySectionDataType, model_type or None, bytes)], packed by the builder itself."""
    b = lbm.LitertLmFileBuilder()
    for dtype, mt, data in parts:
        md = [lbm.Metadata(key="model_type", value=mt, dtype=lbm.DType.STRING)] if mt else []
        b._sections.append(lbm._SectionObject(metadata=md, data_type=dtype,
                                              data_writer=(lambda s, d=data: s.write(d)), source_path=None))
    with open(path, "wb") as f:
        b.build(f)


def sections(path):
    """[(section type, model_type, bytes)] in file order, read from the file's own header."""
    raw = pathlib.Path(path).read_bytes()
    loc = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET
    end = int.from_bytes(raw[loc:loc + 8], "little")
    meta = schema.LiteRTLMMetaDataT.InitFromObj(
        schema.LiteRTLMMetaData.GetRootAs(bytearray(raw[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:end]), 0))
    out = []
    for o in meta.sectionMetadata.objects:
        mt = [k.value.value for k in (o.items or []) if (k.key.decode() if isinstance(k.key, bytes) else k.key)
              == "model_type"]
        mt = mt[0].decode() if mt and isinstance(mt[0], bytes) else (mt[0] if mt else None)
        out.append((litertlm_core.any_section_data_type_to_string(o.dataType), mt, raw[o.beginOffset:o.endOffset]))
    return out


class SectionSwap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = pathlib.Path(cls.tmp.name)
        pb = llm_metadata_pb2.LlmMetadata()
        pb.max_num_tokens = 2048
        cls.pd0, cls.adp0 = os.urandom(70000), os.urandom(9000)
        cls.parts = [(T.LlmMetadataProto, None, pb.SerializeToString()), (T.TFLiteModel, PD, cls.pd0),
                     (T.TFLiteWeights, PD, os.urandom(40000)), (T.TFLiteModel, VENC, os.urandom(50000)),
                     (T.TFLiteModel, VADP, cls.adp0)]
        write_old(d / "old.litertlm", cls.parts)
        for name, data in (("pd0.tflite", cls.pd0), ("adp0.tflite", cls.adp0), ("pd1.tflite", os.urandom(30000)),
                           ("venc1.tflite", os.urandom(20000)), ("adp1.tflite", os.urandom(12000))):
            (d / name).write_bytes(data)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def swap(self, name, *args, ok=True):
        r = tool("litertlm_section_swap.py", self.d / "old.litertlm", self.d / name, *args)
        if ok:
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("SWAP_OK", r.stdout)
        return r

    def test_named_section_only(self):
        self.swap("pd.litertlm", "--section", f"{PD}={self.d / 'pd1.tflite'}")
        got = sections(self.d / "pd.litertlm")
        want = [(litertlm_core.any_section_data_type_to_string(t), mt, data) for t, mt, data in self.parts]
        want[1] = (want[1][0], PD, (self.d / "pd1.tflite").read_bytes())
        self.assertEqual([g[:2] for g in got], [w[:2] for w in want])
        for i, (g, w) in enumerate(zip(got, want)):
            self.assertEqual(g[2], w[2], f"section {i} ({g[0]} {g[1]})")

    def test_same_as_vision_swap(self):
        ids = ("--ids-from", self.d / "old.litertlm")
        self.swap("v_section.litertlm", "--section", f"{VENC}={self.d / 'venc1.tflite'}",
                  "--section", f"{VADP}={self.d / 'adp1.tflite'}", *ids)
        r = tool("litertlm_vision_swap.py", self.d / "old.litertlm", self.d / "v_vision.litertlm",
                 "--venc", self.d / "venc1.tflite", "--vadp", self.d / "adp1.tflite", *ids)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(sha(self.d / "v_section.litertlm"), sha(self.d / "v_vision.litertlm"))

    def test_old_from_its_own_graph(self):
        self.swap("old_again.litertlm", "--section", f"{PD}={self.d / 'pd0.tflite'}",
                  "--ids-from", self.d / "old.litertlm")
        self.assertEqual(sha(self.d / "old_again.litertlm"), sha(self.d / "old.litertlm"))

    @unittest.skipUnless(APFS, "--clone needs APFS")
    def test_clone_old_from_its_own_last_section(self):
        self.swap("old_clone.litertlm", "--clone", "--section", f"{VADP}={self.d / 'adp0.tflite'}",
                  "--ids-from", self.d / "old.litertlm")
        self.assertEqual(sha(self.d / "old_clone.litertlm"), sha(self.d / "old.litertlm"))

    def refused(self, name, args, text):
        r = self.swap(name, *args, ok=False)
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn(text, r.stderr)
        self.assertFalse((self.d / name).exists())
        self.assertFalse((self.d / (name + ".partial")).exists())

    def test_refusals(self):
        pd1 = self.d / "pd1.tflite"
        self.refused("r_none.litertlm", ["--section", f"tf_lite_audio_encoder={pd1}"], "OLD has none of the sections")
        self.refused("r_one_missing.litertlm", ["--section", f"{PD}={pd1}", "--section", f"tf_lite_audio_encoder={pd1}"],
                     "expected each named section exactly once")
        self.refused("r_twice.litertlm", ["--section", f"{PD}={pd1}", "--section", f"{PD}={pd1}"],
                     "bad or repeated --section")
        self.refused("r_clone.litertlm", ["--clone", "--section", f"{PD}={pd1}"],
                     "--clone needs the replaced sections last")


if __name__ == "__main__":
    unittest.main(verbosity=2)
