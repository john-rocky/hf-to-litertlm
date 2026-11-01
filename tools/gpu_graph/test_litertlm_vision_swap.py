#!/usr/bin/env python3
"""Checks of litertlm_vision_swap.py --ids-from on small files written here with the installed litert_lm_builder
(0.19.0 when written; no download): an OLD with four sections (the two vision sections last) and a REF = OLD with new vision bytes, written
by the tool without the option. Without --ids-from a rebuild of REF has REF's sections but its own uuid and time; with
REF's ids (the whole file, or only its first 64 KiB) it is REF byte for byte, with and without --clone; OLD rebuilt
from its own vision bytes with OLD's ids is OLD byte for byte; a FILE that ends inside its header is refused.
--clone needs APFS, so that case runs on macOS only (skipped elsewhere).

    python tools/gpu_graph/test_litertlm_vision_swap.py
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
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder.runtime.proto import llm_metadata_pb2

HERE = pathlib.Path(__file__).resolve().parent
APFS = platform.system() == "Darwin"


def swap(*args):
    return subprocess.run([sys.executable, str(HERE / "litertlm_vision_swap.py"), *map(str, args)],
                          capture_output=True, text=True)


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def write_old(path, venc, vadp):
    """LlmMetadata, a decoder-like TFLiteModel, then the two vision sections, packed by the builder itself."""
    pb = llm_metadata_pb2.LlmMetadata()
    pb.max_num_tokens = 2048
    parts = [(schema.AnySectionDataType.LlmMetadataProto, [], pb.SerializeToString()),
             (schema.AnySectionDataType.TFLiteModel, [("model_type", "tf_lite_prefill_decode")], os.urandom(70000)),
             (schema.AnySectionDataType.TFLiteModel, [("model_type", "tf_lite_vision_encoder")], venc),
             (schema.AnySectionDataType.TFLiteModel, [("model_type", "tf_lite_vision_adapter")], vadp)]
    b = lbm.LitertLmFileBuilder()
    for dtype, items, data in parts:
        md = [lbm.Metadata(key=k, value=v, dtype=lbm.DType.STRING) for k, v in items]
        b._sections.append(lbm._SectionObject(metadata=md, data_type=dtype,
                                              data_writer=(lambda s, d=data: s.write(d)), source_path=None))
    with open(path, "wb") as f:
        b.build(f)


class IdsFrom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = pathlib.Path(cls.tmp.name)
        cls.venc0, cls.vadp0 = os.urandom(50000), os.urandom(9000)
        (d / "venc0.tflite").write_bytes(cls.venc0)
        (d / "vadp0.tflite").write_bytes(cls.vadp0)
        (d / "venc1.tflite").write_bytes(os.urandom(30000))  # the new vision files, sizes unlike the old ones
        (d / "vadp1.tflite").write_bytes(os.urandom(12000))
        write_old(d / "old.litertlm", cls.venc0, cls.vadp0)
        r = swap(d / "old.litertlm", d / "ref.litertlm", "--venc", d / "venc1.tflite", "--vadp", d / "vadp1.tflite")
        assert r.returncode == 0 and "SWAP_OK" in r.stdout, r.stdout + r.stderr
        (d / "ref_head.bin").write_bytes((d / "ref.litertlm").read_bytes()[:65536])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rebuild(self, name, *extra, old="old.litertlm", venc="venc1.tflite", vadp="vadp1.tflite"):
        d = self.d
        r = swap(d / old, d / name, "--venc", d / venc, "--vadp", d / vadp, *extra)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("SWAP_OK", r.stdout)
        return d / name

    def test_new_ids_by_default(self):
        out = self.rebuild("plain.litertlm")
        self.assertEqual(out.stat().st_size, (self.d / "ref.litertlm").stat().st_size)
        self.assertNotEqual(sha(out), sha(self.d / "ref.litertlm"))

    def test_ids_from_ref_gives_ref(self):
        out = self.rebuild("from_ref.litertlm", "--ids-from", self.d / "ref.litertlm")
        self.assertEqual(sha(out), sha(self.d / "ref.litertlm"))

    def test_ids_from_ref_head_only(self):
        out = self.rebuild("from_head.litertlm", "--ids-from", self.d / "ref_head.bin")
        self.assertEqual(sha(out), sha(self.d / "ref.litertlm"))

    @unittest.skipUnless(APFS, "--clone needs APFS")
    def test_ids_from_ref_with_clone(self):
        out = self.rebuild("from_ref_clone.litertlm", "--clone", "--ids-from", self.d / "ref_head.bin")
        self.assertEqual(sha(out), sha(self.d / "ref.litertlm"))

    def test_old_from_its_own_parts(self):
        out = self.rebuild("old_again.litertlm", "--ids-from", self.d / "old.litertlm",
                           venc="venc0.tflite", vadp="vadp0.tflite")
        self.assertEqual(sha(out), sha(self.d / "old.litertlm"))

    def test_header_cut_short_is_refused(self):
        d = self.d
        (d / "short.bin").write_bytes((d / "ref.litertlm").read_bytes()[:200])
        r = swap(d / "old.litertlm", d / "short_out.litertlm", "--venc", d / "venc1.tflite", "--vadp",
                 d / "vadp1.tflite", "--ids-from", d / "short.bin")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("before the end of its header", r.stderr)
        self.assertFalse((d / "short_out.litertlm").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
