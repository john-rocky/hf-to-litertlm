"""copy_int4.py INT4_DIR PUBLISHED.litertlm - the weights of the published int4 file written into the data-free int4
TFLite of an export with another loop count, and the two sections the loop count does not change cut out of the
published file.

Why not by tensor name: the TFLite tensor names are the converter's location names with a counter, so the same weight
is named differently in a graph with one loop (and the recurrent weights, read by both passes of the two-loop graph,
carry generated constant names). The match is by bytes instead: INT4_DIR/model_quantized.tflite (quantize_dory.py,
data-free) is quantized by the same recipe from the same float weights as the published file, and GPTQ changed only
the codes (gptq_dory.py keeps the file's block scales), so each INT4 FULLY_CONNECTED weight's fp16 block scales equal
those of exactly one weight of the published file. Key = (weight shape, block size, sha256 of the block-scale bytes).

Checks: 289 INT4 FC weights on each side, keys unique on each side, every weight matched to exactly one published
weight and every published weight matched, equal code byte lengths; per pair the fraction of equal 4-bit codes between
the data-free codes and the published GPTQ codes is recorded, and a pair below 0.5 stops the step (not the same
weight; GPTQ changes about one code in seven). Then INT4_DIR/model_quantized.tflite is renamed
model_quantized_df.tflite, its copy (an APFS clone where the file system has them) becomes model_quantized.tflite, and
the published code bytes are written over each matched weight's range; read back; a full byte comparison with the
data-free file must find differences only inside the written ranges.

Cut out of the published file (the loop count changes neither): its LlmMetadata section as
INT4_DIR/published_llm_metadata_t.pb and its embedder section as INT4_DIR/published_embedder_quantized.tflite.
INT4_DIR's own llm_metadata_t.pb and embedder_quantized.tflite are compared with them (sha256; the export's metadata
holds the same stop tokens, written in the order of Python's string hashing of the run that made it). Writes
INT4_DIR/copy_int4.json.

Usage: python dory_work/copy_int4.py <int4 dir> <published int4 .litertlm>
"""
import collections
import hashlib
import json
import mmap
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np
from ai_edge_litert import schema_py_generated as ts
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as hs

FC = ts.BuiltinOperator.FULLY_CONNECTED
AGREE_MIN = 0.5


def sha(data):
  return hashlib.sha256(data).hexdigest()


def sha_file(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for b in iter(lambda: f.read(64 << 20), b""):
      h.update(b)
  return h.hexdigest()


def sections(mm):
  """The section table of a .litertlm bundle: data type, byte range, string items (model_type)."""
  end = struct.unpack_from("<Q", mm, litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET)[0]
  md = hs.LiteRTLMMetaData.GetRootAs(bytes(mm[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:end]), 0)
  out = []
  for i in range(md.SectionMetadata().ObjectsLength()):
    s = md.SectionMetadata().Objects(i)
    items = {}
    for j in range(s.ItemsLength()):
      it = s.Items(j)
      if it is None or it.Key() is None or it.ValueType() != hs.VData.StringValue:
        continue
      v = hs.StringValue()
      v.Init(it.Value().Bytes, it.Value().Pos)
      items[it.Key().decode()] = (v.Value() or b"").decode()
    out.append(dict(index=i, type=litertlm_core.any_section_data_type_to_string(s.DataType()), begin=s.BeginOffset(),
                    end=s.EndOffset(), items=items))
  return out


def buf_range(m, idx):
  """(begin, end) of a buffer's data relative to the model's first byte: inline vector or offset form."""
  b = m.Buffers(idx)
  if b.DataLength() > 0:
    off = b._tab.Vector(b._tab.Offset(4))  # pylint: disable=protected-access
    return off, off + b.DataLength()
  if b.Size() > 0:
    return b.Offset(), b.Offset() + b.Size()
  return None


def int4_fcs(m, view):
  """weight buffer -> record of the INT4 blockwise FULLY_CONNECTED weights of every subgraph."""
  codes = [max(m.OperatorCodes(i).BuiltinCode(), m.OperatorCodes(i).DeprecatedBuiltinCode())
           for i in range(m.OperatorCodesLength())]
  out, other = {}, collections.Counter()
  for si in range(m.SubgraphsLength()):
    sg = m.Subgraphs(si)
    for oi in range(sg.OperatorsLength()):
      op = sg.Operators(oi)
      if codes[op.OpcodeIndex()] != FC or op.InputsLength() < 2 or op.Inputs(1) < 0:
        continue
      t = sg.Tensors(op.Inputs(1))
      q = t.Quantization()
      if t.Type() != ts.TensorType.INT4 or q is None or q.DetailsType() != ts.QuantizationDetails.BlockwiseQuantization:
        other[f"type{t.Type()}"] += 1
        continue
      if t.Buffer() in out:
        out[t.Buffer()]["reads"] += 1
        continue
      bq = ts.BlockwiseQuantization()
      bq.Init(q.Details().Bytes, q.Details().Pos)
      st = sg.Tensors(bq.Scales())
      sr = buf_range(m, st.Buffer())
      shape = [int(t.Shape(j)) for j in range(t.ShapeLength())]
      key = f"{shape}|{bq.BlockSize()}|{sha(bytes(view[sr[0]:sr[1]]))}"
      out[t.Buffer()] = dict(shape=shape, code_range=buf_range(m, t.Buffer()), key=key, reads=1)
  return out, dict(other)


def nibble_agree(a, b):
  a = np.frombuffer(a, dtype=np.uint8)
  b = np.frombuffer(b, dtype=np.uint8)
  lo = np.count_nonzero((a & 0x0F) == (b & 0x0F))
  hi = np.count_nonzero((a >> 4) == (b >> 4))
  return (lo + hi) / (2.0 * a.size)


def main(int4_dir, published):
  t0 = time.time()
  fp = open(published, "rb")
  pm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
  secs = sections(pm)
  dec = [s for s in secs if s["items"].get("model_type") == "tf_lite_prefill_decode"]
  emb = [s for s in secs if s["items"].get("model_type") == "tf_lite_embedder"]
  meta = [s for s in secs if s["type"] == "LlmMetadataProto"]
  if not (len(dec) == len(emb) == len(meta) == 1):
    raise SystemExit(f"copy_int4: {published} is not a one-decoder, one-embedder bundle: {secs}")
  dv = memoryview(pm)[dec[0]["begin"]:dec[0]["end"]]
  pub, pub_other = int4_fcs(ts.Model.GetRootAs(dv, 0), dv)

  src = os.path.join(int4_dir, "model_quantized.tflite")
  df = os.path.join(int4_dir, "model_quantized_df.tflite")
  if not os.path.exists(df):
    os.rename(src, df)
  if os.path.exists(src):
    os.remove(src)
  fa = open(df, "rb")
  am = mmap.mmap(fa.fileno(), 0, access=mmap.ACCESS_READ)
  own, own_other = int4_fcs(ts.Model.GetRootAs(memoryview(am), 0), memoryview(am))

  rec = dict(published=os.path.abspath(published), published_sha256=sha_file(published),
             published_sections=[dict(index=s["index"], type=s["type"], begin=s["begin"], size=s["end"] - s["begin"],
                                      **s["items"]) for s in secs],
             data_free=os.path.abspath(df), n_published=len(pub), n_data_free=len(own),
             non_int4_fc_weights=dict(published=pub_other, data_free=own_other))
  problems = []
  if len(pub) != 289 or len(own) != 289:
    problems.append(f"INT4 FC weights: published {len(pub)}, data-free {len(own)} (expected 289 each)")
  pkey = collections.defaultdict(list)
  for b, r in pub.items():
    pkey[r["key"]].append(b)
  okey = collections.defaultdict(list)
  for b, r in own.items():
    okey[r["key"]].append(b)
  dup = [v for v in list(pkey.values()) + list(okey.values()) if len(v) > 1]
  if dup:
    problems.append(f"keys not unique: {dup[:5]}")
  pairs, unmatched = [], []
  for b, r in sorted(own.items()):
    hit = pkey.get(r["key"], [])
    if len(hit) != 1:
      unmatched.append(dict(buffer=b, shape=r["shape"], candidates=hit))
      continue
    p = pub[hit[0]]
    oa, ob = r["code_range"]
    pa, pb = p["code_range"]
    if ob - oa != pb - pa:
      unmatched.append(dict(buffer=b, published_buffer=hit[0], why=f"code bytes {ob - oa} != {pb - pa}"))
      continue
    pairs.append(dict(shape=r["shape"], buffer=b, published_buffer=hit[0], code_bytes=ob - oa, code_range=[oa, ob],
                      published_code_range=[pa, pb], agree=round(nibble_agree(am[oa:ob], dv[pa:pb]), 6),
                      reads=r["reads"], published_reads=p["reads"]))
  missing = sorted(set(pub) - {p["published_buffer"] for p in pairs})
  if unmatched:
    problems.append(f"{len(unmatched)} data-free weights without exactly one published match: {unmatched[:5]}")
  if missing:
    problems.append(f"{len(missing)} published weights not matched: {missing[:10]}")
  ag = sorted(p["agree"] for p in pairs)
  low = [p for p in pairs if p["agree"] < AGREE_MIN]
  if low:
    problems.append(f"{len(low)} pairs with code agreement < {AGREE_MIN}: {[(p['buffer'], p['agree']) for p in low[:5]]}")
  rec.update(n_pairs=len(pairs), unmatched=unmatched, published_not_matched=missing,
             agree=dict(min=ag[0], median=ag[len(ag) // 2], max=ag[-1]) if ag else None)
  print(f"MATCH: published {len(pub)} / data-free {len(own)} INT4 FC weights, pairs {len(pairs)}, unmatched "
        f"{len(unmatched)}, published not matched {len(missing)}; equal 4-bit codes data-free vs published "
        f"min {ag[0] if ag else None} median {ag[len(ag) // 2] if ag else None} max {ag[-1] if ag else None}", flush=True)
  out_json = os.path.join(int4_dir, "copy_int4.json")
  if problems:
    rec.update(ok=False, problems=problems, seconds=round(time.time() - t0, 1))
    json.dump(rec, open(out_json, "w"), indent=1)
    for p in problems:
      print("PROBLEM", p, flush=True)
    print("COPY_INT4 FAIL (nothing written)", flush=True)
    return 1

  if subprocess.run(["cp", "-c", df, src]).returncode != 0:  # APFS clone; a plain copy elsewhere
    shutil.copyfile(df, src)
  ranges = []
  with open(src, "r+b") as fh:
    wm = mmap.mmap(fh.fileno(), 0)
    for p in pairs:
      oa, ob = p["code_range"]
      pa, pb = p["published_code_range"]
      wm[oa:ob] = dv[pa:pb]
      ranges.append((oa, ob))
    wm.flush()
    back = all(wm[p["code_range"][0]:p["code_range"][1]] == dv[p["published_code_range"][0]:p["published_code_range"][1]]
               for p in pairs)
    wm.close()
  a_mm = np.memmap(df, dtype=np.uint8, mode="r")
  b_mm = np.memmap(src, dtype=np.uint8, mode="r")
  assert a_mm.size == b_mm.size
  ranges.sort()
  starts = np.array([r[0] for r in ranges], dtype=np.int64)
  ends = np.array([r[1] for r in ranges], dtype=np.int64)
  outside, diff_total, step = 0, 0, 1 << 26
  for k in range(0, a_mm.size, step):
    d = np.nonzero(a_mm[k:k + step] != b_mm[k:k + step])[0]
    if d.size == 0:
      continue
    d = d.astype(np.int64) + k
    diff_total += d.size
    idx = np.searchsorted(starts, d, side="right") - 1
    ok_ = (idx >= 0) & (d < ends[np.clip(idx, 0, None)])
    outside += int((~ok_).sum())
  del a_mm, b_mm
  write = dict(written=len(ranges), bytes_written=int(sum(b - a for a, b in ranges)), bytes_differing=int(diff_total),
               bytes_differing_outside_written_ranges=outside, read_back_equal=bool(back), size=os.path.getsize(src),
               sha256=sha_file(src), data_free_sha256=sha_file(df))

  cut = {}
  for name, s, own_name in (("published_llm_metadata_t.pb", meta[0], "llm_metadata_t.pb"),
                            ("published_embedder_quantized.tflite", emb[0], "embedder_quantized.tflite")):
    dst = os.path.join(int4_dir, name)
    with open(dst + ".part", "wb") as f:
      f.write(pm[s["begin"]:s["end"]])
    os.rename(dst + ".part", dst)
    own_path = os.path.join(int4_dir, own_name)
    c = dict(section=s["index"], size=s["end"] - s["begin"], sha256=sha_file(dst), own=own_name,
             own_size=os.path.getsize(own_path), own_sha256=sha_file(own_path))
    c["own_equal"] = c["own_sha256"] == c["sha256"] and c["own_size"] == c["size"]
    cut[name] = c
    print(f"CUT {name}: section {c['section']} {c['size']} B sha256 {c['sha256']}; the export's own {own_name} "
          f"{'is byte-equal' if c['own_equal'] else 'differs (sha256 ' + c['own_sha256'] + ')'}", flush=True)
  ok = outside == 0 and back
  rec.update(pairs=pairs, write=write, cut=cut, ok=bool(ok), seconds=round(time.time() - t0, 1))
  json.dump(rec, open(out_json, "w"), indent=1)
  print(f"WRITE: {write['written']} weights ({write['bytes_written']} B), differing bytes {diff_total}, outside the "
        f"written ranges {outside}, read back {'equal' if back else 'DIFFERENT'}; {src} sha256 {write['sha256']}",
        flush=True)
  print(f"COPY_INT4 {'OK' if ok else 'FAIL'} ({rec['seconds']} s)", flush=True)
  return 0 if ok else 1


if __name__ == "__main__":
  if len(sys.argv) != 3:
    raise SystemExit(__doc__)
  sys.exit(main(*sys.argv[1:3]))
