"""Where do the bytes go? Group tensors by buffer, print the largest buffers with dtype / quant layout / short name."""
import sys, collections
from ai_edge_litert.tools import flatbuffer_utils as fu
m = fu.read_model(sys.argv[1])
TT = {0:'F32',1:'F16',2:'I32',3:'U8',4:'I64',5:'STR',6:'BOOL',7:'I16',8:'C64',9:'I8',10:'F64',16:'BF16',17:'I4',20:'I2'}
seen = {}
for si, sg in enumerate(m.subgraphs):
    for t in sg.tensors:
        b = m.buffers[t.buffer]; n = 0 if b.data is None else len(b.data)
        if n < 20_000_000: continue
        if t.buffer in seen: seen[t.buffer][2].append(si); continue
        q = t.quantization; qd = ""
        if q is not None and q.scale is not None and len(q.scale): qd = f"per-axis {len(q.scale)} sc"
        if q is not None and getattr(q, 'detailsType', 0): qd += f" details{q.detailsType}"
        name = t.name.decode() if isinstance(t.name, bytes) else t.name
        seen[t.buffer] = [n, f"{TT.get(t.type, t.type)} {list(t.shape)} {qd} :: {name.split('/')[-1][:70]}", [si]]
tot = 0
for b, (n, d, sgs) in sorted(seen.items(), key=lambda kv: -kv[1][0])[:int(sys.argv[2]) if len(sys.argv)>2 else 14]:
    tot += n; print(f"{n/1e9:6.2f} GB  subgraphs {sgs}  {d}")
print("sum of listed buffers %.2f GB" % (tot/1e9))

grp = collections.Counter()
for b, (n, d, sgs) in seen.items():
    grp[(d.split(' :: ')[0].split(' per-axis')[0], tuple(sgs))] += 1
print("--- buffers >= 20 MB grouped by (dtype shape, subgraphs) ---")
for (k, sgs), c in sorted(grp.items(), key=lambda kv: -kv[1]): print(f"{c:3d} x  {k}  in subgraphs {list(sgs)}")
