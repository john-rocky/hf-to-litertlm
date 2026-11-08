"""The paths of the 7.5, 15 and 25 s buckets on top of the scripts of the other files (cwd = phonon2_work; those scripts are not edited): the 7.5, 15 and 25 s buckets beside
5 / 10 / 20 / 30 s. Each bucket's mel frames are the smallest multiple of 8 at or above 100 x its seconds (the encoder's three stride-2 stages need t_mel % 8 == 0:
as for 5 s = 504 in p5_common.py): 7.5 s = 752 frames (encoder T 94, audio input 120,320 samples), 15 s = 1,504 (T 188, 240,640), 25 s = 2,504 (T 313, 400,640).
The host rule stays p1_drive.py's: a window goes to the smallest bucket whose audio input holds it (30 s takes the rest). Bucket keys are numbers (7.5 a float, the others
ints): file names enc7.5_<form>, mel7.5, the signature key enc7.5. Graphs: out/r7 first, then out/r5, then out/r1."""
import os
import sys

K = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(K, "scripts"))
T_NEW = {7.5: 752, 15: 1504, 25: 2504}
T_ALL = {5: 504, 7.5: 752, 10: 1000, 15: 1504, 20: 2000, 25: 2504, 30: 3000}
B7 = (5, 7.5, 10, 15, 20, 25, 30)
OUT7 = "out/r7"
OUT5 = "out/r5"
OUT1 = "out/r1"
for _b, _t in T_ALL.items():
    assert _t % 8 == 0 and _t >= 100 * _b and _t - 100 * _b < 8, (_b, _t)


def num(x):
    """'7.5' -> 7.5, '15' -> 15"""
    v = float(x)
    return int(v) if v == int(v) else v


def parse_buckets(s):
    return tuple(sorted(num(x) for x in s.split(",") if x))


def graph_path(name):
    for d in (OUT7, OUT5):
        p = f"{d}/{name}.tflite"
        if os.path.exists(p):
            return p
    return f"{OUT1}/{name}.tflite"


def patch_drive():
    """p1_drive with the 7 buckets; its functions read these globals at call time. cmd_run parses --buckets as numbers (p1_drive's int() refuses 7.5)."""
    import p1_drive as DR
    DR.T_MEL.update(T_ALL)
    DR.mel_path = lambda b: graph_path(f"mel{b}")
    DR.enc_path = lambda b, form: graph_path(f"enc{b}_{form}")
    DR.DECS["dec"] = graph_path("dec")
    DR.bk_tag = lambda buckets: ("w30" if tuple(buckets) == (30,) else ("b7" if 7.5 in buckets else ("b5" if 5 in buckets else "bkt")))
    _int = int

    def cmd_run(a):
        sets = a.sets.split(",")
        buckets = parse_buckets(a.buckets)
        shard = tuple(_int(x) for x in a.shard.split("/"))
        outs = {s: DR.out_name(a.outdir, a.form, a.backend, buckets, a.dec, s, shard) for s in sets}
        if all(os.path.exists(o + ".DONE") for o in outs.values()):
            print("all done", flush=True)
            return 0
        os.makedirs(a.outdir, exist_ok=True)
        p = DR.Pipeline(a.form, a.backend, a.threads, buckets, dec=a.dec)
        print(f"loaded {a.form}/{a.backend}/{buckets}", flush=True)
        for s in sets:
            if os.path.exists(outs[s] + ".DONE"):
                continue
            if not DR.run_set(p, a, s, outs[s], shard):
                break
        return 0
    DR.cmd_run = cmd_run
    return DR
