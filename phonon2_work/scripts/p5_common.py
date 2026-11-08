"""The 5 s bucket's paths on top of the scripts of the 10 / 20 / 30 s files (cwd = phonon2_work; those scripts are not edited). The 5 s bucket = 504 mel frames (the encoder
needs t_mel % 8 == 0: 63 encoder frames, audio input 80,640 samples; the host rule stays p1_drive.py's: the smallest bucket whose audio input holds the window, so
n <= 80,640 goes to it). Graphs: out/r5 first (mel5, enc5_*), then out/r1 (the 10 / 20 / 30 s files and the decoders)."""
import os
import sys

K = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(K, "scripts"))
T5 = 504
OUT5 = "out/r5"
OUT1 = "out/r1"


def graph_path(name):
    p = f"{OUT5}/{name}.tflite"
    return p if os.path.exists(p) else f"{OUT1}/{name}.tflite"


def patch_drive():
    """p1_drive with the 5 s bucket; its functions read these globals at call time"""
    import p1_drive as DR
    DR.T_MEL[5] = T5
    DR.mel_path = lambda b: graph_path(f"mel{b}")
    DR.enc_path = lambda b, form: graph_path(f"enc{b}_{form}")
    DR.DECS["dec"] = graph_path("dec")
    DR.bk_tag = lambda buckets: "w30" if tuple(buckets) == (30,) else ("b5" if 5 in buckets else "bkt")
    return DR
