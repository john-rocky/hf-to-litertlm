#!/usr/bin/env python3
"""export_audio.py HF_DIR OUT.tflite - LFM3-Omni-2B's audio encoder and projector as the .litertlm's audio section, in
float32, through LFM3-Audio-2B's section (lfm3audio_work/scripts/r1a_audio_section.py) with Omni's tensor names
(`OMNI_KEYMAP`): 16 kHz PCM in 160-sample frames, up to 3,000 of them (30 s) per call, the log-mel spectrogram computed
in the graph with the settings of HF_DIR/processor_config.json, the downsampler, the 17-layer encoder and the projector
to 2048; out = up to 375 rows (one per 80 ms) plus a uint8 mask of the rows that are audio.

r1a_audio_section.py is loaded by path under the module name `lfm3_peer_r1a_audio_section`, which appears in the
graph's tensor names. HF_DIR = the source checkout (config.json, processor_config.json, model.safetensors).
"""
import importlib.util
import pathlib
import sys
import time

R = pathlib.Path(__file__).resolve().parents[1]


def load_by_path(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def main():
  if len(sys.argv) != 3:
    print(__doc__, file=sys.stderr)
    return 2
  hf, out = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve()
  S = load_by_path("lfm3_peer_r1a_audio_section", R / "lfm3audio_work" / "scripts" / "r1a_audio_section.py")
  t0 = time.time()
  sec, _spec, n = S.build_section(str(hf), keymap=S.OMNI_KEYMAP)
  out.parent.mkdir(parents=True, exist_ok=True)
  tmp = out.with_name(out.name + ".part")
  S.export_section(sec, str(tmp))
  tmp.rename(out)
  print(f"exported {out} ({out.stat().st_size} B, {n} tensors) in {time.time() - t0:.0f} s", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
