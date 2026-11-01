#!/usr/bin/env python3
"""export_speech.py --part depth|frame|codec12|codec1 --hf HF_DIR --out OUT.tflite [--window OUT.npy] - LFM3-Omni-2B's
speech-out graphs in float32 through LFM3-Audio-2B's speech graph code (lfm3audio_work/scripts/r1b_graphs.py), whose
keymap "hf" holds Omni's tensor names and whose depth-decoder geometry equals Omni's depth_config (checked here):
  depth    one depth-decoder pass (`depth_step`: hidden, codes, masked -> logits for the 12 codebooks; 3 passes per
           speech frame), from model.depth_decoder.* (its depth_linear, codebook embeddings and 3 layers)
  frame    the frame embedding (`frame_embed`: the 12 codes of a frame -> the sum of their codebook embeddings, the next
           decoder input), from model.codebook_embedding.weight
  codec12  the codec decoder for 12 frames (0.96 s) per call, with its streaming state as inputs and outputs, from
           audio_tokenizer/model.safetensors (the same bytes as LFM3-Audio-2B's codec); --window also writes the
           codec's overlap-add window
  codec1   the same codec decoder for 1 frame per call
One part per process. The module name the graph code is loaded under appears in the graphs' tensor names:
`lfm3_peer_r1b_graphs` for depth and frame, `r1b_graphs` for the codec.
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import time

R = pathlib.Path(__file__).resolve().parents[1]
MODULE_NAME = {"depth": "lfm3_peer_r1b_graphs", "frame": "lfm3_peer_r1b_graphs", "codec12": "r1b_graphs",
               "codec1": "r1b_graphs"}


def load_by_path(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--part", choices=sorted(MODULE_NAME), required=True)
  ap.add_argument("--hf", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--window", default="")
  a = ap.parse_args()
  hf, out = pathlib.Path(a.hf).resolve(), pathlib.Path(a.out).resolve()
  G = load_by_path(MODULE_NAME[a.part], R / "lfm3audio_work" / "scripts" / "r1b_graphs.py")
  cfg = json.load(open(hf / "config.json"))["depth_config"]
  want = dict(layers=cfg["num_hidden_layers"], heads=cfg["num_attention_heads"], kv_heads=cfg["num_key_value_heads"],
              head_dim=cfg["head_dim"], eps=cfg["norm_eps"],
              theta=cfg["rope_parameters"]["full_attention"]["rope_theta"], positions=cfg["max_position_embeddings"])
  assert {k: G.DEPTH_CFG[k] for k in want} == want, (G.DEPTH_CFG, want)
  assert G.KEYMAPS["hf"]["depth"] == "model.depth_decoder.", G.KEYMAPS["hf"]
  out.parent.mkdir(parents=True, exist_ok=True)
  tmp = out.with_name(out.name + ".part")
  t0 = time.time()
  if a.part == "depth":
    G.export_depth_step(G.load_depth_state_dict(str(hf / "model.safetensors"), "hf"), str(tmp), "hf")
  elif a.part == "frame":
    G.export_frame_embed(G.load_frame_state_dict(str(hf / "model.safetensors"), "hf"), str(tmp), "hf")
  else:
    csd = G.load_codec_state_dict(hf / "audio_tokenizer" / "model.safetensors")
    G.export_codec_chunk(csd, int(a.part[5:]), tmp)
    if a.window:
      import numpy as np  # pylint: disable=g-import-not-at-top
      np.save(a.window, csd["decoder.istft.window"].numpy().astype(np.float32))
      print(f"window {a.window} ({pathlib.Path(a.window).stat().st_size} B)", flush=True)
  tmp.rename(out)
  print(f"exported {a.part} {out} ({out.stat().st_size} B) in {time.time() - t0:.0f} s", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
