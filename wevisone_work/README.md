# wevisone_work — tencent/WeVisOne-VL-2B → `.litertlm` (fast_vlm, 512 × 512, DeepStack kept exact)

The scripts behind [`litert-community/WeVisOne-VL-2B`](https://huggingface.co/litert-community/WeVisOne-VL-2B). One command: `bash scripts/ship_wevisone.sh` from the repository root (environments in its header). The method: [docs/deepstack-wide-embedding.md](../docs/deepstack-wide-embedding.md); the recipe and the published sections: REPRODUCE.md "WeVisOne-VL-2B".

| Step | Script | Output (under `out/`, not in git) |
|---|---|---|
| test pictures | `fixtures/fetch_public.sh` (the four NASA photos, pinned URLs; `make_synth.py` draws the two synthetic pictures) | `fixtures/img/` |
| text-only checkpoint | `tools/text_ckpt.sh` (`../tools/vlm_text_ckpt/make_text_ckpt.py`; `PROVE=1` adds the logits proof `tools/prove_text_ckpt.py`) | `text_ckpt/` |
| vision tower | `tools/vision_taps.py consts`, `tools/wv_vision_r2.py export` / `quant` (`LN_SCALES=ln_scales_512.json`) | `vision512/`, `r2/vision/` |
| decoder | `tools/wv_export_decoder.py` (`tools/wv_patch.py` + `tools/mrope_derived_wv.py`, `tools/tfshim/` for litert-torch 0.10.0's TensorFlow schema import), `tools/wv_quant_decoder.py fp16` | `r2/dec_fp32/`, `r2/dec_fp16/` |
| file | `tools/build_wv_bundle.py --build`, `../scripts/add_executor_metadata.py`, `tools/wv_bundle_check.py` | `r2/bundle/WeVisOne-VL-2B_f16.litertlm` |

Checks and gates:

- `tools/run_arms.py`: the reference answers of the source package ([Tencent/WeVisOne-VL](https://github.com/Tencent/WeVisOne-VL) at `40f735b8`, installed with `--no-deps`) in float32 on the CPU, at its own resolution and at 512 / 768.
- `tools/wv_suite.py` (+ `tools/wv_rt_api.py`): every question through the litert-lm 0.18.1 CLI and Python API, one process each; `tools/score_arms.py` and `tools/wv_parse.py` (the source package's parser and counting metric) score boxes, counts, points and OCR; `tools/judge_r2.py` applies the thresholds (the original 90 % bar, and twice the reference's own bfloat16 miss where that miss is over 10 %).
- `tools/wv_readout.py`: the graphs driven the way LiteRT-LM drives them, against the reference's teacher-forced logits; `tools/wv_twoturn_sim.py` and `tools/r6_twoturn_probe.py`: the runtime's second-message stop-token defect (docs "A runtime defect").
- `tools/op_table.py`: operator counts per subgraph (GATHER / GATHER_ND / CUSTOM / FLEX must be 0).

`fixtures/fixtures.json` holds the 18 public questions (two synthetic pictures, four NASA photos); the published gates also used 11 questions on crops of the source package's example figures, which are not redistributed. `ln_scales_512.json` holds the vision tower's LayerNorm scales as published, so the export does not need those crops.
