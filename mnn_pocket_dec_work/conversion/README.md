# Rebuild the LiteRT files

These scripts rebuild the `.tflite` files of this repository from the files of [`taobao-mnn/MNN-Pocket-0.3B-Dec`](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-Dec). The commands below run from the root of a copy of this repository, with `python3.12` and `python3.13` on the path. An export is the largest step, at about 7 GB of memory on a Mac. Step 4 removes each float32 graph when its float16 file is written, so the 20 files take about 10 GB of disk.

## Sources and environments

```bash
python3.12 -m venv venv-reference && venv-reference/bin/pip install -r conversion/requirements-reference.txt
python3.13 -m venv venv-export && venv-export/bin/pip install -r conversion/requirements-export.txt

venv-export/bin/hf download taobao-mnn/MNN-Pocket-0.3B-Dec --revision dfbe0aed3abd4caed175b883a4b462ed35184fad --local-dir src/MNN-Pocket-0.3B-Dec
SRC=src/MNN-Pocket-0.3B-Dec
```

- **Reference** (`requirements-reference.txt`): the versions in the original repository's `requirements.txt` (torch 2.8.0, transformers 4.57.1, peft 0.19.1, safetensors 0.7.0, accelerate 1.14.0, numpy 1.26.4) with their dependencies. It runs the fold of the LoRA adapter and the original `inference.py`.
- **Export** (`requirements-export.txt`): torch 2.14.1, litert-torch 0.10.0, litert-converter 0.5.0, ai-edge-litert 2.3.0, ai-edge-quantizer 0.10.1, numpy 2.5.4, tokenizers 0.23.3, safetensors 0.8.0.

## Commands

```bash
# 1. The files the host reads: copies of the original files and the embedding table (standard library only).
python3 conversion/bundle_files.py --src $SRC --out .

# 2. The original inference.py on the example requests, compared with examples/*.expected.json.
venv-reference/bin/python conversion/reference_examples.py --src $SRC --examples examples

# 3. The LoRA adapter folded into the backbone in float32.
venv-reference/bin/python conversion/merge_lora.py --src $SRC --out work/merged_fp32.safetensors

# 4. Each form at each length: export, store the FULLY_CONNECTED weights as float16, remove the float32 graph.
one_file() {  # one_file <form> <L> <file name without .tflite>
  PYTHONHASHSEED=0 venv-export/bin/python conversion/export_dec.py --weights work/merged_fp32.safetensors --L "$2" --form "$1" --out "work/$3_w32.tflite"
  venv-export/bin/python conversion/quantize_dec.py --src "work/$3_w32.tflite" --out "work/$3.tflite"
  rm "work/$3_w32.tflite"
}
for L in 64 96 128 256 512 768 1024 2048; do
  one_file k4 $L mnn-pocket-0.3b-dec_L${L}_fp16w
  one_file k4_exp_silu $L mnn-pocket-0.3b-dec_L${L}_fp16w_npu
done
for L in 64 96 128 256; do
  one_file mean_norm $L mnn-pocket-0.3b-dec_L${L}_fp16w_tensor
done

# 5. Compare the 20 files with the published ones.
(cd work && grep -E 'fp16w(_npu|_tensor)?\.tflite$' ../SHA256SUMS | shasum -a 256 -c -)
```

Before it exports, `export_dec.py` runs the PyTorch module on the short example requests that fit the length (`readme_request` and `default_request`; the L64 file runs the first only) through the host's inputs, head and answers. It stops unless every winner equals `examples/*.expected.json` and every probability is within 1e-4.

## The files compiled for the Qualcomm SM8850 NPU

The `*_SM8850.tflite` files hold a context binary compiled ahead of time from the `*_npu.tflite` file of the same length. This step needs Docker and QAIRT 2.50.0.260828. QAIRT is Qualcomm's SDK under Qualcomm's license and is not in this repository: download `v2.50.0.260828.zip` from [Qualcomm](https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.50.0.260828/v2.50.0.260828.zip) (2,601,473,189 bytes, SHA-256 `a346ea0e2c8631b46d57261a4969994cd9cc34124a8355bbc7b08b2c8bd859a5`, the version LiteRT 2.3.0 is built against) to the root of this copy.

```bash
# 6. The parts of QAIRT the compile reads, then the image.
unzip -q v2.50.0.260828.zip 'qairt/2.50.0.260828/*.txt' 'qairt/2.50.0.260828/sdk.yaml' 'qairt/2.50.0.260828/LICENSE.pdf' 'qairt/2.50.0.260828/lib/x86_64-linux-clang/*' 'qairt/2.50.0.260828/bin/x86_64-linux-clang/*' 'qairt/2.50.0.260828/include/*' 'qairt/2.50.0.260828/share/*' -d conversion/npu
mv conversion/npu/qairt/2.50.0.260828 conversion/npu/qairt_data && rmdir conversion/npu/qairt
docker build --platform linux/amd64 -t npu-aot-qc:2.3.0-q250 conversion/npu

# 7. One container per file: compile, read the log, name the output.
mkdir -p work/npu
for L in 64 96 128 256 512 768 1024 2048; do
  docker run --rm --platform linux/amd64 --cpus 4 --memory 28g --memory-swap 28g -v "$PWD/work":/m:ro -v "$PWD/conversion/npu":/s:ro -v "$PWD/work/npu":/w npu-aot-qc:2.3.0-q250 python /s/aot_compile_sm8850.py /m/mnn-pocket-0.3b-dec_L${L}_fp16w_npu.tflite /w/L$L qualcomm_log_level=verbose
  python3 conversion/npu/aot_check.py work/npu/L$L
  cp work/npu/L$L/*_apply_plugin.tflite work/mnn-pocket-0.3b-dec_L${L}_fp16w_npu_SM8850.tflite
done

# 8. Compare with the published files (below: the bits the compiler leaves unwritten can differ).
for L in 64 96 128 256 512 768 1024 2048; do
  venv-export/bin/python conversion/npu/aot_compare.py work/mnn-pocket-0.3b-dec_L${L}_fp16w_npu_SM8850.tflite mnn-pocket-0.3b-dec_L${L}_fp16w_npu_SM8850.tflite
done
```

The compile writes the same bytes each time it runs on the same file under the same name, but the file name changes a few thousand bytes of the output. The context binary stores some 16-bit values in 32-bit words, and the compiler does not set the upper 16 bits of those words, so they keep what was in its memory, and that changes with the input's file name. The published files were compiled from the same graphs under other file names. Compiled with the commands above, the L64 file differs from the published one in 6,229 bytes, all of them in such upper bits; given the file name of its first compile, the same command writes the published bytes. `aot_compare.py` passes when two files differ in nothing else: the same size, the same bytes before the context binary (the flatbuffer with the DISPATCH_OP and its options), and every differing 32-bit word with the same lower 16 bits.

`aot_check.py` reads the plugin's log: exit code 0, every op in one partition, a context binary, an output file. QAIRT 2.50's log prints no memory estimate for the NPU's session, so whether a context fits is decided by its first load and first run on the phone. On the Mac the eight compiles took 12 s (L64) to 12 minutes (L2048), with a container's peak memory from 4 GB to 14 GB.

| Step | Script | Environment | What it reads | What it writes |
|---|---|---|---|---|
| 1 | `bundle_files.py` | any Python 3 | the original repository | `tokenizer.json`, `decision_head.safetensors`, `temperature.json`, `LICENSE`, `embedding.safetensors` |
| 2 | `reference_examples.py` | reference | the original repository, `examples/*_request.json` | nothing (`--write` writes a missing `*.expected.json`) |
| 3 | `merge_lora.py` | reference | `base/model.safetensors`, `adapter/` | the fp32 backbone, 1,211,358,392 bytes, SHA-256 `a2f58515fc25ceb1974cf95452f8cd85c72efbbbb550354db1a4118fc2ed38d3` |
| 4 | `export_dec.py` (with `dec_graph.py`, `tflite_scan.py`) | export | the fp32 backbone, the host, the example requests | the float32 graph, about 977 MB, and `<out>.export.json` |
| 4 | `quantize_dec.py` (with `quant_scan.py`) | export | the float32 graph | the published form, about 489 MB, and `<out>.quant.json` |
| 6 | `npu/Dockerfile` | Docker, linux/amd64 | QAIRT, the pinned packages | the image: ai-edge-litert 2.3.0 and its compiler plugin with QAIRT 2.50.0 |
| 7 | `npu/aot_compile_sm8850.py`, `npu/aot_check.py` | the image; any Python 3 | a `*_npu.tflite` file | the compiled file, 494 MB to 606 MB, `aot.json`, `apply_plugin.log`, `aot_check.json` |
| 8 | `npu/aot_compare.py` | export | a compiled file and the published one | nothing (it prints `same bytes`, `same apart from unwritten bits` or `different`) |

## The forms

`dec_graph.py` writes the decoder out in plain PyTorch, with the computation of the Qwen3 layers of transformers 4.57.1. The graph takes the embeddings, the rotary cos and sin, and the additive attention mask as inputs, because the original model computes each question branch from the state with positions and a mask that depend on the request. It returns the output of the final RMSNorm at every position.

`k4` (the CPU and GPU files) computes the RMSNorm of the input and post-attention norms of layers 6 to 27 and of the final norm on `x * 2^-4` with its epsilon times `4^-4`. In float32 this gives the same result, because the scale cancels exactly. It keeps the squares in those norms inside the float16 range on a GPU that computes in 16 bits: the hidden states reach an absolute value of about 2,000 in those layers, whose square does not fit in float16.

`k4_exp_silu` (the files for the Qualcomm NPU) is `k4` with SiLU written as `h * exp(min(h, 0)) / (1 + exp(-|h|))`, which uses no LOGISTIC op and gives every EXP an argument of at most 0. The Galaxy S26's NPU runs this form as one partition.

`mean_norm` (the files for the Google Tensor NPU) computes every RMSNorm on the row divided by 4 times its mean absolute value, `y = x / (4 mean|x|)`, with the epsilon divided by the square of the same number. In exact arithmetic it is the same formula. In these checks the Pixel 11 Pro's NPU computed in float16 and results below `2^-14` became zero (the module dump that the phone's NPU service printed while compiling a test graph shows float16 activations, and a simulation of a norm with that flush matches the phone), so the squares of small values were lost and a norm's gain moved; on this form they stay at most 1e-3 of the sum. The factor 4 keeps the largest row inside float16: at width 896, `|y|` is at most 224 and the sum of `y^2` at most 50,176.

The recorder in `dec_graph.py` (`set_recorder`) is the scan that measured those ranges.
