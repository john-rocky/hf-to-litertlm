# Which layers can go int8: a teacher-forced sensitivity table

A decoder that passes in fp16 and fails in int8 leaves a choice of which fully connected layers to keep wide. Building and gating a whole file per guess costs an export, a bundle and a gate run each. `tools/quant_sensitivity/` measures one group of layers at a time on the CPU instead: quantize only that group to int8 (the rest fp16), feed every case the same tokens (teacher forcing), and see which answers move and by how much at the positions that decide them. The table picks candidates; the candidates still go through the gate on the runtime. Worked example: the text decoder of LiquidAI/LFM3-VL-2B, whose published f16 file came out of this table ([litert-community/LFM3-VL-2B-LiteRT](https://huggingface.co/litert-community/LFM3-VL-2B-LiteRT), the REPRODUCE.md entry "LFM3-VL-2B").

## Steps

1. **A reference form and teacher tokens.** The reference is the form that passes (here every fully connected weight fp16 with an fp16 embedding table). The teacher tokens are the source model's own greedy output on the test cases (here 23 picture questions × thinking off and on, the first 32 tokens of Transformers in float32, `lfm3vl_work/tools/hf_ref.py`). A case where the source model's float32 and bfloat16 disagree says nothing about quantization; mark the others `counted`.
2. **Check the tool on the reference.** Teacher-forced, the reference graph must give the teacher token as its top-1 at every position. LFM3-VL-2B's fp16 graph did at 886 of 886 positions (150 with thinking off, 736 with it on).
3. **One group at a time.** `fc_groups.py list` classifies every FULLY_CONNECTED op of every signature by the Linear it comes from (rules in a small JSON per model) and stops if one weight would need two configs; `recipe` writes, per group, a regex of the exact quantizer scopes of that group's ops. Match exact scopes, not name fragments: the converter fuses names, so one signature's v_proj scope starts with v_proj and another's with self_attn, w3's scope carries w1 after it, and the prefill signatures have no ops for the last layer's output projection and MLP (nothing reads them without logits). LFM3-VL-2B's decoder has 1,048 fully connected ops over 7 signatures and 154 weights.
4. **Read the positions that decide answers.** For each group, `teacher_force.py --ref` records, at every position, whether the reference's top-1 holds, and the change of the margin between the reference's top-1 and top-2 tokens. `sens_table.py` reports the standard deviation of that change over every position (sigma) and its rms only at the positions that can decide a counted case: a reference gap under 0.5, up to the case's first change (near rms). Use near rms: over every position lm_head moved the margin by 0.047 and changed 2 answers, the convolution output projections by 0.139 and changed none.
5. **Measure combinations; do not add the table.** On the CPU the error of two groups is not the sum of their errors position by position (the correlation was 0.47 over 886 positions; two runs of one graph correlated at 0.96), although the variances add (with thinking off, the variances of the six groups and of the embedding table summed to a sigma of 0.467; every fully connected op plus the embedding table in int8 measured 0.470). LFM3-VL-2B: the convolution input projections, the convolution output projections and every op of layers 0 to 7 each kept all 23 thinking-off answers alone. The two convolution groups together changed one (reference gap 0.086); with layers 0 to 7 added, the changed answer was a different one, at a position with a gap of 1.192. Either combination left no answer to spare against the pass mark the file had to meet (21 of 23 thinking-off answers equal to the bfloat16 model's).
6. **Screen close cases first.** The cases with the smallest reference gaps decide (here five thinking-off positions with gaps of 0.027 to 0.089). Run a new form on those first, a few minutes each, and the full set only if they hold.

## LFM3-VL-2B (Mac CPU, LiteRT Interpreter with XNNPACK, against the fp16 graph)

| int8 group | answers kept, thinking off / on (of 23) | near rms, off / on | thinking-off answers that moved (reference gap) |
|---|---|---|---|
| embedding table only | 23 / 22 | 0.011 / 0.010 | none (thinking on: one, gap 0.001) |
| lm_head | 21 / 23 | 0.076 / 0.033 | two boxes (0.027, 0.089) |
| convolution input projections (13) | 23 / 22 | 0.088 / 0.030 | none |
| convolution output projections (13) | 23 / 23 | 0.129 / 0.064 | none |
| attention q, k, v, gate, o (11 layers) | 21 / 23 | 0.152 / 0.072 | a box (0.027), a description (0.086) |
| MLP w1, w3 (24 layers) | 21 / 22 | 0.098 / 0.077 | two boxes (0.027, 0.051) |
| MLP w2 (24 layers) | 20 / 19 | 0.301 / 0.129 | three boxes |
| every op of layers 0–7 | 23 / 20 | 0.121 / 0.067 | none |
| every op of layers 8–15 | 21 / 22 | 0.216 / 0.089 | a box (0.027), a description (0.244) |
| every op of layers 16–23 | 20 / 22 | 0.141 / 0.074 | a description (0.113), a sign (0.066), a box (0.089) |
| every op + embedding table (all int8) | 20 / 17 | 0.178 / 0.110 | a box (0.027), a description (0.086), a box (0.089) |

The embedding table alone moved the near margins less than any group of fully connected layers did: 0.011 with thinking off against 0.076 or more, 0.010 with thinking on against 0.030 or more. Int8 on the embedding table alone, the decoder kept fp16, became `LFM3-VL-2B_f16.litertlm`: 260 MB smaller than the all-fp16 file, the same answers on the Mac GPU, and on the CPU one answer different at the position the table predicted (thinking on, position 22, gap 0.001). Two candidates with more int8 (the convolution projections; those plus every op of layers 0–7, the fastest of the three candidates on the Mac GPU) passed the Mac gate with no answer to spare on at least one backend, and the second changed a description word on a Pixel 8a GPU (LiteRT-LM 0.18.1 on the phone), so the file keeps only the embedding table in int8.

The CPU's dynamic int8 (XNNPACK) gives different results from one LiteRT build to another: on the all-int8 graph the interpreter here and the LiteRT-LM 0.18.1 runtime changed the same 3 thinking-off answers at the same positions, but only 3 of the 6 thinking-on answers the interpreter changed were the same. With no int8 fully connected layer (the fp16 decoder with the int8 embedding table) the two agreed: the one CPU answer that the table changed moved at the same position in both. One build also varies from one process to the next on a dynamic-int8 graph. Two runs of the all-int8 graph gave a sigma of 0.470 and 0.443 with thinking off and changed the same nine answers, each from the same position; a second run of the convolution output projections gave a near rms of 0.120 / 0.070 against the table's 0.129 / 0.064 and again changed none. Read sigma and near rms to about 10 % and the changed answers exactly. The GPU quantizes weights only. Read the table as the CPU's picture and gate each candidate on every backend it ships for.

## Commands (the LFM3-VL-2B example)

`OUT` = the output of `lfm3vl_work/convert_lfm3_vl.sh` with `FORMS=f16` after the steps `fetch template vision float quant` (`OUT/dec/float/` holds the float32 graphs). `REF` = `hf_ref.py --arms B512 --dtype fp32` (and `--dtype bf16` for the `counted` marks).

```bash
C=lfm3vl_work/quant_sens_classes.json; T=tools/quant_sensitivity
python $T/fc_groups.py quantize --src OUT/dec/float/embedder.tflite --recipe lfm3vl_work/recipe_emb_f16.json --out OUT/dec/emb_f16.tflite
python lfm3vl_work/quant_sens_cases.py --teacher REF/hf_B512_fp32.json --bf16 REF/hf_B512_bf16.json --src OUT/src --tower-dir OUT/tower --out-dir TF
python $T/fc_groups.py list --float OUT/dec/float/model.tflite --classes $C --out TF/fc_ops.json
python $T/teacher_force.py --decoder OUT/dec/f16.tflite --embedder OUT/dec/emb_f16.tflite --cases TF/cases.json --label ref --save-ref --out-dir TF
python $T/teacher_force.py --decoder OUT/dec/f16.tflite --embedder OUT/dec/emb_wi8.tflite --cases TF/cases.json --label emb --ref ref --out-dir TF
mkdir -p TF/recipes
python $T/fc_groups.py recipe --ops TF/fc_ops.json --classes $C --int8 conv_out --out TF/recipes/conv_out.json
python $T/fc_groups.py build --float OUT/dec/float/model.tflite --recipe TF/recipes/conv_out.json --ops TF/fc_ops.json --classes $C --out TF/conv_out.tflite --record TF/recipes/conv_out.build.json
python $T/teacher_force.py --decoder TF/conv_out.tflite --embedder OUT/dec/emb_f16.tflite --cases TF/cases.json --label conv_out --ref ref --out-dir TF
python $T/sens_table.py --dir TF --ref ref --recipes TF/recipes --out TF/sens.md
```

Groups for `--int8`: the names under `groups` in the classes file, `L<a>-<b>` (every op of layers a to b) and `<group>@L<a>-<b>`. One teacher-forced run of the fp16 graph took 2 to 5 minutes on an M4 Max (the longer with other jobs loading the machine) and peaked at 49 GB resident; a group build took 28 to 77 seconds.

Related: [int8-image-rows.md](int8-image-rows.md) finds the cause of a dynamic-int8 error at picture positions by emulating XNNPACK's int8 product in PyTorch, group by group, without exporting.
