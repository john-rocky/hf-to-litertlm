# Check a demo scene before you film it

A demo clip says a model can do something. Before filming, fix the photos, the requests and the truth regions; score the model in the form the app ships (the picture it sends, the thinking mode, the room the context leaves for the answer); and run every cheaper route at its best: other small VLMs in their own authors' prompt forms, an open-vocabulary detector with a fixed rule, and the detector's boxes handed to a text LLM. Film only requests the model answers and every cheaper route misses. [`tools/demo_scene_check/`](../tools/demo_scene_check/) does each step; [`tools/card_verify.py`](#check-a-cards-numbers) re-derives a card's numbers from the raw records before the card goes up.

## What the worked example found

The scene in the tools is "where to put it": 10 photos of desks, tables and shelves from Wikimedia Commons (9 CC0, 1 public domain), 30 placement requests such as "Put the mug to the right of the books." (8 left or right of something, 6 between two things, 10 in front of or behind something, 6 on a surface), each with a truth region drawn by one rule, and 23 reference objects asked alone ("Point to the books."). The model was LFM3-VL-2B, the app an Android sample that sends the photo stretched to 512 × 512 and draws a ring at the point.

- **The app's form decides.** The source model (float32, Transformers on the Mac GPU) on the 512 × 512 picture answered 6 of 30 with thinking off and 21 with thinking on (9 and 23 on the source picture, long side 1024). With thinking off, 11 of its 24 misses landed on the named object itself instead of the free space next to it. The demo needs thinking on.
- **So does the context.** With thinking on, the int4 file with a 2,048-token context answered 17 of 30: 7 thoughts reached the end of the context and no answer came. The same recipe with a 4,096-token context answered 21 and ran out on none. The source model's output was 668 tokens at the median and 2,332 at the longest.
- **A cheap pipeline can break the claim.** The line was: every other route at its best stays at least 8 below the model's 21, so at most 13. The four other small VLMs we ran (three families, each at its best condition) stayed under it, and so did the detector's boxes handed to a text LLM (4 of 30 at best). The detector + fixed rule answered 14, so the claim narrowed from "only this model can" to "one model answers a free-text request with a point". The rule arm was handed the object names and the relation kind in advance, in its favour; it answered 10 of the 18 left/right and front/behind requests, 2 of the 6 "between" and 2 of the 6 "on a surface".
- **What is left to film.** 8 requests the source model answered on the app's picture with thinking on while the best condition of every other route missed; on 4 of them every condition of every route missed. The 4,096-token file answered 4 of the 8 (2 of those 4), and a clip of that file comes from those.

Re-run on a Mac from this tree, the tools rebuilt the same fixtures (every photo's sha1, every mask's hash) and plans, and scored the recorded answers to the same verdicts, tables and candidate list as the records the example was first scored with; the detector arm, re-run in full, gave the same 51 boxes and the same 30 rule points (14 of 30); a sample of the model runs (8 transformers cases on the CPU and the Mac GPU, 3 `.litertlm` cases on the Mac GPU, one of them at a 2,048-token budget) gave the same outputs as recorded.

## Steps

Packages: numpy and Pillow for the fixtures, scoring and tables (Pillow also for `card_verify.py` when a checks file reads masks); torch and transformers for `run_hf.py` and `detect.py`; `litert-lm` for `run_litertlm.py`. Measured with Python 3.12, numpy 2.5.4, Pillow 13.0.0, torch 2.14.1, transformers 5.21.0, huggingface_hub 2.3.0 and litert-lm 0.19.0 on a Mac.

```bash
T=tools/demo_scene_check; D=out/demo_scene_check/where_to_put
# 1. fixtures: download the photos (sha1 checked), build the 512 x 512 pictures and the truth masks
python $T/fixtures.py write --out $D            # look at $D/geom/ and $D/gt512/ before any model runs
python $T/fixtures.py check --out $D
# 2. plans: every prompt fixed in plan.py, one file per model
python $T/plan.py --fixtures $D/fixtures.json --model lfm3 --out $D/plan/lfm3.json   # S512 / N x thinking off / on
python $T/plan.py --fixtures $D/fixtures.json --model qwen35 --out $D/plan/qwen35.json
python $T/plan.py --fixtures $D/fixtures.json --model gemma4 --conditions N:off,N:on --forms box,point --out $D/plan/gemma4.json
python $T/plan.py --fixtures $D/fixtures.json --model lfm25_3 --prompt lfm25 --conditions N:off --out $D/plan/lfm25_3.json
python $T/plan.py --fixtures $D/fixtures.json --model lfm3 --objects --out $D/plan/lfm3_obj.json
# 3. runs: the source models in transformers, the .litertlm the way the app sends it
HF_HUB_OFFLINE=1 python $T/run_hf.py --model-path <snapshot> --name lfm3 --revision <sha> --trust-remote-code \
    --plan $D/plan/lfm3.json --out $D/raw/lfm3_mps.jsonl --device mps
python $T/run_litertlm.py <model.litertlm> --plan $D/plan/lfm3.json --label int4_gpu_on --backend gpu --think on \
    --ctx 4096 --prompt-tokens-from $D/raw/lfm3_mps.jsonl --out $D/litertlm
# 4. the pipeline arms: detector + fixed rule, and the detector's boxes as text to an LLM
HF_HUB_DISABLE_XET=1 python $T/detect.py fetch --cache $D/hf_cache
HF_HUB_OFFLINE=1 python $T/detect.py run --fixtures $D/fixtures.json --cache $D/hf_cache --out $D/c/detections.json
python $T/crule.py --fixtures $D/fixtures.json --detections $D/c/detections.json --out $D/c/crule.json
python $T/plan.py --fixtures $D/fixtures.json --model qwen35 --cllm $D/c/detections.json --out $D/plan/cllm_qwen35.json
# 5. score, table, read
python $T/score.py --fixtures $D/fixtures.json --subject lfm3 --raw $D/raw/lfm3_mps.jsonl --raw ... \
    --bundle $D/litertlm/int4_gpu_on.json --crule $D/c/crule.json --out $D/score/all.json
python $T/table.py $D/score/all.json --fixtures $D/fixtures.json --subject-label "lfm3 S512 on std" \
    --app-label "bundle int4_gpu_on" --ctx 4096 --out $D/table.md --json $D/table.json
python $T/misses.py $D/score/all.json --json $D/score/misses.json
python $T/read.py $D/score/all.json --fixtures $D/fixtures.json --labels "lfm3 S512 on std,bundle int4_gpu_on" --out $D/read
```

Run the first three cases of each plan on `--device cpu` and on `mps` into two files (`--only`) and compare the generated ids; use the faster device for the rest only when they are equal. Run one model at a time.

A label is `<model> <arm> <think> <form>` (`lfm3 S512 on std`, `gemma4 N on box`, `crule CRULE off rule`, `bundle int4_gpu_on`). Arm `S512` = the 512 × 512 stretch, `N` = the source picture (long side 1024) through the model's own processor, `CLLM` = the detector's boxes as text, `CRULE` = the fixed rule.

## Rules

1. **Score the shipped form before you build the scene.** The picture the app sends, the thinking mode and the room the context leaves for the output change the count: here 6 against 21 for thinking off and on, and 17 against 21 for a 2,048- and a 4,096-token context.
2. **Fix the material before any model answers.** Photos whose licence allows redistribution (CC0 or public domain, the original's sha1 checked against Commons), the requests, the truth regions drawn by one written rule, the prompt texts, the reading rules and the lines. Look at every region over its photo before the first run: on the example a grid read in pixels instead of 0–1000 units put a bed 100–200 units off, caught on the overlay. Before you fix a line, compute its odds from the author's published scores; a line can sit above what those scores make likely.
3. **Run each rival at its best.** Its author's prompt form, thinking off and on, the source picture and the app's; read its answer in its own format and in its favour (a 0–1 scale read × 1000, x and y swapped, pixel coordinates divided by the image size); its best condition counts. Hand the model's wins to the rivals first: one rival PASS closes the task, one seed's FAIL does not.
4. **Add the pipeline arms.** An open-vocabulary detector with a fixed rule, and the detector's boxes as text to an LLM. When one of them breaks the line, narrow the claim to what that arm cannot do.
5. **Read every answer.** `read.py` writes each answer next to its verdict and marks the ones the rule cannot settle (several points, no point in a non-empty answer, a point outside 0–1000). Judge right and wrong from the score file, not from a scaled-down picture.
6. **Count a thought that never closed as no answer, and read the ones that never opened apart.** A model asked to think can answer without a thought; the rule counts that as no answer, so count those answers separately when they change the best condition.
7. **Keep the whole set.** A clip shows one request; the count over the set is what it stands for.
8. **For a model that hears, add a transcript arm and an audio arm.** A speech recognizer whose transcript goes to a text model of the same kind (here the model family's text-only sibling), and another small model that takes the audio itself. Include recognizers that run on the phone, not only a large desktop one, and get a phone timing for every arm that matches the answers (measured, or from its model card): a speed claim needs all of them. In twelve scenes for an audio decision model, wherever it answered from what was said, a desktop recognizer's transcript did too; through a phone line, so did three phone recognizers (the smallest on two of three clips in one scene) and an audio LLM given the clip. With a background voice as loud as the caller or 3 dB louder, the model that hears got no more clips right than any transcript arm: 3 of 6 against 3 to 5 of 6, and through a phone line 1 of 6 against 2 to 5 of 6 (the audio LLM 0 of 6). The non-speech sounds it was asked about (how many beeps, a clip compared with stored ones, running against dripping water) it missed itself.

## Voice-cloning scenes (text to speech)

The same check applies to a text-to-speech model that clones a voice from a short recording. A search for a scene that only Audio8-TTS-Realistic-0.6b (23 languages) could make found none; three of the reasons carry over to any voice-cloning scene:

- **A rival's card is not its ability.** The older Audio8-TTS-Preview-0.6b, whose card lists no Swedish, run as its original weights from the same reference clip, read both Swedish scene sentences with 0 or 1 word errors (one sentence in 2 of 3 runs, the other in 3 of 3), and Qwen3-TTS-12Hz-0.6B-Base made one of them with its language set to German (a rival makes a clip when any one of its runs has at most one word error and a speaker cosine of at least 0.48). Run each rival on the scene's own sentences and reference, the cheapest first, and score it before any other check.
- **The clone copies the recording.** The FLEURS references we used (UTMOS22 2.13 to 3.19) gave outputs with every word right that still stayed under a 3.70 UTMOS22 line (Turkish 9 of 9, Arabic 4 of 6, Hindi 1 of 4); LibriSpeech references score 4.06 to 4.44. Score the reference clip first, and count the clips in the scene's language that pass a quality line (we used UTMOS22 ≥ 3.80), a verbatim transcript (the ASR reads the clip's text with no difference) and a licence that allows cloning, before you write the scene. For Turkish, Vietnamese and Arabic we found none in the open sets we tried.
- **Use a reference in the language it will speak.** With an English reference, Audio8-TTS-Realistic-0.6b broke words in 9 of the 10 other languages we screened (Arabic, Finnish, Hindi, Indonesian, Malay, Norwegian, Swedish, Thai, Turkish, Vietnamese; only Malay held), and its original weights did the same (runs with at most one word error: 9 of 30 for the LiteRT files, 13 of 30 for the original weights, one seed each), so the breakage comes with the model, not with the conversion; we did not measure why.

## Check a card's numbers

`tools/card_verify.py` checks every number of a model card draft against the raw run records, with code that shares nothing with the scorer, the card writer or the table writer. A checks file re-derives each value from the records and names the exact text the card must contain; the verifier reports each check found or not.

```bash
python tools/card_verify.py --card README.md --checks lfm3vl_work/card_checks_int4_ctx4096.py \
    --sources sources.json --opt phones --out verify.json
```

`sources.json` maps the checks file's source names (its `SOURCES`) to paths: the uploaded file, the Mac timing records, the fixtures, the placement runs, the section comparison, the phone session directories of [tools/devicebench](../tools/devicebench/). Exit 0 = every check found, 1 = a check failed (the line names the check and the text it wanted), 2 = the card still has an open `{{SLOT}}`, 3 = a source is missing.

The example holds the 51 checks of the [LFM3-VL-2B card](https://huggingface.co/litert-community/LFM3-VL-2B-LiteRT) after the 4,096-token file was added: the file's size and sha256, the Mac rows (medians taken from the per-run records), the placement table (each answer scored here against its truth mask), the weights comparison, and the Galaxy S26 and Pixel 11 Pro sessions (each leg's LiteRT-LM BenchmarkInfo and reply). The published card passes 51 of 51. On its first run the verifier counted the source model at 20 of 30, not 21: the source model's text holds its thought, and the first `point_2d` it found was inside the thought. Reading the answer after the last `</think>`, as the scoring rule does, gave 21. The verifier was wrong, not the card; read each record the way its own rule reads it.

Write the checks before you read the draft's numbers. A failed check is a number to fix in the card, or a record to read again, never a check to loosen.
