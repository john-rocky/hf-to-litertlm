# card_facts — check every number of a model card against the files it came from

`facts_check.py` reads a table with one row per number on a card (`card/facts.tsv`), finds each row's quote in the card (`card/README.md`), reads the value again from the files the row names (JSON, logs, TSV) and prints PASS or FAIL for each row. It shares no code with whatever wrote the card or the result files, so a number copied wrong, rounded wrong or taken from the wrong run fails. Python 3, standard library only. [`tools/card_verify.py`](../card_verify.py) does the same job with a checks file written in Python; this tool takes a table and no code. The card of [litert-community/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/litert-community/nemotron-3.5-asr-streaming-0.6b) was checked this way: 283 rows, each matching its files.

## Run

```bash
cd <the directory the source paths are relative to>
python3 <this repo>/tools/card_facts/facts_check.py [card/README.md] [card/facts.tsv]
```

It prints one line per row, `PASS n 'printed'` or `FAIL n 'printed' — why` (the quote is not in the card, the printed value is not in the quote, the files give another value, or a source cannot be read), then `facts_check: P / N PASS`. Exit 0 when every row passes, 1 otherwise.

## The table

Tab-separated, with a header line; one row per number; lines that start with `#` are skipped.

| column | what it holds |
|---|---|
| `n` | the row's number |
| `card_quote` | text the card must contain verbatim (a sentence part, a table row) |
| `printed` | the number as the card prints it; it must appear inside `card_quote` |
| `sources` | one or more `path#key` items joined by ` ; `; each path is relative to the current directory, a `.gz` file is read as text |
| `op` | how the values become the printed number (below) |
| `fmt` | a Python format spec for the result (`.3f`, `,d`, …; `minmax`, `minmax_k`, `minmax_of_max`, `minmax_of_max_gb` and `join` take `SPEC\|SEP`); a trailing `×` is appended as text. The formatted result must equal `printed` |
| `note` | free text, not read |

| key | the values it gives |
|---|---|
| `j:a.b.0` | the value at a JSON path (list items by index) |
| `re:REGEX` | every match of the regex's group 1 (multi-line mode) |
| `re1:REGEX` | the first match only |
| `lit:TEXT` | TEXT, if the file contains it; nothing otherwise |
| `tsv:COL=VAL:OUT` | column OUT of the TSV row whose column COL is VAL |

A value that reads as a number once its commas are removed becomes that number. The values of all the sources of a row are taken in order.

## Ops (32)

| op | result |
|---|---|
| `one` | the single value |
| `str` | the single value as text (`fmt` not used) |
| `min`, `max`, `sum` | the smallest, the largest, the sum |
| `minmax` | the smallest and the largest, joined by SEP |
| `mean` | the mean |
| `inv` | 1 / the single value |
| `inv_mean` | 1 / the mean (a rate averaged over runs) |
| `ratio` | v0 / v1 |
| `diff` | v0 − v1 |
| `k` | the single value / 1000 (ms → s) |
| `minmax_k` | the smallest and the largest, each / 1000 |
| `div16000` | the single value / 16000 (samples → seconds at 16 kHz) |
| `minghz` | the smallest value / 1e6 (kHz → GHz) |
| `kb2gb` | the single value in kB as decimal GB (× 1024 / 1e9) |
| `max_kb2gb` | the largest value in kB as decimal GB |
| `kb2gb_diff` | v0 − v1 in kB as decimal GB |
| `sumgb` | the sum / 1e9 (bytes → decimal GB) |
| `minmax_of_max` | each source's largest value, then the smallest and the largest of those |
| `minmax_of_max_gb` | the same, in kB as decimal GB |
| `maxdelta` | values in pairs (device, reference, device, reference, …): the largest device − reference |
| `maxdelta_ceil2` | the same, rounded up to 0.01 |
| `rtf_mean` | n run times in ms, then the audio length in s: mean(times) / 1000 / length |
| `inv_rtf_mean` | 1 / that |
| `pct_vs_first` | n times, n more times, then one value that is not used: (mean of the second n / mean of the first n − 1) × 100 |
| `ncsv` | the number of comma-separated items in the single value |
| `word` | the single value as an English word (one to ten) |
| `join` | every value formatted with SPEC, joined by SEP |
| `present` | `fmt` itself when every source matched, `absent` otherwise (write the claim's wording in both `fmt` and `printed`) |
| `equal` | `fmt` itself when every source gave one value, all are equal, and that value is the printed number (as text, or as a number once its commas are removed); `agree on V, not on 'P'` or `differ` otherwise |
| `agree` | `fmt` itself when every source gave one value and all are equal, `differ` otherwise: for a wording the agreement carries (`the same bytes`; write it in both `fmt` and `printed`). This is what `equal` did before it compared the value with `printed`, so a table written for that `equal` needs `agree` on its wording rows |

## Example

A card, three result files and the table (the table's columns are separated by tabs):

```text
card/README.md     On a Pixel 8a the median step took 41.7 ms (the mean of two runs) over 1,200 steps per run.
                   The file is 1.19 GB.
runs/a1.json       {"p50_ms": 41.52}
runs/a2.json       {"p50_ms": 41.88}
runs/a1.log        done steps=1200 wall_ms=50112
files.tsv          name	bytes
                   model.tflite	1191615072
```

```text
n	card_quote	printed	sources	op	fmt	note
1	the median step took 41.7 ms (the mean of two runs)	41.7	runs/a1.json#j:p50_ms ; runs/a2.json#j:p50_ms	mean	.1f	two runs of one setting
2	over 1,200 steps per run	1,200	runs/a1.log#re1:steps=(\d+)	one	,d	the runner's last line
3	The file is 1.19 GB.	1.19	files.tsv#tsv:name=model.tflite:bytes	sumgb	.2f	bytes / 1e9
```

```console
$ python3 <this repo>/tools/card_facts/facts_check.py
PASS 1 '41.7'
PASS 2 '1,200'
PASS 3 '1.19'
facts_check: 3 / 3 PASS
```

Then break it on purpose, once on each side. With 41.8 in the card instead of 41.7, row 1 prints `FAIL 1 '41.7' — quote not in the card`; with the card restored and `runs/a2.json` at 42.08, it prints `FAIL 1 '41.7' — primary gives '41.8'`. Both runs end in `facts_check: 2 / 3 PASS` and exit 1. A table that never fails has not been tested.
