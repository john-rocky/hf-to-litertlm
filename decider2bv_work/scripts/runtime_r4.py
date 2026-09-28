"""Round 4 runtime legs: LiteRT-LM 0.17.1 (PyPI, out/venv-readout) on the three weight-form bundles x {cpu, gpu}.

  run   : --variant V --leg cpu|gpu. Rows = scripts/r3_rows.py IMAGE_ROWS (the 12 round-3 rows, fixed before any
          round-3 runtime result). Strictly one row per process and one process at a time (the driver waits for each),
          stdin /dev/null, one cache dir per bundle (out/runtime_cache/decider-2b-vision_<V>_r4, reused by every row and
          leg of that bundle), each bundle in its own directory (out/bundle_r4/<V>/). Per-row stderr log
          logs/runtime_r4/<V>/<leg>_<row>.log, per-row JSON results/runtime_r4_rows/<V>/<leg>/<row>.json.
          Leg `gpu` = decoder AND vision on the GPU (WebGPU delegate of the PyPI build). If engine creation fails, a
          second row is attempted (determinism), then the leg stops; rows not attempted are listed as such.
  judge : per variant, leg and row (i) runtime prefill tokens == len(oracle input_ids); (ii) the first streamed token
          against the fp32 oracle's full-vocabulary top-1 (== the RELU-free fp32 graph's top-1, checked) AND against
          the same variant's CPU graph readout top-1 (results/readout_r4_<V>.json); 'Validation error' / 'Shape mismatch'
          counts; for the gpu leg the delegate lines in full (unsupported-op listing, the "N operations will run on the
          GPU, and the remaining M operations will run on the CPU" lines, the error block).
          Pass rule, as round 3, is read for `fp16` only: all 12 rows (i) and first token == fp16 graph top-1 == oracle
          top-1. `dyn8` / `v7c` rows are tabled against both references without a pass line.

    out/venv-readout/bin/python -B scripts/runtime_r4.py run --variant fp16 --leg cpu [--rows a,b] [--force]
    out/venv-readout/bin/python -B scripts/runtime_r4.py judge
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ROOT, read_json, write_json, sha256_file          # noqa: E402
from r3_rows import IMAGE_ROWS, oracle_rows                          # noqa: E402
from runtime_r3 import log_facts                                     # noqa: E402

PY = ROOT / 'out/venv-readout/bin/python'
VARIANTS = ('fp16', 'dyn8', 'v7c')
LEGS = {'cpu': ('cpu', 'cpu'), 'gpu': ('gpu', 'gpu')}


def bundle_path(v):
    return ROOT / f'out/bundle_r4/{v}/decider-2b-vision_{v}.litertlm'


def cache_dir(v):
    return ROOT / f'out/runtime_cache/decider-2b-vision_{v}_r4'


def run(args):
    backend, vision = LEGS[args.leg]
    rows = args.rows.split(',') if args.rows else IMAGE_ROWS
    logdir = ROOT / f'logs/runtime_r4/{args.variant}'
    os.makedirs(logdir, exist_ok=True)
    bundle = bundle_path(args.variant)
    assert bundle.exists(), bundle
    engine_failures = 0
    for i, rid in enumerate(rows):
        out = ROOT / f'results/runtime_r4_rows/{args.variant}/{args.leg}/{rid}.json'
        if out.exists() and not args.force and json.loads(out.read_text()).get('status') == 'DONE':
            print('skip', args.variant, args.leg, rid, flush=True)
            continue
        if args.leg == 'gpu' and engine_failures >= 2:
            print('stop', args.variant, args.leg, 'after two engine-creation failures; not attempted:', rows[i:], flush=True)
            break
        log = logdir / f'{args.leg}_{rid}.log'
        cmd = [str(PY), '-B', str(ROOT / 'scripts/runtime_row_r3.py'), '--row', rid, '--backend', backend,
               '--vision-backend', vision, '--cache-dir', str(cache_dir(args.variant)), '--out', str(out),
               '--bundle', str(bundle)]
        t0 = time.monotonic()
        with open(log, 'w') as f:
            f.write('# ' + time.strftime('%F %T %Z') + ' ' + ' '.join(cmd) + '\n')
            f.flush()
            p = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, timeout=1800)
        rec = json.loads(out.read_text()) if out.exists() else {}
        if rec.get('status') != 'DONE' and 'engine_create_seconds' not in rec:
            engine_failures += 1
        print(f'{args.variant} {args.leg} {rid} rc={p.returncode} status={rec.get("status")} first={rec.get("first_chunk")!r} '
              f'prefill={rec.get("benchmark", {}).get("last_prefill_token_count")} {time.monotonic() - t0:.0f}s', flush=True)


def delegate_lines(path):
    """The GPU delegate's own lines, verbatim: the unsupported-op listing block, the GPU/CPU split lines, the
    'fully delegated' hint, delegate init / replacing lines and the engine failure line."""
    text = path.read_text(errors='replace') if path.exists() else ''
    lines = text.splitlines()
    listing = []
    for i, ln in enumerate(lines):
        if 'Following operations are not supported by GPU delegate' in ln:
            block = [ln]
            for nxt in lines[i + 1:]:
                block.append(nxt)
                if 'operations will run on the GPU' in nxt:
                    break
            listing.append(block)
    keys = ('operations will run on the GPU', 'fully delegated', 'Replacing', 'Initializing WebGPU-based API',
            'Failed to create engine', 'Created a WebGPU environment', 'Selected adapter', 'Deserialization failed',
            'Not supported op')
    return dict(unsupported_listing_blocks=listing,
                split_lines=[ln for ln in lines if 'operations will run on the GPU' in ln],
                key_lines=[ln for ln in lines if any(k in ln for k in keys)][:200])


def judge(args):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(ROOT / 'out/src/decider-2b-vision/tokenizer.json'))
    orows = oracle_rows(read_json)
    fp32 = read_json('results/graph_parity_r4_fp32.json')
    fp32_rows = {r['row_id']: r for r in fp32['rows']}
    res = dict(status='RUNNING', rows_rule='scripts/r3_rows.py IMAGE_ROWS (the round-3 rows, fixed before any round-3 runtime result)',
               references=dict(oracle='fixtures/oracle_fp32*.json g256_mrope vocab_top1_id',
                               fp32_graph='results/graph_parity_r4_fp32.json arm i_tflite_vision (RELU-free fp32 decoder)',
                               variant_graph='results/readout_r4_<variant>.json arm <variant> (CPU graph of the same weight form)'),
               pass_rule='fp16 only, per leg: all 12 rows (i) prefill == len(oracle input_ids) AND (ii) first streamed token == '
                         'fp16 CPU graph top-1 == oracle vocab top-1; dyn8 / v7c are tabled against both references, no pass line',
               variants={})
    for v in VARIANTS:
        rpath = ROOT / f'results/readout_r4_{v}.json'
        if not rpath.exists() or not (ROOT / f'results/runtime_r4_rows/{v}').is_dir():
            continue
        ro = read_json(f'results/readout_r4_{v}.json')
        vrows = {r['row_id']: r for r in ro['rows']}
        vres = dict(bundle=dict(path=os.path.relpath(bundle_path(v), ROOT), bytes=os.path.getsize(bundle_path(v))),
                    readout=f'results/readout_r4_{v}.json', readout_status=ro['status'], legs={})
        for leg in sorted(os.listdir(ROOT / f'results/runtime_r4_rows/{v}')):
            rows = []
            for rid in IMAGE_ROWS:
                p = ROOT / f'results/runtime_r4_rows/{v}/{leg}/{rid}.json'
                if not p.exists():
                    rows.append(dict(row_id=rid, status='NOT_ATTEMPTED'))
                    continue
                rr = json.loads(p.read_text())
                log = ROOT / f'logs/runtime_r4/{v}/{leg}_{rid}.log'
                oslot = orows[rid]['forward']['slots'][0]
                vslot = vrows[rid]['arms'][v]['slots'][0]
                fslot = fp32_rows[rid]['arms']['i_tflite_vision']['slots'][0]
                entry = dict(row_id=rid, family=orows[rid]['fixture']['family'], purpose=orows[rid]['fixture']['purpose'],
                             status=rr['status'], oracle_n_input_ids=len(orows[rid]['input_ids']), **log_facts(log),
                             delegate=delegate_lines(log),
                             oracle_top1_id=oslot['vocab_top1_id'], oracle_top1_text=tk.decode([oslot['vocab_top1_id']]),
                             fp32_graph_top1_id=fslot['vocab_top1_id'], fp32_graph_equals_oracle=fslot['vocab_top1_id'] == oslot['vocab_top1_id'],
                             variant_graph_top1_id=vslot['vocab_top1_id'], variant_graph_top1_text=tk.decode([vslot['vocab_top1_id']]),
                             variant_graph_top2_id=vslot['vocab_top2_id'], variant_graph_top1_top2_logit_gap=vslot['vocab_top1_top2_logit_gap'],
                             fp32_graph_top1_top2_logit_gap=fslot['vocab_top1_top2_logit_gap'],
                             variant_graph_letter_probs=vslot['probs'], oracle_letter_probs=oslot['probs'])
                if rr['status'] != 'DONE':
                    entry.update(error_type=rr.get('error_type'), error=rr.get('error'),
                                 engine_created='engine_create_seconds' in rr)
                    rows.append(entry)
                    continue
                first = rr['first_chunk']
                bench = rr['benchmark']
                ids = rr['first_chunk_runtime_ids']
                entry.update(engine_created=True, runtime_prefill_tokens=bench['last_prefill_token_count'],
                             decode_tokens=bench['last_decode_token_count'],
                             pass_i=bench['last_prefill_token_count'] == entry['oracle_n_input_ids'],
                             first_chunk=first, first_chunk_runtime_ids=ids, response_text=rr['response_text'], chunks=rr['chunks'],
                             first_equals_oracle_top1=first == entry['oracle_top1_text'] and ids == [oslot['vocab_top1_id']],
                             first_equals_variant_graph_top1=first == entry['variant_graph_top1_text'] and ids == [vslot['vocab_top1_id']],
                             runtime_render_equal_expected=rr['runtime_render_equal_expected'],
                             runtime_text_ids_equal_oracle=rr['runtime_text_ids_equal_oracle'],
                             ttft_s=bench['time_to_first_token_in_second'], engine_create_s=rr['engine_create_seconds'],
                             wall_s=rr['wall_seconds'])
                entry['pass_ii'] = entry['first_equals_variant_graph_top1'] and entry['first_equals_oracle_top1']
                rows.append(entry)
            done = [r for r in rows if r['status'] == 'DONE']
            att = [r for r in rows if r['status'] != 'NOT_ATTEMPTED']
            summ = dict(n_rows=len(IMAGE_ROWS), n_attempted=len(att), n_done=len(done),
                        engine_created=sum(r.get('engine_created', False) for r in att),
                        n_pass_i=sum(r.get('pass_i', False) for r in done),
                        n_first_equals_oracle_top1=sum(r.get('first_equals_oracle_top1', False) for r in done),
                        n_first_equals_variant_graph_top1=sum(r.get('first_equals_variant_graph_top1', False) for r in done),
                        validation_error_lines=sum(r['validation_error_lines'] for r in att),
                        shape_mismatch_lines=sum(r['shape_mismatch_lines'] for r in att),
                        webgpu_delegate_init_lines=sum(r['webgpu_delegate_init_lines'] for r in att),
                        partial_delegation_lines=sum(r['partial_delegation_lines'] for r in att),
                        unsupported_op_lines=sum(r['unsupported_op_lines'] for r in att),
                        engine_errors=sorted(set(r.get('error', '') for r in att if r['status'] != 'DONE')))
            if v == 'fp16':
                summ['pass'] = summ['n_done'] == len(IMAGE_ROWS) and summ['n_pass_i'] == len(IMAGE_ROWS) and \
                    sum(r.get('pass_ii', False) for r in done) == len(IMAGE_ROWS)
            vres['legs'][leg] = dict(backend=LEGS[leg][0], vision_backend=LEGS[leg][1], summary=summ, rows=rows)
        res['variants'][v] = vres
    res['disk'] = {d.name: {f: os.path.getsize(d / f) for f in sorted(os.listdir(d))}
                   for d in sorted((ROOT / 'out/runtime_cache').glob('*_r4')) if d.is_dir()}
    res['status'] = 'DONE'
    write_json('results/runtime_r4.json', res)
    for v, vr in res['variants'].items():
        for leg, lr in vr['legs'].items():
            print('RUNTIME_R4', v, leg, {k: lr['summary'][k] for k in ('n_attempted', 'n_done', 'engine_created', 'n_pass_i',
                                                                           'n_first_equals_oracle_top1', 'n_first_equals_variant_graph_top1',
                                                                           'validation_error_lines')}, lr['summary'].get('pass', '-'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('run', 'judge'))
    ap.add_argument('--variant', choices=VARIANTS)
    ap.add_argument('--leg', choices=sorted(LEGS), default='cpu')
    ap.add_argument('--rows', default='')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    run(args) if args.mode == 'run' else judge(args)


if __name__ == '__main__':
    main()
