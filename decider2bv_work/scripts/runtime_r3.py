"""Round 3 runtime leg: drive scripts/runtime_row_r3.py one row per process, then judge the rows against the fp16
graph readout -> results/runtime_r3.json.

  run   : rows of r3_rows (IMAGE_ROWS, or TEXT_ROWS with --text) on one backend, strictly sequential (one runtime
          process on the bundle directory at a time; one cache dir for the bundle, reused by every row and backend),
          stdin /dev/null, per-row stderr log in logs/runtime_r3/<leg>_<row>.log, per-row JSON in
          results/runtime_r3_rows/<leg>/<row>.json. A row with a DONE JSON is skipped unless --force.
  judge : per leg and row, (i) the runtime's prefill token count == len(oracle input_ids) (image: 64 soft tokens
          included), (ii) the first streamed token == the graph readout's full-vocabulary top-1 at that slot
          (results/fp16_readout_r3.json, arm A_fp16 for image rows; both text arms for the no-image rows) which must
          itself equal the oracle's vocab_top1_id; plus the render / tokenization facts, 'Validation error' counts
          and the error block of any leg whose engine could not be created.

    out/venv-readout/bin/python -B scripts/runtime_r3.py run --leg cpu [--text] [--rows a,b]
    out/venv-readout/bin/python -B scripts/runtime_r3.py run --leg gpu --rows color_red
    out/venv-readout/bin/python -B scripts/runtime_r3.py judge
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
from r3_rows import IMAGE_ROWS, TEXT_ROWS                             # noqa: E402

PY = ROOT / 'out/venv-readout/bin/python'
CACHE = 'out/runtime_cache/decider-2b-vision_fp16'
LEGS = {'cpu': ('cpu', 'cpu'), 'gpu': ('gpu', 'gpu'), 'cpu_visgpu': ('cpu', 'gpu')}
READOUT = 'results/fp16_readout_r3.json'


def run(args):
    backend, vision = LEGS[args.leg]
    rows = args.rows.split(',') if args.rows else (TEXT_ROWS if args.text else IMAGE_ROWS)
    os.makedirs(ROOT / 'logs/runtime_r3', exist_ok=True)
    for rid in rows:
        out = ROOT / f'results/runtime_r3_rows/{args.leg}/{rid}.json'
        if out.exists() and not args.force and json.loads(out.read_text()).get('status') == 'DONE':
            print('skip', args.leg, rid, flush=True)
            continue
        log = ROOT / f'logs/runtime_r3/{args.leg}_{rid}.log'
        cmd = [str(PY), '-B', str(ROOT / 'scripts/runtime_row_r3.py'), '--row', rid, '--backend', backend,
               '--vision-backend', vision, '--cache-dir', str(ROOT / CACHE), '--out', str(out)]
        t0 = time.monotonic()
        with open(log, 'w') as f:
            f.write('# ' + ' '.join(cmd) + '\n')
            f.flush()
            p = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, timeout=1800)
        print(f'{args.leg} {rid} rc={p.returncode} {time.monotonic() - t0:.0f}s', flush=True)


def log_facts(path):
    text = path.read_text(errors='replace') if path.exists() else ''
    lines = text.splitlines()
    # the whole error block: from the first ERROR / engine failure / traceback line to the end of the log, minus the
    # accelerator (de)registration chatter
    start = next((i for i, ln in enumerate(lines) if ln.startswith('ERROR') or 'Failed to create engine' in ln
                  or ln.startswith('Traceback')), None)
    err = [] if start is None else [ln for ln in lines[start:] if not re.match(r'INFO: \[(accelerator_registry|cpu_registry|gpu_registry)', ln)]
    return dict(log=os.path.relpath(path, ROOT), validation_error_lines=text.count('Validation error'),
                shape_mismatch_lines=text.count('Shape mismatch'), error_block=err[:80],
                accelerators_registered=sorted(set(re.findall(r'name=(GPU WebGPU|GPU Metal|CpuAccelerator)', text))),
                webgpu_environment='Created a WebGPU environment' in text,
                webgpu_delegate_init_lines=text.count('Initializing WebGPU-based API'),
                partial_delegation_lines=text.count('operations will run on the CPU'),
                unsupported_op_lines=sum('Not supported op' in ln or 'not supported by GPU delegate' in ln for ln in text.splitlines()))


def judge(args):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(ROOT / 'out/src/decider-2b-vision/tokenizer.json'))
    ro = read_json(READOUT)
    graph = {r['row_id']: r for r in ro['rows']}
    res = dict(status='RUNNING', bundle=read_json('results/bundle_r3.json')['bundle'], readout=READOUT,
               readout_status=ro['status'], rows_rule='scripts/r3_rows.py (fixed before any runtime result)',
               pass_rule='per leg: all 12 image rows have (i) runtime prefill tokens == len(oracle input_ids) AND '
                         '(ii) first streamed token == graph (A_fp16) full-vocabulary top-1 == oracle vocab_top1_id',
               legs={})
    for leg in sorted(os.listdir(ROOT / 'results/runtime_r3_rows')):
        d = ROOT / f'results/runtime_r3_rows/{leg}'
        rows = []
        for rid in IMAGE_ROWS + TEXT_ROWS:
            p = d / f'{rid}.json'
            if not p.exists():
                continue
            rr = json.loads(p.read_text())
            g = graph[rid]
            image = g['image']
            entry = dict(row_id=rid, image=image, family=g['family'], purpose=g['purpose'], status=rr['status'],
                         oracle_n_input_ids=len(read_json_row_ids(rid)), **log_facts(ROOT / f'logs/runtime_r3/{leg}_{rid}.log'))
            if rr['status'] != 'DONE':
                entry.update(error_type=rr.get('error_type'), error=rr.get('error'))
                rows.append(entry)
                continue
            arms = ['A_fp16'] if image else ['text_from_65', 'text_from_0_informational']
            slot = {a: g['arms'][a]['slots'][0] for a in arms}
            first = rr['first_chunk']
            bench = rr['benchmark']
            entry.update(
                runtime_prefill_tokens=bench['last_prefill_token_count'], decode_tokens=bench['last_decode_token_count'],
                prefill_equal=bench['last_prefill_token_count'] == entry['oracle_n_input_ids'],
                response_text=rr['response_text'], chunks=rr['chunks'], first_chunk=first,
                first_chunk_runtime_ids=rr['first_chunk_runtime_ids'],
                runtime_render_equal_expected=rr['runtime_render_equal_expected'],
                runtime_text_ids_equal_oracle=rr['runtime_text_ids_equal_oracle'],
                engine_bos_token_id=rr['engine_bos_token_id'], engine_eos_token_ids=rr['engine_eos_token_ids'],
                ttft_s=bench['time_to_first_token_in_second'], engine_create_s=rr['engine_create_seconds'], wall_s=rr['wall_seconds'],
                graph={a: dict(vocab_top1_id=s['vocab_top1_id'], vocab_top1_text=tk.decode([s['vocab_top1_id']]),
                               oracle_vocab_top1_id=s['ref_vocab_top1_id'], vocab_top2_id=s['vocab_top2_id'],
                               top1_top2_logit_gap=s['vocab_top1_top2_logit_gap'], letter_probs=s['probs'], oracle_probs=s['ref_probs'])
                       for a, s in slot.items()})
            for a, s in slot.items():
                entry[f'first_equal_{a}'] = (first == tk.decode([s['vocab_top1_id']]) and rr['first_chunk_runtime_ids'] == [s['vocab_top1_id']])
            if image:
                entry['graph_top1_equals_oracle'] = slot['A_fp16']['vocab_top1_id'] == slot['A_fp16']['ref_vocab_top1_id']
                entry['pass_i'] = entry['prefill_equal']
                entry['pass_ii'] = entry['first_equal_A_fp16'] and entry['graph_top1_equals_oracle']
            rows.append(entry)
        img = [r for r in rows if r['image']]
        done = [r for r in img if r['status'] == 'DONE']
        summ = dict(n_image_rows=len(img), n_done=len(done), n_pass_i=sum(r.get('pass_i', False) for r in img),
                    n_pass_ii=sum(r.get('pass_ii', False) for r in img),
                    validation_error_lines=sum(r['validation_error_lines'] for r in rows),
                    shape_mismatch_lines=sum(r['shape_mismatch_lines'] for r in rows),
                    rows_with_webgpu_environment=sum(r['webgpu_environment'] for r in rows),
                    webgpu_delegate_init_lines=sum(r['webgpu_delegate_init_lines'] for r in rows),
                    partial_delegation_lines=sum(r['partial_delegation_lines'] for r in rows),
                    unsupported_op_lines=sum(r['unsupported_op_lines'] for r in rows),
                    engine_errors=sorted(set(r.get('error', '') for r in rows if r['status'] != 'DONE')))
        summ['pass'] = len(img) == len(IMAGE_ROWS) and summ['n_pass_i'] == summ['n_pass_ii'] == len(IMAGE_ROWS)
        res['legs'][leg] = dict(backend=LEGS.get(leg, ('?', '?'))[0], vision_backend=LEGS.get(leg, ('?', '?'))[1],
                                summary=summ, rows=rows)
    cpu = res['legs'].get('cpu', {}).get('summary', {})
    res['status'] = 'PASS' if cpu.get('pass') else 'FAIL'
    res['legs_informational'] = ['cpu_visgpu']
    # disk: the runtime cache dir for the bundle and the readout weight caches (file name -> bytes)
    res['disk'] = {d: {f: os.path.getsize(ROOT / d / f) for f in sorted(os.listdir(ROOT / d))}
                   for d in (CACHE, 'out/xnn_cache') if (ROOT / d).is_dir()}
    du = {d: int(subprocess.run(['du', '-sk', str(ROOT / d)], capture_output=True, text=True).stdout.split()[0])
          for d in ('out', 'out/fp16_r3', 'out/bundle', 'out/runtime_cache', 'out/xnn_cache') if (ROOT / d).exists()}
    res['disk_du_kib'] = dict(du, out_before_round3=27254728, out_before_round3_source='du -sk out at 00:04 JST, before any round-3 write',
                              new_in_round3_gb=round((du['out'] - 27254728) * 1024 / 1e9, 1),
                              note='du counts every file in full (APFS clones included); out/xnn_cache also holds round 2\'s fp32 cache')
    write_json('results/runtime_r3.json', res)
    for leg, v in res['legs'].items():
        print('RUNTIME_JUDGE', leg, {k: v['summary'][k] for k in ('n_image_rows', 'n_done', 'n_pass_i', 'n_pass_ii', 'validation_error_lines', 'pass')})


_ORACLE_IDS = {}


def read_json_row_ids(rid):
    if not _ORACLE_IDS:
        from r3_rows import oracle_rows
        for k, v in oracle_rows(read_json).items():
            _ORACLE_IDS[k] = v['input_ids']
    return _ORACLE_IDS[rid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('run', 'judge'))
    ap.add_argument('--leg', choices=sorted(LEGS), default='cpu')
    ap.add_argument('--rows', default='')
    ap.add_argument('--text', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    run(args) if args.mode == 'run' else judge(args)


if __name__ == '__main__':
    main()
