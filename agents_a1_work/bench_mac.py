"""Mac p256/d256 benchmark: three measured runs, preflights and GPU rests."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
from runtime_helpers import HERE, ROOT, configure_output, run_cli, save_json, cli_success
from export_float import sha256


def stamp():
    return datetime.now(timezone.utc).isoformat()


def preflight(pattern):
    uptime = subprocess.check_output(['uptime'], text=True).strip()
    match = re.search(r'load averages?:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)', uptime)
    assert match, uptime
    loads = [float(x) for x in match.groups()]
    processes = []
    lines = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,comm=,args='], text=True).splitlines()
    parent_map = {int(f[0]): int(f[1]) for f in (line.strip().split(None,3) for line in lines) if len(f)==4}
    own_chain = {os.getpid()}; current = os.getpid()
    while current in parent_map and parent_map[current] not in own_chain:
        current = parent_map[current]; own_chain.add(current)
    for line in lines:
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            continue
        pid, ppid, comm, args = fields
        if int(pid) in own_chain:
            continue
        if re.search(pattern, comm + ' ' + args, re.I):
            processes.append({'pid': int(pid), 'ppid': int(ppid), 'command': comm, 'args': args})
    return {'date': stamp(), 'uptime': uptime, 'load_1m': loads[0], 'load_5m': loads[1],
            'load_15m': loads[2], 'blocking_processes': processes,
            'quiet': loads[0] < 3.0 and not processes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundles', type=Path, default=ROOT)
    p.add_argument('--cli', default=os.environ.get('LITERT_LM', 'litert-lm'))
    p.add_argument('--out', type=Path, default=ROOT / 'results/bench_mac.json')
    p.add_argument('--device', required=True, help='Measured host hardware label')
    p.add_argument('--lock', type=Path, help='Optional GPU lock path agreed on for this host')
    p.add_argument('--busy-pattern', default=r'\blitert[-_]lm\b|python.*benchmark')
    a = p.parse_args(); configure_output(a.out)
    version = subprocess.check_output([a.cli, '--version'], text=True).strip()
    flights, cells = [], []
    lock = a.lock or a.out.parent / 'gpu-benchmark.lock'
    acquired = False
    token = 'agents-a1-4b benchmark ' + str(os.getpid())
    try:
        for variant, backend in [('int8', 'gpu'), ('mixed_int4', 'gpu'), ('int8', 'cpu'), ('mixed_int4', 'cpu')]:
            model = a.bundles / ('Agents-A1-4B_' + variant + '.litertlm')
            gate = json.loads((a.out.parent / f'gate8q_{variant}_{backend}_0.17.1.json').read_text())
            assert gate['verdict'] == 'PASS' and gate['manual_prompt_relevance'] is True
            assert gate['model_sha256'] == sha256(model), 'Generation evidence is for a different file'
            if backend == 'gpu':
                for _ in range(5):
                    print('GPU rest: 60 seconds', flush=True)
                    time.sleep(60)
            started_wait = time.monotonic()
            while True:
                flight = preflight(a.busy_pattern); flights.append({'variant': variant, 'backend': backend, **flight})
                save_json(a.out.with_name('bench_mac_preflight.json'), flights)
                if flight['quiet'] or time.monotonic() - started_wait >= 1200:
                    break
                print('Waiting for quiet host; load', flight['load_1m'], flush=True)
                time.sleep(60)
            if backend == 'gpu' and not acquired:
                lock.parent.mkdir(parents=True, exist_ok=True)
                with lock.open('x') as stream:
                    stream.write(token + '\n')
                acquired = True
            observation = a.out.parent / 'logs' / f'bench_{variant}_{backend}.runs.json'
            before = {x.name for x in model.parent.glob(model.name + '_*_mldrift_*_cache.bin')}
            command = [a.cli, 'benchmark', str(model), '-p', '256', '-d', '256', '--runs', '3', '--cache', 'no', '--backend', backend]
            execution = run_cli(command, f'bench_{variant}_{backend}', env_overrides={
                'AGENTS_A1_BENCH_OBSERVATION': str(observation.resolve()),
                'PYTHONPATH': str(HERE / 'bench_observer') + os.pathsep + os.environ.get('PYTHONPATH', '')})
            calls = json.loads(observation.read_text())['calls'] if observation.exists() else []
            row = {'variant': variant, 'backend': backend, 'device': a.device, 'runtime': version, 'date': stamp(),
                   'prompt_tokens': 256, 'decode_tokens': 256, 'runs': 3, 'cache': 'no',
                   'preflight': flight, 'contention': 'quiet' if flight['quiet'] else 'contended',
                   'gpu_rest_seconds_minimum': 300 if backend == 'gpu' else None,
                   'execution': execution, 'warmup': calls[0] if calls else None, 'per_run': calls[1:]}
            row['verdict'] = 'PASS' if cli_success(execution) and len(calls) == 4 else 'FAIL'
            if row['verdict'] == 'PASS':
                for key, field in [('prefill_tokens_per_second', 'last_prefill_tokens_per_second'),
                                   ('decode_tokens_per_second', 'last_decode_tokens_per_second'),
                                   ('ttft_seconds', 'time_to_first_token_in_second')]:
                    values = [r[field] for r in calls[1:]]
                    row[key] = statistics.mean(values); row[key + '_range'] = [min(values), max(values)]
            deleted = []
            for path in model.parent.glob(model.name + '_*_mldrift_*_cache.bin'):
                if path.name not in before:
                    deleted.append({'name': path.name, 'bytes': path.stat().st_size}); path.unlink()
            row['cache_check'] = {'unexpected_generated_caches_deleted': deleted, 'preexisting_preserved': sorted(before)}
            cells.append(row); save_json(a.out, {'protocol': 'p256/d256, three runs after default warmup, cache no', 'cells': cells})
            if variant == 'mixed_int4' and backend == 'gpu':
                assert lock.read_text().strip() == token
                lock.unlink(); acquired = False
    finally:
        if acquired:
            assert lock.read_text().strip() == token
            lock.unlink()
    raise SystemExit(0 if all(r['verdict'] == 'PASS' for r in cells) else 1)


if __name__ == '__main__':
    main()
