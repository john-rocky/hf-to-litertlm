"""Round 6 Mac measurements for the card: fp16 and v7c x LiteRT-LM 0.17.1 PyPI CPU / GPU, one decision per process,
three processes per cell, plus the reference readout (CompiledModel CPU) on v7c and fp16.

Before every process: `uptime` and the processes above 120 % CPU other than this driver's own children are recorded;
if any is present the driver waits (poll 30 s) up to --max-wait seconds for it to go, then runs anyway and marks the
row contended. GPU processes get --gpu-rest seconds since the previous GPU process ended (the first GPU process of
the session --gpu-first-rest). Each process runs under /usr/bin/time -l with stdin /dev/null; its stderr is kept
(logs/r6/bench/<cell>_<i>.log) and 'Validation error' / 'Shape mismatch' lines are counted.

    python3 -B scripts/r6_mac_bench.py [--cells fp16:cpu,fp16:gpu,v7c:cpu,v7c:gpu,ref:v7c,ref:fp16] [--runs 3]
        -> results/r6/mac_bench.json (+ one JSON per process under results/r6/bench/)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_RT = ROOT / 'out/venv-readout/bin/python'          # litert-lm 0.17.1 PyPI
PY_REF = ROOT / 'out/venv-ref/bin/python'             # numpy, pillow, tokenizers, ai-edge-litert only


def bundle(v):
    return ROOT / f'out/bundle_r4/{v}/decider-2b-vision_{v}.litertlm'


def busy_peers(own_pids):
    out = subprocess.run(['ps', '-Ao', 'pid,ppid,pcpu,rss,etime,comm'], capture_output=True, text=True).stdout.splitlines()[1:]
    peers = []
    for line in out:
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid, ppid, pcpu = int(parts[0]), int(parts[1]), float(parts[2])
        if pcpu > 120.0 and pid not in own_pids and ppid not in own_pids:
            peers.append(dict(pid=pid, pcpu=pcpu, rss_kb=int(parts[3]), etime=parts[4], comm=parts[5][-80:]))
    return peers


def uptime():
    return subprocess.run(['uptime'], capture_output=True, text=True).stdout.strip()


def parse_time_l(text):
    def grab(label):
        m = re.search(r'^\s*(\d+)\s+' + re.escape(label) + r'\s*$', text, re.M)
        return int(m.group(1)) if m else None
    real = re.search(r'^\s*([\d.]+) real', text, re.M)
    return dict(max_rss_bytes=grab('maximum resident set size'), peak_footprint_bytes=grab('peak memory footprint'),
                real_s=float(real.group(1)) if real else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='fp16:cpu,fp16:gpu,v7c:cpu,v7c:gpu,ref:v7c,ref:fp16')
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--row', default='game_pong_atari_up')
    ap.add_argument('--max-wait', type=int, default=300)
    ap.add_argument('--first-wait', type=int, default=1200,
                    help='wait for a quiet window before the first process; if none comes, run the rest without waiting')
    ap.add_argument('--gpu-first-rest', type=int, default=300)
    ap.add_argument('--gpu-rest', type=int, default=120)
    ap.add_argument('--out', default='results/r6/mac_bench.json')
    ap.add_argument('--redo-contended', action='store_true',
                    help='run again only the rows recorded as contended; the old record moves to superseded_runs')
    args = ap.parse_args()
    cells = [tuple(c.split(':')) + (('nocache',) if c.count(':') == 1 else ()) for c in args.cells.split(',')]
    (ROOT / 'logs/r6/bench').mkdir(parents=True, exist_ok=True)
    (ROOT / 'results/r6/bench').mkdir(parents=True, exist_ok=True)
    out_path = ROOT / args.out
    res = json.loads(out_path.read_text()) if out_path.exists() else dict(runs=[])
    res.update(machine=subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip(),
               model=subprocess.run(['sysctl', '-n', 'hw.model'], capture_output=True, text=True).stdout.strip(),
               memsize=int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True).stdout.strip()),
               os=subprocess.run(['sw_vers', '-productVersion'], capture_output=True, text=True).stdout.strip() + ' ' +
               subprocess.run(['sw_vers', '-buildVersion'], capture_output=True, text=True).stdout.strip(),
               protocol='one decision per process, fresh process each run; runtime cache_dir :nocache; CPU 8 threads; '
                        'GPU = LiteRT-LM Backend.GPU() for the decoder and the vision encoder; reference = CompiledModel CPU '
                        '8 threads with its section folder and weight cache already on disk')
    last_gpu_end = None
    first = True
    # interleave: run index outer, cells inner (a GPU process is never directly after another GPU process when a CPU
    # cell sits between them)
    for i in range(args.runs):
        for kind, arg, cache in cells:
            name = f'{kind}_{arg}_{i}' + ('' if cache == 'nocache' else f'_{cache}')
            prev = [r for r in res['runs'] if r['name'] == name and r.get('rc') == 0]
            if prev and not (args.redo_contended and prev[0]['contended']):
                print('skip', name, flush=True)
                continue
            if prev:
                res.setdefault('superseded_runs', []).append(dict(prev[0], superseded_because='contended; re-run'))
            gpu = arg == 'gpu'
            if gpu:
                need = args.gpu_first_rest if last_gpu_end is None else args.gpu_rest
                since = None if last_gpu_end is None else time.monotonic() - last_gpu_end
                wait = need if since is None else max(0.0, need - since)
                if wait > 0:
                    print(f'{name}: GPU rest {wait:.0f}s', flush=True)
                    time.sleep(wait)
            waited, t_w = 0, time.monotonic()
            peers = busy_peers(set())
            limit = args.first_wait if first else args.max_wait
            while peers and time.monotonic() - t_w < limit:
                time.sleep(30)
                peers = busy_peers(set())
            waited = time.monotonic() - t_w
            if first and peers:
                args.max_wait = 0                       # no quiet window within --first-wait: stop waiting, mark rows
                res['no_quiet_window_within_s'] = args.first_wait
            first = False
            pre = dict(uptime=uptime(), peers_over_120=peers, waited_s=round(waited, 1))
            j = ROOT / f'results/r6/bench/{name}.json'
            log = ROOT / f'logs/r6/bench/{name}.log'
            if kind == 'ref':
                cmd = [str(PY_REF), '-B', str(ROOT / 'scripts/r6_ref_bench_row.py'), '--bundle', str(bundle(arg)),
                       '--row', args.row, '--out', str(j)]
            else:
                cmd = [str(PY_RT), '-B', str(ROOT / 'scripts/r6_bench_row.py'), '--bundle', str(bundle(kind)),
                       '--backend', arg, '--row', args.row, '--out', str(j)]
                if cache == 'disk':                     # the runtime's default mode: a cache folder, cold at run 0
                    cdir = ROOT / f'out/r6_bench_cache/{kind}_{arg}'
                    if i == 0 and cdir.exists():
                        shutil.rmtree(cdir)
                    cdir.mkdir(parents=True, exist_ok=True)
                    cmd += ['--cache-dir', str(cdir)]
            t0 = time.monotonic()
            during, stop = [], threading.Event()

            def sample(child):
                while not stop.wait(5.0):
                    own = {os.getpid(), child.pid}
                    for q in busy_peers(own):
                        if q['comm'].endswith(('Python', 'python', 'python3')) and 'r6_' in q['comm']:
                            continue
                        during.append(dict(q, t=round(time.monotonic() - t0, 1)))
            with open(log, 'w') as lf:
                lf.write('# ' + ' '.join(['/usr/bin/time', '-l'] + cmd) + '\n')
                lf.flush()
                p = subprocess.Popen(['/usr/bin/time', '-l'] + cmd, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT)
                th = threading.Thread(target=sample, args=(p,), daemon=True)
                th.start()
                p.wait()
                stop.set()
                th.join()
            wall = time.monotonic() - t0
            if gpu:
                last_gpu_end = time.monotonic()
            text = log.read_text(errors='replace')
            post_peers = busy_peers(set())
            row = dict(name=name, kind=kind, arg=arg, cache=cache, run=i, rc=p.returncode, driver_wall_s=wall, log=str(log.relative_to(ROOT)),
                       json=str(j.relative_to(ROOT)), pre=pre, post_peers_over_120=post_peers,
                       during_peers_over_120=during, contended=bool(peers or post_peers or during), validation_errors=text.count('Validation error'),
                       shape_mismatch=text.count('Shape mismatch'), time_l=parse_time_l(text),
                       finished=time.strftime('%Y-%m-%d %H:%M:%S %Z'))
            if j.exists():
                row['result'] = json.loads(j.read_text())
            res['runs'] = [r for r in res['runs'] if r['name'] != name] + [row]
            out_path.write_text(json.dumps(res, indent=1) + '\n')
            rr = row.get('result', {})
            print(f"{name}: rc {p.returncode} contended {row['contended']} ttft {rr.get('ttft_wall_s')} "
                  f"first-decide {rr.get('first_decide_wall_s')} resp {rr.get('response')!r} ok {rr.get('first_token_ok')} "
                  f"rss {row['time_l']['max_rss_bytes']} VE {row['validation_errors']} wall {wall:.0f}s", flush=True)
    print('MAC_BENCH done', flush=True)


if __name__ == '__main__':
    main()
