"""Round 6: summarize the Mac measurements for the card and the manifest (medians and every run, per cell).

Reads results/r6/mac_bench.json (cache no) and results/r6/mac_bench_disk.json (cache folder; run 0 cold, runs 1-2
warm). Per cell: engine creation wall (Python constructor), send-to-first-token wall, runtime-reported TTFT, prefill
tokens and tok/s, peak RSS (/usr/bin/time -l "maximum resident set size") and peak memory footprint, first token,
Validation-error lines, contended flag. The reference readout cells report its constructor wall and its first and
second decision walls.

    python3 -B scripts/r6_perf_table.py -> results/r6/perf_summary.json (+ a markdown table on stdout)
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def f(x, spec):
    return '-' if x is None else format(x, spec)


def main():
    runs = []
    for name in ('mac_bench.json', 'mac_bench_disk.json'):
        p = ROOT / 'results/r6' / name
        if p.exists():
            d = json.loads(p.read_text())
            for r in d['runs']:
                r['series'] = name
                runs.append(r)
            meta = {k: d[k] for k in ('machine', 'model', 'memsize', 'os', 'protocol') if k in d}
    cells = {}
    for r in runs:
        key = (r['kind'], r['arg'], r.get('cache', 'nocache'))
        cells.setdefault(key, []).append(r)
    out = dict(meta=meta, cells=[])
    for (kind, arg, cache), rs in sorted(cells.items()):
        rs = sorted(rs, key=lambda r: r['run'])
        res = [r.get('result', {}) for r in rs]
        cell = dict(kind=kind, arg=arg, cache=cache, n=len(rs), rc=[r['rc'] for r in rs],
                    contended=[r['contended'] for r in rs], finished=[r['finished'] for r in rs],
                    json=[r['json'] for r in rs], log=[r['log'] for r in rs],
                    max_rss_gb=[round(r['time_l']['max_rss_bytes'] / 1e9, 2) if r['time_l']['max_rss_bytes'] else None for r in rs],
                    peak_footprint_gb=[round(r['time_l']['peak_footprint_bytes'] / 1e9, 2) if r['time_l']['peak_footprint_bytes'] else None for r in rs],
                    validation_errors=[r['validation_errors'] for r in rs])
        if kind == 'ref':
            cell.update(constructor_s=[x.get('constructor_wall_s') for x in res], first_decide_s=[x.get('first_decide_wall_s') for x in res],
                        second_decide_s=[x.get('second_decide_wall_s') for x in res],
                        choice=[(x.get('answer') or [{}])[0].get('choice') for x in res])
            sel = [i for i in range(len(rs)) if not rs[i]['contended']]
            cell['median_runs'] = [rs[i]['run'] for i in sel]
            cell['median_idx'] = sel
            pick = lambda k: [cell[k][i] for i in sel]
            cell['median'] = dict(constructor_s=med(pick('constructor_s')), first_decide_s=med(pick('first_decide_s')),
                                  second_decide_s=med(pick('second_decide_s')), max_rss_gb=med(pick('max_rss_gb')))
        else:
            b = [x.get('benchmark', {}) for x in res]
            cell.update(engine_create_s=[x.get('engine_create_wall_s') for x in res], ttft_wall_s=[x.get('ttft_wall_s') for x in res],
                        ttft_runtime_s=[y.get('time_to_first_token_in_second') for y in b],
                        init_runtime_s=[y.get('init_time_in_second') for y in b],
                        prefill_tokens=[y.get('last_prefill_token_count') for y in b],
                        prefill_tps=[y.get('last_prefill_tokens_per_second') for y in b],
                        response=[x.get('response') for x in res], first_token_ok=[x.get('first_token_ok') for x in res])
            sel = [i for i in range(len(rs)) if (cache == 'nocache' or rs[i]['run'] > 0) and not rs[i]['contended']]
            # medians over uncontended runs only (disk cells: warm runs only); every run stays listed above
            pick = lambda k: [cell[k][i] for i in sel]
            cell['median_runs'] = [rs[i]['run'] for i in sel]
            cell['median_idx'] = sel
            cell['median'] = {k: med(pick(k)) for k in ('engine_create_s', 'ttft_wall_s', 'ttft_runtime_s', 'prefill_tps',
                                                        'max_rss_gb', 'peak_footprint_gb', 'init_runtime_s')}
            if cache == 'disk':
                cell['cold_run0'] = {k: cell[k][0] for k in ('engine_create_s', 'ttft_wall_s', 'max_rss_gb')}
        out['cells'].append(cell)
    (ROOT / 'results/r6/perf_summary.json').write_text(json.dumps(out, indent=1) + '\n')
    for c in out['cells']:
        m = c['median']
        if c['kind'] == 'ref':
            print(f"ref {c['arg']:5s} (clean runs {c['median_runs']}): constructor {f(m['constructor_s'], '.1f')} s, first decision {f(m['first_decide_s'], '.2f')} s, second {f(m['second_decide_s'], '.2f')} s, "
                  f"RSS {m['max_rss_gb']} GB | runs {['%.2f' % x for x in c['first_decide_s']]} contended {c['contended']}")
        else:
            print(f"{c['kind']:5s} {c['arg']} {c['cache']:7s} (clean runs {c['median_runs']}): engine {f(m['engine_create_s'], '.1f')} s, TTFT {f(m['ttft_wall_s'], '.2f')} s "
                  f"(runtime {f(m['ttft_runtime_s'], '.2f')}), prefill {f(m['prefill_tps'], '.0f')} tok/s, RSS {m['max_rss_gb']} GB, footprint {m['peak_footprint_gb']} GB | "
                  f"ttft runs {['%.2f' % x for x in c['ttft_wall_s']]} engine runs {['%.1f' % x for x in c['engine_create_s']]} "
                  f"rss {c['max_rss_gb']} ok {c['first_token_ok']} VE {c['validation_errors']} contended {c['contended']}")


if __name__ == '__main__':
    main()
