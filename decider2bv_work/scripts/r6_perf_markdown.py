"""Round 6: print the card's Performance tables from results/r6/perf_summary.json (medians over uncontended runs,
the runs themselves in brackets). The printed markdown is pasted into publish/README.md and checked back by
scripts/r6_verify_card.py.

    python3 -B scripts/r6_perf_markdown.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runs(c, key, fmt):
    return ', '.join(format(c[key][i], fmt) for i in c['median_idx'])


def main():
    ps = json.loads((ROOT / 'results/r6/perf_summary.json').read_text())
    order = [(k, a, cache) for k in ('fp16', 'v7c') for a in ('cpu', 'gpu') for cache in ('nocache', 'disk')]
    cells = {(c['kind'], c['arg'], c['cache']): c for c in ps['cells']}
    print('| File | Backend | Runtime cache folder | TTFT s | Engine creation s | Prefill tok/s | Peak RSS GB | Peak footprint GB |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for key in order:
        c = cells.get(key)
        if not c:
            continue
        m = c['median']
        label = {'nocache': 'none', 'disk': 'yes, warm'}[key[2]]
        fp = [c['peak_footprint_gb'][i] for i in c['median_idx']]
        fpm = sorted(fp)[len(fp) // 2] if len(fp) % 2 else sum(sorted(fp)[len(fp) // 2 - 1:len(fp) // 2 + 1]) / 2
        print(f"| {key[0]} | {key[1].upper()} | {label} | {m['ttft_wall_s']:.2f} ({runs(c, 'ttft_wall_s', '.2f')}) | "
              f"{m['engine_create_s']:.1f} | {m['prefill_tps']:.0f} | {m['max_rss_gb']:.1f} | {fpm:.1f} |")
    print()
    print('| File | Constructor s | First decision s | Second decision s | Peak RSS GB |')
    print('| --- | --- | --- | --- | --- |')
    for v in ('fp16', 'v7c'):
        c = cells.get(('ref', v, 'nocache'))
        if not c:
            continue
        m = c['median']
        print(f"| {v} | {m['constructor_s']:.1f} | {m['first_decide_s']:.2f} ({runs(c, 'first_decide_s', '.2f')}) | "
              f"{m['second_decide_s']:.2f} | {m['max_rss_gb']:.1f} |")
    for c in ps['cells']:
        if c['kind'] != 'ref' and c['cache'] == 'disk':
            print(f"cold run 0 {c['kind']} {c['arg']}: {c.get('cold_run0')}")
        print(c['kind'], c['arg'], c['cache'], 'clean runs used', c['median_runs'], 'of', c['n'], 'contended', c['contended'])


if __name__ == '__main__':
    main()
