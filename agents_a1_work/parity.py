"""Use the existing big-model parity math with an explicit dataset and JSON gate."""
import argparse
import importlib.util
import json
from pathlib import Path
from runtime_helpers import REPO, ROOT, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['pt', 'lt', 'cmp'])
    p.add_argument('--hf', default='src_models/Agents-A1-4B')
    p.add_argument('--dataset', type=Path, default=Path('evaldata/gsm8k_test.jsonl'))
    p.add_argument('--tflite')
    p.add_argument('--ids', default=str(ROOT / 'results/parity_pt.npz'))
    p.add_argument('--pt', default=str(ROOT / 'results/parity_pt.npz'))
    p.add_argument('--lt', default=str(ROOT / 'results/parity_lt.npz'))
    p.add_argument('--out', type=Path)
    p.add_argument('--n', type=int, default=48)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('bigmodel_parity', REPO / 'scripts/parity_logits_bigmodel.py')
    rail = importlib.util.module_from_spec(spec); spec.loader.exec_module(rail)
    a.tag = 'Agents-A1-4B fp32 CPU'
    a.out = a.out or ROOT / ('results/parity_' + a.stage + ('.json' if a.stage == 'cmp' else '.npz'))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.stage == 'pt':
        def build_ids(hf, n):
            from transformers import AutoTokenizer
            from download_gsm8k import SHA256
            import hashlib
            assert hashlib.sha256(a.dataset.read_bytes()).hexdigest() == SHA256
            question = json.loads(a.dataset.read_text().splitlines()[0])['question']
            text = f'<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n'
            return AutoTokenizer.from_pretrained(hf)(text, add_special_tokens=False)['input_ids'][:n]
        rail.build_ids = build_ids
        rail.cmd_pt(a)
    elif a.stage == 'lt':
        if not a.tflite:
            p.error('--tflite is required for lt')
        rail.cmd_lt(a)
    else:
        import numpy as np
        pt, lt = np.load(a.pt)['pt'], np.load(a.lt)['lt']
        assert pt.shape == lt.shape and len(pt) == 48
        rail.cmd_cmp(a)
        p1, l1 = pt.argmax(-1), lt.argmax(-1)
        p5 = np.argsort(-pt, -1)[:, :5]
        P, Q = rail.softmax(pt), rail.softmax(lt)
        rows = []
        for i in range(len(pt)):
            rows.append({'position': i, 'top1': bool(p1[i] == l1[i]), 'top5': bool(l1[i] in p5[i]),
                         'pearson': float(np.corrcoef(pt[i], lt[i])[0, 1]),
                         'kl_nats': float(np.sum(P[i] * (np.log(P[i] + 1e-9) - np.log(Q[i] + 1e-9)))),
                         'max_abs_diff': float(np.max(np.abs(pt[i] - lt[i])))})
        checks = {'finite': bool(np.isfinite(pt).all() and np.isfinite(lt).all()),
                  'top1_48': sum(r['top1'] for r in rows) == 48,
                  'top5_48': sum(r['top5'] for r in rows) == 48,
                  'pearson_min': min(r['pearson'] for r in rows) >= .9999,
                  'mean_kl': bool(np.mean([r['kl_nats'] for r in rows]) <= .001)}
        save_json(a.out, {'positions': rows, 'checks': checks, 'verdict': 'PASS' if all(checks.values()) else 'FAIL'})
        raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
