"""HF joint tokenization of decoded oracle ids; never edits the oracle."""
import importlib.metadata
from transformers import AutoTokenizer
from common import ROOT, snapshot_path, read_json, write_json, file_fact


def main():
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path(), local_files_only=True)
    rows = []
    for source in read_json('fixtures/oracle_fp32.json')['rows']:
        ids = source['ids']
        raw = tokenizer.decode(ids)
        joint = tokenizer.encode(raw, add_special_tokens=False)
        clean = joint == ids
        divergence = None
        if not clean:
            pos = next((i for i, (a, b) in enumerate(zip(ids, joint)) if a != b), min(len(ids), len(joint)))
            def context(seq):
                return [dict(position=i, id=seq[i], token=tokenizer.convert_ids_to_tokens(seq[i]),
                             decoded=tokenizer.decode([seq[i]]))
                        for i in range(max(0, pos-3), min(len(seq), pos+5))]
            divergence = dict(position=pos, piecewise=context(ids), joint=context(joint))
        label_ids = [tokenizer.encode(text, add_special_tokens=False) for text in source['label_texts']]
        assert label_ids == [[i] for i in source['label_token_ids']]
        rows.append(dict(row_id=source['row_id'], family=source['family'], boundary_clean=clean,
                         n_tokens=len(ids), joint_n_tokens=len(joint), raw_text=raw, joint_ids=joint,
                         first_divergence=divergence, label_text_single_token_check=True))
    result = dict(status='PASS', method='tok.encode(tok.decode(ids), add_special_tokens=False)',
                  tokenizer_class=type(tokenizer).__name__, transformers=importlib.metadata.version('transformers'),
                  source_oracle=file_fact(ROOT/'fixtures/oracle_fp32.json'), rows=rows,
                  row_count=len(rows), boundary_clean_rows=sum(r['boundary_clean'] for r in rows),
                  differing_rows=sum(not r['boundary_clean'] for r in rows),
                  differing_row_ids=[r['row_id'] for r in rows if not r['boundary_clean']])
    write_json('results/tokenization_joint_vs_piecewise.json', result)
    print({k: result[k] for k in ['status', 'row_count', 'boundary_clean_rows', 'differing_rows', 'differing_row_ids']}, flush=True)


if __name__ == '__main__':
    main()
