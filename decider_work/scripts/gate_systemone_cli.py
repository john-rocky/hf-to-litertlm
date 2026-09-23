"""Execute the actual JSON CLI on all 40 fixtures, then compare upstream answers."""
import json
import subprocess
import sys
import time
from decimal import Decimal
from common import ROOT,read_json,write_json,snapshot_path,status


def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b):return [dict(path=path,structural=True,oracle=a,actual=b)]
        return [d for k in a for d in differences(a[k],b[k],path+'/'+k)]
    if type(a) in (int,float) and type(b) in (int,float):
        delta=abs(Decimal(str(a))-Decimal(str(b)))
        return [] if not delta else [dict(path=path,structural=False,oracle=a,actual=b,absolute_difference=float(delta))]
    return [] if a==b else [dict(path=path,structural=True,oracle=a,actual=b)]


def main():
    start=time.monotonic();status('RUNNING','Round 4 A2: actual System One CLI on both final files, 40 fixtures / 120 rows each.')
    fixtures=read_json('fixtures/fixtures.json');oracle=read_json('fixtures/oracle_fp32.json')
    official={f['id']:f['official'] for f in oracle['fixtures']}
    source_rows={f['id']:[r for r in oracle['rows'] if r['fixture_id']==f['id']] for f in fixtures}
    request_file=ROOT/'fixtures/systemone_requests.jsonl'
    request_file.write_text(''.join(json.dumps({k:f[k] for k in ('state','questions')},ensure_ascii=False)+'\n' for f in fixtures))
    one=ROOT/'fixtures/systemone_example.json';one.write_text(json.dumps({k:fixtures[0][k] for k in ('state','questions')},ensure_ascii=False,indent=2)+'\n')
    result=dict(status='RUNNING',device='Apple M4 Max CPU',num_threads=4,timing='contended (5 other Codex runs)',
                implementation='scripts/systemone_litert.py; verbatim vendored upstream prompt/systemone + infer excerpt',bundles={})
    for name in ('fp16','int8'):
        before=time.monotonic();bundle=ROOT/read_json('results/final_bundles.json')['bundles'][name]['bundle']['path']
        command=[sys.executable,'-B','scripts/systemone_litert.py','--bundle',str(bundle),'--tokenizer',str(snapshot_path()),
                 '--requests-jsonl',str(request_file),'--trace',f'results/systemone_{name}_trace.json']
        stdout=ROOT/f'logs/systemone_{name}.jsonl';stderr=ROOT/f'logs/systemone_{name}.stderr.log'
        with stdout.open('w') as out,stderr.open('w') as err:run=subprocess.run(command,stdout=out,stderr=err)
        if run.returncode:raise RuntimeError(f'CLI failed; see {stderr}')
        actual=[json.loads(line) for line in stdout.read_text().splitlines()]
        assert len(actual)==len(fixtures)==40
        trace=read_json(f'results/systemone_{name}_trace.json')['requests'];rows=[]
        for f,response,tr in zip(fixtures,actual,trace):
            src=official[f['id']];ds=differences(src['answers'],response['answers'])
            maximum=max((d.get('absolute_difference',0) for d in ds),default=0)
            for recorded,source in zip(tr['rows'],source_rows[f['id']]):
                assert recorded['ids']==source['ids'] and recorded['label_token_ids']==source['label_token_ids']
            assert len(tr['rows'])==len(source_rows[f['id']]) and response['usage']==src['usage']
            rows.append(dict(fixture_id=f['id'],family=f['family'],exact=response['answers']==src['answers'],
                max_answer_difference=maximum,structural_differences=any(d['structural'] for d in ds),differences=ds,
                within_1e4=maximum<=0.0001 and not any(d['structural'] for d in ds),usage_exact=True,response=response,oracle=src))
        exact=sum(r['exact'] for r in rows);within=sum(r['within_1e4'] for r in rows)
        result['bundles'][name]=dict(status=('PASS' if within==40 else 'FAIL') if name=='fp16' else 'INFORMATIONAL',fixtures=40,rows=120,
            exact_fixtures=exact,within_1e4_fixtures=within,max_answer_difference=max(r['max_answer_difference'] for r in rows),
            usage_exact_fixtures=40,piecewise_ids_exact=True,all_finite=True,command=command,
            stdout_evidence=str(stdout.relative_to(ROOT)),stderr_evidence=str(stderr.relative_to(ROOT)),fixtures_detail=rows,
            wall_seconds_contended=time.monotonic()-before)
        write_json('results/systemone_cli_gate.json',result)
        print(name,'exact',exact,'within 1e-4',within,'max',result['bundles'][name]['max_answer_difference'],flush=True)
    # Exercise the documented single-request flag too; the main sweep uses the same function.
    command=[sys.executable,'-B','scripts/systemone_litert.py','--bundle',str(ROOT/'exports/final/decider-0.8b_fp16.litertlm'),
             '--tokenizer',str(snapshot_path()),'--request',str(one)]
    with open(ROOT/'logs/systemone_single.json','w') as out,open(ROOT/'logs/systemone_single.stderr.log','w') as err:
        run=subprocess.run(command,stdout=out,stderr=err)
    assert run.returncode==0
    single=read_json('logs/systemone_single.json');assert single==result['bundles']['fp16']['fixtures_detail'][0]['response']
    result.update(status=result['bundles']['fp16']['status'],single_request_command=command,single_request_stdout_exact=True,
                  wall_seconds_contended=time.monotonic()-start)
    write_json('results/systemone_cli_gate.json',result)


if __name__=='__main__':main()
