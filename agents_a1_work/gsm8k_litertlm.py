"""Fresh-process GSM8K; preserve the original extraction protocol and raw answers."""
import argparse,hashlib,json,os,re
from pathlib import Path
from runtime_helpers import HERE,ROOT,run_cli,save_json,configure_output,cli_success
from export_float import sha256
from download_gsm8k import SHA256
DATA = os.environ.get('GSM8K_DATA','evaldata/gsm8k_test.jsonl')

COT = ("\n\nSolve this step by step. After your reasoning, write the final answer on its own "
       "line in the exact form:\n#### <number>")

def load_q(n):
    out = []
    for line in open(DATA):
        d = json.loads(line)
        out.append((d["question"], d["answer"].split("####")[-1].strip().replace(",", "")))
        if len(out) >= n:
            break
    return out

def extract(text):
    """GSM8K-standard: prefer '#### N', then 'answer is/: N', then the last number."""
    if not text:
        return None
    t = text.replace(",", "")
    m = re.findall(r"####\s*\$?(-?\d+(?:\.\d+)?)", t)
    if m: return m[-1].rstrip(".0") if "." in m[-1] else m[-1]
    m = re.findall(r"\\boxed\{\s*\$?(-?\d+(?:\.\d+)?)", t)  # OLMo-2 etc. mark the final answer with \boxed{}
    if m: return m[-1].rstrip(".0") if "." in m[-1] else m[-1]
    m = re.findall(r"(?:answer|total|result)\s*(?:is|:|=)\s*\$?(-?\d+(?:\.\d+)?)", t, re.I)
    if m: return m[-1].rstrip(".0") if "." in m[-1] else m[-1]
    m = re.findall(r"-?\d+(?:\.\d+)?", t)
    if not m: return None
    v = m[-1]
    return v.rstrip(".0") if "." in v else v

def norm(x):
    if x is None: return None
    x = x.replace(",", "").lstrip("$")
    try:
        f = float(x); return str(int(f)) if f == int(f) else str(f)
    except: return x


def main():
    global DATA
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['int8','mixed_int4'],required=True)
    p.add_argument('--model',type=Path)
    p.add_argument('--cli',default=os.environ.get('LITERT_LM','litert-lm'))
    p.add_argument('--dataset',default=DATA)
    p.add_argument('--out',type=Path)
    p.add_argument('--n',type=int,default=100)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--gate8q',type=Path,required=True)
    p.add_argument('--sweep',type=Path,required=True)
    p.add_argument('--keep-cache',action='store_true',help='Keep cache until the second variant finishes; default removes generated cache files.')
    a=p.parse_args();DATA=a.dataset
    assert 0<a.n<=100
    assert hashlib.sha256(Path(DATA).read_bytes()).hexdigest()==SHA256
    model=(a.model or ROOT/('Agents-A1-4B_'+a.variant+'.litertlm')).resolve()
    fingerprint=sha256(model)
    gate=json.loads(a.gate8q.read_text());sweep=json.loads(a.sweep.read_text())
    assert gate['correct']==8 and gate['degenerate']==0 and gate['manual_prompt_relevance'] is True
    assert sweep['clean']==20 and sweep['verdict']=='PASS'
    assert all(d['model_sha256']==fingerprint and d['backend']=='gpu' for d in [gate,sweep])
    out=a.out or ROOT/('results/gsm8k_'+a.variant+'.json');configure_output(out)
    rows=[]
    if out.exists():
        assert a.resume,'Use --resume to continue saved answers without rerunning them.'
        previous=json.loads(out.read_text())
        assert previous['model_sha256']==fingerprint and previous['dataset_sha256']==SHA256
        rows=previous['results']
        assert [r['q'] for r in rows]==list(range(len(rows))) and len(rows)<=a.n
    summary={'tag':a.variant,'n':len(rows),'planned_n':a.n,'correct':sum(r['ok'] for r in rows),'results':rows,
             'backend':'gpu','runtime':'0.17.1','cache':'disk','thinking':False,'max_context_tokens':4096,'decode_token_cap':None,
             'dataset':str(DATA),'dataset_sha256':SHA256,'model_sha256':fingerprint,'context_limit_count':0,'verdict':'RUNNING'}
    cache_before={x.name for x in model.parent.glob(model.name+'_*_mldrift_*_cache.bin')}
    try:
        for i,(q,gold) in enumerate(load_q(a.n)):
            if i<len(rows):continue
            observation=out.parent/'observations'/f'{a.variant}_q{i+1}.json'
            observation.parent.mkdir(parents=True,exist_ok=True)
            env={'AGENTS_A1_GSM_OBSERVATION':str(observation.resolve()),
                 'PYTHONPATH':str(HERE/'gsm_observer')+os.pathsep+os.environ.get('PYTHONPATH','')}
            execution=run_cli([a.cli,'run',str(model),'--prompt',q+COT,'--backend','gpu','--thinking','false',
                              '--temperature','0','--seed','0','--cache','disk'],f'gsm8k_{a.variant}_q{i+1}',env_overrides=env)
            raw_answer=execution['stdout'].strip();txt=raw_answer if cli_success(execution) else ''
            ok = norm(extract(txt)) == norm(gold)
            row={'q': i, 'ok': bool(ok), 'pred': norm(extract(txt)), 'gold': norm(gold)}
            obs=json.loads(observation.read_text()) if observation.exists() else {'context_limit_reached':None}
            if 'prompt_sha256' in obs:assert obs['prompt_sha256']==hashlib.sha256((q+COT).encode()).hexdigest()
            row.update(question=q,answer=raw_answer,answer_characters=len(raw_answer),answer_utf8_bytes=len(raw_answer.encode()),
                       scoring_text=txt,context_limit_reached=obs['context_limit_reached'],native_token_observation=obs,execution=execution)
            rows.append(row)
            summary.update(n=len(rows),correct=sum(r['ok'] for r in rows),acc=sum(r['ok'] for r in rows)/len(rows),
                           context_limit_count=sum(r['context_limit_reached'] is True for r in rows),
                           context_limit_unknown=sum(r['context_limit_reached'] is None for r in rows))
            save_json(out,summary)
            if (i+1)%10==0:print(a.variant,summary['correct'],'/',len(rows),flush=True)
            if not cli_success(execution):break
        summary.update(n=len(rows),correct=sum(r['ok'] for r in rows),acc=sum(r['ok'] for r in rows)/len(rows) if rows else None,context_limit_count=sum(r['context_limit_reached'] is True for r in rows),context_limit_unknown=sum(r['context_limit_reached'] is None for r in rows))
        summary['verdict']='COMPLETE' if len(rows)==a.n else 'PARTIAL'
    finally:
        summary.update(n=len(rows),correct=sum(r['ok'] for r in rows),acc=sum(r['ok'] for r in rows)/len(rows) if rows else None,
                       context_limit_count=sum(r['context_limit_reached'] is True for r in rows),
                       context_limit_unknown=sum(r['context_limit_reached'] is None for r in rows))
        if summary['verdict']=='RUNNING':summary['verdict']='PARTIAL'
        deleted=[]
        if not a.keep_cache:
            for path in model.parent.glob(model.name+'_*_mldrift_*_cache.bin'):
                if path.name not in cache_before:
                    deleted.append({'name':path.name,'bytes':path.stat().st_size});path.unlink()
        summary['cache_cleanup']={'deleted':deleted,'preexisting_preserved':sorted(cache_before),'keep_cache':a.keep_cache}
        save_json(out,summary)
    print(json.dumps({k:summary[k] for k in ['tag','n','correct','verdict','context_limit_count']}))
if __name__=='__main__':main()
