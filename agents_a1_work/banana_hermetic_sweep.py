"""One fresh engine per fill length; fixed BANANA prompt and acceptance."""
import argparse,re
from pathlib import Path
from export_float import sha256
from runtime_helpers import add_runtime_args,configure_output,run_cli,save_json,cli_success

def main():
    p=argparse.ArgumentParser(description=__doc__);add_runtime_args(p,'banana.json')
    p.add_argument('--backend',choices=['cpu','gpu'],required=True)
    p.add_argument('--fill-lo',type=int,default=12);p.add_argument('--fill-hi',type=int)
    a=p.parse_args();configure_output(a.out)
    hi=a.fill_hi if a.fill_hi is not None else 51 if a.backend=='cpu' else 31
    assert hi>=a.fill_lo
    rows=[];data={'model':a.model,'backend':a.backend,'runtime':a.runtime,'n':hi-a.fill_lo+1,'results':rows,'model_sha256':sha256(a.model)}
    for fill in range(a.fill_lo,hi+1):
        prompt='well '*fill+'Reply with only the single word BANANA.'
        record=run_cli([a.cli,'run',a.model,'--prompt',prompt,'--backend',a.backend,'--cache','no','--temperature','0','--seed','0','--thinking','false'],Path(a.out).stem+'_fill'+str(fill))
        text=record['stdout'].strip();ok=cli_success(record) and bool(re.search(r'\bBANANA\b',text)) and len(text)<200
        rows.append({'fill':fill,'ok':ok,'text':text,'characters':len(text),'execution':record})
        data.update(clean=sum(r['ok'] for r in rows),executed=len(rows));save_json(a.out,data)
        print(fill,ok,repr(text),flush=True)
        if not cli_success(record):break
    data['verdict']='PASS' if data['clean']==data['n'] else 'FAIL';save_json(a.out,data)
    raise SystemExit(0 if data['verdict']=='PASS' else 1)
if __name__=='__main__':main()
