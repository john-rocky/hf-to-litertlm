"""Rescore saved answers; preserve extraction order and normalize numerically."""
import argparse,hashlib,json,re
from pathlib import Path
from runtime_helpers import ROOT,save_json
from gsm8k_litertlm import norm

def extract_corrected(text):
    if not text:return None
    t=text.replace(',','')
    for pattern,flags in [(r'####\s*\$?(-?\d+(?:\.\d+)?)',0),
                          (r'\\boxed\{\s*\$?(-?\d+(?:\.\d+)?)',0),
                          (r'(?:answer|total|result)\s*(?:is|:|=)\s*\$?(-?\d+(?:\.\d+)?)',re.I),
                          (r'-?\d+(?:\.\d+)?',0)]:
        matches=re.findall(pattern,t,flags)
        if matches:return norm(matches[-1])
    return None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--int8',type=Path,default=ROOT/'results/gsm8k_int8.json')
    p.add_argument('--mixed-int4',type=Path,default=ROOT/'results/gsm8k_mixed_int4.json')
    p.add_argument('--out',type=Path,default=ROOT/'results/gsm8k_ab_corrected.json')
    p.add_argument('--original-out',type=Path,default=ROOT/'results/gsm8k_ab.json')
    args=p.parse_args()
    result={'protocol':'saved answers only; same extraction order (####, boxed, answer/total/result, last number); numeric float normalization replaces rstrip(".0")','index_base':0,'backend':'gpu','runtime':'0.17.1','variants':{},'changed_verdict_rows':[]}
    maps={}
    for variant in ['int8','mixed_int4']:
        source=args.int8 if variant=='int8' else args.mixed_int4;data=json.loads(source.read_text());rows=[]
        for r in data['results']:
            text=r['scoring_text'];pred=extract_corrected(text);ok=pred==norm(r['gold'])
            row={'q':r['q'],'question_number':r['q']+1,'original_ok':r['ok'],'corrected_ok':ok,'original_pred':r['pred'],'corrected_pred':pred,'gold':r['gold'],'context_limit_reached':r['context_limit_reached']}
            rows.append(row)
            if ok!=r['ok']:result['changed_verdict_rows'].append({'variant':variant,**row,'raw_answer':r['answer']})
        maps[variant]={r['q']:r for r in rows}
        count=sum(r['corrected_ok'] for r in rows)
        result['variants'][variant]={'n':data['n'],'planned_n':100,'original_correct':data['correct'],'correct':count,'acc':count/data['n'] if data['n'] else None,'verdict':data['verdict'],'context_limit_count':data['context_limit_count'],'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rows':rows}
    a,b=maps['int8'],maps['mixed_int4'];paired=sorted(a.keys()&b.keys());result['paired_n']=len(paired)
    for name,oa,ob in [('int8_right_int4_wrong',True,False),('int8_wrong_int4_right',False,True),('both_wrong',False,False),('both_right',True,True)]:
        result[name]=[q for q in paired if a[q]['corrected_ok']==oa and b[q]['corrected_ok']==ob]
    result['context_limit_count']=sum(v['context_limit_count'] for v in result['variants'].values())
    save_json(args.out,result)
    original={'variants':{k:{'n':v['n'],'correct':v['original_correct'],'context_limit_count':v['context_limit_count']} for k,v in result['variants'].items()},'paired_n':len(paired)}
    for name,oa,ob in [('int8_right_int4_wrong',True,False),('int8_wrong_int4_right',False,True),('both_wrong',False,False),('both_right',True,True)]:
        original[name]=[q for q in paired if maps['int8'][q]['original_ok']==oa and maps['mixed_int4'][q]['original_ok']==ob]
    original['context_limit_count']=result['context_limit_count']
    save_json(args.original_out,original)
    print(json.dumps({'variants':{k:{kk:v[kk] for kk in ['n','original_correct','correct']} for k,v in result['variants'].items()},'changed_verdict_rows':[{k:r[k] for k in ['variant','question_number','original_pred','corrected_pred']} for r in result['changed_verdict_rows']]}),flush=True)
if __name__=='__main__':main()
