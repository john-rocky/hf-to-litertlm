"""Eight fixed questions, fresh process per question; inspect relevance as well."""
import argparse,re
import json
from pathlib import Path
from export_float import sha256
from runtime_helpers import add_runtime_args,configure_output,run_cli,save_json,cli_success
SUFFIX = " Answer briefly."
QUESTIONS = [
    ("17+25=42",        "What is 17 + 25?",                                  r"\b42\b"),
    ("capital=Tokyo",   "What is the capital of Japan?",                     r"tokyo"),
    ("opp(hot)=cold",   'What is the opposite of "hot"?',                    r"\bcold\b"),
    ("days/week=7",     "How many days are in a week?",                      r"\bseven\b|\b7\b"),
    ("thanks(fr)=merci", 'How do you say "thank you" in French?',            r"merci"),
    ("8*7=56",          "What is 8 times 7?",                                r"\b56\b"),
    ("0.9>0.11",        "Which is larger: 0.9 or 0.11?",                     r"0\.9"),
    ("rhyme=blue",      'Complete the rhyme: "Roses are red, violets are ___"', r"\bblue\b"),
]

def degenerate(text):
    words = re.findall(r"\w+", text.lower())
    if len(words) >= 12:
        from collections import Counter
        top = Counter(words).most_common(1)[0][1]
        if top / len(words) > 0.5:
            return True
    return len(text.strip()) == 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    add_runtime_args(p,'gate8q.json')
    p.add_argument('--backend',choices=['cpu','gpu'])
    p.add_argument('--review',type=Path)
    p.add_argument('--on-topic',choices=['yes','no'])
    a=p.parse_args()
    if a.review:
        if not a.on_topic:p.error('--review requires --on-topic yes or no after reading every answer')
        data=json.loads(a.review.read_text());data['manual_prompt_relevance']=a.on_topic=='yes'
        data['verdict']='PASS' if data['automatic_verdict']=='PASS' and data['manual_prompt_relevance'] else 'FAIL'
        save_json(a.review,data);return
    if not a.backend:p.error('--backend is required for generation')
    configure_output(a.out)
    rows=[]
    data={'model':a.model,'backend':a.backend,'runtime':a.runtime,'results':rows,'correct':0,'degenerate':0,'manual_prompt_relevance':'PENDING','model_sha256':sha256(a.model)}
    for i,(label,q,pattern) in enumerate(QUESTIONS,1):
        command=[a.cli,'run',a.model,'--prompt',q+SUFFIX,'--backend',a.backend,'--cache','no','--temperature','0','--seed','0','--thinking','false']
        record=run_cli(command,Path(a.out).stem+'_q'+str(i))
        text=record['stdout'].strip();ok=cli_success(record) and bool(re.search(pattern,text,re.I));dg=degenerate(text)
        rows.append({'label':label,'question':q+SUFFIX,'pattern':pattern,'ok':ok,'degenerate':dg,'text':text,'execution':record})
        data.update(correct=sum(r['ok'] for r in rows),degenerate=sum(r['degenerate'] for r in rows))
        save_json(a.out,data);print(label,ok,repr(text),flush=True)
        if not cli_success(record):break
    data['automatic_verdict']='PASS' if data['correct']==8 and data['degenerate']==0 else 'FAIL'
    data['verdict']='REVIEW_REQUIRED' if data['automatic_verdict']=='PASS' else 'FAIL'
    save_json(a.out,data)
    raise SystemExit(0 if data['automatic_verdict']=='PASS' else 1)
if __name__=='__main__':main()
