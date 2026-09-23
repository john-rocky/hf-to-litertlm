"""System One reference: upstream formatting + independent LiteRT graph rows.

Public command: python -B scripts/systemone_litert.py --bundle FILE
  --tokenizer SNAPSHOT_DIR --request REQUEST.json
The optional JSONL mode executes that same request function for the fixture gate.
Mirrors pinned decider/infer.py:141-176; source hashes: results/decider_vendored.json.
"""
import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from bundle_cache import unpack_bundle
from litert_readout import Readout,softmax32
from decider_vendored.infer import Example,Q,neutralize_options
from decider_vendored.prompt import build,label_table,MAX_OPTIONS
from decider_vendored.systemone import render_state,render_question,plan_rows,assemble,unique_tokens


@contextlib.contextmanager
def diagnostics_to_stderr():
    """Keep native runtime diagnostics out of the JSON stdout protocol."""
    sys.stdout.flush();saved=os.dup(1)
    try:
        os.dup2(2,1)
        with contextlib.redirect_stdout(sys.stderr):yield
    finally:
        sys.stderr.flush();os.dup2(saved,1);os.close(saved)


class HFTokenizer:
    def __init__(self,path):self.inner=Tokenizer.from_file(str(Path(path)/'tokenizer.json'))
    def encode(self,text,add_special_tokens=False):return self.inner.encode(text,add_special_tokens=add_special_tokens).ids
    def decode(self,ids):return self.inner.decode(ids,skip_special_tokens=False)


class Keep:
    def shuffle(self,x):pass
    def sample(self,xs,k):return xs[:k]


class SystemOneLiteRT:
    def __init__(self,bundle,tokenizer):
        self.name=Path(bundle).name
        self.config=json.loads((Path(tokenizer)/'decider_config.json').read_text())
        assert self.config['temperature']==1.03 and self.config['isolated_levels'] is True
        assert self.config['schema_first'] is False and self.config['neutralize_none'] is False
        self.tok=HFTokenizer(tokenizer)
        self.labels,self.label_ids,_=label_table(self.tok)
        with diagnostics_to_stderr():
            self.cache,self.tflite,_,self.bundle_sha256=unpack_bundle(bundle)
            self.runner=Readout(self.tflite)
        self.last_rows=[]

    def plan(self,state,questions):
        ctx=render_state(state)
        rqs={k:render_question(v) for k,v in questions.items()}
        flat,index=plan_rows(rqs,isolated=self.config['isolated_levels'])
        if not flat:raise ValueError('At least one question is required')
        items=[]
        for r in flat:
            opts=neutralize_options(r['options'])[0] if self.config['neutralize_none'] else list(r['options'])
            item=build(Example(ctx,[Q(r['question'],opts,0)]),self.tok,Keep(),max_options=MAX_OPTIONS,
                       max_ctx_tokens=self.config['max_state_tokens'],layout='state_first')
            assert item['slots']==[len(item['ids'])-1]
            if len(item['ids'])>4096:raise ValueError(f"Graph cache limit is 4096 tokens per row; got {len(item['ids'])}")
            items.append(item)
        return rqs,index,items

    def system_one(self,state,questions):
        rqs,index,items=self.plan(state,questions)
        probs=[];self.last_rows=[]
        with diagnostics_to_stderr():
            for item in items:
                logits,chunks,steps,finite,seconds=self.runner.score(item['ids'])
                n=item['nopts'][0];labels=self.label_ids[:n]
                raw=logits[labels];p=softmax32(raw)
                if not finite or not np.isfinite(p).all():raise RuntimeError('Non-finite graph output')
                probs.append(p.tolist())
                self.last_rows.append(dict(ids=item['ids'],slot_index=item['slots'][0],nopts=n,label_token_ids=labels,
                    raw_label_logits_fp32=raw.tolist(),probabilities=p.tolist(),finite=True,chunk_plan=chunks,
                    wall_seconds_contended=seconds))
        return dict(model=self.name,answers=assemble(rqs,index,probs),
                    usage=dict(input_tokens=unique_tokens(items),output_tokens=0))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',required=True);ap.add_argument('--tokenizer',required=True)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--request');group.add_argument('--requests-jsonl')
    ap.add_argument('--trace',help='Optional gate evidence; no effect on stdout responses')
    args=ap.parse_args()
    requests=[json.loads(Path(args.request).read_text())] if args.request else [json.loads(line) for line in Path(args.requests_jsonl).read_text().splitlines() if line.strip()]
    runtime=SystemOneLiteRT(args.bundle,args.tokenizer)
    trace=[]
    for request in requests:
        result=runtime.system_one(request['state'],request['questions'])
        print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
        if args.trace:trace.append(dict(request=request,response=result,rows=runtime.last_rows))
    if args.trace:
        from common import write_json
        write_json(args.trace,dict(bundle_sha256=runtime.bundle_sha256,cache=str(runtime.cache),requests=trace))


if __name__=='__main__':main()
