"""Pinned upstream system_one oracle and an independently assembled row capture.

Mirrors installed decider/infer.py (c4daaac...), lines 151-175:
  151: ctx = render_state(state); rqs = {k: render_question(v) ...}
  153: flat, index = plan_rows(rqs, isolated)
  154: rows = [[r] for r in flat] if independent else [flat]
  158-159: items = [build(Example(ctx, [Q(...) ...]), self.m.tok, _Keep(),
                        max_options=MAX_OPTIONS, max_ctx_tokens=max_state_tokens, layout=layout) ...]
  164: probs = []; per = max(1, max_fwd_tokens // max(len(it['ids']) for it in items))
  169-171: bt = collate(...); lg = self.m.slot_logits(...); pr = torch.softmax(lg / self.T, -1).cpu()
  174-176: flatp = [...]; return {'answers': assemble(rqs, index, flatp),
                               'usage': {'input_tokens': unique_tokens(items), 'output_tokens': 0}}

Two forwards per fixture: the untouched public method, then the same package's
row functions. Only the replay returns raw logits for capture. No custom prompt,
temperature, answer rounding, or delta-rule math is introduced.
"""
import functools
import inspect
import json
import time
import traceback
from collections import Counter
import numpy as np
import torch
from decider.infer import Decider, Example, Q
from decider.model import collate
from decider.prompt import build, MAX_OPTIONS, label_table
from decider.systemone import render_state, render_question, plan_rows, assemble, unique_tokens
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
from common import ROOT, snapshot_path, read_json, write_json, sha256, status
from make_fixtures import Keep


def main():
    start = time.monotonic()
    torch.set_num_threads(4)
    assert torch.get_num_threads() == 4
    fixtures = read_json('fixtures/fixtures.json')
    d = Decider(str(snapshot_path()), device='cpu', dtype=torch.float32, use_graphs=None)
    assert d.T == 1.03 and d.isolated_levels is True and d.schema_first is False
    assert d.neutralize_none is False and d.eng is None
    assert all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in d.m.parameters())
    counters = Counter()
    kernel_shapes = Counter()
    kernel_functions = {}
    # Record actual calls through the modeling globals; wrappers only delegate.
    for name in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule','causal_conv1d_fn','causal_conv1d_update']:
        original = getattr(modeling, name)
        fn = inspect.unwrap(original)
        kernel_functions[name] = dict(module=fn.__module__,qualname=fn.__qualname__,
                                      file=inspect.getsourcefile(fn), first_line=inspect.getsourcelines(fn)[1])
        def instrument(fn, key):
            @functools.wraps(fn)
            def tracked(*args, **kwargs):
                counters[key] += 1
                first = args[0] if args else next(iter(kwargs.values()))
                if isinstance(first,torch.Tensor):
                    kernel_shapes[f'{key}:{list(first.shape)}:{first.dtype}:{first.device}'] += 1
                return fn(*args, **kwargs)
            return tracked
        setattr(modeling,name,instrument(original,name))
    labels, label_ids, _ = label_table(d.m.tok)
    assert d.m.letters.tolist() == label_ids
    result = dict(status='RUNNING', model='Mapika/decider-0.8b', dtype='float32',device='Apple M4 Max CPU',
                  torch_threads=4, temperature=d.T, independent=True, isolated_levels=True,layout='state_first',
                  fixture_sha256=sha256(ROOT/'fixtures/fixtures.json'), fixtures=[],rows=[],rejections=[])
    status('RUNNING','Pinned source verified; capturing fp32 system_one oracle and Gate 0 on 40 fixtures.')
    for fixture in fixtures:
        elapsed_start=time.monotonic()
        try:
            official=d.system_one(fixture['state'],fixture['questions'])
        except ValueError as error:
            rejection=dict(fixture_id=fixture['id'],family=fixture['family'],error=repr(error),traceback=traceback.format_exc())
            result['rejections'].append(rejection)
            write_json('fixtures/oracle_fp32.json',result)
            print('UPSTREAM REJECTION',json.dumps(rejection),flush=True)
            continue
        official_seconds=time.monotonic()-elapsed_start
        ctx=render_state(fixture['state'])
        rqs={k:render_question(v) for k,v in fixture['questions'].items()}
        flat,index=plan_rows(rqs,True)
        items=[build(Example(ctx,[Q(r['question'],list(r['options']),0)]),d.m.tok,Keep(),
                     max_options=MAX_OPTIONS,max_ctx_tokens=32768,layout='state_first') for r in flat]
        assert all(it['slots']==[len(it['ids'])-1] for it in items)
        per=max(1,65536//max(len(it['ids']) for it in items))
        captured_logits=[]
        captured_probs=[]
        batches=[]
        with torch.no_grad():
            for i in range(0,len(items),per):
                bt=collate(items[i:i+per],d.m.tok.pad_token_id)
                lg=d.m.slot_logits(*[bt[k].to(d.dev) for k in ('input_ids','attention_mask','slot_idx','slot_batch','nopts')])
                pr=torch.softmax(lg/d.T,-1).cpu()
                captured_logits.extend(lg.cpu())
                captured_probs.extend(pr.tolist())
                batches.append(dict(shape=list(bt['input_ids'].shape),slot_idx=bt['slot_idx'].tolist(),nopts=bt['nopts'].tolist()))
        assembled=assemble(rqs,index,captured_probs)
        gate0=assembled==official['answers']
        assert official['usage']==dict(input_tokens=unique_tokens(items),output_tokens=0)
        fixture_result=dict(id=fixture['id'],family=fixture['family'],official=official,
                            rendered_state=ctx,rendered_questions=rqs,plan_rows=flat,index=index,
                            assembled=assembled,gate0_exact=gate0,replay_batches=batches,
                            official_wall_seconds_contended=official_seconds,
                            total_wall_seconds_contended=time.monotonic()-elapsed_start)
        result['fixtures'].append(fixture_result)
        for i,(item,lg,p) in enumerate(zip(items,captured_logits,captured_probs)):
            n=item['nopts'][0]
            raw=lg[:n].tolist()
            p=p[:n]
            qid,kind,first,count=next(entry for entry in index if entry[2]<=i<entry[2]+entry[3])
            arr=np.array(p)
            ordered=np.sort(arr)
            gap=float(ordered[-1]-ordered[-2])
            finite=bool(np.isfinite(raw).all() and np.isfinite(arr).all())
            assert finite
            row=dict(row_id=f"{fixture['id']}/{i:02}",fixture_id=fixture['id'],family=fixture['family'],
                     row_index=i,question_id=qid,question_type=rqs[qid]['type'],level_index=i-first if kind=='iso' else None,
                     rendered_question=flat[i],ids=item['ids'],slot_index=item['slots'][0],n_tokens=len(item['ids']),nopts=n,
                     slot_token_id=item['ids'][-1],slot_token_text=d.m.tok.decode([item['ids'][-1]]),
                     label_token_ids=label_ids[:n],label_texts=labels[:n],raw_slot_logits_fp32=raw,probabilities=p,
                     argmax=int(arr.argmax()),oracle_top2_gap=gap,tie=gap<=1e-4,finite=finite,
                     oracle_usage_input_tokens=official['usage']['input_tokens'],
                     oracle_usage_scope='fixture; shared row prefix counted once by upstream unique_tokens',
                     single_row_unique_tokens=unique_tokens([item]),
                     token_zero_positions=[j for j,token in enumerate(item['ids']) if token==0])
            result['rows'].append(row)
        write_json('fixtures/oracle_fp32.json',result)
        print(json.dumps(dict(fixture=fixture['id'],rows=len(items),usage=official['usage'],gate0=gate0,
                              elapsed_seconds_contended=fixture_result['total_wall_seconds_contended'])),flush=True)
        if not gate0:
            write_json('results/gate0_mismatch.json',fixture_result)
            raise AssertionError('Gate 0 mismatch; inspect script before proceeding (30-minute repair cap)')
    assert counters['torch_chunk_gated_delta_rule'] > 0, 'kernel instrumentation did not observe execution'
    family={}
    for f in sorted(set(r['family'] for r in result['rows'])):
        rows=[r for r in result['rows'] if r['family']==f]
        lengths=[r['n_tokens'] for r in rows]
        family[f]=dict(rows=len(rows),min_tokens=min(lengths),median_tokens=float(np.median(lengths)),max_tokens=max(lengths),
                       ties=sum(r['tie'] for r in rows))
    result.update(status='PASS',wall_seconds_contended=time.monotonic()-start,family_summary=family)
    write_json('fixtures/oracle_fp32.json',result)
    write_json('results/gate0.json',dict(status='PASS',fixtures=len(result['fixtures']),rows=len(result['rows']),
                                      exact_fixtures=sum(f['gate0_exact'] for f in result['fixtures']),
                                      family_summary=family,rejections=result['rejections'],
                                      max_abs_label_logit=max(abs(x) for r in result['rows'] for x in r['raw_slot_logits_fp32']),
                                      all_finite=all(r['finite'] for r in result['rows']),
                                      ties=[r['row_id'] for r in result['rows'] if r['tie']],
                                      slots={repr(r['slot_token_text']):r['slot_token_id'] for r in result['rows']},
                                      wall_seconds_contended=result['wall_seconds_contended']))
    env=read_json('results/environment_oracle.json')
    env.update(delta_rule_kernel_path='transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule (CPU torch fallback)',
               kernel_functions=kernel_functions,actual_kernel_call_counts=dict(counters),actual_kernel_shapes=dict(kernel_shapes),
               attn_implementation=d.m.lm.config._attn_implementation,
               model_parameters=sum(p.numel() for p in d.m.parameters()),temperature=d.T,
               isolated_levels=d.isolated_levels,schema_first=d.schema_first,neutralize_none=d.neutralize_none,
               torch_num_threads=torch.get_num_threads())
    write_json('results/environment_oracle.json',env)
    print('GATE 0 PASS',len(result['fixtures']),'fixtures',len(result['rows']),'rows',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        (ROOT/'logs/oracle.traceback.txt').write_text(traceback.format_exc())
        raise
