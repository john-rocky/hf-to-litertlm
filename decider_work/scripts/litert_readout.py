"""Raw option-letter readout, CompiledModel CPU only, one final decode per row.

Uses the named-buffer write/run_by_name/read pattern from the copied owner
scripts/parity_logits_bigmodel.py. Every non-token/position/mask state is carried;
each row starts from zero state. Exact-fit chunks never insert pad positions.
"""
import json
import time
import traceback
from collections import Counter
import numpy as np
from tokenizers import Tokenizer
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.options import Options, CpuOptions
from common import ROOT, LADDER, snapshot_path, read_json, write_json, sha256, status


def chunk_plan(n):
    plan=[]
    p=0
    for length in LADDER:
        while n-p>=length:
            plan.append(dict(signature=f'prefill_{length}',start=p,length=length))
            p+=length
    assert p==n
    return plan


def softmax32(logits):
    z=np.asarray(logits,np.float32)/np.float32(1.03)
    e=np.exp(z-z.max())
    return e/e.sum(dtype=np.float32)


class Readout:
    def __init__(self,path,expected_cache_length=4096):
        start=time.monotonic()
        self.model=CompiledModel.from_file(str(path),options=Options(
            hardware_accelerators=HardwareAccelerator.CPU,cpu_options=CpuOptions(num_threads=4)))
        self.load_seconds=time.monotonic()-start
        self.signatures=self.model.get_signature_list()
        assert set(self.signatures)=={'decode',*(f'prefill_{n}' for n in LADDER)}
        self.inputs={key:self.model.get_input_tensor_details(key) for key in self.signatures}
        self.outputs={key:self.model.get_output_tensor_details(key) for key in self.signatures}
        self.states={n:d for n,d in self.inputs['decode'].items() if n not in ('tokens','input_pos','mask')}
        assert len(self.states)==48
        for key in self.signatures:
            assert set(self.inputs[key])=={'tokens','input_pos','mask',*self.states}
            assert set(self.outputs[key])==set(self.states)|({'logits'} if key=='decode' else set())
            for name,detail in self.states.items():
                assert np.dtype(detail['dtype'])==np.float32
                assert list(self.inputs[key][name]['shape'])==list(detail['shape'])
                assert list(self.outputs[key][name]['shape'])==list(detail['shape'])
        self.cache_length=int(self.inputs['decode']['mask']['shape'][-1])
        assert self.cache_length==expected_cache_length

    def zero_states(self):
        return {name:np.zeros(detail['shape'],dtype=detail['dtype']) for name,detail in self.states.items()}

    def step(self,key,ids,start,states):
        length=len(ids)
        inp=self.inputs[key]
        out=self.outputs[key]
        positions=np.arange(start,start+length,dtype=inp['input_pos']['dtype'])
        allowed=np.arange(self.cache_length)[None,:]<=positions[:,None]
        mask=np.where(allowed,np.float32(0.0),np.float32(-1e30)).reshape(inp['mask']['shape'])
        inputs=dict(states,tokens=np.asarray(ids,dtype=inp['tokens']['dtype']).reshape(inp['tokens']['shape']),
                    input_pos=positions.reshape(inp['input_pos']['shape']),mask=mask)
        input_buffers={}
        output_buffers={}
        try:
            for name,arr in inputs.items():
                assert list(arr.shape)==list(inp[name]['shape']),(key,name,arr.shape,inp[name]['shape'])
                b=self.model.create_input_buffer_by_name(key,name)
                input_buffers[name]=b
                b.write(np.ascontiguousarray(arr))
            output_buffers={name:self.model.create_output_buffer_by_name(key,name) for name in out}
            self.model.run_by_name(key,input_buffers,output_buffers)
            values={name:np.array(output_buffers[name].read(int(np.prod(detail['shape'])),detail['dtype']),
                                  dtype=detail['dtype'],copy=True).reshape(detail['shape']) for name,detail in out.items()}
            next_states={name:values[name] for name in self.states}
            finite=all(np.isfinite(v).all() for v in values.values())
            logits=values.get('logits')
            return next_states,None if logits is None else logits.reshape(-1),bool(finite)
        finally:
            for b in [*input_buffers.values(),*output_buffers.values()]:
                b.destroy()

    def score(self,ids):
        assert 1<=len(ids)<=self.cache_length
        start=time.monotonic()
        states=self.zero_states()
        plan=chunk_plan(len(ids)-1)
        all_finite=True
        timings=[]
        for chunk in plan:
            p=chunk['start']; length=chunk['length']; before=time.monotonic()
            states,logits,finite=self.step(chunk['signature'],ids[p:p+length],p,states)
            assert logits is None
            all_finite &= finite
            timings.append(dict(chunk,wall_seconds_contended=time.monotonic()-before,finite=finite))
        before=time.monotonic()
        states,logits,finite=self.step('decode',[ids[-1]],len(ids)-1,states)
        all_finite &= finite
        assert logits.size==248320
        timings.append(dict(signature='decode',start=len(ids)-1,length=1,
                            wall_seconds_contended=time.monotonic()-before,finite=finite))
        return logits,plan,timings,all_finite,time.monotonic()-start


def summarize(rows):
    non_ties=[r for r in rows if not r['tie']]
    return dict(rows=len(rows),non_tie_rows=len(non_ties),argmax_identical=sum(r['argmax_identical'] for r in rows),
                non_tie_argmax_identical=sum(r['argmax_identical'] for r in non_ties),ties=sum(r['tie'] for r in rows),
                max_abs_probability_error=max(r['max_abs_probability_error'] for r in rows),
                p95_abs_probability_error=float(np.percentile([r['max_abs_probability_error'] for r in rows],95)),
                max_abs_label_logit_error=max(r['max_abs_label_logit_error'] for r in rows),
                oracle_max_abs_label_logit=max(r['oracle_max_abs_label_logit'] for r in rows),
                lite_max_abs_label_logit=max(r['lite_max_abs_label_logit'] for r in rows),
                all_finite=all(r['finite'] for r in rows),
                row_length_tokens=dict(min=min(r['n_tokens'] for r in rows),median=float(np.median([r['n_tokens'] for r in rows])),max=max(r['n_tokens'] for r in rows)),
                wall_seconds_contended=sum(r['wall_seconds_contended'] for r in rows))


def main():
    start=time.monotonic()
    oracle=read_json('fixtures/oracle_fp32.json')
    exported=read_json('results/export.json')
    assert oracle['status']=='PASS' and exported['status']=='PASS'
    tflite=ROOT/exported['tflite']['path']
    assert sha256(tflite)==exported['tflite']['sha256']
    tok=Tokenizer.from_file(str(snapshot_path()/'tokenizer.json'))
    status('RUNNING','Float export inspected; testing every oracle row through CompiledModel CPU at 4 threads.')
    runner=Readout(tflite)
    exported.update(compiled_model_load='PASS',compiled_model_load_seconds_contended=runner.load_seconds)
    write_json('results/export.json',exported)
    output=dict(status='RUNNING',device='Apple M4 Max CPU',os='macOS 27.0 (26A428)',runtime='ai-edge-litert 2.2.0 CompiledModel',
                cpu_num_threads=4,temperature=1.03,timing='contended, informational',
                build=exported['build'],tflite=exported['tflite'],bundle=exported['bundle'],
                load_seconds_contended=runner.load_seconds,oracle_sha256=sha256(ROOT/'fixtures/oracle_fp32.json'),rows=[])
    print('COMPILED CPU LOAD PASS',runner.load_seconds,'s contended',flush=True)
    write_json('results/parity_fp.json',output)
    for source in oracle['rows']:
        logits,plan,steps,finite,elapsed=runner.score(source['ids'])
        labels=logits[source['label_token_ids']]
        p=softmax32(labels)
        oracle_p=np.asarray(source['probabilities'],np.float64)
        oracle_lg=np.asarray(source['raw_slot_logits_fp32'],np.float64)
        delta_p=np.abs(p.astype(np.float64)-oracle_p)
        delta_lg=np.abs(labels.astype(np.float64)-oracle_lg)
        finite=bool(finite and np.isfinite(p).all() and np.isfinite(delta_p).all())
        full_top1=int(logits.argmax())
        row=dict(row_id=source['row_id'],fixture_id=source['fixture_id'],family=source['family'],n_tokens=source['n_tokens'],
                 nopts=source['nopts'],chunk_plan=plan,steps=steps,slot_index=source['slot_index'],
                 label_token_ids=source['label_token_ids'],argmax_oracle=source['argmax'],argmax_lite=int(p.argmax()),
                 argmax_identical=int(p.argmax())==source['argmax'],tie=source['tie'],oracle_top2_gap=source['oracle_top2_gap'],
                 max_abs_probability_error=float(delta_p.max()) if finite else None,
                 max_abs_label_logit_error=float(delta_lg.max()) if finite else None,
                 oracle_max_abs_label_logit=float(np.abs(oracle_lg).max()),
                 lite_max_abs_label_logit=float(np.abs(labels).max()) if finite else None,
                 full_vocab_top1_token=full_top1,full_vocab_top1_text=tok.decode([full_top1],skip_special_tokens=False),
                 finite=finite,wall_seconds_contended=elapsed,raw_label_logits_fp32=labels.tolist() if finite else None,
                 probabilities=p.tolist() if finite else None,
                 oracle_usage_input_tokens=source['oracle_usage_input_tokens'])
        output['rows'].append(row)
        write_json('results/parity_fp.json',output)
        print(json.dumps({k:row[k] for k in ['row_id','n_tokens','nopts','argmax_oracle','argmax_lite','tie','max_abs_probability_error','max_abs_label_logit_error','finite','wall_seconds_contended']}),flush=True)
        if not finite:
            output['status']='FAIL'
            write_json('results/parity_fp.json',output)
            raise AssertionError('Nonfinite graph result recorded; no numerical tuning permitted')
    summary=summarize(output['rows'])
    summary['wide_255_row_present']=any(r['nopts']==255 for r in output['rows'])
    passed=(summary['non_tie_argmax_identical']==summary['non_tie_rows'] and
            summary['max_abs_probability_error']<=1e-3 and summary['all_finite'] and summary['wide_255_row_present'] and
            len(output['rows'])==len(oracle['rows']))
    output.update(status='PASS' if passed else 'FAIL',summary=summary,
                  per_family={f:summarize([r for r in output['rows'] if r['family']==f]) for f in sorted(set(r['family'] for r in output['rows']))},
                  total_wall_seconds_contended=time.monotonic()-start,
                  ties=[r['row_id'] for r in output['rows'] if r['tie']],
                  non_tie_argmax_mismatches=[r['row_id'] for r in output['rows'] if not r['tie'] and not r['argmax_identical']])
    write_json('results/parity_fp.json',output)
    print('PARITY',output['status'],json.dumps(summary),flush=True)
    assert passed,'Round 1 fixed parity acceptance failed; supervisor decision required'


if __name__=='__main__':
    try:
        main()
    except Exception:
        (ROOT/'logs/parity_fp.traceback.txt').write_text(traceback.format_exc())
        raise
