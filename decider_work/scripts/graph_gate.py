"""Unchanged exact-fit row protocol, selected CPU/GPU CompiledModel backend."""
import argparse, faulthandler, json, os, resource, time, traceback
faulthandler.enable()
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import numpy as np
from tokenizers import Tokenizer
from common import ROOT,read_json,write_json,sha256
from litert_readout import Readout,softmax32,summarize


def main():
    ap=argparse.ArgumentParser();ap.add_argument('variant');ap.add_argument('backend',choices=['cpu','gpu'])
    ap.add_argument('--tflite',required=True);ap.add_argument('--row');ap.add_argument('--output',required=True)
    a=ap.parse_args();start=time.monotonic()
    oracle=read_json('fixtures/oracle_fp32.json');sources=oracle['rows']
    if a.row:sources=[r for r in sources if r['row_id']==a.row]
    tok=Tokenizer.from_file(str(ROOT/'fixtures/tokenizer.json'))
    out=dict(status='RUNNING',variant=a.variant,backend=a.backend,device='Apple M4 Max',os='macOS 27.0 (26A428)',
             runtime='ai-edge-litert 2.2.0 CompiledModel',cpu_threads=4 if a.backend=='cpu' else None,
             gpu_options={'enforce_f32':True} if a.backend=='gpu' else None,pid=os.getpid(),
             tflite=a.tflite,temperature=1.03,oracle_sha256=sha256(ROOT/'fixtures/oracle_fp32.json'),
             timing='contended; informational',compiled=False,rows=[],requested_rows=len(sources))
    write_json(a.output,out)
    try:
        print('COMPILATION START',a.backend,flush=True)
        runner=Readout(ROOT/a.tflite,backend=a.backend)
        out.update(compiled=True,load_seconds=runner.load_seconds,
                   is_fully_accelerated=bool(runner.model.is_fully_accelerated()),
                   accelerator_name='not exposed by Python API; see native log',
                   peak_rss_after_init_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(a.output,out);print('COMPILED',json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
        for source in sources:
            logits,plan,steps,finite,elapsed=runner.score(source['ids'])
            labels=logits[source['label_token_ids']];p=softmax32(labels)
            op=np.asarray(source['probabilities'],np.float64);ol=np.asarray(source['raw_slot_logits_fp32'],np.float64)
            dp=np.abs(p.astype(np.float64)-op);dl=np.abs(labels.astype(np.float64)-ol)
            finite=bool(finite and np.isfinite(p).all() and np.isfinite(dp).all());top=int(logits.argmax())
            row=dict(row_id=source['row_id'],fixture_id=source['fixture_id'],family=source['family'],n_tokens=len(source['ids']),
                     nopts=source['nopts'],chunk_plan=plan,steps=steps,slot_index=len(source['ids'])-1,
                     label_token_ids=source['label_token_ids'],label_texts=source['label_texts'],
                     argmax_oracle=source['argmax'],argmax_lite=int(p.argmax()),argmax_identical=int(p.argmax())==source['argmax'],
                     tie=source['tie'],oracle_top2_gap=source['oracle_top2_gap'],finite=finite,
                     max_abs_probability_error=float(dp.max()) if finite else None,
                     max_abs_label_logit_error=float(dl.max()) if finite else None,
                     oracle_max_abs_label_logit=float(np.abs(ol).max()),lite_max_abs_label_logit=float(np.abs(labels).max()) if finite else None,
                     full_vocab_top1_token=top,full_vocab_top1_text=tok.decode([top],skip_special_tokens=False),
                     predicted_label=source['label_texts'][int(p.argmax())],wall_seconds_contended=elapsed,
                     raw_label_logits_fp32=labels.tolist() if finite else None,probabilities=p.tolist() if finite else None)
            out['rows'].append(row);write_json(a.output,out)
            print(json.dumps({k:row[k] for k in ['row_id','argmax_identical','max_abs_probability_error','finite','wall_seconds_contended']}),flush=True)
            assert finite,'Nonfinite values recorded; no tuning'
        summary=summarize(out['rows']);errors=[r['max_abs_probability_error'] for r in out['rows']]
        summary.update(median_abs_probability_error=float(np.median(errors)),rows_above_002=sum(e>0.02 for e in errors))
        out.update(status='PASS',summary=summary,acceptance='Measurement completeness only; supervisor owns numerical decisions',
                   peak_rss_at_stage_end_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   total_wall_seconds=time.monotonic()-start)
    except Exception:
        out.update(status='FAIL',error=traceback.format_exc(),total_wall_seconds=time.monotonic()-start)
        print(out['error'],flush=True)
    write_json(a.output,out)
    print('RESULT',a.variant,a.backend,out['status'],json.dumps(out.get('summary')),flush=True)
    if out['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
