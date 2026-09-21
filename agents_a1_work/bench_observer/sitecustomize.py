"""Observe the CLI's unchanged Benchmark.run results, including its warmup."""
import os, sys
_out=os.environ.get('AGENTS_A1_BENCH_OBSERVATION')
if _out:
    import dataclasses, functools, json
    from pathlib import Path
    import litert_lm
    _path=Path(_out).resolve(); _path.parent.mkdir(parents=True,exist_ok=True)
    _original=litert_lm.Benchmark.run
    _runs=[]
    @functools.wraps(_original)
    def observed_run(self):
        info=_original(self)
        _runs.append(dataclasses.asdict(info))
        _path.write_text(json.dumps({'method':'Return values of unchanged Benchmark.run; first is CLI warmup','calls':_runs},indent=2)+'\n')
        return info
    litert_lm.Benchmark.run=observed_run
