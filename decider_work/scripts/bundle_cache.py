"""Content-addressed, run-local bundle unpacking for the reference readout."""
import subprocess
import sys
import tomllib
from pathlib import Path
from common import ROOT, sha256, write_json


def unpack_bundle(bundle):
    bundle=Path(bundle).resolve()
    digest=sha256(bundle)
    folder=ROOT/'exports/readout_cache'/digest
    marker=folder/'complete.json'
    if not marker.exists():
        if folder.exists():
            raise RuntimeError(f'Incomplete bundle cache requires inspection: {folder}')
        folder.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-B',str(Path(sys.executable).parent/'litert-lm'),
                 'unpack',str(bundle),'--output-dir',str(folder)]
        result=subprocess.run(command,capture_output=True,text=True)
        log=ROOT/'logs'/f'unpack_cache_{digest}.log'
        log.write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(f'Bundle unpack failed: {log}')
        write_json(marker,dict(bundle_sha256=digest,command=command,log=str(log.relative_to(ROOT))))
    config=tomllib.loads((folder/'model.toml').read_text())
    sections=[s for s in config['section'] if s['section_type']=='TFLiteModel']
    assert len(sections)==1
    return folder,folder/sections[0]['data_path'],config,digest
