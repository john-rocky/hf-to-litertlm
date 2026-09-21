"""Portable serial CLI execution with raw streams and explicit output roots."""
import json
import os
from pathlib import Path
import selectors
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ROOT = Path(os.environ.get('AGENTS_A1_OUTPUT', 'out/agents-a1-4b')).resolve()
LOGS = ROOT / 'logs'


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def configure_output(path):
    global LOGS
    LOGS = Path(path).resolve().parent / 'logs'
    LOGS.mkdir(parents=True, exist_ok=True)


def run_cli(command, stem, combined_path=None, env_overrides=None, **unused):
    LOGS.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = LOGS / (stem + '.stdout.txt'), LOGS / (stem + '.stderr.txt')
    environment = dict(os.environ)
    environment.update(env_overrides or {})
    record = {'command': [str(x) for x in command], 'stdin': 'DEVNULL',
              'stdout_path': str(stdout_path), 'stderr_path': str(stderr_path)}
    if combined_path:
        combined_path = Path(combined_path)
        combined_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
        child = subprocess.Popen(record['command'], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 env=environment, bufsize=0)
        combined = combined_path.open('wb') if combined_path else None
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ, stdout)
                selector.register(child.stderr, selectors.EVENT_READ, stderr)
                while selector.get_map():
                    for key, _ in selector.select():
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        key.data.write(chunk)
                        key.data.flush()
                        if combined:
                            combined.write(chunk)
                            combined.flush()
        finally:
            if combined:
                combined.close()
        record['returncode'] = child.wait()
    record['stdout'] = stdout_path.read_text(errors='replace')
    record['stderr'] = stderr_path.read_text(errors='replace')
    record['stderr_tail'] = record['stderr'][-8000:]
    record['engine_error'] = any(x in record['stderr'] + record['stdout'] for x in
                                 ['No adapters found', 'Failed to create LiteRT-LM engine', 'An error occurred'])
    save_json(LOGS / (stem + '.process.json'), record)
    return record


def add_runtime_args(parser, default_output, model=None):
    parser.add_argument('--cli', default=os.environ.get('LITERT_LM', 'litert-lm'))
    parser.add_argument('--runtime', default=os.environ.get('RUNTIME_LABEL', 'unspecified'))
    parser.add_argument('--model', default=str(model or ROOT / 'Agents-A1-4B_int8.litertlm'))
    parser.add_argument('--out', default=str(ROOT / 'results' / default_output))


def cli_success(record):
    return record['returncode'] == 0 and not record['engine_error']
