"""hf-to-litertlm: a Hugging Face model repo to a .litertlm bundle in one command, on a pinned toolchain.

    hf-to-litertlm <org>/<model> [--revision SHA] [--recipe int8|int4] [--check] [--smoke] ...
    hf-to-litertlm verify out/<model>/<model>.lock.json   rebuild from the lock: MATCH = the same bytes
    hf-to-litertlm check out/<model>/<model>_int8.litertlm   load it on litert-lm (CPU), an echo prompt

The first form is scripts/convert.py of https://github.com/john-rocky/hf-to-litertlm: it writes
out/<model>/<model>_<recipe>.litertlm, <model>.lock.json and convert_report.json under the current
directory. --check runs the entry gate only (no download); --smoke runs the one-prompt check instead of
the 8-question gate. `hf-to-litertlm <org>/<model> --help` lists every flag; it runs inside the
toolchain environment, so on a new machine it installs that first. Exit code of the first form: 0 =
converted and passed the gate, 1 = the gate or the check failed, 2 = refused or stopped before any
download, 3 = any later failure. verify exits 0 on MATCH, check exits 0 on PASS.

This package carries those scripts with the repo's pyproject.toml and uv.lock. Each run puts them in
~/.cache/hf-to-litertlm/<version>/ (HF_TO_LITERTLM_HOME replaces ~/.cache/hf-to-litertlm) and runs
them there with `uv run --frozen`: the packages are the ones uv.lock pins, the environment `uv sync`
makes in a clone, so a lock the package wrote verifies in a clone and a clone's lock verifies from the
package, when both run the same Python patch version. The first run installs that environment there.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from . import __version__

SUBCOMMANDS = {"verify": "scripts/verify_lock.py", "check": "scripts/check_bundle.py"}
PAYLOAD = Path(__file__).with_name("payload.zip")
# a user's own venv or project env would put the toolchain somewhere uv.lock does not describe
DROP_ENV = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")


def home():
    root = os.environ.get("HF_TO_LITERTLM_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(str(Path.home()), ".cache"), "hf-to-litertlm")
    return Path(root) / __version__


def unpack():
    """Write every payload file whose bytes are not the manifest's, then return the directory."""
    d = home()
    with zipfile.ZipFile(PAYLOAD) as z:
        manifest = json.loads(z.read("MANIFEST.json"))
        for rel, sha in manifest["files"].items():
            p = d / rel
            if not (p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == sha):
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
                tmp.write_bytes(z.read(rel))
                os.replace(tmp, p)
            # an entry keeps its git mode: verify_quality.py runs litert_lm_cli_verifier.py as a program
            if (z.getinfo(rel).external_attr >> 16) & 0o111 and not os.access(p, os.X_OK):
                os.chmod(p, (p.stat().st_mode & 0o777) | 0o111)
    return d


def uv_bin():
    try:
        from uv import find_uv_bin
        return find_uv_bin()
    except (ImportError, FileNotFoundError):
        found = shutil.which("uv")
    if not found:
        sys.exit("FAIL setup: uv not found -> next: pip install uv, or run uvx hf-to-litertlm")
    return found


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"hf-to-litertlm {__version__}")
        return 0
    script = SUBCOMMANDS[argv.pop(0)] if argv[0] in SUBCOMMANDS else "scripts/convert.py"
    d = unpack()
    env = {k: v for k, v in os.environ.items() if k not in DROP_ENV}
    env["HF_TO_LITERTLM_PACKAGE"] = __version__
    cmd = [uv_bin(), "run", "--frozen", "--project", str(d), "python", str(d / script), *argv]
    if os.name == "nt":
        return subprocess.call(cmd, env=env)
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    sys.exit(main())
