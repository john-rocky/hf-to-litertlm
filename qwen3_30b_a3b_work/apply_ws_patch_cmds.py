#!/usr/bin/env python3
"""Apply the patch_cmds of one http_archive in a WORKSPACE file to a local checkout, the way bazel runs them.

usage: apply_ws_patch_cmds.py <WORKSPACE> <repo name> <checkout dir>

Bazel runs each patch_cmds entry with bash in the extracted archive. The commands are written for GNU sed
(`sed -i -e ...`); on macOS this script puts GNU sed (gsed) first on PATH as `sed`, so the edits match the
Linux reading and leave no `-e` backup files. Prints each command and whether it changed the tree.
"""
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile

ws, name, checkout = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
text = open(ws, encoding="utf-8").read()
m = re.search(r'http_archive\(\s*name\s*=\s*"' + re.escape(name) + r'",\s*patch_cmds\s*=\s*(\[.*?\n    \]),', text, re.S)
if not m:
    sys.exit(f"no http_archive(name = \"{name}\", patch_cmds = [...]) in {ws}")
body = re.sub(r"^\s*#.*$", "", m.group(1), flags=re.M)  # Starlark comments
cmds = ast.literal_eval(body)
gsed = shutil.which("gsed")
if not gsed:
    sys.exit("gsed not found")
shim = tempfile.mkdtemp()
os.symlink(gsed, os.path.join(shim, "sed"))
env = dict(os.environ, PATH=shim + os.pathsep + os.environ["PATH"])


def status():
    return subprocess.run(["git", "-C", checkout, "status", "--porcelain"], capture_output=True, text=True).stdout


for i, cmd in enumerate(cmds, 1):
    before = subprocess.run(["git", "-C", checkout, "diff"], capture_output=True, text=True).stdout
    r = subprocess.run(["bash", "-c", cmd], cwd=checkout, env=env, capture_output=True, text=True)
    after = subprocess.run(["git", "-C", checkout, "diff"], capture_output=True, text=True).stdout
    print(f"[{i}/{len(cmds)}] rc={r.returncode} changed={before != after}: {cmd}")
    if r.returncode != 0:
        print(r.stderr)
        sys.exit(1)
print(f"applied {len(cmds)} patch_cmds of @{name}; changed files:\n{status()}")
