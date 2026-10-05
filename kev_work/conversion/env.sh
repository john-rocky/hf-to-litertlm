# Source from the work directory W (the parent of scripts/):  cd W && source scripts/env.sh
# Every cache and temporary file stays inside W; the Hugging Face cache is W/hf.
export W="$PWD"
export HF_HOME="$W/hf" HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME="$W/cache/xdg" TORCH_HOME="$W/cache/torch" TMPDIR="$W/cache/tmp" PIP_CACHE_DIR="$W/cache/pip" UV_CACHE_DIR="$W/cache/uv"
mkdir -p "$W/cache/xdg" "$W/cache/torch" "$W/cache/tmp" "$W/cache/pip" "$W/cache/uv"
# The three interpreters (see README.md): ORACLE (requirements-oracle.txt), EXPORT (requirements-export.txt),
# HOST (host/requirements-host.txt), e.g.  export ORACLE=$W/.venv-oracle/bin/python
