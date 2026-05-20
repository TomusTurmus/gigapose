#!/usr/bin/env bash
set -euo pipefail

# Source conda
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  export PATH=/opt/conda/bin:$PATH
fi

conda activate gigapose

# If HF token provided, login non-interactively
if [ ! -z "${HF_TOKEN-}" ]; then
  python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
fi

# Ensure MPLCONFIGDIR writable
: ${MPLCONFIGDIR:=/tmp/.mplconfig}
mkdir -p "$MPLCONFIGDIR"
export MPLCONFIGDIR

# Run passed command, drop to shell, or keep container alive when detached.
if [ $# -eq 0 ]; then
  if [ -t 0 ]; then
    exec bash
  else
    exec tail -f /dev/null
  fi
else
  exec "$@"
fi
