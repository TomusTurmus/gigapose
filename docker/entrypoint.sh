#!/usr/bin/env bash
set -euo pipefail

# Source conda
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  export PATH=/opt/conda/bin:$PATH
fi

conda activate gigapose-py310

# conda-forge pinocchio is built with a newer GCC and needs CXXABI_1.3.15, which the
# Ubuntu 22.04 base image's system libstdc++ lacks ("version `CXXABI_1.3.15' not found").
# Preload ONLY the env's newer libstdc++ (from libstdcxx-ng). Do NOT add $CONDA_PREFIX/lib
# to LD_LIBRARY_PATH wholesale: that shadows torch's bundled cu118 CUDA libs and breaks
# cudnn ("Could not load library libcudnn_cnn_infer.so.8 ... libnvrtc.so not found").
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

# pip torch (cu118) bundles its CUDA libs under site-packages/nvidia/*/lib. cudnn
# dlopen()s "libnvrtc.so" by its unversioned name at runtime; add those dirs to the
# loader path so it is found ("Could not load library libcudnn_cnn_infer.so.8 ...
# libnvrtc.so: cannot open shared object file"). Only torch's own nvidia libs are
# added here -- NOT $CONDA_PREFIX/lib, which would shadow them and break CUDA.
NV_LIB_DIRS="$(python -c "import os,nvidia; d=os.path.dirname(nvidia.__file__); print(':'.join(os.path.join(d,s,'lib') for s in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d,s,'lib'))))" 2>/dev/null || true)"
if [ -n "$NV_LIB_DIRS" ]; then
  export LD_LIBRARY_PATH="${NV_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

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
