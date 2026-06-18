FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl ca-certificates unzip pkg-config \
    libglib2.0-0 libgl1-mesa-glx libsm6 libxrender1 libxext6 ffmpeg python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -qO /tmp/Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /workspace

# Copy environment file (created earlier) and repo files.
# NOTE: this COPY relies on .dockerignore excluding gigaPose_datasets/, outputs/,
# .git/, media/, local_data/ (38 GB+). Without it, COPY . bakes all of that into
# an image layer. Only code (src/, configs/, setup.*) is needed at build time for
# the editable install; data/weights are bind-mounted at runtime.
COPY environment.yml /workspace/environment.yml
COPY . /workspace

# Create conda env (will use environment.yml). We'll install PyTorch/CUDA afterwards to ensure compatibility.
RUN ( /opt/conda/bin/conda env create -f /workspace/environment.yml -n gigapose-py310 \
   || /opt/conda/bin/conda env update -n gigapose-py310 -f /workspace/environment.yml ) \
 && /opt/conda/bin/conda clean -ya

# Install a CUDA-enabled PyTorch. environment.yml pins torch UNPINNED, which resolves
# to a CPU-only wheel (torch.cuda.is_available() == False -> Lightning raises
# "No supported gpu backend found!"). Pin to torch 2.0.0, the version xformers==0.0.18
# requires; the cu118 wheel runs on the host driver (535 / CUDA 12.2) and bundles its
# own CUDA runtime, so it is independent of the CUDA-12.1 base image.
RUN /opt/conda/bin/conda run -n gigapose-py310 pip install --no-cache-dir --force-reinstall \
    torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118

# cudnn dlopen()s the unversioned "libnvrtc.so" at runtime, but the cu118 wheel only
# ships the versioned "libnvrtc.so.11.2" -> "Could not load library libcudnn_cnn_infer.so.8
# ... libnvrtc.so: cannot open shared object file". Create the unversioned symlink.
# (entrypoint.sh adds these site-packages/nvidia/*/lib dirs to LD_LIBRARY_PATH.)
RUN set -e; NV="$(/opt/conda/bin/conda run -n gigapose-py310 python -c 'import os,nvidia; print(os.path.dirname(nvidia.__file__))')"; \
    real="$(ls "$NV"/cuda_nvrtc/lib/libnvrtc.so.* 2>/dev/null | head -1)"; \
    [ -n "$real" ] && ln -sf "$(basename "$real")" "$NV/cuda_nvrtc/lib/libnvrtc.so"

# MKL 2024.1+ dropped the iJIT_NotifyEvent symbol that libtorch_cpu.so links against,
# causing "ImportError: undefined symbol: iJIT_NotifyEvent". Pin MKL to 2024.0.
RUN /opt/conda/bin/conda run -n gigapose-py310 /opt/conda/bin/conda install -y -c conda-forge "mkl=2024.0" \
 && /opt/conda/bin/conda clean -ya

# Pinocchio (Stack-of-Tasks robotics lib) provides pin.SE3, used by megapose's
# lib3d.transform. The PyPI "pinocchio" package is a DIFFERENT, unrelated project,
# so install the real one from conda-forge.
RUN /opt/conda/bin/conda run -n gigapose-py310 /opt/conda/bin/conda install -y -c conda-forge pinocchio \
 && /opt/conda/bin/conda clean -ya

# Ensure hf cli is available
RUN /opt/conda/bin/conda run -n gigapose-py310 pip install --no-cache-dir --upgrade huggingface_hub

# setuptools 81+ removed the bundled pkg_resources module that several deps still
# import ("ModuleNotFoundError: No module named 'pkg_resources'"). Pin to <81.
RUN /opt/conda/bin/conda run -n gigapose-py310 pip install --no-cache-dir "setuptools<81"

# Post-install steps the README requires but environment.yml omits:
#  - bop_toolkit_lib: imported by test.py / refine.py / the dataset scripts
#  - megapose (pip install -e .): refine.py does `from megapose...` (package_dir=src)
#  - trimesh: used by the prepare/visualize scripts
#  - panda3d: only needed to RENDER templates in-container (optional otherwise)
#  - webdataset: the test dataloader reads tar shards via IterableWebSceneDataset.
#    Pinned to 1.0.2 because the local group_by_keys patch targets its trailing
#    meta-entry behavior (see REALSENSE_CUP_PIPELINE.md).
RUN /opt/conda/bin/conda run -n gigapose-py310 pip install --no-cache-dir \
    git+https://github.com/thodan/bop_toolkit.git trimesh transforms3d "webdataset==1.0.2" \
 && /opt/conda/bin/conda run -n gigapose-py310 pip install --no-cache-dir -e /workspace \
 && /opt/conda/bin/conda run -n gigapose-py310 pip install --no-cache-dir \
    --pre --extra-index-url https://archive.panda3d.org/ panda3d==1.11.0.dev3233 || true

# Vendored megapose (src/megapose) ships no requirements file, and environment.yml omits
# its runtime deps -> refine.py fails with a cascade of ModuleNotFoundError (joblib,
# simplejson, bokeh, roma, open3d, ...). These are the third-party modules imported across
# the megapose / refiner import chain that aren't already covered above.
# (teaserpp_python is intentionally excluded: it's only used by the point-cloud
# teaserpp_refiner, not the RGB refine path, and needs a from-source build.)
# torch/torchvision are repeated here as PINS, not to reinstall: torchnet/torchgeometry
# declare a loose `torch` dependency, so without this pip silently upgrades torch to the
# latest CUDA-13 build (driver-too-old at runtime). Pinning keeps the cu118 2.0.0 build.
RUN /opt/conda/bin/conda run -n gigapose-py310 pip install --no-cache-dir \
    torch==2.0.0 torchvision==0.15.1 \
    joblib simplejson bokeh roma open3d seaborn imageio meshcat psutil colorama \
    xarray torchgeometry torchnet ipython

# NumPy/SciPy pin MUST be the LAST package step (mirrors the working host env).
# torch 2.0.0 / torchvision / pinocchio are built against NumPy 1.x; NumPy 2.x triggers
# "_ARRAY_API not found" segfaults and pytorch-lightning 1.8.1's removed `np.Inf`.
# The conda installs above (pinocchio in particular) pull NumPy 2.x AND a scipy built
# against it back in, so an earlier pin gets clobbered and leaves scipy ABI-mismatched
# ("All ufuncs must have type numpy.ufunc"). Reinstall both from PyPI wheels (a mutually
# compatible pair) with --force-reinstall --no-deps so they win over conda's copies.
RUN /opt/conda/bin/conda run -n gigapose-py310 pip install --no-cache-dir \
    --force-reinstall --no-deps "numpy==1.26.4" "scipy==1.15.3"

# Fallback so the top-level `megapose` package resolves even under the bind mount.
ENV PYTHONPATH=/workspace/src
ENV CONDA_DEFAULT_ENV=gigapose-py310

# Keep the entrypoint outside /workspace so a bind mount over /workspace can't shadow it.
COPY docker/entrypoint.sh /opt/entrypoint.sh

ENTRYPOINT ["bash", "/opt/entrypoint.sh"]