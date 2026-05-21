FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl ca-certificates unzip pkg-config \
    libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -qO /tmp/Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /workspace

# Copy the repository-root environment file and source tree from the build context.
COPY ./environment.yml /workspace/environment.yml
COPY . /workspace

# Create conda env from a build-local copy that only adjusts pip for compatibility with the original dependencies.
RUN cp /workspace/environment.yml /tmp/environment.build.yml \
    && python -c "from pathlib import Path; path = Path('/tmp/environment.build.yml'); text = path.read_text(); path.write_text(text.replace('  - pip\n', '  - pip<24.1\n', 1))"
RUN python -c "from pathlib import Path; path = Path('/tmp/environment.build.yml'); text = path.read_text(); path.write_text(text.replace('  - python=3.10\n', '  - python=3.10\n  - numpy<2\n', 1))"
RUN python -c "from pathlib import Path; path = Path('/tmp/environment.build.yml'); text = path.read_text(); text = text.replace('    - torch\n', '    - torch==2.0.0\n', 1); path.write_text(text)"
RUN python -c "from pathlib import Path; path = Path('/tmp/environment.build.yml'); text = path.read_text(); text = text.replace('    - torchvision\n', '    - torchvision==0.15.1\n', 1); path.write_text(text)"
RUN python -c "from pathlib import Path; path = Path('/tmp/environment.build.yml'); text = path.read_text(); path.write_text(text.replace('    - opencv-python \n', '    - opencv-python<4.13 \n', 1))"
RUN /opt/conda/bin/conda env create -f /tmp/environment.build.yml -n gigapose || /opt/conda/bin/conda env update -n gigapose -f /tmp/environment.build.yml
RUN /opt/conda/bin/conda install -n gigapose -c conda-forge pinocchio=2.6.20
RUN /opt/conda/bin/conda run -n gigapose pip install --no-cache-dir --force-reinstall "numpy<2" "opencv-python<4.13"
RUN /opt/conda/bin/conda run -n gigapose pip uninstall -y setuptools \
    && /opt/conda/bin/conda run -n gigapose pip install --no-cache-dir "setuptools<81"
RUN /opt/conda/bin/conda run -n gigapose pip install -e /workspace
RUN /opt/conda/bin/conda run -n gigapose pip install git+https://github.com/thodan/bop_toolkit.git
# Ensure hf cli is available
RUN /opt/conda/bin/conda run -n gigapose pip install --no-cache-dir --upgrade huggingface_hub

ENV CONDA_DEFAULT_ENV=gigapose

COPY docker/entrypoint.sh /entrypoint.sh

ENTRYPOINT ["bash", "/entrypoint.sh"]