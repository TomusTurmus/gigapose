# Running GigaPose on Custom RGB Images (`realsense_cup`)

End-to-end **RGB → 6D pose** on custom images, using the `realsense_cup` example
(a mug captured with a RealSense, CAD from the KITchen dataset). No depth and no
CNOS install required — detections are reused from existing per-frame masks.

Pipeline: **RGB frames → detections (from masks) → coarse pose (GigaPose) → refined pose (MegaPose) → CAD overlay**

---

## 0. Inputs used

| Input | Source | Notes |
|-------|--------|-------|
| RGB (640×360) | `~/dipl/FoundationPose/demo_data/realsense_cup/rgb/` | 30 frames, 25 with masks |
| Masks (binary) | `.../realsense_cup/masks/` | 25 frames → become detections |
| CAD | `~/dipl/datasets/KITchen/models/obj_000077.ply` | FruitTea mug, **mm units** |
| Texture | `~/dipl/datasets/KITchen/models/FruitTea_800_tex.png` | PLY references it as `obj_000077.png` |
| Intrinsics | `fx=fy=456.5, cx=320, cy=180` | at 640×360 |

> **Object id is renumbered 77 → 1.** GigaPose indexes templates as
> `template_data[obj_id-1]` and derives labels as `f"{index+1}"`, so object ids
> must be contiguous starting at 1.

---

## 1. How to run

> **WHERE EACH STEP RUNS** — markers used below:
> - **🖥️ [HOST]** = host conda env `gigapose` (`/home/pose/dipl/gigapose`). Has a display
>   (panda3d rendering) and reads the external `~/dipl/...` source data.
> - **🐳 [CONTAINER]** = Docker env `gigapose-py310`. Has the GPU + the pinned CUDA/torch
>   stack, but **no display**.
>
> The split: **build + render on the HOST** (steps 1–3), **coarse pose in the CONTAINER**
> (step 4), **refine on the HOST** (step 5, needs GPU **and** display), **visualize anywhere**
> (step 6). See §1b for the full rationale.
>
> Simplest alternative: once the host env has a CUDA torch + display (see host setup in
> §1b), **all six steps can run on the HOST** — skip the container entirely.

```bash
# ============================ 🖥️ HOST (env: gigapose) ============================
source /home/pose/miniconda3/etc/profile.d/conda.sh && conda activate gigapose
DST=gigaPose_datasets/datasets/realsense_cup

# 1. [HOST] Build dataset: copies RGB + CAD (+texture), generates models_info.json,
#    and converts the 25 masks into a CNOS-format detections JSON.
python -m src.scripts.prepare_realsense_cup

# 2. [HOST] Convert to the webdataset format GigaPose reads (tar shards + key_to_shard.json).
python -m src.scripts.convert_scenewise_to_imagewise --input $DST/scenewise --output $DST/imagewise --nprocs 1
python -m src.scripts.convert_imagewise_to_webdataset --input $DST/imagewise --output $DST/test

# 3. [HOST] Render templates from the CAD (162 poses × {rgb,depth} = 324 images).
#    Needs a display (panda3d); fails in a headless container with "Could not open window".
python -m src.scripts.render_custom_templates custom_dataset_name=realsense_cup
```

```bash
# ========================= 🐳 CONTAINER (env: gigapose-py310) =========================
# Launch:  sudo docker run --gpus all -it --rm \
#            -v /home/pose/dipl/gigapose:/workspace \
#            -v /home/pose/dipl/gigapose/gigaPose_datasets:/workspace/gigaPose_datasets \
#            gigapose:latest

# 4. [CONTAINER] Coarse pose estimation (GPU, no display needed — uses pre-rendered templates).
python test.py test_dataset_name=realsense_cup run_id=cup test_setting=detection
```

```bash
# ====================== 🖥️ HOST (env: gigapose) — needs GPU + display ======================
# 5. [HOST] MegaPose refinement. Renders pose hypotheses (panda3d) -> needs a display,
#    AND needs the GPU -> host torch must be the cu118 CUDA build (see §1b host setup).
#    Container alternative: prefix with `xvfb-run -a -s "-screen 0 1280x1024x24"`.
python refine.py test_dataset_name=realsense_cup run_id=cup test_setting=detection

# 6. [HOST or CONTAINER] Visualize (CAD projected onto each frame; no GPU, no display).
python src/scripts/visualize_realsense_cup.py \
  gigaPose_datasets/results/large_cup/refined_multiple_predictions/large-pbrreal-rgb-mmodel_realsense_cup-test_cupMultiHypothesis.csv
```

### Outputs
- Coarse: `gigaPose_datasets/results/large_cup/predictions/...test_cup.csv`
- Refined: `gigaPose_datasets/results/large_cup/refined_multiple_predictions/...MultiHypothesis.csv`
- Overlays: `gigaPose_datasets/results/large_cup/viz/`

---

## 1a. Detections: mask-derived vs CNOS

GigaPose consumes a precomputed detections JSON (it does not segment). Two sources
are wired in; the loader (`src/utils/inout.py`) **prefers the CNOS file** if present:

| File | Source | Script |
|------|--------|--------|
| `default_detections/realsense_cup/realsense_cup.json` | FoundationPose binary masks | `prepare_realsense_cup.py` |
| `default_detections/realsense_cup/realsense_cup_cnos.json` | **SAM-6D ISM (= CNOS)** detections | `prepare_cnos_detections_cup.py` |

CNOS (the authors' segmenter) is what they use for best BOP accuracy. SAM-6D's
Instance Segmentation Model is built on CNOS, so its per-frame
`output/obj_000077/sam6d_results/detection_XXXXXX.json` are already in GigaPose's
schema — the converter just fixes `image_id` (from the filename; the internal value
is 0), sets `scene_id=1`, and keeps the top-1 detection per frame.

```bash
python -m src.scripts.prepare_cnos_detections_cup   # writes realsense_cup_cnos.json
```

> On this dataset both detection sources give near-identical pose scores (top-5
> median ≈ 0.48) because they localise the same small region; the quality limiter
> is object size + dark templates, not the detector.

## 1b. Full best-performance config (authors' setup)

The authors' best BOP result comes from coarse → MegaPose refine **top-1** and
**top-5**, selecting the top-5 (`refined_multiple`) result. The RGB refiner uses
`n_iterations=5` and `megapose-1.0-RGB-multi-hypothesis` (default in
`configs/model/refiner/rgb.yaml`). One-shot script:

```bash
bash run_full_cup_pipeline.sh
```

which runs:
```bash
python test.py   test_dataset_name=realsense_cup run_id=cup test_setting=detection
python refine.py test_dataset_name=realsense_cup run_id=cup test_setting=detection use_multiple=False  # top-1
python refine.py test_dataset_name=realsense_cup run_id=cup test_setting=detection use_multiple=True   # top-5 (best)
```

Observed median scores on the cup: coarse 0.01 → refined top-1 0.14 → **refined top-5 0.49**.

### Running in Docker (intended environment)

The container uses env **`gigapose-py310`**, CUDA 12.1, `WORKDIR=/workspace`.
`docker/docker-compose.yml` bind-mounts the repo (`../:/workspace`) and
`gigaPose_datasets`, so **host edits, scripts and the built dataset are live —
no image rebuild needed for code/data changes**.

**Host vs container — what runs where (steps 1–3 host, 4–6 container):**

| Step | Where | Why |
|------|-------|-----|
| 1. `prepare_realsense_cup.py`, `prepare_cnos_detections_cup.py` | **HOST** (env `gigapose`) | read `~/dipl/FoundationPose`, `~/dipl/datasets/KITchen`, `~/dipl/SAM-6D` — not mounted in the container unless you add `-v /home/pose/dipl:/home/pose/dipl` |
| 2. the two converters | **HOST** | only touch `gigaPose_datasets/`, so they *could* run in-container — but running as host user `pose` avoids the **root-owned files** the container (runs as root) writes back through the bind mount |
| 3. `render_custom_templates` | **HOST** (hard requirement) | panda3d needs a display/GL; a headless container fails with `Could not open window`. (In-container alternative: `xvfb-run`.) |
| 4. `test.py` (coarse pose) | **CONTAINER** | needs the GPU; matches against pre-rendered templates, so **no display required**. |
| 5. `refine.py` (MegaPose refine) | **HOST**, or container via `xvfb-run` | needs **both** GPU **and** a display — MegaPose renders pose hypotheses with panda3d, so a headless container fails with `Could not open window`. |
| 6. `visualize_realsense_cup.py` | either | just projects CAD vertices into RGB with cv2/matplotlib and writes PNGs — no GPU, no display. |

> **`refine.py` needs a GPU + a display at once.** The host has both (once its torch
> is a CUDA build matching the driver — see host setup below); the container has the GPU
> but no display unless you wrap the command in `xvfb-run -a -s "-screen 0 1280x1024x24"`.

> **The two environments are independent.** Host `gigapose` (where you build +
> render) and container `gigapose-py310` (where you run inference) do **not** share
> packages — a fix applied in one is not present in the other.

**One-time host setup** (env `gigapose`) — needed for steps 1–3 **and `refine.py`**:

```bash
conda activate gigapose
# CUDA torch matching the host driver (535 / CUDA 12.2). cu118 needs only a >=11.8 driver.
# Required for refine.py on the host; pin it so later pip installs can't bump it (see caution).
pip install --force-reinstall torch==2.0.0 torchvision==0.15.1 \
  --index-url https://download.pytorch.org/whl/cu118
pip install transforms3d       # imported by megapose/lib3d/rotations.py (render worker)
# pinocchio must be importable here (provides pin.SE3); install via conda-forge if missing:
# conda install -c conda-forge pinocchio
# NumPy/SciPy LAST (matched pair). torch/pinocchio are built for NumPy 1.x; NumPy 2.x ->
# "_ARRAY_API not found" segfault in the render worker, and breaks scipy ABI.
pip install --force-reinstall --no-deps "numpy==1.26.4" "scipy==1.15.3"
```

> **Caution — the host env has no torch pin in any manifest.** Any `pip install <pkg>`
> that depends on torch (e.g. `torchnet`, `torchgeometry`, installed for `refine.py`) can
> silently upgrade torch to the latest CUDA-13 wheel → `RuntimeError: The NVIDIA driver on
> your system is too old (found version 12020)`. After installing host deps, re-assert
> `torch==2.0.0 torchvision==0.15.1` (cu118) and re-pin numpy/scipy. Verify with
> `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` → want
> `2.0.0+cu118 True`.

The render step **swallows worker crashes** and still prints `Finished for 1/1`
with zero templates — always verify output exists:
`ls gigaPose_datasets/datasets/templates/realsense_cup/000001/` (expect `000000.png` … ).
If an earlier in-container attempt left a **root-owned** `templates/.../000001` dir,
clear it first: `sudo rm -rf gigaPose_datasets/datasets/templates/realsense_cup/000001`.

**One-time container setup** — `environment.yml` omits several README post-install
deps, so a fresh `gigapose-py310` is missing them (symptom:
`ModuleNotFoundError: No module named 'bop_toolkit_lib'`). These are now baked into
`docker/Dockerfile`; if running an older image, install them once:

```bash
pip install git+https://github.com/thodan/bop_toolkit.git trimesh
pip install -e /workspace        # installs the top-level `megapose` package (refine.py needs it)
# panda3d only if you intend to render templates in-container:
# pip install --pre --extra-index-url https://archive.panda3d.org/ panda3d==1.11.0.dev3233
```

**Run the full best-performance config in the container** (dataset already built on host):

```bash
docker compose -f docker/docker-compose.yml run --rm gigapose bash run_full_cup_pipeline.sh
```

Notes:
- Do **not** run the `prepare_*`/`render_*` scripts in the container — build the
  dataset on the host first (see §1, §1a), then run inference in the container.
- If `docker` gives a permission error, add your user to the `docker` group (or use
  sudo); the host conda env `gigapose` is an equivalent fallback for everything.

## 1c. Docker environment — every fix baked in (and why)

The stock `environment.yml` does **not** produce a working inference env; getting a clean
container required the fixes below. All are now baked into `docker/Dockerfile` and
`docker/entrypoint.sh` **in a specific order** — the NumPy/SciPy pin must be the *last*
package step, because conda installs (pinocchio) and pip deps (open3d) drag NumPy 2.x back
in and clobber an earlier pin. After a rebuild, smoke-test before relying on it:

```bash
python - <<'PY'
import torch, numpy, scipy
from scipy.spatial.transform import Rotation
import src.dataloader.test, src.models.gigaPose, refine
import webdataset, pinocchio, transforms3d, bop_toolkit_lib, joblib, open3d
assert torch.cuda.is_available(), "CUDA not available — launch with --gpus all"
assert torch.__version__.startswith("2.0.0+cu"), torch.__version__
assert numpy.__version__ == "1.26.4", numpy.__version__
print("ALL OK:", torch.__version__, numpy.__version__, scipy.__version__)
PY
```

| # | Symptom | Cause | Fix (where) |
|---|---------|-------|-------------|
| 1 | `/workspace/entrypoint.sh: No such file` | bind mount over `/workspace` shadows the baked entrypoint | entrypoint lives at `/opt/entrypoint.sh` (Dockerfile) |
| 2 | scripts can't find `~/dipl/...` data | only `/workspace` mounted | mount `-v /home/pose/dipl:/home/pose/dipl` (FoundationPose-style) |
| 3 | `ImportError: undefined symbol: iJIT_NotifyEvent` | MKL 2024.1+ dropped the symbol torch links | pin `mkl=2024.0` (conda-forge) |
| 4 | `No module named 'pkg_resources'` | setuptools 81+ removed it | pin `setuptools<81` |
| 5 | `pinocchio has no attribute 'SE3'` | PyPI `pinocchio` is an unrelated package | install real one from **conda-forge** |
| 6 | `No module named 'transforms3d'` / `webdataset` | omitted from `environment.yml` | pip install (webdataset pinned `==1.0.2` for the local patch) |
| 7 | `No supported gpu backend found!` / torch CPU-only | `environment.yml` leaves `torch` unpinned → CPU wheel | install `torch==2.0.0 torchvision==0.15.1` **cu118** (matches `xformers==0.0.18` + driver) |
| 8 | `_ARRAY_API not found` segfault; `np.Inf` removed; `All ufuncs must have type numpy.ufunc` | NumPy 2.x vs torch/scipy built for 1.x; conda/pip split | `--force-reinstall --no-deps numpy==1.26.4 scipy==1.15.3` **as the last step** |
| 9 | `CXXABI_1.3.15 not found` (pinocchio) | conda pinocchio needs newer libstdc++ than Ubuntu 22.04 ships | `LD_PRELOAD` the env's `libstdc++.so.6` (entrypoint) — **not** whole `$CONDA_PREFIX/lib` |
| 10 | `libcudnn_cnn_infer.so.8 ... libnvrtc.so: cannot open` | cu118 wheel ships only versioned `libnvrtc.so.11.2` | symlink unversioned `libnvrtc.so` (Dockerfile) + add `nvidia/*/lib` to `LD_LIBRARY_PATH` (entrypoint) |
| 11 | `refine.py`: cascade of `ModuleNotFoundError` (joblib, simplejson, bokeh, roma, open3d…) | vendored megapose ships no requirements file | dedicated megapose-deps pip step (Dockerfile), **with `torch==2.0.0` pinned** so `torchnet`/`torchgeometry` can't bump torch |
| 12 | container loses GPU mid-run, `Failed to initialize NVML` | nvidia-toolkit + systemd cgroup bug on host `daemon-reload` | start a fresh container; prevent via `native.cgroupdriver=cgroupfs` (see §6) |
| 13 | image ballooned (38 GB) | `COPY . /workspace` baked `gigaPose_datasets/` | `.dockerignore` excludes data/weights/.git; `conda clean -ya` per layer |

**Two independent envs, two failure modes for the same root cause:** the NumPy-1.x and
torch-cu118 constraints apply to **both** the host `gigapose` env and the container
`gigapose-py310` env, but each must be fixed separately (the Dockerfile only fixes the
container; the host is fixed by the §1b host-setup commands). Neither env pins torch in a
manifest, so installing extra deps in *either* can silently bump torch and re-break CUDA.

## 2. Scripts added

| File | Purpose |
|------|---------|
| `src/scripts/prepare_realsense_cup.py` | Builds the dataset (RGB, CAD renumbered to obj 1, texture, `models_info.json`) and converts masks → detections JSON. |
| `src/scripts/visualize_realsense_cup.py` | Projects CAD vertices into each RGB frame with the predicted pose → overlay PNGs. |

---

## 3. Code changes required (existing files)

These were needed to make the custom pipeline work; each is also useful for any
small / RGB-only / custom dataset.

| File | Change | Reason |
|------|--------|--------|
| `src/scripts/render_bop_templates.py` | `num_gpus 4 → 1` | Single GPU; was routing to nonexistent GPUs 1–3 |
| `src/scripts/render_custom_templates.py` | `num_gpus 4 → 1`; mesh paths made absolute | Same GPU fix + panda3d resolves model paths against its own model-path, not CWD |
| `src/scripts/convert_scenewise_to_imagewise.py` | GT detected via `scene_gt.json` existence | Custom dataset has no GT |
| `src/utils/inout.py` | `realsense_cup` branch in `load_test_list_and_cnos_detections` | Loader rejected non-builtin dataset names |
| `src/custom_megapose/web_scene_dataset.py` | Load depth only if `depth.png` present | RGB-only data |
| `src/megapose/utils/webdataset.py` | `group_by_keys` skips entries missing `fname`/`data` | webdataset 1.0.2 emits a trailing meta entry |
| `test.py` | `log_interval = max(1, len//30)` | 25 frames → div-by-zero |

---

## 4. Adapting to a new object / capture

1. Edit the paths and intrinsics at the top of `prepare_realsense_cup.py`
   (`FP_SRC`, `KITCHEN_CAD`, `CAM_K`, texture name).
2. Keep the **target object id = 1**.
3. Provide one binary mask per RGB frame (any segmenter; here reused from
   FoundationPose). Only frames with a mask are processed.
4. Rerun steps 1–6 above.

---

## 5. Quality notes

- Masks on far frames are tiny (~13×19 px); templates render dark. This yields
  low/variable coarse scores (0–0.34) and inconsistent depth.
- Refinement helps on close frames — e.g. frame 25 (Z≈654 mm) aligns well on the cup.
- **For better results:** capture frames where the object is larger in view, and/or
  brighten the template rendering lighting.

## 6. Notes / gotchas

- CAD units are **mm** (BOP standard) — no conversion for GigaPose (unlike
  FoundationPose, which divides KITchen PLYs by 1000).
- The existing HOPE `test/` scene dirs are root-owned (`drwxr-x---`); the custom
  `realsense_cup` data is created as user `pose` and is fully readable.
- `convert_scenewise_to_imagewise.py` must be run with `--nprocs 1`; its serial
  path has an unrelated bug (missing `image_tkey` argument).
- **GPU disappears from a running container** (`nvidia-smi` → `Failed to initialize
  NVML: Unknown Error`, and torch reports `torch.cuda.is_available() == False` /
  Lightning `No supported gpu backend found!`). This is the nvidia-container-toolkit +
  systemd cgroup bug: a host `systemctl daemon-reload` (or any cgroup refresh) revokes
  the GPU from already-running containers. The device cgroup is only set up at container
  **start**, so it cannot be restored in place — start a fresh container with
  `--gpus all`. **Prevent it** by switching Docker off the systemd cgroup driver — add to
  `/etc/docker/daemon.json`:

  ```json
  { "exec-opts": ["native.cgroupdriver=cgroupfs"] }
  ```

  then `sudo systemctl restart docker`. After this, running GPU containers survive host
  `daemon-reload`s. (Quick alternative: just don't run `daemon-reload` while GPU
  containers are up.) Note the GPU host stays healthy throughout — verify with
  `nvidia-smi` on the host; only the container loses access.
