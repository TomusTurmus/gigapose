# Cues for Wiring a Custom Pose Estimation Pipeline — GigaPose edition

Updated lessons after getting **GigaPose** running RGB→pose on custom RealSense
data (`realsense_cup`, KITchen CAD). This supersedes the original SAM-6D +
FoundationPose cues for GigaPose work — most of the *meta* advice still holds,
but the concrete details differ. Read this before starting; it will save hours.

The biggest time sinks this round were things I only discovered mid-run:
**(a) GigaPose wants a webdataset, not raw BOP folders; (b) object ids must be
contiguous from 1; (c) you can skip CNOS entirely by converting existing masks.**

---

## 0. Collect this before writing any code

Same as before — paste outputs into context at session start.

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
nvcc --version
conda env list
conda run -n <env> python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available())"
```

**New, GigaPose-specific things to confirm upfront** (each cost me a round-trip):
- **Do you have a precomputed detections JSON, or per-frame masks, or neither?**
  GigaPose does *not* segment — it consumes a CNOS-format detections JSON. If you
  already have masks (e.g. from a FoundationPose/SAM-6D run), convert them and skip CNOS.
- **What is the object id, and is it contiguous from 1?** If not, renumber (see §3).
- **Is the data RGB-only?** GigaPose coarse + MegaPose-RGB refiner don't need depth,
  but the loader assumes depth exists unless patched (§7).
- **Number of test frames** — small datasets (<30) trip `len//30 == 0` bugs (§7).
- **CAD: units and texture.** `trimesh.load(ply).extents` should match real size in mm.
  Check the PLY header for `comment TextureFile <name>.png` and whether that file exists.

---

## 1. Units — settle first (GigaPose differs from FoundationPose!)

| What | Unit | Notes |
|---|---|---|
| **GigaPose / MegaPose CAD (BOP)** | **mm** | KITchen/BOP PLYs are already mm — **no /1000** (unlike FoundationPose) |
| GigaPose pose output `t` | **mm** | in the BOP results CSV |
| Camera intrinsics | pixels | tied to resolution; halve fx,fy,cx,cy when halving resolution |
| FoundationPose mesh | metres | divide KITchen PLY by 1000 (only relevant to the *other* pipeline) |

Check `mesh.extents` after loading; for the cup it was `[116.9, 92.9, 81.3]` mm — sane.

---

## 2. Data format — GigaPose reads a WebDataset, not raw BOP

This was the key surprise. `WebSceneDataset` reads **tar shards + `key_to_shard.json`**,
not `rgb/000001.png` + `scene_camera.json`. You must convert:

```
raw BOP scenewise            ->   imagewise            ->   webdataset
test/000001/rgb/*.png             000001_000001.rgb.png      shard-000000.tar
test/000001/scene_camera.json     000001_000001.camera.json  key_to_shard.json
                                                              (lands in datasets/<name>/test/)
```

Scripts: `convert_scenewise_to_imagewise.py` then `convert_imagewise_to_webdataset.py`.

- Run the first with `--nprocs 1` — its serial path has a bug (missing `image_tkey`).
- GT/masks/depth are all **optional** in these converters; RGB + camera.json suffice.
- The webdataset goes to `datasets/<name>/test/`; that's where `WebSceneDataset` looks.

**Detections JSON** (what replaces CNOS), one entry per instance per frame:
```json
{"scene_id":1,"image_id":3,"category_id":1,"bbox":[x,y,w,h],
 "score":1.0,"time":0.0,"segmentation":{"counts":[...],"size":[h,w]}}
```
- `bbox` is `[x,y,w,h]`; `segmentation` is COCO RLE via
  `bop_toolkit_lib.pycoco_utils.binary_mask_to_rle` (uncompressed list counts, column-major).
- The loader expects it under `datasets/default_detections/<name>/<name>.json` and you
  must add a branch in `load_test_list_and_cnos_detections` (it hardcodes dataset names).

---

## 3. Object ids MUST be contiguous from 1

GigaPose indexes templates as `template_data[obj_id - 1]` (gigaPose.py) and builds
template labels as `f"{index+1}"` (template.py). So a single object **must be id 1**,
not its native dataset id. Symptom if you don't: `KeyError: '1'` in
`get_object_templates`. Renumber everywhere consistently:
- CAD file `obj_000001.ply`, `models_info.json` key `"1"`, detections `category_id=1`,
  templates dir `templates/<name>/000001`.
- Keep the texture named to match the **PLY's internal** `TextureFile` reference
  (it still says the original id), not the renamed mesh.

---

## 4. Templates — panda3d, single GPU, absolute paths

`render_custom_templates.py` / `render_bop_templates.py`:
- `num_gpus` is **hardcoded to 4** → on a 1-GPU box it routes objects to nonexistent
  GPUs (`CUDA_VISIBLE_DEVICES=1/2/3`). Set to 1.
- panda3d resolves model paths against **its own model-path, not CWD** → pass an
  **absolute** mesh path (`root_dir` config is `./gigaPose_datasets`, relative). Symptom:
  `Couldn't load file ...ply: not found on model path`.
- Missing texture → renders untextured/gray (warning, not fatal). Find the real texture
  (KITchen textures are named by object, e.g. `FruitTea_800_tex.png`) and copy it to the
  name the PLY header expects.
- Output: `162 poses × {rgb,depth} = 324` PNGs per object. Verify the count.
- HOPE/most datasets use panda3d; only `tless`/`itodd` use blenderproc.

---

## 5. Working directory & paths

- Run everything from the repo root; configs use relative `./gigaPose_datasets`.
- Always activate the env in the *same* shell call:
  `source .../conda.sh && conda activate gigapose && python ...` (cwd resets between tool calls).
- panda3d model loads need absolute paths (§4); writers (cv2.imwrite) are fine with relative.

---

## 6. GPU memory

A2000 12GB was plenty here (coarse ~0.04 s/img, refine ~0.5 s/img on 25 frames).
If OOM: reduce `machine.num_workers`, `max_num_dets_per_forward`, or template parallelism.

---

## 7. Upstream bugs/assumptions to patch for custom + small + RGB-only data

| File | Issue | Fix |
|---|---|---|
| `src/utils/inout.py` | `load_test_list_and_cnos_detections` raises `NotImplementedError` for unknown dataset | add a branch pointing at your detections JSON |
| `src/custom_megapose/web_scene_dataset.py` | loads `depth.png` unconditionally → `KeyError: 'depth.png'` | `if load_depth and depth_format in sample` |
| `src/megapose/utils/webdataset.py` | `group_by_keys` assumes every entry has `fname` | webdataset 1.0.2 emits a trailing meta entry → skip if `fname`/`data` missing |
| `test.py` | `log_interval = len//30 == 0` on <30 frames → `ZeroDivisionError` | `max(1, len//30)` |
| `src/dataloader/test.py` | `load_init_loc` reads `loc["instance_id"]` → `KeyError` for top-1 refine (`use_multiple=False`) | assign `instance_id=idx` when absent (only the MultiHypothesis CSV has it) |
| `convert_scenewise_to_imagewise.py` | forces GT for non-hope names; serial path missing arg | detect GT by file existence; run `--nprocs 1` |
| `render_*_templates.py` | `num_gpus=4`; relative mesh path | set 1; `.resolve()` the paths |

---

## 8. The mask → detection bridge (skip CNOS)

The cheapest detection source is masks you already have. Per frame:
```python
from bop_toolkit_lib import pycoco_utils
import cv2, numpy as np
m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE); b = (m>0).astype(np.uint8)
ys,xs = np.where(b); bbox=[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)]
det = {"scene_id":1,"image_id":im_id,"category_id":1,"bbox":bbox,
       "score":1.0,"time":0.0,"segmentation":pycoco_utils.binary_mask_to_rle(b)}
```
Only frames with a mask should go into the webdataset too — a frame present in the
webdataset but absent from the detections/test_list will `KeyError` at collate time.

**Even better than home-made masks: reuse a CNOS run you already have.**
SAM-6D's Instance Segmentation Model *is* CNOS, and its per-frame
`detection_XXXXXX.json` are already in GigaPose's exact schema (scene_id, image_id,
category_id, bbox, score, time, segmentation RLE-as-list). To reuse: merge the
per-frame files, set `image_id` from the **filename** (the internal value is 0),
`scene_id` to your scene, keep top-1 per frame; `category_id` is already 1. This
gives real CNOS-quality detections (SAM proposals + DINOv2 matching) for free.
Note: detection quality only helps if the object is large enough — on a tiny/far
object, CNOS and crude masks localise the same region and yield the same poses.

---

## 9. Debugging order that worked

1. **Env**: torch+CUDA import, panda3d import.
2. **Format discovery**: read `WebSceneDataset` / `get_split_name` *before* building data
   (this is where I learned about the webdataset requirement).
3. **Units**: `mesh.extents` in mm; intrinsics match resolution.
4. **Build dataset** → run the two converters → confirm shard + `key_to_shard.json`.
5. **Templates**: render one object, open a rendered PNG (is it the right shape? textured?).
6. **Single pass of `test.py`**: fix errors one at a time — they surfaced in this order:
   depth KeyError → template KeyError('1') → log_interval div0 → webdataset fname.
7. **Sanity-check poses**: `det(R)≈1`, translation plausible (mm).
8. **Overlay**: project CAD vertices with `K @ (R@V + t)` onto RGB; eyeball alignment.

---

## 9b. Full pipeline & best-performance config

The authors' canonical eval is `src/scripts/eval_bop.py` → `run_bop`: per dataset it
runs **coarse → MegaPose refine top-1 (`use_multiple=False`) → refine top-5
(`use_multiple=True`) → BOP-toolkit eval**, and reports all three. The **best result
is `refined_multiple` (top-5)**. The RGB refiner default is
`megapose-1.0-RGB-multi-hypothesis`, `n_iterations=5` (`configs/model/refiner/rgb.yaml`).
For best BOP accuracy the input detections are the official **CNOS** dets
(cnos-sam for BOP24/HOPE, cnos-fastsam for BOP23 core), not home-made masks.

On the cup, score progression matched expectation: coarse 0.01 → top-1 0.14 → top-5 0.49 (median).

**Docker is the intended runtime.** Container env is `gigapose-py310` (not the host
`gigapose`), CUDA 12.1, `WORKDIR=/workspace`. docker-compose bind-mounts the repo and
`gigaPose_datasets`, so host edits are live (no rebuild). Dataset *building*
(prepare/convert/render scripts) runs on the host where the source RGB/CAD live
(they aren't mounted in the container, and rendering needs panda3d); *inference*
(test/refine/visualize) runs in-container. The `pose` user may not be in the `docker`
group — check before assuming you can drive the daemon.

**Container env gap:** the repo's `environment.yml` omits README post-install deps,
so a fresh `gigapose-py310` lacks `bop_toolkit_lib` (imported by test/refine), the
top-level `megapose` package (`pip install -e .`, `package_dir=src`), and trimesh.
Symptom: `ModuleNotFoundError: No module named 'bop_toolkit_lib'`. Bake these into the
Dockerfile (or `pip install git+.../bop_toolkit.git trimesh && pip install -e /workspace`).
A `PYTHONPATH=/workspace/src` makes the `megapose` import resolve without `-e .`.

## 10. Quality reality check

A working pipeline ≠ good poses. Here the masks were tiny (~13×19 px on far frames)
and templates dark, so coarse scores were low (0–0.34) and depth varied wildly;
refinement only locked on for close frames. If results are poor, suspect the **inputs**
(object too small/far in frame, dark/untextured templates, sloppy masks) before the model.

---

## 11. Document as you go

Every non-obvious fix gets a line immediately. See `REALSENSE_CUP_PIPELINE.md` for the
exact run sequence and the full list of files changed/added this session.
