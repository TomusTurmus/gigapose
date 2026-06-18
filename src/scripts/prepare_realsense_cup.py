"""Prepare the custom `realsense_cup` dataset for GigaPose.

Reuses the RealSense capture + masks from the FoundationPose project and the
KITchen CAD model (obj_000077, a cup, in mm). Produces:

  datasets/realsense_cup/
    scenewise/000001/rgb/*.png            (only frames that have a mask)
    scenewise/000001/scene_camera.json
    models/obj_000077.ply
    models/models_info.json               (generated from the mesh)
  datasets/default_detections/realsense_cup/realsense_cup.json
                                          (CNOS-format dets built from masks)

After this, run the scenewise->imagewise->webdataset converters to create
datasets/realsense_cup/test/ (tar shards + key_to_shard.json).

Units: KITchen mesh is in millimetres (BOP standard) -> no conversion for GigaPose.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import cv2
import trimesh
from scipy.spatial.distance import cdist

from bop_toolkit_lib import pycoco_utils

# --- paths -------------------------------------------------------------------
FP_SRC = Path("/home/pose/dipl/FoundationPose/demo_data/realsense_cup")
KITCHEN_CAD = Path("/home/pose/dipl/datasets/KITchen/models/obj_000077.ply")
GIGA_ROOT = Path("/home/pose/dipl/gigapose/gigaPose_datasets/datasets")

# GigaPose indexes templates as template_data[obj_id - 1] and derives template
# labels as f"{index+1}", so object ids MUST be contiguous from 1. With a single
# object it must be 1 (KITchen's source id 77 is only used to locate the files).
OBJ_ID = 1
SRC_OBJ_ID = 77
SCENE_ID = 1
CAM_K = [456.5, 0.0, 320.0, 0.0, 456.5, 180.0, 0.0, 0.0, 1.0]
DEPTH_SCALE = 1.0

DST = GIGA_ROOT / "realsense_cup"
SCENE_DIR = DST / "scenewise" / f"{SCENE_ID:06d}"
MODELS_DIR = DST / "models"
DET_DIR = GIGA_ROOT / "default_detections" / "realsense_cup"


def build_scenewise():
    rgb_out = SCENE_DIR / "rgb"
    rgb_out.mkdir(parents=True, exist_ok=True)

    mask_files = sorted((FP_SRC / "masks").glob("[0-9]*.png"))
    scene_camera = {}
    used = []
    for mp in mask_files:
        im_id = int(mp.stem)
        rgb_src = FP_SRC / "rgb" / f"{im_id:06d}.png"
        if not rgb_src.exists():
            print(f"  skip {im_id}: no rgb")
            continue
        shutil.copy(rgb_src, rgb_out / f"{im_id:06d}.png")
        scene_camera[str(im_id)] = {"cam_K": CAM_K, "depth_scale": DEPTH_SCALE}
        used.append(im_id)

    (SCENE_DIR / "scene_camera.json").write_text(json.dumps(scene_camera, indent=2))
    print(f"scenewise: {len(used)} frames -> {SCENE_DIR}")
    return used


def build_models_info():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst_ply = MODELS_DIR / f"obj_{OBJ_ID:06d}.ply"
    shutil.copy(KITCHEN_CAD, dst_ply)

    # The PLY header references a texture "obj_000077.png" that doesn't exist
    # under that name in KITchen (textures are named by object, e.g.
    # FruitTea_800_tex.png for obj 77). Copy it under the name the PLY expects
    # (the original source id), so the renderer finds it.
    tex_src = KITCHEN_CAD.parent / "FruitTea_800_tex.png"
    if tex_src.exists():
        shutil.copy(tex_src, MODELS_DIR / f"obj_{SRC_OBJ_ID:06d}.png")

    mesh = trimesh.load(dst_ply)
    mn = mesh.bounds[0]
    ext = mesh.extents
    # BOP diameter = max pairwise vertex distance; approximate via convex hull
    hull = mesh.convex_hull.vertices
    diameter = float(cdist(hull, hull).max())
    info = {
        str(OBJ_ID): {
            "diameter": diameter,
            "min_x": float(mn[0]), "min_y": float(mn[1]), "min_z": float(mn[2]),
            "size_x": float(ext[0]), "size_y": float(ext[1]), "size_z": float(ext[2]),
        }
    }
    (MODELS_DIR / "models_info.json").write_text(json.dumps(info, indent=2))
    print(f"models_info: extents={ext} diameter={diameter:.2f} mm -> {MODELS_DIR}")


def build_detections(used):
    DET_DIR.mkdir(parents=True, exist_ok=True)
    dets = []
    for im_id in used:
        m = cv2.imread(str(FP_SRC / "masks" / f"{im_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
        binary = (m > 0).astype(np.uint8)
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            print(f"  skip det {im_id}: empty mask")
            continue
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
        rle = pycoco_utils.binary_mask_to_rle(binary)
        dets.append({
            "scene_id": SCENE_ID,
            "image_id": im_id,
            "category_id": OBJ_ID,
            "bbox": [x, y, w, h],
            "score": 1.0,
            "time": 0.0,
            "segmentation": rle,
        })
    out = DET_DIR / "realsense_cup.json"
    out.write_text(json.dumps(dets))
    print(f"detections: {len(dets)} dets -> {out}")


if __name__ == "__main__":
    used = build_scenewise()
    build_models_info()
    build_detections(used)
    print("Done. Next: run scenewise->imagewise->webdataset converters.")
