"""Overlay estimated poses (coarse or refined) onto the RGB frames.

Projects the CAD model vertices into the image using the predicted pose and the
camera intrinsics, drawing them as a translucent green point cloud so the pose
alignment can be checked visually.
"""
import sys
from pathlib import Path

import numpy as np
import cv2
import trimesh

ROOT = Path("/home/pose/dipl/gigapose/gigaPose_datasets/datasets")
RGB_DIR = ROOT / "realsense_cup" / "scenewise" / "000001" / "rgb"
MESH = ROOT / "realsense_cup" / "models" / "obj_000001.ply"
K = np.array([[456.5, 0, 320], [0, 456.5, 180], [0, 0, 1]], dtype=np.float64)
OUT = Path("/home/pose/dipl/gigapose/gigaPose_datasets/results/large_cup/viz")


def load_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    poses = {}
    for _, r in df.iterrows():
        R = np.array(list(map(float, r["R"].split()))).reshape(3, 3)
        t = np.array(list(map(float, r["t"].split())))
        poses[int(r["im_id"])] = (R, t, float(r["score"]))
    return poses


def main(csv_path):
    OUT.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(MESH)
    V = np.asarray(mesh.vertices)  # mm
    if len(V) > 3000:
        V = V[np.random.choice(len(V), 3000, replace=False)]
    poses = load_csv(csv_path)

    for im_id, (R, t, score) in sorted(poses.items()):
        rgb = cv2.imread(str(RGB_DIR / f"{im_id:06d}.png"))
        if rgb is None:
            continue
        Xc = (R @ V.T).T + t  # mm in camera frame
        z = Xc[:, 2]
        ok = z > 1e-6
        uv = (K @ Xc[ok].T).T
        uv = uv[:, :2] / uv[:, 2:3]
        for u, v in uv.astype(int):
            if 0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0]:
                rgb[v, u] = (0, 255, 0)
        cv2.putText(rgb, f"f{im_id} s={score:.2f} Z={t[2]:.0f}mm", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(OUT / f"{im_id:06d}.png"), rgb)
    print(f"wrote {len(poses)} overlays to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
