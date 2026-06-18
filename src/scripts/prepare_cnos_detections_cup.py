"""Convert SAM-6D ISM (CNOS-style) detections into GigaPose's detections JSON.

SAM-6D's Instance Segmentation Model is built on CNOS (SAM proposals + DINOv2
template matching), so its per-frame `detection_XXXXXX.json` files are already in
the schema GigaPose consumes (scene_id, image_id, category_id, bbox, score, time,
segmentation RLE). This merges them into one file, fixing:
  - image_id (the per-file value is 0; the real frame id is in the filename)
  - scene_id (-> 1 to match the realsense_cup scene)
category_id is already 1 (matches the renumbered obj_000001). We keep the top-1
detection per frame for this single-object scene.
"""
import json
from pathlib import Path

SAM6D_OUT = Path("/home/pose/dipl/SAM-6D/SAM-6D/output/obj_000077/sam6d_results")
OUT = Path(
    "/home/pose/dipl/gigapose/gigaPose_datasets/datasets/"
    "default_detections/realsense_cup/realsense_cup_cnos.json"
)
SCENE_ID = 1
TOPK = 1  # single object per frame -> keep the best-matching detection

dets = []
empty = []
for f in sorted(SAM6D_OUT.glob("detection_0*.json")):
    im_id = int(f.stem.split("_")[1])
    frame_dets = json.loads(f.read_text())
    if not frame_dets:
        empty.append(im_id)
        continue
    frame_dets.sort(key=lambda d: d["score"], reverse=True)
    for d in frame_dets[:TOPK]:
        d["scene_id"] = SCENE_ID
        d["image_id"] = im_id  # real frame id from filename
        dets.append(d)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(dets))
print(f"wrote {len(dets)} CNOS detections to {OUT}")
print(f"empty frames (no detection): {empty}")
print(f"score range: {min(d['score'] for d in dets):.3f} .. {max(d['score'] for d in dets):.3f}")
