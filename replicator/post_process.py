import torch
import torchvision.transforms.v2 as T
import argparse
import pathlib
import glob
import random
from PIL import Image
import re
import json
import numpy as np
from typing import Dict

from yolo_pose.datasets.segmentation_dataset.segmentation_dataset import SegmentationSample


def get_id(path: pathlib.Path) -> str:
    match = re.search(r"_(\d+)\.", str(path))
    if match:
        id = match.group(1)
        return id
    else:
        raise ValueError("no match for id")


def parse_seg_value(s: str) -> (int, int, int, int):
    return [int(x) for x in s[1:-1].split(",")]


def post_process(rgb_path: pathlib.Path, background_path: pathlib.Path,
                 in_dir: pathlib.Path, background_dir: pathlib.Path, out_dir: pathlib.Path, class_names: Dict[str, int]):
    id = get_id(rgb_path)

    seg_path = (in_dir / f"instance_segmentation_{id}").with_suffix(".png")
    seg_instance_path = (in_dir / f"instance_segmentation_mapping_{id}").with_suffix(".json")

    bbox_path = (in_dir / f"bounding_box_2d_loose_{id}").with_suffix(".npy")
    bbox_classification_path = (in_dir / f"bounding_box_2d_loose_labels_{id}").with_suffix(".json")
    bbox_instance_path = (in_dir / f"bounding_box_2d_loose_prim_paths_{id}").with_suffix(".json")

    img_pil = Image.open(rgb_path)
    background_pil = Image.open(background_path)
    seg_pil = Image.open(seg_path)
    seg_raw = T.ToTensor()(seg_pil)

    background_pil.paste(img_pil, (0, 0), img_pil)

    img = T.ToTensor()(background_pil)

    w, h = img_pil.size

    bboxes = np.load(bbox_path)
    with open(bbox_classification_path, "r") as fp:
        bbox_classifications = json.load(fp)
    with open(bbox_instance_path, "r") as fp:
        bbox_instances = json.load(fp)
    with open(seg_instance_path, "r") as fp:
        seg_instances = json.load(fp)

    n_detections = len(bboxes)

    seg_instances = {v: k for k, v in seg_instances.items()}

    valid = torch.full((n_detections,), fill_value=True, dtype=torch.bool)
    classifications = torch.zeros(n_detections, dtype=torch.long)
    bounding_boxes = torch.zeros((n_detections, 4), dtype=torch.float)
    seg = torch.full((h, w), fill_value=255, dtype=torch.uint8)

    for i in range(len(bboxes)):
        bbox_class, x0, y0, x1, y1, _ = bboxes[i]

        bbox_x = ((x0 + x1) / 2) / w
        bbox_y = ((y0 + y1) / 2) / h

        bbox_w = abs(x1 - x0) / w
        bbox_h = abs(y1 - y0) / h

        bbox_class_name = bbox_classifications[str(bbox_class)]["class"].split(",")[-1]

        class_id = class_names[bbox_class_name]

        classifications[i] = class_id
        bounding_boxes[i] = torch.Tensor([bbox_y, bbox_x, bbox_h, bbox_w])

        if bbox_instances[i] in seg_instances:
            seg_value = parse_seg_value(seg_instances[bbox_instances[i]])
            seg_mask = seg_raw == (torch.Tensor(seg_value).unsqueeze(1).unsqueeze(2) / 255)

        seg[seg_mask[0] & seg_mask[1] & seg_mask[2] & seg_mask[3]] = i

    sample = SegmentationSample(
        img=img,
        seg=seg,
        valid=valid,
        classifications=classifications,
        bounding_boxes=bounding_boxes,
    )

    out_id = id.zfill(8)
    sample.save(out_dir, out_id)

def run(in_dir: pathlib.Path, background_dir: pathlib.Path, out_dir: pathlib.Path):
    rgb_paths = glob.glob("rgb_*.png", root_dir=in_dir)
    rgb_paths = [in_dir / rgb_path for rgb_path in rgb_paths]

    background_paths = glob.glob("*.png", root_dir=background_dir)
    background_paths = [background_dir / background_path for background_path in background_paths]

    # TODO: READ CLASS NAMES
    class_names = {"torpedo_target": 0, "torpedo_target_open": 1, "torpedo_target_closed": 2}

    for rgb_path in rgb_paths:
        background_path = random.choice(background_paths)
        post_process(rgb_path, background_path, in_dir, background_dir, out_dir, class_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir")
    parser.add_argument("background_dir")
    parser.add_argument("out_dir")

    args = parser.parse_args()

    in_dir = pathlib.Path(args.in_dir).expanduser()
    background_dir = pathlib.Path(args.background_dir).expanduser()
    out_dir = pathlib.Path(args.out_dir).expanduser()

    assert in_dir.is_dir()
    assert background_dir.is_dir()

    if not out_dir.exists():
        out_dir.mkdir()

    run(in_dir, background_dir, out_dir)


if __name__ == "__main__":
    main()