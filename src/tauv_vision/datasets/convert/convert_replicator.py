import argparse
from pathlib import Path
from typing import List, Set
import numpy as np
import human_id
import glob
import re
from PIL import Image
from tqdm import tqdm
import json
import random
import dirhash
import datetime
import matplotlib.pyplot as plt

"""
Example seg output:

...
bounding_box_2d_tight_0_9999.npy
bounding_box_2d_tight_labels_0_9999.json
bounding_box_2d_tight_prim_paths_0_9999.json
instance_segmentation_0_9999.png
instance_segmentation_mapping_0_9999.json
instance_segmentation_semantics_mapping_0_9999.json
rgb_0_9999.png
...

sample id will be content between "rgb_" and ".png". So "0_9999"
"""

selected_class_ids = ["torpedo_22_circle", "torpedo_22_trapezoid", "torpedo_22_star", "buoy_23_abydos_1", "buoy_23_abydos_2", "buoy_23_earth_1", "buoy_23_earth_2"]

def get_sample_ids(replicator_out_dir: Path) -> List[str]:
    rgb_names = glob.glob("rgb*", root_dir=replicator_out_dir)

    sample_id_re = re.compile(r"(?<=rgb_)(.*?)(?=\.png)")

    sample_ids = []
    for rgb_name in rgb_names:
        match = re.search(sample_id_re, rgb_name)
        if match is not None:
            sample_ids.append(match.group(1))
        else:
            raise ValueError("malformed rgb file name: {rgb_name}")

    return sample_ids


def split(pop: List, splits: List[float]) -> List[List]:
    out_splits = []

    pop_size = len(pop)
    for split in splits[:-1]:
        out_split = random.sample(pop, int(split * pop_size))
        pop = [x for x in pop if x not in out_split]
        out_splits.append(out_split)

    out_splits.append(pop)

    return out_splits

def parse_seg_value(s: str) -> (int, int, int, int):
    return [int(x) for x in s[1:-1].split(",")]


def convert_sample_seg(replicator_out_dir: Path, dataset_dir: Path, sample_id: str) -> Set[str]:
    rgb_path = replicator_out_dir / f"rgb_{sample_id}.png"

    bbox_path = replicator_out_dir / f"bounding_box_2d_tight_{sample_id}.npy"
    bbox_class_path = replicator_out_dir / f"bounding_box_2d_tight_labels_{sample_id}.json"
    bbox_instance_path = replicator_out_dir / f"bounding_box_2d_tight_prim_paths_{sample_id}.json"

    seg_path = replicator_out_dir / f"instance_segmentation_{sample_id}.png"
    seg_instance_path = replicator_out_dir / f"instance_segmentation_mapping_{sample_id}.json"

    rgb_out_path = dataset_dir / "data" / f"{sample_id}.png"
    seg_out_path = dataset_dir / "data" / f"{sample_id}_seg.png"
    json_out_path = dataset_dir / "data" / f"{sample_id}.json"

    rgb_raw_pil = Image.open(rgb_path)
    seg_raw_pil = Image.open(seg_path)

    bboxes_raw = np.load(bbox_path)
    with open(bbox_class_path, "r") as fp:
        bbox_classes_raw = json.load(fp)
    with open(bbox_instance_path, "r") as fp:
        bbox_instances_raw = json.load(fp)
    with open(seg_instance_path, "r") as fp:
        seg_instances_raw = json.load(fp)

    seg_instances_raw = {v: k for k, v in seg_instances_raw.items()}

    n_objects = len(bboxes_raw)

    w, h = rgb_raw_pil.size

    objects = []

    seg_raw_np = np.array(seg_raw_pil)
    seg_np = np.full((h, w), fill_value=255, dtype=np.uint8)

    class_ids = set()

    for object_i in range(n_objects):
        bbox_class_index, x0, y0, x1, y1, _ = bboxes_raw[object_i]

        bbox_x = ((x0 + x1) / 2) / w
        bbox_y = ((y0 + y1) / 2) / h

        bbox_w = abs(x1 - x0) / w
        bbox_h = abs(y1 - y0) / h

        bbox_class_id = bbox_classes_raw[str(bbox_class_index)]["class"].split(",")[-1]

        if bbox_class_id not in selected_class_ids:
            continue

        if bbox_instances_raw[object_i] in seg_instances_raw:
            # seg_value = parse_seg_value(seg_instances_raw[bbox_instances_raw[object_i]])
            seg_value = int(seg_instances_raw[bbox_instances_raw[object_i]])

            seg_np = np.where(
                seg_raw_np == seg_value,
                object_i,
                seg_np,
            )

        objects.append({
            "class_id": bbox_class_id,
            "bbox": {
                "y": round(bbox_y, 4),
                "x": round(bbox_x, 4),
                "h": round(bbox_h, 4),
                "w": round(bbox_w, 4)
            },
        })

        class_ids.add(bbox_class_id)

    seg_pil = Image.fromarray(seg_np)

    json_data = {
        "objects": objects,
    }

    rgb_raw_pil.save(rgb_out_path)
    seg_pil.save(seg_out_path)

    with open(json_out_path, "w") as fp:
        json.dump(json_data, fp, indent="  ")

    return class_ids


def convert(replicator_out_dir: Path, datasets_dir: Path, dataset_type: str, splits: List[float], email: str, description: str):
    if not np.isclose(sum(splits), 1):
        raise ValueError(f"Error: splits must sum to 1")

    if not replicator_out_dir.exists() or not replicator_out_dir.is_dir():
        raise ValueError(f"Error: {replicator_out_dir} does not exist")

    if not datasets_dir.exists() or not datasets_dir.is_dir():
        raise ValueError(f"Error: {replicator_out_dir} does not exist")

    dataset_id = human_id.generate_id(word_count=3)

    dataset_dir = datasets_dir / dataset_id

    meta_json_path = dataset_dir / "meta.json"
    splits_json_path = dataset_dir / "splits.json"
    classes_json_path = dataset_dir / "classes.json"

    if dataset_dir.exists():
        raise ValueError(f"Error: {dataset_dir} already exists")

    print(f"Creating dataset {dataset_dir}...")
    print(f"Input: {replicator_out_dir}")
    print(f"Type: {dataset_type}")
    print(f"Author: {email}")
    print(f"Description: {description}")

    dataset_dir.mkdir()
    (dataset_dir / "data").mkdir()

    sample_ids = get_sample_ids(replicator_out_dir)

    class_ids = set()

    for sample_id in tqdm(sample_ids):
        new_class_ids = convert_sample_seg(replicator_out_dir, dataset_dir, sample_id)
        class_ids = class_ids.union(new_class_ids)

    sample_id_splits = split(sample_ids, splits)
    splits_json_data = {
        "splits": {
            "train": sample_id_splits[0],
            "val": sample_id_splits[1],
            "test": sample_id_splits[2]
        }
    }

    with open(splits_json_path, "w") as fp:
        json.dump(splits_json_data, fp, indent="  ")

    classes_json_data = {
        "classes": [{"id": class_id} for class_id in class_ids]
    }

    with open(classes_json_path, "w") as fp:
        json.dump(classes_json_data, fp, indent="  ")

    md5 = dirhash.dirhash(dataset_dir, "md5")

    meta_json_data = {
        "author": email,
        "type": "seg",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "description": description,
        "md5": md5,
    }

    with open(meta_json_path, "w") as fp:
        json.dump(meta_json_data, fp, indent="  ")

    print(f"Created dataset {dataset_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replicator_out_dir")
    parser.add_argument("datasets_dir")
    parser.add_argument("--dataset_type", choices=["seg"], required=True)
    parser.add_argument("--splits", type=float, nargs=3, required=True)

    args = parser.parse_args()

    # email = input("Your email:")
    # description = input("Description:")
    email = "theo@chemel.net"
    description = "A test dataset."

    replicator_out_dir = Path(args.replicator_out_dir).expanduser()
    datasets_dir = Path(args.datasets_dir).expanduser()

    convert(replicator_out_dir, datasets_dir, args.dataset_type, args.splits, email, description)


if __name__ == "__main__":
    main()