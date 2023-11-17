import torch
from torch.utils.data import Dataset
import pathlib
import glob
import re
import numpy as np
import torchvision.transforms.v2 as T
import json
from PIL import Image
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from yolact.model.boxes import box_xy_swap


class SegmentationDatasetSet(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass
class SegmentationSample:
    img: torch.Tensor
    seg: torch.Tensor
    valid: torch.Tensor
    classifications: torch.Tensor
    bounding_boxes: torch.Tensor

    def save(self, set_path: pathlib.Path, id: str):
        json_path = (set_path / id).with_suffix(".json")
        img_path = (set_path / id).with_suffix(".png")
        seg_path = (set_path / f"{id}_seg").with_suffix(".png")

        detections = []

        n_detections = len(self.valid)

        for i in range(n_detections):
            classification = self.classifications[i]
            y, x, h, w = self.bounding_boxes[i]
            detections.append({
                "classification": int(classification),
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h)
            })

        meta = {"detections": detections}

        with open(json_path, "w") as fp:
            json.dump(meta, fp)

        to_pil = T.ToPILImage()

        img_pil = to_pil(self.img)
        img_pil.save(img_path)

        seg = torch.stack((
            self.seg,
            255 - self.seg,
            255 - self.seg,
        ), dim=0)
        seg_pil = to_pil(seg)
        seg_pil.save(seg_path)

    @classmethod
    def load(cls, set_path: pathlib.Path, id: str, transform: Optional = None):
        json_path = (set_path / id).with_suffix(".json")
        img_path = (set_path / id).with_suffix(".png")
        seg_path = (set_path / f"{id}_seg").with_suffix(".png")

        meta = None
        with open(json_path, "r") as fp:
            meta = json.load(fp)

        img_pil = Image.open(img_path)
        seg_pil = Image.open(seg_path)

        img_np = np.array(img_pil)
        seg_np = np.array(seg_pil)[:, :, 0]

        n_detections = len(meta["detections"])

        valid = torch.full((n_detections,), fill_value=True, dtype=torch.bool)
        classifications_np = np.zeros(n_detections, dtype=np.int64)
        bounding_boxes_np = np.zeros((n_detections, 4))

        for i, detection in enumerate(meta["detections"]):
            classifications_np[i] = detection["classification"] + 1
            bounding_boxes_np[i] = np.array([
                detection["x"],
                detection["y"],
                detection["w"],
                detection["h"],
            ])

        if transform is not None:
            transformed = transform(
                image=img_np,
                mask=seg_np,
                bboxes=bounding_boxes_np,
                classifications=classifications_np,
            )

            img_np = transformed["image"]
            seg_np = transformed["mask"]
            bounding_boxes_np = transformed["bboxes"]

        n_detections = len(bounding_boxes_np)

        img = T.ToTensor()(img_np)
        seg = T.ToTensor()(seg_np)[0]
        seg = (255 * seg).to(torch.uint8)
        classifications = torch.Tensor(classifications_np).to(torch.long)

        if n_detections == 0:
            valid = torch.Tensor([False])
            classifications = torch.Tensor([0]).to(torch.long)
            bounding_boxes = torch.Tensor([[0, 0, 0, 0]])

            sample = cls(
                img=img,
                seg=seg,
                valid=valid,
                classifications=classifications,
                bounding_boxes=bounding_boxes,
            )

            return sample

        bounding_boxes = box_xy_swap(torch.Tensor(bounding_boxes_np).unsqueeze(0)).squeeze(0)

        sample = cls(
            img=img,
            seg=seg,
            valid=valid,
            classifications=classifications,
            bounding_boxes=bounding_boxes,
        )

        return sample


class SegmentationDataset(Dataset):

    def __init__(self, root: pathlib.Path, set: SegmentationDatasetSet, transform: Optional = None):
        super().__init__()

        self._root_path: pathlib.Path = root
        self._set: SegmentationDatasetSet = set

        self._transform: Optional = transform

        if not self._root_path.is_dir():
            raise ValueError(f"No such directory: {self._root_path}")

        self._set_path: pathlib.Path = self._root_path / self._set.value

        if not self._set_path.is_dir():
            raise ValueError(f"No such directory: {self._set_path}")

        self._ids: [str] = SegmentationDataset.get_ids(self._set_path)

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, i: int) -> SegmentationSample:
        id = self._ids[i]

        return SegmentationSample.load(self._set_path, id, transform=self._transform)

    @staticmethod
    def get_ids(path: pathlib.Path) -> [str]:
        json_names = glob.glob("*.json", root_dir=path)

        ids = list(filter(lambda id: id is not None, [SegmentationDataset.get_id(name) for name in json_names]))

        return ids

    @staticmethod
    def get_id(name: str) -> Optional[str]:
        match = re.search(r"(\d+)\.json", name)
        if match:
            return match.group(1)
        else:
            return None


def main():
    ds = SegmentationDataset(
        root=pathlib.Path("~/Documents/torpedo_target_2").expanduser(),
        set=SegmentationDatasetSet.TRAIN,
    )

    print(f"Length: {len(ds)}")
    for sample in ds:
        print(sample)


if __name__ == "__main__":
    main()