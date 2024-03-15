from enum import Enum
import pathlib
import random
import json
from torch.utils.data import Dataset
from dataclasses import dataclass
import torch
from typing import Optional, Dict
from typing_extensions import Self
from PIL import Image
import torchvision.transforms.v2 as T
import torch.nn.functional as F


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class PoseSample:
    img: torch.Tensor # [batch_size, 3, in_h, in_w]

    valid: torch.Tensor  # [batch_size, n_objects]
    label: torch.Tensor  # [batch_size, n_objects]
    center: torch.Tensor  # [batch_size, n_objects, 2]
    size: torch.Tensor  # [batch_size, n_objects, 2]

    roll: Optional[torch.Tensor]  # [batch_size, n_objects]
    pitch: Optional[torch.Tensor]  # [batch_size, n_objects]
    yaw: Optional[torch.Tensor]  # [batch_size, n_objects]
    depth: Optional[torch.Tensor]  # [batch_size, n_objects]

    def to(self, device: torch.device) -> Self:
        return PoseSample(
            img=self.img.to(device),
            valid=self.valid.to(device),
            label=self.label.to(device),
            center=self.center.to(device),
            size=self.size.to(device),
            roll=self.roll.to(device) if self.roll is not None else None,
            pitch=self.pitch.to(device) if self.pitch is not None else None,
            yaw=self.yaw.to(device) if self.yaw is not None else None,
            depth=self.depth.to(device) if self.depth is not None else None,
        )

    @classmethod
    def load(cls, data_path: pathlib.Path, id: str, label_id_to_index: Dict[str, int]) -> Self:
        json_path = (data_path / id).with_suffix(".json")
        img_path = (data_path / id).with_suffix(".png")

        with open(json_path, "r") as fp:
            data = json.load(fp)

        img_pil = Image.open(img_path).convert("RGB")

        tf = T.Compose([T.ToImageTensor(), T.ConvertImageDtype()])
        img = tf(img_pil)

        filtered_objects = [object for object in data["objects"] if object["label"] in label_id_to_index]

        n_objects = len(filtered_objects)

        valid = torch.full((n_objects,), fill_value=True, dtype=torch.bool)

        label = torch.zeros(n_objects, dtype=torch.long)
        center = torch.zeros((n_objects, 2), dtype=torch.float32)
        size = torch.zeros((n_objects, 2), dtype=torch.float32)

        roll = torch.zeros(n_objects, dtype=torch.float32)
        pitch = torch.zeros(n_objects, dtype=torch.float32)
        yaw = torch.zeros(n_objects, dtype=torch.float32)
        depth = torch.zeros(n_objects, dtype=torch.float32)

        for i, object in enumerate(filtered_objects):
            label[i] = label_id_to_index[object["label"]]

            center[i, 0] = object["bbox"]["y"]
            center[i, 1] = object["bbox"]["x"]

            size[i, 0] = object["bbox"]["h"]
            size[i, 1] = object["bbox"]["w"]

            roll[i] = object["pose"]["roll"]
            pitch[i] = object["pose"]["pitch"]
            yaw[i] = object["pose"]["yaw"]
            depth[i] = object["pose"]["distance"]

        sample = PoseSample(
            img=img.unsqueeze(0),
            valid=valid.unsqueeze(0),
            label=label.unsqueeze(0),
            center=center.unsqueeze(0),
            size=size.unsqueeze(0),
            roll=roll.unsqueeze(0),
            pitch=pitch.unsqueeze(0),
            yaw=yaw.unsqueeze(0),
            depth=depth.unsqueeze(0),
        )

        return sample

    @classmethod
    def collate(cls, samples: [Self]) -> Self:
        n_detections = [sample.valid.size(1) for sample in samples]
        max_n_detections = max(n_detections)

        img = torch.cat([sample.img for sample in samples], dim=0)

        valid = torch.cat([
            F.pad(sample.valid, (0, max_n_detections - sample.valid.size(1)), value=False)
            for sample in samples
        ], dim=0)
        label = torch.cat([
            F.pad(sample.label, (0, max_n_detections - sample.label.size(1)), value=0)
            for sample in samples
        ], dim=0)
        center = torch.cat([
            F.pad(sample.center, (0, 0, 0, max_n_detections - sample.center.size(1)), value=0)
            for sample in samples
        ], dim=0)
        size = torch.cat([
            F.pad(sample.size, (0, 0, 0, max_n_detections - sample.size.size(1)), value=0)
            for sample in samples
        ], dim=0)
        roll = torch.cat([
            F.pad(sample.roll, (0, max_n_detections - sample.roll.size(1)), value=0)
            for sample in samples
        ], dim=0)
        pitch = torch.cat([
            F.pad(sample.pitch, (0, max_n_detections - sample.pitch.size(1)), value=0)
            for sample in samples
        ], dim=0)
        yaw = torch.cat([
            F.pad(sample.yaw, (0, max_n_detections - sample.yaw.size(1)), value=0)
            for sample in samples
        ], dim=0)
        depth = torch.cat([
            F.pad(sample.depth, (0, max_n_detections - sample.depth.size(1)), value=0)
            for sample in samples
        ], dim=0)

        result = PoseSample(
            img=img,
            valid=valid,
            label=label,
            center=center,
            size=size,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            depth=depth,
        )

        return result


class PoseDataset(Dataset):

    def __init__(self, root: pathlib.Path, split: Split, label_id_to_index: Dict[str, int]):
        super().__init__()

        self._root_path: pathlib.Path = root
        self._split: Split = split

        if not self._root_path.is_dir():
            raise ValueError(f"No such directory: {self._root_path}")

        self._data_path: pathlib.Path = self._root_path / "data"

        if not self._data_path.is_dir():
            raise ValueError(f"No such directory: {self._data_path}")

        self._ids: [str] = self._get_ids()
        random.shuffle(self._ids)

        self._label_id_to_index: Dict[str, int] = label_id_to_index

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, i: int):
        return PoseSample.load(self._data_path, self._ids[i], self._label_id_to_index)

    def _get_ids(self) -> [str]:
        splits_json_path = self._root_path / "splits.json"

        with open(splits_json_path, "r") as fp:
            splits_data = json.load(fp)

        return splits_data["splits"][self._split.value]


def main():
    label_id_to_index = {
        "torpedo_22_circle": 0,
        "torpedo_22_trapezoid": 1,
    }

    ds = PoseDataset(
        root=pathlib.Path("~/Documents/TAUV-Datasets/test").expanduser(),
        split=Split.TEST,
        label_id_to_index=label_id_to_index,
    )

    print(f"{len(ds)}")

    for sample in ds:
        print(sample)


if __name__ == "__main__":
    main()