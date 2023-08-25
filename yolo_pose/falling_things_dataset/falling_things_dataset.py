import torch
from pathlib import Path
from enum import Enum
from torch.utils.data import Dataset


class FallingThingsVariant(Enum):
    SINGLE = "single"
    MIXED = "mixed"

def get_all_directories_recursive(dir):
    dirs = []
    for item in dir.iterdir():
        if item.is_dir():
            dirs.append(item)
            dirs.extend(get_all_directories_recursive(item))
    return dirs


class FallingThingsDataset(Dataset):

    """
    Directory structure, from https://pytorch.org/vision/0.15/_modules/torchvision/datasets/_stereo_matching.html#FallingThingsStereo

    <dir>
        single
            dir1
                scene1
                    _object_settings.json
                    _camera_settings.json
                    image1.left.depth.png
                    image1.right.depth.png
                    image1.left.jpg
                    image1.right.jpg
                    image2.left.depth.png
                    image2.right.depth.png
                    image2.left.jpg
                    image2.right
                    ...
                scene2
            ...
        mixed
            scene1
                _object_settings.json
                _camera_settings.json
                image1.left.depth.png
                image1.right.depth.png
                image1.left.jpg
                image1.right.jpg
                image2.left.depth.png
                image2.right.depth.png
                image2.left.jpg
                image2.right
                ...
            scene2
    """

    def __init__(self, root: str, variant: FallingThingsVariant):
        super().__init__()

        self._root: Path = Path(root)
        self._variant: FallingThingsVariant = variant

        variant_dir = (self._root / self._variant.value).expanduser()

        if (not variant_dir.exists()) or (not variant_dir.is_dir()):
            raise ValueError(f"{variant_dir} does not exist")

        sample_dirs = get_all_directories_recursive(variant_dir)

        print(sample_dirs)


    def __len__(self) -> int:
        return 0

    def __getitem__(self, i: int) -> (torch.Tensor, ...):
        return None


def main():
    dataset = FallingThingsDataset("~/Documents/falling_things", FallingThingsVariant.SINGLE)


if __name__ == "__main__":
    main()