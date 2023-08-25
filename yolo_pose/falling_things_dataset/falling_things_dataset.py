import torch
import json
from PIL import Image
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path
from enum import Enum
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FallingThingsVariant(Enum):
    SINGLE = "single"
    MIXED = "mixed"


class FallingThingsEnvironment(Enum):
    Kitchen0 = "kitchen_0"
    Kitchen1 = "kitchen_0"
    Kitchen2 = "kitchen_0"
    Kitchen3 = "kitchen_0"
    Kitchen4 = "kitchen_0"
    KiteDemo0 = "kitedemo_0"
    KiteDemo1 = "kitedemo_1"
    KiteDemo2 = "kitedemo_2"
    KiteDemo3 = "kitedemo_3"
    KiteDemo4 = "kitedemo_4"
    Temple0 = "temple_0"
    Temple1 = "temple_0"
    Temple2 = "temple_0"
    Temple3 = "temple_0"
    Temple4 = "temple_0"


class FallingThingsObject(Enum):
    MasterChefCan = "002_master_chef_can_16k"
    CrackerBox = "003_cracker_box_16k"
    SugarBox = "004_sugar_box_16k"
    TomatoSoupCan = "005_tomato_soup_can_16k"
    MustardBottle = "006_mustard_bottle_16k"
    TunaFishCan = "007_tuna_fish_can_16k"
    PuddingBox = "008_pudding_box_16k"
    GelatinBox = "009_gelatin_box_16k"
    PottedMeatCan = "010_potted_meat_can_16k"
    Banana = "011_banana_16k"
    PitcherBase = "019_pitcher_base_16k"
    BleachCleanser = "021_bleach_cleanser_16k"
    Bowl = "024_bowl_16k"
    Mug = "025_mug_16k"
    PowerDrill = "035_power_drill_16k"
    WoodBlock = "036_wood_block_16k"
    Scissors = "037_scissors_16k"
    LargeMarker = "040_large_marker_16k"
    LargeClamp = "051_large_clamp_16k"
    ExtraLargeClamp = "052_extra_large_clamp_16k"
    FoamBrick = "061_foam_brick_16k"


falling_things_object_ids = {member.value: index for index, member in enumerate(FallingThingsObject)}


@dataclass
class FallingThingsSample:
    intrinsics: torch.Tensor
    classifications: torch.Tensor
    bounding_boxes: torch.Tensor
    poses: torch.Tensor
    cuboids: torch.Tensor
    projected_cuboids: torch.Tensor
    img: torch.Tensor
    seg: torch.Tensor
    depth: torch.Tensor


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

    def __init__(self,
                 root: str,
                 variant: FallingThingsVariant,
                 environments: List[FallingThingsEnvironment],
                 objects: Optional[List[FallingThingsObject]]
                 ):
        super().__init__()

        self._root: Path = Path(root)
        self._variant: FallingThingsVariant = variant
        self._environments = environments

        if variant != FallingThingsVariant.SINGLE and objects is not None:
            raise ValueError("objects must be specified for variant SINGLE and cannot be specified for variant MIXED")

        self._objects = objects

        variant_dir = (self._root / self._variant.value).expanduser()

        if (not variant_dir.exists()) or (not variant_dir.is_dir()):
            raise ValueError(f"{variant_dir} does not exist")

        if variant == FallingThingsVariant.SINGLE:
            assert objects is not None
            object_dirs = [variant_dir / obj.value for obj in objects]
        elif variant == FallingThingsVariant.MIXED:
            object_dirs = [variant_dir]

        environment_dirs = []
        for object_dir in object_dirs:
            environment_dirs.extend([object_dir / environment.value for environment in environments])

        id_paths = self._get_id_paths(environment_dirs)
        id_paths = [value for sublist in id_paths.values() for value in sublist]
        self._id_paths: List[Path] = id_paths

    def __len__(self) -> int:
        return len(self._id_paths)

    def __getitem__(self, i: int) -> (torch.Tensor, ...):
        id_path = self._id_paths[i]

        camera_json_path = id_path.with_name("_camera_settings.json")
        object_json_path = id_path.with_name("_object_settings.json")

        left_json_path = id_path.with_suffix(".left.json")
        left_img_path = id_path.with_suffix(".left.jpg")
        left_seg_path = id_path.with_suffix(".left.seg.png")
        left_depth_path = id_path.with_suffix(".left.depth.png")

        camera_data = self._get_json(camera_json_path)
        object_data = self._get_json(object_json_path)
        left_data = self._get_json(left_json_path)

        to_tensor = transforms.ToTensor()

        intrinsics = [
            camera_data["camera_settings"][0]["intrinsic_settings"]["fx"],
            camera_data["camera_settings"][0]["intrinsic_settings"]["fy"],
            camera_data["camera_settings"][0]["intrinsic_settings"]["cx"],
            camera_data["camera_settings"][0]["intrinsic_settings"]["cy"],
        ]
        intrinsics = torch.Tensor(intrinsics)

        classifications = [
            falling_things_object_ids[object["class"].lower()] for object in left_data["objects"]
        ]
        classifications = torch.Tensor(classifications)

        bounding_boxes = [
            object["bounding_box"]["top_left"] + object["bounding_box"]["bottom_right"] for object in left_data["objects"]
        ]
        bounding_boxes = torch.Tensor(bounding_boxes)

        cuboids = [
            object["cuboid"] for object in left_data["objects"]
        ]
        cuboids = torch.Tensor(cuboids)

        projected_cuboids = [
            object["projected_cuboid"] for object in left_data["objects"]
        ]
        projected_cuboids = torch.Tensor(projected_cuboids)

        poses = [
            object["location"] + object["quaternion_xyzw"] for object in left_data["objects"]
        ]
        poses = torch.Tensor(poses)

        img = to_tensor(Image.open(left_img_path))
        seg = to_tensor(Image.open(left_seg_path))[0]
        depth = to_tensor(Image.open(left_depth_path))[0]

        for object in object_data["exported_objects"]:
            seg = torch.where(seg == object["segmentation_class_id"], falling_things_object_ids[object["class"].lower()], seg)

        depth = depth.float()
        depth[depth == 65535] = torch.nan
        depth = depth / 1e4

        sample = FallingThingsSample(
            intrinsics=intrinsics,
            classifications=classifications,
            bounding_boxes=bounding_boxes,
            poses=poses,
            cuboids=cuboids,
            projected_cuboids=projected_cuboids,
            img=img,
            seg=seg,
            depth=depth,
        )

        return sample


    def _get_id_paths(self, dirs: List[Path]) -> Dict[Path, List[int]]:
        id_paths = {}

        for dir in dirs:
            filenames = [file.name for file in dir.iterdir() if file.is_file()]

            unique_id_paths = set()
            for filename in filenames:
                if len(filename) >= 6 and filename[:6].isdigit():
                    unique_id_paths.add(dir / filename[:6])

            id_paths[dir] = list(unique_id_paths)

        return id_paths

    def _get_json(self, path: Path) -> Dict:
        with open(path, "r") as file:
            return json.load(file)

def main():
    import matplotlib.pyplot as plt


    single_dataset = FallingThingsDataset(
        "~/Documents/falling_things/fat",
        FallingThingsVariant.SINGLE,
        [
            FallingThingsEnvironment.Kitchen0,
            FallingThingsEnvironment.Kitchen1,
            FallingThingsEnvironment.Kitchen2,
            FallingThingsEnvironment.Kitchen3,
            FallingThingsEnvironment.Kitchen4,
        ],
        [
            FallingThingsObject.CrackerBox,
        ]
    )

    mixed_dataset = FallingThingsDataset(
        "~/Documents/falling_things/fat",
        FallingThingsVariant.MIXED,
        [
            FallingThingsEnvironment.Kitchen0,
            FallingThingsEnvironment.Kitchen1,
            FallingThingsEnvironment.Kitchen2,
            FallingThingsEnvironment.Kitchen3,
            FallingThingsEnvironment.Kitchen4,
        ],
        None
    )

    mixed_sample = mixed_dataset[0]
    plt.figure()
    plt.imshow(mixed_sample.img.permute(1, 2, 0))
    plt.figure()
    plt.imshow(mixed_sample.seg)
    plt.figure()
    plt.imshow(mixed_sample.depth)

    # TODO: Draw cuboids and shit, make sure everything checks out

    plt.show()


if __name__ == "__main__":
    main()