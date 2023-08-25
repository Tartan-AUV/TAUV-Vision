import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import List

from yolo_pose.model.config import Config
from yolo_pose.model.loss import loss
from yolo_pose.model.model import YoloPose
from yolo_pose.falling_things_dataset.falling_things_dataset import FallingThingsDataset, FallingThingsVariant, FallingThingsEnvironment, FallingThingsSample


config = Config(
    in_w=512,
    in_h=512,
    feature_depth=256,
    n_classes=21,
    n_prototype_masks=32,
    n_prototype_points=64,
    n_points=8,
    n_masknet_layers_pre_upsample=1,
    n_masknet_layers_post_upsample=1,
    n_pointnet_layers_pre_upsample=1,
    n_pointnet_layers_post_upsample=1,
    n_prediction_head_layers=1,
    n_fpn_downsample_layers=2,
    anchor_scales=(24, 48, 96, 192, 384),
    anchor_aspect_ratios=(1 / 2, 1, 2),
    iou_pos_threshold=0.5,
    iou_neg_threshold=0.4,
    negative_example_ratio=3,
)

lr = 1e-2
momentum = 0.9
weight_decay = 0.9
n_epochs = 1000
train_split = 0.9
batch_size = 4

hue_jitter = 0.5
saturation_jitter = 0.5
brightness_jitter = 0.5

img_mean = (0.485, 0.456, 0.406)
img_stddev = (0.229, 0.224, 0.225)

trainval_environments = [
    FallingThingsEnvironment.Kitchen0,
    FallingThingsEnvironment.Kitchen1,
    FallingThingsEnvironment.Kitchen2,
    FallingThingsEnvironment.Kitchen3,
    FallingThingsEnvironment.Kitchen4,
]

falling_things_root = "~/Documents/falling_things/fat"

def collate_samples(samples: List[FallingThingsSample]) -> FallingThingsSample:
    n_detections = [sample.valid.size(0) for sample in samples]
    max_n_detections = max(n_detections)

    intrinsics = torch.stack([sample.intrinsics for sample in samples], dim=0)
    valid = torch.stack([
        F.pad(sample.valid, (0, max_n_detections - sample.valid.size(0)), value=False)
        for sample in samples
    ], dim=0)
    classifications = torch.stack([
        F.pad(sample.classifications, (0, max_n_detections - sample.classifications.size(0)), value=False)
        for sample in samples
    ], dim=0)
    bounding_boxes = torch.stack([
        F.pad(sample.bounding_boxes, (0, 0, 0, max_n_detections - sample.bounding_boxes.size(0)), value=False)
        for sample in samples
    ], dim=0)
    poses = torch.stack([
        F.pad(sample.poses, (0, 0, 0, max_n_detections - sample.poses.size(0)), value=False)
        for sample in samples
    ], dim=0)
    cuboids = torch.stack([
        F.pad(sample.cuboids, (0, 0, 0, 0, 0, max_n_detections - sample.cuboids.size(0)), value=False)
        for sample in samples
    ], dim=0)
    projected_cuboids = torch.stack([
        F.pad(sample.projected_cuboids, (0, 0, 0, 0, 0, max_n_detections - sample.projected_cuboids.size(0)), value=False)
        for sample in samples
    ], dim=0)
    img = torch.stack([sample.img for sample in samples], dim=0)
    seg = torch.stack([sample.seg for sample in samples], dim=0)
    depth = torch.stack([sample.depth for sample in samples], dim=0)

    sample = FallingThingsSample(
        intrinsics=intrinsics,
        valid=valid,
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


def transform_sample(sample: FallingThingsSample) -> FallingThingsSample:
    # TODO Crop all of these, and apply basic color transformations to img

    # crop_transforms = transforms.Compose([
    #     transforms.Resize(min(config.in_h, config.in_w), interpolation=transforms.InterpolationMode.BILINEAR),
    #     transforms.CenterCrop((config.in_h, config.in_w)),
    # ])
    #
    # seg_crop_transforms = transforms.Compose([
    #     transforms.Resize(min(config.in_h, config.in_w), interpolation=transforms.InterpolationMode.NEAREST),
    #     transforms.CenterCrop((config.in_h, config.in_w)),
    # ])

    color_transforms = transforms.Compose([
        transforms.ColorJitter(hue=hue_jitter, saturation=saturation_jitter, brightness=brightness_jitter),
        transforms.Normalize(mean=img_mean, std=img_stddev),
    ])

    img = color_transforms(sample.img)
    seg = sample.seg
    depth = sample.depth

    transformed_sample = FallingThingsSample(
        intrinsics=sample.intrinsics,
        valid=sample.valid,
        classifications=sample.classifications,
        bounding_boxes=sample.bounding_boxes,
        poses=sample.poses,
        cuboids=sample.cuboids,
        projected_cuboids=sample.projected_cuboids,
        img=img,
        seg=seg,
        depth=depth,
    )

    return transformed_sample


def run_train_epoch(epoch_i: int, model: YoloPose, optimizer: torch.optim.Optimizer, data_loader: DataLoader, device: torch.device):
    model.train()

    for batch_i, batch in enumerate(data_loader):
        print(f"train epoch {epoch_i}, batch {batch_i}")

        optimizer.zero_grad()

        img = batch.img.to(device)
        truth = (
            batch.valid.to(device),
            batch.classifications.to(device),
            batch.bounding_boxes.to(device),
            batch.seg.to(device),
            None,
            None,
        )

        prediction = model.forward(img)

        total_loss, (classification_loss, box_loss) = loss(prediction, truth, config)

        print(f"total loss: {float(total_loss)}")
        print(f"classification loss: {float(classification_loss)}, box loss: {float(box_loss)}")

        total_loss.backward()

        optimizer.step()

def run_validation_epoch(epoch_i: int, model: YoloPose, data_loader: DataLoader, device: torch.device):
    model.eval()

    for batch_i, batch in enumerate(data_loader):
        print(f"val epoch {epoch_i}, batch {batch_i}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YoloPose(config).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    trainval_dataset = FallingThingsDataset(
        falling_things_root,
        FallingThingsVariant.MIXED,
        trainval_environments,
        None,
        transform_sample
    )

    train_size = int(train_split * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size
    train_dataset, val_dataset = random_split(trainval_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_samples,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_samples,
    )

    for epoch_i in range(n_epochs):
        run_train_epoch(epoch_i, model, optimizer, train_dataloader, device)

        run_validation_epoch(epoch_i, model, val_dataloader, device)

        if epoch_i % 10 == 0:
            # Save shit
            pass



if __name__ == "__main__":
    main()
