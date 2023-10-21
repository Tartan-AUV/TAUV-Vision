import torch
import os
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import List
from pathlib import Path
from dataclasses import asdict
import wandb

from yolo_pose.model.config import Config
from yolo_pose.model.loss import loss
from yolo_pose.model.model import YoloPose
from yolo_pose.model.weights import initialize_weights
from yolo_pose.falling_things_dataset.falling_things_dataset import FallingThingsDataset, FallingThingsVariant, FallingThingsEnvironment, FallingThingsSample, FallingThingsObject

torch.autograd.set_detect_anomaly(True)


config = Config(
    in_w=960,
    in_h=480,
    feature_depth=256,
    n_classes=23,
    n_prototype_masks=32,
    n_point_maps=32,
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

lr = 1e-3
momentum = 0.9
weight_decay = 0
n_epochs = 500
weight_save_interval = 10
train_split = 0.9
batch_size = 12

hue_jitter = 0.2
saturation_jitter = 0.2
brightness_jitter = 0.2

img_mean = (0.485, 0.456, 0.406)
img_stddev = (0.229, 0.224, 0.225)

trainval_environments = [
    FallingThingsEnvironment.Kitchen0,
    FallingThingsEnvironment.Kitchen1,
    FallingThingsEnvironment.Kitchen2,
    FallingThingsEnvironment.Kitchen3,
    FallingThingsEnvironment.Kitchen4,
]

trainval_objects = [
    FallingThingsObject.CrackerBox,
]

falling_things_root = "~/Documents/falling_things/fat"
results_root = "~/Documents/yolo_pose_runs"

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
    camera_pose = torch.stack([sample.camera_pose for sample in samples], dim=0)
    img = torch.stack([sample.img for sample in samples], dim=0)
    seg_map = torch.stack([sample.seg_map for sample in samples], dim=0)
    depth_map = torch.stack([sample.depth_map for sample in samples], dim=0)
    position_map = torch.stack([sample.position_map for sample in samples], dim=0)

    sample = FallingThingsSample(
        intrinsics=intrinsics,
        valid=valid,
        classifications=classifications,
        bounding_boxes=bounding_boxes,
        camera_pose=camera_pose,
        poses=poses,
        cuboids=cuboids,
        projected_cuboids=projected_cuboids,
        img=img,
        seg_map=seg_map,
        depth_map=depth_map,
        position_map=position_map,
    )

    return sample


def transform_sample(sample: FallingThingsSample) -> FallingThingsSample:
    color_transforms = transforms.Compose([
        transforms.ColorJitter(hue=hue_jitter, saturation=saturation_jitter, brightness=brightness_jitter),
        transforms.Normalize(mean=img_mean, std=img_stddev),
    ])

    img = color_transforms(sample.img)
    seg_map = sample.seg_map
    depth_map = sample.depth_map
    position_map = sample.position_map

    transformed_sample = FallingThingsSample(
        intrinsics=sample.intrinsics,
        valid=sample.valid,
        classifications=sample.classifications,
        bounding_boxes=sample.bounding_boxes,
        camera_pose=sample.camera_pose,
        poses=sample.poses,
        cuboids=sample.cuboids,
        projected_cuboids=sample.projected_cuboids,
        img=img,
        seg_map=seg_map,
        depth_map=depth_map,
        position_map=position_map,
    )

    return transformed_sample


def run_train_epoch(epoch_i: int, model: YoloPose, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, data_loader: DataLoader, device: torch.device):
    model.train()

    for batch_i, batch in enumerate(data_loader):
        print(f"train epoch {epoch_i}, batch {batch_i}")

        optimizer.zero_grad()

        img = batch.img.to(device)
        truth = (
            batch.valid.to(device),
            batch.classifications.to(device),
            batch.bounding_boxes.to(device),
            batch.seg_map.to(device),
            batch.position_map.to(device),
        )

        prediction = model(img)

        total_loss, (classification_loss, box_loss, mask_loss, point_loss) = loss(prediction, truth, config)

        print(f"total loss: {float(total_loss)}")
        print(f"classification loss: {float(classification_loss)}, box loss: {float(box_loss)}, mask loss: {float(mask_loss)}, point loss: {float(point_loss)}")

        total_loss.backward()

        optimizer.step()
        scheduler.step(epoch_i)

        wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})

        wandb.log({"train_total_loss": total_loss})
        wandb.log({"train_classification_loss": classification_loss})
        wandb.log({"train_box_loss": box_loss})
        wandb.log({"train_mask_loss": mask_loss})
        wandb.log({"train_point_loss": point_loss})


def run_validation_epoch(epoch_i: int, model: YoloPose, data_loader: DataLoader, device: torch.device):
    model.eval()

    avg_losses = torch.zeros(5, dtype=torch.float)
    n_batch = torch.zeros(1, dtype=torch.float)

    for batch_i, batch in enumerate(data_loader):
        print(f"val epoch {epoch_i}, batch {batch_i}")

        with torch.no_grad():
            img = batch.img.to(device)
            truth = (
                batch.valid.to(device),
                batch.classifications.to(device),
                batch.bounding_boxes.to(device),
                batch.seg_map.to(device),
                batch.position_map.to(device),
            )

            prediction = model.forward(img)

            total_loss, (classification_loss, box_loss, mask_loss, point_loss) = loss(prediction, truth, config)

            wandb.log({"val_total_loss": total_loss})
            wandb.log({"val_classification_loss": classification_loss})
            wandb.log({"val_box_loss": box_loss})
            wandb.log({"val_mask_loss": mask_loss})
            wandb.log({"val_point_loss": point_loss})

        print(f"total loss: {float(total_loss)}")
        print(f"classification loss: {float(classification_loss)}, box loss: {float(box_loss)}, mask loss: {float(mask_loss)}, point loss: {float(point_loss)}")

        avg_losses += torch.Tensor((total_loss, classification_loss, box_loss, mask_loss, point_loss))
        n_batch += 1

    avg_losses /= n_batch
    avg_total_loss, avg_classification_loss, avg_box_loss, avg_mask_loss, avg_point_loss = avg_losses

    print("validation averages:")
    print(f"total loss: {float(avg_total_loss)}")
    print(f"classification loss: {float(avg_classification_loss)}, box loss: {float(avg_box_loss)}, mask loss: {float(avg_mask_loss)}, point loss: {float(avg_point_loss)}")

    wandb.log({"val_avg_total_loss": avg_total_loss})
    wandb.log({"val_avg_classification_loss": avg_classification_loss})
    wandb.log({"val_avg_box_loss": avg_box_loss})
    wandb.log({"val_avg_mask_loss": avg_mask_loss})
    wandb.log({"val_avg_point_loss": avg_point_loss})


def main():
    save_dir = Path(results_root).expanduser()
    for checkpoint in save_dir.iterdir():
        checkpoint.unlink()

    wandb.init(
        project="yolo_pose",
        config={
            **asdict(config),
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "n_epochs": n_epochs,
            "weight_save_interval": weight_save_interval,
            "train_split": train_split,
            "batch_size": batch_size,
            "hue_jitter": hue_jitter,
            "saturation_jitter": saturation_jitter,
            "brightness_jitter": brightness_jitter,
            "img_mean": img_mean,
            "img_stddev": img_stddev,
            "trainval_environments": trainval_environments,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model = YoloPose(config).to(device)
    initialize_weights(model, [model._backbone])

    wandb.watch(model, log="all", log_freq=1)

    def lr_lambda(epoch):
        if epoch < 50:
            return (epoch + 1) / 50
        else:
            return 1

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trainval_dataset = FallingThingsDataset(
        falling_things_root,
        FallingThingsVariant.SINGLE,
        trainval_environments,
        trainval_objects,
        transform_sample
    )

    # train_size = int(train_split * len(trainval_dataset))
    train_size = batch_size
    val_size = len(trainval_dataset) - train_size
    train_dataset, val_dataset = random_split(trainval_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_samples,
        shuffle=True,
        num_workers=4,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_samples,
        shuffle=True,
        num_workers=4,
    )

    for epoch_i in range(n_epochs):
        if epoch_i % weight_save_interval == 0:
            save_path = save_dir / f"{epoch_i}.pt"
            torch.save(model.state_dict(), save_path)
            artifact = wandb.Artifact('model', type='model')
            artifact.add_dir(save_dir)
            wandb.log_artifact(artifact)

        run_train_epoch(epoch_i, model, optimizer, scheduler, train_dataloader, device)

        # run_validation_epoch(epoch_i, model, val_dataloader, device)

    save_path = save_dir / f"{epoch_i}.pt"
    torch.save(model.state_dict(), save_path)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_dir(save_dir)
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()
