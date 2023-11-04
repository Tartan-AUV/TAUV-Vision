import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from typing import List, Tuple, Optional
from pathlib import Path
import pathlib
from dataclasses import asdict
import wandb
import matplotlib.pyplot as plt

from yolact.model.config import Config
from yolact.model.loss import loss
from yolact.model.model import Yolact
from yolact.model.weights import initialize_weights
from yolact.model.boxes import box_decode
from yolact.model.masks import assemble_mask
from datasets.segmentation_dataset.segmentation_dataset import SegmentationDataset, SegmentationSample, SegmentationDatasetSet
from yolact.utils.plot import save_plot, plot_prototype, plot_mask, plot_detection

torch.autograd.set_detect_anomaly(True)


config = Config(
    in_w=960,
    in_h=480,
    feature_depth=256,
    n_classes=3,
    n_prototype_masks=32,
    n_masknet_layers_pre_upsample=1,
    n_masknet_layers_post_upsample=1,
    n_prediction_head_layers=1,
    n_fpn_downsample_layers=2,
    anchor_scales=(24, 48, 96, 192, 384),
    anchor_aspect_ratios=(1 / 2, 1, 2),
    iou_pos_threshold=0.5,
    iou_neg_threshold=0.4,
    negative_example_ratio=3,
)


lr = 1e-4
momentum = 0.9
weight_decay = 0
n_epochs = 500
# n_warmup_epochs = 10
n_warmup_epochs = 0
weight_save_interval = 10
train_split = 0.9
batch_size = 8

hue_jitter = 0.2
saturation_jitter = 0.2
brightness_jitter = 0.2

img_mean = (0.485, 0.456, 0.406)
img_stddev = (0.229, 0.224, 0.225)

dataset_root = pathlib.Path("~/Documents/torpedo_target_2").expanduser()
results_root = pathlib.Path("~/Documents/yolact_runs").expanduser()


def collate_samples(samples: List[SegmentationSample]) -> SegmentationSample:
    n_detections = [sample.valid.size(0) for sample in samples]
    max_n_detections = max(n_detections)

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
    img = torch.stack([sample.img for sample in samples], dim=0)
    seg = torch.stack([sample.seg for sample in samples], dim=0)

    sample = SegmentationSample(
        img=img,
        seg=seg,
        valid=valid,
        classifications=classifications,
        bounding_boxes=bounding_boxes,
    )

    return sample


def transform_sample(sample: SegmentationSample) -> SegmentationSample:
    color_transforms = T.Compose([
        T.ColorJitter(hue=hue_jitter, saturation=saturation_jitter, brightness=brightness_jitter),
        T.GaussianBlur(kernel_size=7),
        T.Normalize(mean=img_mean, std=img_stddev),
    ])

    img = color_transforms(sample.img)

    transformed_sample = SegmentationSample(
        img=img,
        seg=sample.seg,
        valid=sample.valid,
        classifications=sample.classifications,
        bounding_boxes=sample.bounding_boxes,
    )

    return transformed_sample


def prepare_batch(batch: SegmentationSample, device: torch.device) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    img = batch.img.to(device)

    truth = (
        batch.valid.to(device),
        batch.classifications.to(device),
        batch.bounding_boxes.to(device),
        batch.seg.to(device),
    )

    return img, truth


def run_train_epoch(epoch_i: int, model: Yolact, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, data_loader: DataLoader, device: torch.device):
    model.train()

    for batch_i, batch in enumerate(data_loader):
        print(f"train epoch {epoch_i}, batch {batch_i}")
        batch = transform_sample(batch)

        img, truth = prepare_batch(batch, device=device)

        optimizer.zero_grad()

        prediction = model(img)

        total_loss, (classification_loss, box_loss, mask_loss) = loss(prediction, truth, config)

        print(f"total loss: {float(total_loss)}")
        print(f"classification loss: {float(classification_loss)}, box loss: {float(box_loss)}, mask loss: {float(mask_loss)}")

        total_loss.backward()

        optimizer.step()
        scheduler.step(epoch_i)

        wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})

        wandb.log({"train_total_loss": total_loss})
        wandb.log({"train_classification_loss": classification_loss})
        wandb.log({"train_box_loss": box_loss})
        wandb.log({"train_mask_loss": mask_loss})


def plot_validation_batch(epoch_i: int, batch_i: int, img: torch.Tensor, prediction: Tuple[torch.Tensor, ...], truth: Tuple[torch.Tensor, ...], save_dir: Optional[pathlib.Path] = None):
    classification, box_encoding, mask_coeff, anchor, mask_prototype = prediction
    truth_valid, truth_classification, truth_box, truth_seg_map = truth

    n_batch = img.size(0)

    for sample_i in range(n_batch):
        classification_max = torch.argmax(classification[sample_i], dim=-1).squeeze(0)
        detections = classification_max.nonzero().squeeze(-1)[:100]

        prototype_fig = plot_prototype(mask_prototype[sample_i])
        prototype_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Prototypes")
        prototype_fig.set_size_inches(16, 10)
        wandb.log({f"val_prototype_{batch_i}_{sample_i}": prototype_fig})
        save_plot(prototype_fig, save_dir, f"val_prototype_{epoch_i}_{batch_i}_{sample_i}")

        if len(detections) > 0:
            mask = assemble_mask(mask_prototype[sample_i], mask_coeff[sample_i, detections])
            mask_fig = plot_mask(None, mask)
            mask_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Masks")
            mask_fig.set_size_inches(16, 10)
            wandb.log({f"val_mask_{batch_i}_{sample_i}": mask_fig})
            save_plot(mask_fig, save_dir, f"val_mask_{epoch_i}_{batch_i}_{sample_i}")

            mask_overlay_fig = plot_mask(img[sample_i], mask)
            mask_overlay_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Mask Overlays")
            mask_overlay_fig.set_size_inches(16, 10)
            wandb.log({f"val_mask_overlay_{batch_i}_{sample_i}": mask_overlay_fig})
            save_plot(mask_overlay_fig, save_dir, f"val_mask_overlay_{epoch_i}_{batch_i}_{sample_i}")

            box = box_decode(box_encoding, anchor)
            detection_fig = plot_detection(
                img[sample_i],
                classification_max[detections],
                box[sample_i, detections],
                truth_valid[sample_i],
                truth_classification[sample_i],
                truth_box[sample_i],
            )
            detection_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Detections")
            wandb.log({f"val_detection_{batch_i}_{sample_i}": detection_fig})
            save_plot(detection_fig, save_dir, f"val_detection_{epoch_i}_{batch_i}_{sample_i}")


def run_validation_epoch(epoch_i: int, model: Yolact, data_loader: DataLoader, device: torch.device):
    model.eval()

    avg_losses = torch.zeros(4, dtype=torch.float)
    n_batch = torch.zeros(1, dtype=torch.float)

    for batch_i, batch in enumerate(data_loader):
        print(f"val epoch {epoch_i}, batch {batch_i}")

        with torch.no_grad():
            img, truth = prepare_batch(batch, device=device)

            prediction = model.forward(img)

            if batch_i == 0:
                plot_validation_batch(epoch_i, batch_i, img, prediction, truth, save_dir=pathlib.Path("./out"))

            total_loss, (classification_loss, box_loss, mask_loss) = loss(prediction, truth, config)

            wandb.log({"val_total_loss": total_loss})
            wandb.log({"val_classification_loss": classification_loss})
            wandb.log({"val_box_loss": box_loss})
            wandb.log({"val_mask_loss": mask_loss})

        print(f"total loss: {float(total_loss)}")
        print(f"classification loss: {float(classification_loss)}, box loss: {float(box_loss)}, mask loss: {float(mask_loss)}")

        avg_losses += torch.Tensor((total_loss, classification_loss, box_loss, mask_loss))
        n_batch += 1

    avg_losses /= n_batch
    avg_total_loss, avg_classification_loss, avg_box_loss, avg_mask_loss = avg_losses

    print("validation averages:")
    print(f"total loss: {float(avg_total_loss)}")
    print(f"classification loss: {float(avg_classification_loss)}, box loss: {float(avg_box_loss)}, mask loss: {float(avg_mask_loss)}")

    wandb.log({"val_avg_total_loss": avg_total_loss})
    wandb.log({"val_avg_classification_loss": avg_classification_loss})
    wandb.log({"val_avg_box_loss": avg_box_loss})
    wandb.log({"val_avg_mask_loss": avg_mask_loss})


def main():
    save_dir = Path(results_root).expanduser()
    for checkpoint in save_dir.iterdir():
        checkpoint.unlink()

    wandb.init(
        project="yolact",
        config={
            **asdict(config),
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "n_epochs": n_epochs,
            "n_warmup_epochs": n_warmup_epochs,
            "weight_save_interval": weight_save_interval,
            "train_split": train_split,
            "batch_size": batch_size,
            "hue_jitter": hue_jitter,
            "saturation_jitter": saturation_jitter,
            "brightness_jitter": brightness_jitter,
            "img_mean": img_mean,
            "img_stddev": img_stddev,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Yolact(config).to(device)
    initialize_weights(model, [model._backbone])

    wandb.watch(model, log="all", log_freq=1)

    def lr_lambda(epoch):
        if epoch < n_warmup_epochs:
            return (epoch + 1) / n_warmup_epochs
        else:
            return 1

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_dataset = SegmentationDataset(dataset_root, SegmentationDatasetSet.TRAIN)
    val_dataset = SegmentationDataset(dataset_root, SegmentationDatasetSet.VALIDATION)

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
        shuffle=False,
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

        run_validation_epoch(epoch_i, model, val_dataloader, device)

    save_path = save_dir / f"{epoch_i}.pt"
    torch.save(model.state_dict(), save_path)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_dir(save_dir)
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()
