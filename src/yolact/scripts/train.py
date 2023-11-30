import cv2
import torch
import glob
from torch.utils.data import DataLoader, random_split, ConcatDataset
import itertools
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from typing import List, Tuple, Optional
from pathlib import Path
import pathlib
from dataclasses import asdict
import wandb
import matplotlib.pyplot as plt
import albumentations as A

from yolact.model.config import Config
from yolact.model.loss import loss
from yolact.model.model import Yolact
from yolact.model.weights import initialize_weights
from yolact.model.boxes import box_decode, box_xy_swap, box_to_corners, corners_to_box
from yolact.model.masks import assemble_mask
from datasets.segmentation_dataset.segmentation_dataset import SegmentationDataset, SegmentationSample, SegmentationDatasetSet
from yolact.utils.plot import save_plot, plot_prototype, plot_mask, plot_detection
from yolact.utils.overlay import Overlay

# torch.autograd.set_detect_anomaly(True)


config = Config(
    # in_w=1280,
    # in_h=720,
    in_w=640,
    in_h=360,
    feature_depth=64,
    n_classes=2,
    n_prototype_masks=16,
    n_masknet_layers_pre_upsample=1,
    n_masknet_layers_post_upsample=1,
    n_prediction_head_layers=1,
    n_classification_layers=0,
    n_box_layers=0,
    n_mask_layers=0,
    n_fpn_downsample_layers=2,
    anchor_scales=(24, 48, 96, 192, 384), # TODO: Check this. Like really check it.
    # anchor_aspect_ratios=(1 / 2, 1, 2),
    anchor_aspect_ratios=(1,),
    iou_pos_threshold=0.4,
    iou_neg_threshold=0.3,
    negative_example_ratio=3,
)


lr = 1e-3
# lr = 1e-4
momentum = 0.9
weight_decay = 0
n_epochs = 200
n_warmup_epochs = 0
# n_warmup_epochs = 10
weight_save_interval = 1
batch_size = 96
epoch_n_batches = 100
grad_max_norm = 1e0

img_mean = (0.485, 0.456, 0.406)
img_stddev = (0.229, 0.224, 0.225)

train_dataset_roots = [
    # pathlib.Path("~/Documents/2023-11-05").expanduser(),
    # pathlib.Path("~/Documents/torpedo_22_2_small").expanduser(),
    pathlib.Path("~/Documents/torpedo_22_1_small").expanduser(),
]
val_dataset_root = pathlib.Path("~/Documents/2023-11-05").expanduser()
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
    img_valid = torch.stack([sample.img_valid for sample in samples], dim=0)

    corners = box_to_corners(bounding_boxes)
    corners = torch.clamp(corners, min=0, max=1)
    bounding_boxes = corners_to_box(corners)

    sample = SegmentationSample(
        img=img,
        seg=seg,
        valid=valid,
        classifications=classifications,
        bounding_boxes=bounding_boxes,
        img_valid=img_valid,
    )

    return sample


def prepare_batch(batch: SegmentationSample, device: torch.device) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    img = batch.img.to(device)

    truth = (
        batch.valid.to(device),
        batch.classifications.to(device),
        batch.bounding_boxes.to(device),
        batch.seg.to(device),
        batch.img_valid.to(device),
    )

    return img, truth


def plot_train_batch(epoch_i: int, batch_i: int, img: torch.Tensor, prediction: Tuple[torch.Tensor, ...], truth: Tuple[torch.Tensor, ...], save_dir: Optional[pathlib.Path] = None):
    classification, box_encoding, mask_coeff, anchor, mask_prototype = prediction
    truth_valid, truth_classification, truth_box, truth_seg_map, truth_img_valid = truth

    n_batch = img.size(0)

    for sample_i in [0]:
        classification_max = torch.argmax(classification[sample_i], dim=-1).squeeze(0)
        detections = classification_max.nonzero().squeeze(-1)[:100]

        prototype_fig = plot_prototype(mask_prototype[sample_i])
        prototype_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Prototypes")
        prototype_fig.set_size_inches(16, 10)
        wandb.log({f"train_prototype_{batch_i}_{sample_i}": prototype_fig})
        save_plot(prototype_fig, save_dir, f"train_prototype_{epoch_i}_{batch_i}_{sample_i}")

        if len(detections) > 0:
            mask = assemble_mask(mask_prototype[sample_i], mask_coeff[sample_i, detections], box=None)
            mask_fig = plot_mask(None, mask)
            mask_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Masks")
            mask_fig.set_size_inches(16, 10)
            wandb.log({f"train_mask_{batch_i}_{sample_i}": mask_fig})
            save_plot(mask_fig, save_dir, f"train_mask_{epoch_i}_{batch_i}_{sample_i}")

            mask_overlay_fig = plot_mask(img[sample_i], mask)
            mask_overlay_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Mask Overlays")
            mask_overlay_fig.set_size_inches(16, 10)
            wandb.log({f"train_mask_overlay_{batch_i}_{sample_i}": mask_overlay_fig})
            save_plot(mask_overlay_fig, save_dir, f"train_mask_overlay_{epoch_i}_{batch_i}_{sample_i}")

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
            wandb.log({f"train_detection_{batch_i}_{sample_i}": detection_fig})
            save_plot(detection_fig, save_dir, f"train_detection_{epoch_i}_{batch_i}_{sample_i}")

            plt.close("all")

        plt.close("all")


# From https://stackoverflow.com/questions/47714643/pytorch-data-loader-multiple-iterations
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def run_train_epoch(epoch_i: int, model: Yolact, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, data_loader: DataLoader, device: torch.device):
    model.train()

    data_loader_cycle = iter(cycle(data_loader))

    for batch_i, batch in enumerate(data_loader_cycle):
        if batch_i >= epoch_n_batches:
            break

        print(f"train epoch {epoch_i}, batch {batch_i}")
        img, truth = prepare_batch(batch, device=device)

        optimizer.zero_grad()

        prediction = model(img)

        if batch_i == epoch_n_batches - 1:
            plot_train_batch(epoch_i, batch_i, img, prediction, truth, save_dir=pathlib.Path("./out"))

        classification, box_encoding, mask_coeff, anchor, mask_prototype = prediction
        n_detections = torch.sum(torch.argmax(classification, dim=-1) > 0)

        total_loss, (classification_loss, box_loss, mask_loss) = loss(prediction, truth, config)

        print(f"total loss: {float(total_loss)}")
        print(f"classification loss: {float(classification_loss)}, box loss: {float(box_loss)}, mask loss: {float(mask_loss)}")

        print(f"n detections: {float(n_detections)}")
        wandb.log({"n_detections": n_detections})

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None)

        optimizer.step()
        scheduler.step(epoch_i)

        wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})

        wandb.log({"train_total_loss": total_loss})
        wandb.log({"train_classification_loss": classification_loss})
        wandb.log({"train_box_loss": box_loss})
        wandb.log({"train_mask_loss": mask_loss})


def plot_validation_batch(epoch_i: int, batch_i: int, img: torch.Tensor, prediction: Tuple[torch.Tensor, ...], truth: Tuple[torch.Tensor, ...], save_dir: Optional[pathlib.Path] = None):
    classification, box_encoding, mask_coeff, anchor, mask_prototype = prediction
    truth_valid, truth_classification, truth_box, truth_seg_map, truth_img_valid = truth

    n_batch = img.size(0)

    for sample_i in range(max(n_batch, 4)):
        classification_max = torch.argmax(classification[sample_i], dim=-1).squeeze(0)
        detections = classification_max.nonzero().squeeze(-1)[:100]

        prototype_fig = plot_prototype(mask_prototype[sample_i])
        prototype_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Prototypes")
        prototype_fig.set_size_inches(16, 10)
        wandb.log({f"val_prototype_{batch_i}_{sample_i}": prototype_fig})
        save_plot(prototype_fig, save_dir, f"val_prototype_{epoch_i}_{batch_i}_{sample_i}")

        if len(detections) > 0:
            mask = assemble_mask(mask_prototype[sample_i], mask_coeff[sample_i, detections], box=None)
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

            plt.close("all")

        plt.close("all")


def run_validation_epoch(epoch_i: int, model: Yolact, data_loader: DataLoader, device: torch.device):
    model.eval()

    avg_losses = torch.zeros(4, dtype=torch.float)
    n_batch = torch.zeros(1, dtype=torch.float)

    for batch_i, batch in enumerate(data_loader):
        print(f"val epoch {epoch_i}, batch {batch_i}")

        with torch.no_grad():
            img, truth = prepare_batch(batch, device=device)

            prediction = model.forward(img)

            if batch_i == 0 and epoch_i > 0:
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
            "batch_size": batch_size,
            "img_mean": img_mean,
            "img_stddev": img_stddev,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Yolact(config).to(device)
    initialize_weights(model, [model._backbone])
    # model.load_state_dict(torch.load("/home/theo/Downloads/14.pt", map_location=device))

    def lr_lambda(epoch):
        if epoch < n_warmup_epochs:
            return (epoch + 1) / n_warmup_epochs
        else:
            return 1

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    reference_image_paths = glob.glob("/home/theo/Documents/2023-11-05/images/*.png")

    train_transform = A.Compose(
        [
            # A.ChannelShuffle(p=0.5),
            A.Resize(height=360, width=640, always_apply=True),
            # Streaks(
            #     shape=(360, 640),
            #     intensity=(128, 255),
            #     p=0.5
            # ),
            # A.ElasticTransform(
            #     border_mode=cv2.BORDER_CONSTANT,
            #     value=0,
            #     mask_value=254,
            #     p=1,
            # ),
            # A.RandomBrightnessContrast(
            #     brightness_limit=0.2,
            #     contrast_limit=(-0.5, 0),
            #     p=1,
            # ),
            # A.HistogramMatching(
            #     reference_images=reference_image_paths,
            # ),
            Overlay(
                paths=glob.glob("/home/theo/Documents/noise_textures/*.png"),
                intensity=(0.5, 1),
                p=1,
            ),
            A.ColorJitter(
                p=1,
            ),
            A.GaussNoise(p=1),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.MotionBlur(p=1),
                A.MedianBlur(blur_limit=3, p=1),
                A.Blur(blur_limit=3, p=1),
            ], p=0.5),
            # A.OneOf([
            #     A.GridDistortion(
            #         border_mode=cv2.BORDER_CONSTANT,
            #         value=0,
            #         mask_value=254,
            #         p=1,
            #     ),
            #     A.OpticalDistortion(
            #         border_mode=cv2.BORDER_CONSTANT,
            #         value=0,
            #         mask_value=254,
            #         p=1,
            #     ),
            # ], p=0.5),
            # A.Affine(
            #     scale=(0.9, 1.1),
            #     keep_ratio=True,
            #     translate_percent=(-0.25, 0.25),
            #     rotate=(-30, 30),
            #     shear=(-10, 10),
            #     mode=cv2.BORDER_CONSTANT,
            #     cval=0,
            #     cval_mask=254,
            #     p=1
            # ),
            A.ShiftScaleRotate(
                shift_limit=(-0.25, 0.25),
                scale_limit=(-0.1, 0.1),
                rotate_limit=(-30, 30),
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=254,
                p=1,
            ),
            A.Perspective(
                scale=(0, 0.1),
                pad_mode=cv2.BORDER_CONSTANT,
                pad_val=0,
                mask_pad_val=254,
                p=1,
            ),
            A.Normalize(mean=img_mean, std=img_stddev, always_apply=True),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["classifications"], min_visibility=0.1),
    )

    val_transform = A.Compose(
        [
            A.Resize(height=360, width=640),
            A.Normalize(mean=img_mean, std=img_stddev),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["classifications"]),
    )

    train_datasets = [
        SegmentationDataset(dataset_root, SegmentationDatasetSet.TRAIN, transform=train_transform)
        for dataset_root in train_dataset_roots
    ]
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = SegmentationDataset(val_dataset_root, SegmentationDatasetSet.VALIDATION, transform=val_transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_samples,
        shuffle=True,
        num_workers=10,
    )

    wandb.watch(model, log="all", log_freq=1000)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_samples,
        shuffle=False,
        num_workers=10,
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
