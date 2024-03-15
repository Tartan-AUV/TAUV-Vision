import torch
from math import pi
import kornia.augmentation as A
import pathlib
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import wandb

from tauv_vision.centernet.model.centernet import Centernet, initialize_weights
from tauv_vision.centernet.model.backbones.dla import DLABackbone
from tauv_vision.centernet.model.loss import gaussian_splat, loss
from tauv_vision.centernet.model.config import ObjectConfig, ObjectConfigSet, AngleConfig, ModelConfig, TrainConfig
from tauv_vision.datasets.load.pose_dataset import PoseDataset, PoseSample, Split

torch.autograd.set_detect_anomaly(True)

model_config = ModelConfig(
    in_h=360,
    in_w=640,
    backbone_heights=[2, 2, 2, 2, 2, 2],
    backbone_channels=[32, 32, 32, 32, 32, 32, 32],
    downsample_ratio=2,
    angle_bin_overlap=pi / 3,
)

train_config = TrainConfig(
    lr=1e-3,
    heatmap_focal_loss_a=2,
    heatmap_focal_loss_b=4,
    heatmap_sigma_factor=0.1,
    batch_size=16,
    n_batches=1000,
    n_epochs=10000,
    n_workers=4,
    loss_lambda_size=0,
    loss_lambda_offset=0, # Not set up to train properly
    loss_lambda_angle=0,
    loss_lambda_depth=0,
)


object_config = ObjectConfigSet(
    configs=[
        ObjectConfig(
            id="torpedo_22_circle",
            yaw=AngleConfig(
                train=True,
                modulo=2 * pi,
            ),
            pitch=AngleConfig(
                train=True,
                modulo=2 * pi,
            ),
            roll=AngleConfig(
                train=False,
                modulo=None,
            ),
            train_depth=True,
        ),
        ObjectConfig(
            id="torpedo_22_trapezoid",
            yaw=AngleConfig(
                train=True,
                modulo=2 * pi,
            ),
            pitch=AngleConfig(
                train=True,
                modulo=2 * pi,
            ),
            roll=AngleConfig(
                train=False,
                modulo=None,
            ),
            train_depth=True,
        ),
    ]
)

train_dataset_roots = [
    pathlib.Path("~/Documents/TAUV-Datasets/bring-great-service").expanduser(),
]
val_dataset_root = pathlib.Path("~/Documents/TAUV-Datasets/bring-great-service").expanduser()
results_root = pathlib.Path("~/Documents/yolact_runs").expanduser()


# From https://stackoverflow.com/questions/47714643/pytorch-data-loader-multiple-iterations

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def run_train_epoch(epoch_i: int, centernet: Centernet, optimizer, data_loader, train_config, device):
    centernet.train()

    data_loader_cycle = iter(cycle(data_loader))

    for batch_i, batch in enumerate(data_loader_cycle):
        if batch_i >= train_config.n_batches:
            break

        print(f"train epoch {epoch_i}, batch {batch_i}")

        optimizer.zero_grad()

        batch = batch.to(device)

        img = batch.img

        prediction = centernet(img)

        total_loss = loss(prediction, batch, model_config, train_config, object_config)

        total_loss.backward()

        for name, param in centernet.named_parameters():
            if param.grad is None:
                continue

            if torch.any(torch.isnan(param.grad)):
                print(f"{name} is nan")

        optimizer.step()

        if batch_i % 10 == 0:
            plt.imshow(F.sigmoid(prediction.heatmap[0, 0]).detach().cpu())
            plt.colorbar()
            plt.show()

        print(f"total loss: {float(total_loss)}")

        wandb.log({"train_total_loss": total_loss})

def main():
    wandb.init(
        project="centernet",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"running on {device}")

    dla_backbone = DLABackbone(model_config.backbone_heights, model_config.backbone_channels)
    centernet = Centernet(dla_backbone, object_config).to(device)

    centernet.train()
    initialize_weights(centernet, [])

    optimizer = torch.optim.Adam(centernet.parameters(), lr=1e-3)

    color_transforms = A.AugmentationSequential(
        A.ColorJitter(hue=0.0, saturation=0.0, brightness=0.0),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.3, 0.3, 0.3)),
    )

    train_datasets = [
        PoseDataset(dataset_root, Split.TRAIN, object_config.label_id_to_index)
        for dataset_root in train_dataset_roots
    ]
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = PoseDataset(val_dataset_root, Split.VAL, object_config.label_id_to_index)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        collate_fn=PoseSample.collate,
        shuffle=True,
        num_workers=train_config.n_workers,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        collate_fn=PoseSample.collate,
        shuffle=True,
        num_workers=train_config.n_workers,
    )

    for epoch_i in range(train_config.n_epochs):

        run_train_epoch(epoch_i, centernet, optimizer, train_dataloader, train_config, device)


if __name__ == "__main__":
    main()