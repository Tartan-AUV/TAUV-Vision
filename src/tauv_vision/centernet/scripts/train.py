import torch
from math import pi
import pathlib
from torch.utils.data import DataLoader, ConcatDataset
import wandb
import albumentations as A

from tauv_vision.centernet.model.centernet import Centernet, initialize_weights
from tauv_vision.centernet.model.backbones.dla import DLABackbone
from tauv_vision.centernet.model.loss import loss
from tauv_vision.centernet.model.config import ObjectConfig, ObjectConfigSet, AngleConfig, ModelConfig, TrainConfig
from tauv_vision.datasets.load.pose_dataset import PoseDataset, PoseSample, Split

torch.autograd.set_detect_anomaly(True)

model_config = ModelConfig(
    in_h=360,
    in_w=640,
    backbone_heights=[2, 2, 2, 2, 2, 2],
    backbone_channels=[128, 128, 128, 128, 128, 128, 128],
    downsamples=2,
    angle_bin_overlap=pi / 3,
)

train_config = TrainConfig(
    lr=1e-3,
    heatmap_focal_loss_a=2,
    heatmap_focal_loss_b=4,
    heatmap_sigma_factor=0.1,
    batch_size=12,
    n_batches=1000,
    n_epochs=10000,
    loss_lambda_size=1,
    loss_lambda_offset=0.0,
    loss_lambda_angle=0.0,
    loss_lambda_depth=0.0,
    n_workers=4,
    weight_save_interval=1,
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
results_root = pathlib.Path("~/Documents/centernet_runs").expanduser()


# From https://stackoverflow.com/questions/47714643/pytorch-data-loader-multiple-iterations

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def run_train_epoch(epoch_i: int, centernet: Centernet, optimizer, data_loader, train_config, device):
    centernet.train()


    for batch_i, batch in enumerate(data_loader):
        if batch_i >= train_config.n_batches:
            break

        print(f"train epoch {epoch_i}, batch {batch_i}")

        optimizer.zero_grad()

        batch = batch.to(device)

        img = batch.img

        prediction = centernet(img)

        losses = loss(prediction, batch, model_config, train_config, object_config)

        total_loss = losses.total

        total_loss.backward()

        optimizer.step()

        wandb.log({"train_total_loss": losses.total})
        wandb.log({"train_heatmap_loss": losses.heatmap})
        wandb.log({"train_size_loss": losses.size})
        wandb.log({"train_offset_loss": losses.offset})
        wandb.log({"train_roll_loss": losses.roll})
        wandb.log({"train_pitch_loss": losses.pitch})
        wandb.log({"train_yaw_loss": losses.yaw})
        wandb.log({"train_depth_loss": losses.depth})

        wandb.log({"train_avg_size_error": losses.avg_size_error})
        wandb.log({"train_max_size_error": losses.max_size_error})


def run_validation_epoch(epoch_i, centernet, data_loader, device):
    centernet.eval()

    avg_losses = torch.zeros(8, dtype=torch.float32)
    n_batch = torch.zeros(1, dtype=torch.float)

    for batch_i, batch in enumerate(data_loader):
        print(f"val epoch {epoch_i}, batch {batch_i}")

        with torch.no_grad():
            batch = batch.to(device)

            img = batch.img

            prediction = centernet(img)

            losses = loss(prediction, batch, model_config, train_config, object_config)

            wandb.log({"val_total_loss": losses.total})
            wandb.log({"val_heatmap_loss": losses.heatmap})
            wandb.log({"val_size_loss": losses.size})
            wandb.log({"val_offset_loss": losses.offset})
            wandb.log({"val_roll_loss": losses.roll})
            wandb.log({"val_pitch_loss": losses.pitch})
            wandb.log({"val_yaw_loss": losses.yaw})
            wandb.log({"val_depth_loss": losses.depth})

            avg_losses += torch.Tensor((
                losses.total.cpu(),
                losses.heatmap.cpu(),
                losses.size.cpu(),
                losses.offset.cpu(),
                losses.roll.cpu(),
                losses.pitch.cpu(),
                losses.yaw.cpu(),
                losses.depth.cpu(),
            ))
            n_batch += 1

    avg_losses /= n_batch

    wandb.log({"val_avg_total_loss": avg_losses[0]})
    wandb.log({"val_avg_heatmap_loss": avg_losses[1]})
    wandb.log({"val_avg_size_loss": avg_losses[2]})
    wandb.log({"val_avg_offset_loss": avg_losses[3]})
    wandb.log({"val_avg_roll_loss": avg_losses[4]})
    wandb.log({"val_avg_pitch_loss": avg_losses[5]})
    wandb.log({"val_avg_yaw_loss": avg_losses[6]})
    wandb.log({"val_avg_depth_loss": avg_losses[7]})


def main():
    for checkpoint in results_root.iterdir():
        checkpoint.unlink()

    wandb.init(
        project="centernet",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"running on {device}")

    train_transform = A.Compose(
        [
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.1,
                always_apply=True,
            ),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.3, 0.3, 0.3), always_apply=True)
        ]
    )

    val_transform = A.Compose(
        [
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.3, 0.3, 0.3), always_apply=True)
        ]
    )


    dla_backbone = DLABackbone(model_config.backbone_heights, model_config.backbone_channels, model_config.downsamples)
    centernet = Centernet(dla_backbone, object_config).to(device)

    centernet.train()
    initialize_weights(centernet, [])

    optimizer = torch.optim.Adam(centernet.parameters(), lr=1e-3)

    train_datasets = [
        PoseDataset(dataset_root, Split.TRAIN, object_config.label_id_to_index, train_transform)
        for dataset_root in train_dataset_roots
    ]
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = PoseDataset(val_dataset_root, Split.VAL, object_config.label_id_to_index, val_transform)

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

    train_dataloader_cycle = iter(cycle(train_dataloader))

    for epoch_i in range(train_config.n_epochs):
        if epoch_i % train_config.weight_save_interval == 0:
            save_path = results_root / f"{epoch_i}.pt"
            torch.save(centernet.state_dict(), save_path)
            artifact = wandb.Artifact('model', type='model')
            artifact.add_dir(results_root)
            wandb.log_artifact(artifact)

        run_train_epoch(epoch_i, centernet, optimizer, train_dataloader_cycle, train_config, device)

        run_validation_epoch(epoch_i, centernet, val_dataloader, device)

    save_path = results_root / f"{epoch_i}.pt"
    torch.save(centernet.state_dict(), save_path)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_dir(results_root)
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()