from typing import Optional, Dict

from dataclasses import dataclass


@dataclass
class ModelConfig:
    backbone_heights: [int]
    backbone_channels: [int]

    in_h: int
    in_w: int

    downsample_ratio: int

    angle_bin_overlap: float

    @property
    def out_h(self) -> int:
        return self.in_h // self.downsample_ratio

    @property
    def out_w(self) -> int:
        return self.in_w // self.downsample_ratio


@dataclass
class TrainConfig:
    lr: float

    batch_size: int
    n_batches: int
    n_epochs: int

    n_workers: int

    heatmap_focal_loss_a: float
    heatmap_focal_loss_b: float
    heatmap_sigma_factor: float

    loss_lambda_size: float
    loss_lambda_offset: float
    loss_lambda_angle: float
    loss_lambda_depth: float


@dataclass
class AngleConfig:
    train: bool
    modulo: Optional[float]


@dataclass
class ObjectConfig:
    id: str

    yaw: AngleConfig
    pitch: AngleConfig
    roll: AngleConfig

    train_depth: bool


class ObjectConfigSet:

    def __init__(self, configs: [ObjectConfig]):
        self.configs: [ObjectConfig] = configs

    @property
    def train_yaw(self) -> bool:
        return any([config.yaw.train for config in self.configs])

    @property
    def train_pitch(self) -> bool:
        return any([config.pitch.train for config in self.configs])

    @property
    def train_roll(self) -> bool:
        return any([config.roll.train for config in self.configs])

    @property
    def train_depth(self) -> bool:
        return any([config.train_depth for config in self.configs])

    @property
    def n_labels(self) -> int:
        return len(self.configs)

    @property
    def label_id_to_index(self) -> Dict[str, int]:
        return {config.id: i for (i, config) in enumerate(self.configs)}
