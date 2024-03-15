from typing import List, Optional
from typing_extensions import Self
from dataclasses import dataclass
from math import pi

from tauv_vision.centernet.model.backbones.dla import DLABackbone
from tauv_vision.centernet.model.config import ObjectConfig, ObjectConfigSet, AngleConfig

import torch
import torch.nn as nn


@dataclass
class Prediction:
	heatmap: torch.Tensor                      # [batch_size, n_labels, out_h, out_w]
	size: torch.Tensor                         # [batch_size, out_h, out_w, 2]
	offset: torch.Tensor                       # [batch_size, out_h, out_w, 2]

	roll_bin: Optional[torch.Tensor]           # [batch_size, out_h, out_w, 4]
	roll_offset: Optional[torch.Tensor]        # [batch_size, out_h, out_w, 4]
	pitch_bin: Optional[torch.Tensor]          # [batch_size, out_h, out_w, 4]
	pitch_offset: Optional[torch.Tensor]       # [batch_size, out_h, out_w, 4]
	yaw_bin: Optional[torch.Tensor]            # [batch_size, out_h, out_w, 4]
	yaw_offset: Optional[torch.Tensor]         # [batch_size, out_h, out_w, 4]

	depth: Optional[torch.Tensor]              # [batch_size, out_h, out_w]


@dataclass
class Truth:
	heatmap: torch.Tensor                      # [batch_size, n_labels, out_h, out_w]
	valid: torch.Tensor                        # [batch_size, n_objects]
	label: torch.Tensor                        # [batch_size, n_objects]
	center: torch.Tensor                       # [batch_size, n_objects, 2]
	size: torch.Tensor                         # [batch_size, n_objects, 2]

	roll: Optional[torch.Tensor]               # [batch_size, n_objects]
	pitch: Optional[torch.Tensor]              # [batch_size, n_objects]
	yaw: Optional[torch.Tensor]                # [batch_size, n_objects]
	depth: Optional[torch.Tensor]              # [batch_size, n_objects]

	def to(self, device: torch.device) -> Self:
		return Truth(
			heatmap=self.heatmap.to(device),
			valid=self.valid.to(device),
			label=self.label.to(device),
			center=self.center.to(device),
			size=self.size.to(device),
			roll=self.roll.to(device) if self.roll is not None else None,
			pitch=self.pitch.to(device) if self.pitch is not None else None,
			yaw=self.yaw.to(device) if self.yaw is not None else None,
			depth=self.depth.to(device) if self.depth is not None else None,
		)


class Centernet(nn.Module):

	def __init__(self, backbone: nn.Module, object_config: ObjectConfigSet):
		super().__init__()

		self.backbone = backbone

		heads = []

		out_channels = get_head_channels(object_config)

		for out_channel in out_channels:
			head = nn.Sequential(
				nn.Conv2d(
					in_channels=backbone.out_channels,
					out_channels=2*backbone.out_channels,
					kernel_size=3,
					padding=1,
				),
				nn.LeakyReLU(),
				nn.Conv2d(
					in_channels=2*backbone.out_channels,
					out_channels=out_channel,
					kernel_size=1,
				),
			)

			heads.append(head)

		self.heads = nn.ModuleList(heads)

		self.object_config = object_config

	def forward(self, img: torch.Tensor) -> Prediction:
		# img is [batch_size, 3, in_h, in_w]

		features = self.backbone(img)

		out = []

		for head in self.heads:
			out.append(head(features))

		prediction = Prediction(
			heatmap=out.pop(0),
			size=out.pop(0).permute(0, 2, 3, 1),
			offset=out.pop(0).permute(0, 2, 3, 1),
			roll_bin=out.pop(0).permute(0, 2, 3, 1) if self.object_config.train_roll else None,
			roll_offset=out.pop(0).permute(0, 2, 3, 1) if self.object_config.train_roll else None,
			pitch_bin=out.pop(0).permute(0, 2, 3, 1) if self.object_config.train_pitch else None,
			pitch_offset=out.pop(0).permute(0, 2, 3, 1) if self.object_config.train_pitch else None,
			yaw_bin=out.pop(0).permute(0, 2, 3, 1) if self.object_config.train_yaw else None,
			yaw_offset=out.pop(0).permute(0, 2, 3, 1) if self.object_config.train_yaw else None,
			depth=out.pop(0).permute(0, 2, 3, 1) if self.object_config.train_depth else None,
		)

		return prediction


def is_child(child_module: nn.Module, parent_modules: List[nn.Module]) -> bool:
	for parent_module in parent_modules:
		if child_module in parent_module.modules():
			return True

	return False


def initialize_weights(module: nn.Module, excluded_modules: List[nn.Module]):
	for name, submodule in module.named_modules():
		if isinstance(submodule, nn.Conv2d) or isinstance(submodule, nn.ConvTranspose2d) and not is_child(submodule, excluded_modules):
			print(f"Initializing {name}")
			nn.init.xavier_uniform_(submodule.weight)
			if submodule.bias is not None:
				nn.init.zeros_(submodule.bias)
		else:
			print(f"Skipping {name}")


def get_head_channels(object_config: ObjectConfigSet) -> [int]:
	n_heatmaps = len(object_config.configs)
	n_size = 2
	n_offset = 2

	head_channels = [n_heatmaps, n_size, n_offset]

	n_angle_bin = 4
	n_angle_offset = 4
	n_depth = 1

	if object_config.train_yaw:
		head_channels.extend((n_angle_bin, n_angle_offset))
	if object_config.train_pitch:
		head_channels.extend((n_angle_bin, n_angle_offset))
	if object_config.train_roll:
		head_channels.extend((n_angle_bin, n_angle_offset))
	if object_config.train_depth:
		head_channels.append(n_depth)

	return head_channels


def main():
	backbone_heights = [2, 2, 2]
	backbone_channels = [64, 64, 64, 64]

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
					modulo=2 * pi,
				),
				train_depth=True,
			)
		]
	)

	backbone = DLABackbone(backbone_heights, backbone_channels)
	centernet = Centernet(backbone, object_config)

	img = torch.rand(1, 3, 360, 640)
	prediction = centernet(img)

	pass


if __name__ == "__main__":
	main()