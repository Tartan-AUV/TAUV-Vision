from typing import Tuple, List

from tauv_vision.centernet.model.backbones.dla import DLABackbone

import torch
import torch.nn as nn
import torch.nn.functional as F


class Centernet(nn.Module):

	def __init__(self, backbone: nn.Module, out_channels: List[int]):
		super().__init__()

		self.backbone = backbone

		heads = []

		for out_channel in out_channels:
			head = nn.Sequential(
				nn.Conv2d(
					in_channels=backbone.out_channels,
					out_channels=backbone.out_channels,
					kernel_size=3,
					padding=1,
				),
				nn.ReLU(),
				nn.Conv2d(
					in_channels=backbone.out_channels,
					out_channels=out_channel,
					kernel_size=1,
				),
			)

			heads.append(head)

		self.heads = nn.ModuleList(heads)

	def forward(self, img: torch.Tensor) -> List[torch.Tensor]:
		features = self.backbone(img)

		out = []

		for head in self.heads:
			out.append(head(features))

		return out


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


def main():
	backbone_heights = [2, 2, 2]
	backbone_channels = [64, 64, 64, 64]
	out_channels = [1, 2, 8]

	backbone = DLABackbone(backbone_heights, backbone_channels)
	centernet = Centernet(backbone, out_channels)

	img = torch.rand(1, 3, 512, 512)
	prediction = centernet(img)

	pass


if __name__ == "__main__":
	main()