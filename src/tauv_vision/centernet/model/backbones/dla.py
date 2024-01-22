from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class ResidualBlock(nn.Module):

	def __init__(self, in_channels: int, out_channels: int, stride: int):
		super().__init__()

		self.conv1 = nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=3,
			stride=stride,
			padding=1,
		)
		self.bn1 = nn.BatchNorm2d(out_channels)

		self.conv2 = nn.Conv2d(
			in_channels=out_channels,
			out_channels=out_channels,
			kernel_size=3,
			stride=1,
			padding=1,
		)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.conv_residual = nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=1,
			stride=stride,
		)
		self.bn_residual = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		residual = self.conv_residual(x)
		residual = self.bn_residual(residual)

		y = self.conv1(x)
		y = self.bn1(y)
		y = F.relu(y)

		y = self.conv2(y)
		y = self.bn2(y)
		y += residual
		y = F.relu(y)

		return y


# When it's time for deform conv stuff, use MMCV package
# https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/csrc/pytorch/deform_conv.cpp

class Root(nn.Module):

	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()

		self.conv = nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=1,
			stride=1
		)
		self.bn = nn.BatchNorm2d(out_channels)

	def forward(self, children: List[torch.Tensor]) -> torch.Tensor:
		x = self.conv(torch.cat(children, 1))
		x = self.bn(x)
		x = F.relu(x)

		return x


class Tree(nn.Module):

	def __init__(self, in_channels: int, out_channels: int, height: int, root_channels: Optional[int], block: nn.Module, stride: int):
		super().__init__()

		self.height = height

		if root_channels is None:
			root_channels = 2 * out_channels

		if height == 1:
			self.tree_l = block(
				in_channels=in_channels,
				out_channels=out_channels,
				stride=stride,
			)
			self.tree_r = block(
				in_channels=out_channels,
				out_channels=out_channels,
				stride=1,
			)
			self.root = Root(
				in_channels=root_channels,
				out_channels=out_channels
			)
		else:
			self.tree_l = Tree(
				in_channels=in_channels,
				out_channels=out_channels,
				height=height - 1,
				root_channels=None,
				block=block,
				stride=stride,
			)
			self.tree_r = Tree(
				in_channels=out_channels,
				out_channels=out_channels,
				height=height - 1,
				root_channels=root_channels + out_channels,
				block=block,
				stride=1,
			)
			self.root = None

	def forward(self, x: torch.Tensor, children: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
		if children is None:
			children = []

		xl = self.tree_l(x)

		if self.height == 1:
			xr = self.tree_r(xl)
			x = self.root(children + [xl, xr])
		else:
			x = self.tree_r(xl, children=children + [xl])

		return x


class DLADown(nn.Module):

	def __init__(self, heights: List[int], channels: List[int], block: nn.Module):
		super().__init__()

		self.projection_layer = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				out_channels=channels[0],
				kernel_size=7,
				stride=1,
				padding=3,
			),
			nn.BatchNorm2d(channels[0]),
			nn.ReLU(),
		)

		self.block_layer = ResidualBlock(
			in_channels=channels[0],
			out_channels=channels[0],
			stride=2,
		)

		tree_layers = []

		for tree_layer_i in range(len(heights)):
			layer = Tree(
				in_channels=channels[tree_layer_i],
				out_channels=channels[tree_layer_i + 1],
				height=heights[tree_layer_i],
				root_channels=None,
				block=block,
				stride=2,
			)

			tree_layers.append(layer)

		self.tree_layers = nn.ModuleList(tree_layers)

	def forward(self, img: torch.Tensor) -> List[torch.Tensor]:
		x = self.projection_layer(img)
		x = self.block_layer(x)

		y = [x]

		for tree_layer in self.tree_layers:
			x = tree_layer(x)
			y.append(x)

		return y


class IDAUp(nn.Module):

	def __init__(self, feature_channels: List[int], scales: List[int]):
		super().__init__()

		projection_layers = []
		output_layers = []
		upsample_layers = []

		for feature_i in range(1, len(feature_channels)):
			projection_layer = nn.Conv2d(
				in_channels=feature_channels[feature_i - 1],
				out_channels=feature_channels[feature_i],
				kernel_size=3,
				padding=1,
				stride=1,
			)

			output_layer = nn.Conv2d(
				in_channels=feature_channels[feature_i],
				out_channels=feature_channels[feature_i],
				kernel_size=3,
				padding=1,
				stride=1,
			)

			upsample_layer = nn.ConvTranspose2d(
				in_channels=feature_channels[feature_i],
				out_channels=feature_channels[feature_i],
				kernel_size=scales[feature_i - 1],
				stride=scales[feature_i - 1],
			)

			projection_layers.append(projection_layer)
			output_layers.append(output_layer)
			upsample_layers.append(upsample_layer)

		self.projection_layers = nn.ModuleList(projection_layers)
		self.output_layers = nn.ModuleList(output_layers)
		self.upsample_layers = nn.ModuleList(upsample_layers)

	def forward(self, features: List[torch.Tensor], start_i: int, reverse: bool):
		# for i in range(len(features) - 1, start_i - 1, -1):
		for i in range(len(features) - start_i):
			if reverse:
				feature_i = len(features) - 1 - i
			else:
				feature_i = start_i + i

			layer_i = len(features) - start_i - 1 - i if reverse else i

			project = self.projection_layers[layer_i]
			output = self.output_layers[layer_i]
			upsample = self.upsample_layers[layer_i]

			features[feature_i] = output(upsample(features[feature_i]) + project(features[feature_i - 1]))


class MultiIDAUp(nn.Module):

	def __init__(self, channels: List[int]):
		super().__init__()

		ida_up_layers = []

		for i in range(1, len(channels)):
			ida_up_layer = IDAUp(
				feature_channels=channels[i - 1:],
				scales=[2 for _ in range(len(channels[i:]))],
			)

			ida_up_layers.append(ida_up_layer)

		self.ida_up_layers = nn.ModuleList(ida_up_layers)

	def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
		out = [features[-1]] # Copy lowest-size feature directly

		for i in range(len(features) - 1):
			self.ida_up_layers[i](features, i + 1, reverse=True)
			out.append(features[-1])

		return list(reversed(out))


def main():
	img = torch.rand((2, 3, 512, 512))

	heights = [1, 2, 2, 1]
	channels = [64, 128, 128, 256, 512]
	block = ResidualBlock

	dla_down = DLADown(heights, channels, block)

	y = dla_down.forward(img)

	multi_ida_up = MultiIDAUp(channels)

	z = multi_ida_up.forward(y)

	ida_up = IDAUp(
		feature_channels=[channels[-1] for _ in channels[1:]],
		scales=[z[1].size(2) // z[i].size(2) for i in range(2, len(channels))],
	)

	ida_up(z, 2, reverse=False)

	out = z[-1]

	pass


if __name__ == "__main__":
	main()