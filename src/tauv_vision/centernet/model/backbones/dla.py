from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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

		self.downsample_residual = nn.MaxPool2d(
			kernel_size=stride,
			stride=stride,
		)
		self.conv_residual = nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=1,
			stride=1,
		)
		self.bn_residual = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		residual = self.downsample_residual(x)
		residual = self.conv_residual(residual)
		residual = self.bn_residual(residual)

		y = self.conv1(x)
		y = self.bn1(y)
		y = F.relu(y)

		y = self.conv2(y)
		y = self.bn2(y)
		y += residual
		y = F.relu(y)

		return y


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

		y = [x]

		for tree_layer in self.tree_layers:
			x = tree_layer(x)
			y.append(x)

		return y


class DLAUp(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
		pass


def main():
	img = torch.rand((2, 3, 360, 640))

	heights = [1, 2, 2]
	channels = [128, 128, 256, 512]
	block = ResidualBlock

	dla_down = DLADown(heights, channels, block)

	y = dla_down.forward(img)

	pass


if __name__ == "__main__":
	main()