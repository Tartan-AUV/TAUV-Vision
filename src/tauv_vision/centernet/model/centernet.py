from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Centernet(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, ...]:
		pass


def main():
	pass


if __name__ == "__main__":
	main()