import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck

from yolact.model.config import Config


class PredictionHead(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self._config = config

        self._bottleneck = Bottleneck(config.feature_depth, config.feature_depth // 4)

        self._conv = nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=1)
        self._batchnorm = nn.BatchNorm2d(config.feature_depth)

        self._classification_extra_layers = nn.ModuleList([
            nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=3, stride=1, padding=1)
            for _ in range(self._config.n_classification_layers)
        ])

        self._box_extra_layers = nn.ModuleList([
            nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=3, stride=1, padding=1)
            for _ in range(self._config.n_box_layers)
        ])

        self._mask_extra_layers = nn.ModuleList([
            nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=3, stride=1, padding=1)
            for _ in range(self._config.n_mask_layers)
        ])

        self._classification_layer = nn.Conv2d(
            config.feature_depth,
            len(config.anchor_aspect_ratios) * (config.n_classes + 1),
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self._box_encoding_layer = nn.Conv2d(
            config.feature_depth,
            len(config.anchor_aspect_ratios) * 4,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self._mask_coeff_layer = nn.Conv2d(
            config.feature_depth,
            len(config.anchor_aspect_ratios) * config.n_prototype_masks,
            kernel_size=3,
            padding=1,
            stride=1,
        )

    def forward(self, fpn_output: torch.Tensor) -> (torch.Tensor, ...):
        a = self._bottleneck(fpn_output)

        b = self._conv(fpn_output)
        b = self._batchnorm(b)
        b = F.relu(b)

        x = a + b

        classification = x

        for classification_layer in self._classification_extra_layers:
            classification = classification_layer(classification)
            classification = F.relu(classification)

        classification = self._classification_layer(classification)
        classification = classification.permute(0, 2, 3, 1)
        classification = classification.reshape(classification.size(0), -1, self._config.n_classes + 1)

        box_encoding = x

        for box_layer in self._box_extra_layers:
            box_encoding = box_layer(box_encoding)
            box_encoding = F.relu(box_encoding)

        box_encoding = self._box_encoding_layer(box_encoding)
        box_encoding = box_encoding.permute(0, 2, 3, 1)
        box_encoding = box_encoding.reshape(box_encoding.size(0), -1, 4)
        box_encoding[:, :, 0:2] = F.sigmoid(box_encoding[:, :, 0:2]) - 0.5
        box_encoding[:, :, 0] /= x.size(2)
        box_encoding[:, :, 1] /= x.size(3)

        mask_coeff = x
        for mask_layer in self._mask_extra_layers:
            mask_coeff = mask_layer(mask_coeff)
            mask_coeff = F.relu(mask_coeff)

        mask_coeff = self._mask_coeff_layer(x)
        mask_coeff = mask_coeff.permute(0, 2, 3, 1)
        mask_coeff = mask_coeff.reshape(mask_coeff.size(0), -1, self._config.n_prototype_masks)
        mask_coeff = F.tanh(mask_coeff)

        return classification, box_encoding, mask_coeff
