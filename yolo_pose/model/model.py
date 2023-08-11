import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo_pose.model.config import Config
from yolo_pose.model.backbone import Resnet101Backbone
from yolo_pose.model.feature_pyramid import FeaturePyramid
from yolo_pose.model.masknet import Masknet
from yolo_pose.model.pointnet import Pointnet
from yolo_pose.model.prediction_head import PredictionHead
from yolo_pose.model.anchors import get_anchor


class YoloPose(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self._config = config

        self._backbone = Resnet101Backbone(self._config)
        self._feature_pyramid = FeaturePyramid(self._backbone.depths, self._config)
        self._masknet = Masknet(self._config)
        self._pointnet = Pointnet(self._config)
        self._prediction_head = PredictionHead(self._config)

    def forward(self, img: torch.Tensor) -> (torch.Tensor, ...):
        backbone_outputs = self._backbone(img)

        fpn_outputs = self._feature_pyramid(backbone_outputs)

        mask_prototype = self._masknet(fpn_outputs[0])
        point_prototype, direction_prototype = self._pointnet(fpn_outputs[0])

        classifications = []
        boxes = []
        masks = []
        points = []
        anchors = []

        for fpn_i, fpn_output in enumerate(fpn_outputs):
            classification, box, mask, point = self._prediction_head(fpn_output)

            anchor = get_anchor(fpn_i, tuple(fpn_output.size()[2:4]), self._config)

            classifications.append(classification)
            boxes.append(box)
            masks.append(mask)
            points.append(point)
            anchors.append(anchor)

        classification = torch.cat(classifications, dim=-1)
        box = torch.cat(boxes, dim=-1)
        mask = torch.cat(masks, dim=-1)
        point = torch.cat(points, dim=-1)
        anchor = torch.cat(anchors, dim=-1)

        return classification, box, mask, point, anchor, mask_prototype, point_prototype, direction_prototype


def main():
    config = Config(
        in_w=512,
        in_h=512,
        feature_depth=256,
        n_classes=3,
        n_prototype_masks=32,
        n_prototype_points=64,
        n_points=8,
        n_masknet_layers_pre_upsample=1,
        n_masknet_layers_post_upsample=1,
        n_pointnet_layers_pre_upsample=1,
        n_pointnet_layers_post_upsample=1,
        n_prediction_head_layers=1,
        n_fpn_downsample_layers=2,
        anchor_scales=(24, 48, 96, 192, 384),
        anchor_aspect_ratios=(1/2, 1, 2),
    )
    model = YoloPose(config)

    img = torch.rand(1, 3, config.in_h, config.in_w)
    out = model(img)

if __name__ == "__main__":
    main()