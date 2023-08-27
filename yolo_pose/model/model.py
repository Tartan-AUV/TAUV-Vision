import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms

from yolo_pose.model.config import Config
from yolo_pose.model.backbone import Resnet101Backbone
from yolo_pose.model.feature_pyramid import FeaturePyramid
from yolo_pose.model.masknet import Masknet
from yolo_pose.model.pointnet import Pointnet
from yolo_pose.model.prediction_head import PredictionHead
from yolo_pose.model.anchors import get_anchor
from yolo_pose.model.loss import loss
from yolo_pose.model.boxes import box_to_mask
from torchviz import make_dot


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
        box_encodings = []
        mask_coeffs = []
        point_coeffs = []
        anchors = []

        for fpn_i, fpn_output in enumerate(fpn_outputs):
            classification, box_encoding, mask_coeff, point_coeff = self._prediction_head(fpn_output)

            anchor = get_anchor(fpn_i, tuple(fpn_output.size()[2:4]), self._config).detach()
            anchor = anchor.to(box_encoding.device)

            classifications.append(classification)
            box_encodings.append(box_encoding)
            mask_coeffs.append(mask_coeff)
            point_coeffs.append(point_coeff)
            anchors.append(anchor)

        classification = torch.cat(classifications, dim=1)
        box_encoding = torch.cat(box_encodings, dim=1)
        mask_coeff = torch.cat(mask_coeffs, dim=1)
        point_coeff = torch.cat(point_coeffs, dim=1)
        anchor = torch.cat(anchors, dim=1)

        return classification, box_encoding, mask_coeff, point_coeff, anchor, mask_prototype, point_prototype, direction_prototype


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
        iou_pos_threshold=0.5,
        iou_neg_threshold=0.4,
        negative_example_ratio=3,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YoloPose(config).to(device)

    cmu_img = Image.open("../../img/cmu.png").convert("RGB")
    img = transforms.ToTensor()(cmu_img).to(device)
    img = img.unsqueeze(0).tile((6, 1, 1, 1))

    # img = torch.rand(3, 3, config.in_h, config.in_w).to(device)

    truth_valid = torch.tensor([
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
    ]).to(device)
    truth_classification = torch.tensor([
        [1, 2],
        [1, 2],
        [1, 2],
        [1, 2],
        [1, 2],
        [1, 2],
    ]).to(device)
    truth_box = torch.tensor([
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.7, 0.7, 0.3, 0.3],
        ],
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.7, 0.7, 0.3, 0.3],
        ],
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.7, 0.7, 0.3, 0.3],
        ],
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.7, 0.7, 0.3, 0.3],
        ],
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.7, 0.7, 0.3, 0.3],
        ],
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.7, 0.7, 0.3, 0.3],
        ],
    ]).to(device)
    truth_mask = torch.zeros(6, 2, config.in_h, config.in_w).to(device)
    for batch_i in range(truth_mask.size(0)):
        for detection_i in range(truth_mask.size(1)):
            truth_mask[batch_i, detection_i] = box_to_mask(truth_box[batch_i, detection_i], (config.in_h, config.in_w))
    truth_point = torch.zeros(6, 2, 9, config.in_h, config.in_w).to(device)
    truth_direction = torch.full((6, 2, 8, config.in_h, config.in_w), 1, dtype=torch.float).to(device)

    truth = (truth_valid, truth_classification, truth_box, truth_mask, truth_point, truth_direction)

    model.train()

    prediction = model.forward(img)

    l = loss(prediction, truth, config)
    total_loss, _ = l

    make_dot(total_loss, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render("dag", format="svg")

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    for iteration_i in range(1000):
        optimizer.zero_grad()

        prediction = model.forward(img)

        l = loss(prediction, truth, config)
        total_loss, _ = l
        print(l)

        total_loss.backward()

        optimizer.step()


if __name__ == "__main__":
    main()