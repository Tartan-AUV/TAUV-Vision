import torch
torch.autograd.detect_anomaly(check_nan=True)
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms

from yolo_pose.model.config import Config
from yolo_pose.model.weights import initialize_weights
from yolo_pose.model.backbone import Resnet101Backbone
from yolo_pose.model.feature_pyramid import FeaturePyramid
from yolo_pose.model.masknet import Masknet
from yolo_pose.model.pointnet import Pointnet
from yolo_pose.model.prediction_head import PredictionHead
from yolo_pose.model.anchors import get_anchor
from yolo_pose.model.loss import loss
from yolo_pose.model.boxes import box_to_mask


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
        belief_prototype, affinity_prototype = self._pointnet(fpn_outputs[0])

        classifications = []
        box_encodings = []
        mask_coeffs = []
        belief_coeffs = []
        affinity_coeffs = []
        anchors = []

        for fpn_i, fpn_output in enumerate(fpn_outputs):
            classification, box_encoding, mask_coeff, belief_coeff, affinity_coeff = self._prediction_head(fpn_output)

            anchor = get_anchor(fpn_i, tuple(fpn_output.size()[2:4]), self._config).detach()
            anchor = anchor.to(box_encoding.device)

            classifications.append(classification)
            box_encodings.append(box_encoding)
            mask_coeffs.append(mask_coeff)
            belief_coeffs.append(belief_coeff)
            affinity_coeffs.append(affinity_coeff)
            anchors.append(anchor)

        classification = torch.cat(classifications, dim=1)
        box_encoding = torch.cat(box_encodings, dim=1)
        mask_coeff = torch.cat(mask_coeffs, dim=1)
        belief_coeff = torch.cat(belief_coeffs, dim=1)
        affinity_coeff = torch.cat(affinity_coeffs, dim=1)
        anchor = torch.cat(anchors, dim=1)

        return classification, box_encoding, mask_coeff, belief_coeff, affinity_coeff, anchor, mask_prototype, belief_prototype, affinity_prototype


def main():
    config = Config(
        in_w=960,
        in_h=480,
        feature_depth=256,
        n_classes=23,
        n_prototype_masks=32,
        n_masknet_layers_pre_upsample=1,
        n_masknet_layers_post_upsample=1,
        pointnet_layers=[
            (1, 3, 6, 512),
            (3, 7, 10, 128),
            (3, 7, 10, 128),
            (3, 7, 10, 128),
            (3, 7, 10, 128),
            (3, 7, 10, 128),
        ],
        pointnet_feature_depth=128,
        prototype_belief_depth=18,
        prototype_affinity_depth=32,
        belief_depth=9,
        affinity_depth=16,
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
    initialize_weights(model, [model._backbone])

    img0 = Image.open("../../img/000000.left.jpg").convert("RGB")
    img0 = transforms.ToTensor()(img0).to(device)
    img1 = Image.open("../../img/000001.left.jpg").convert("RGB")
    img1 = transforms.ToTensor()(img1).to(device)
    img = torch.stack((img0, img1), dim=0)

    # img = torch.rand(3, 3, config.in_h, config.in_w).to(device)

    truth_valid = torch.tensor([
        [True, True],
        [True, True],
    ]).to(device)
    truth_classification = torch.tensor([
        [1, 2],
        [3, 4],
    ], dtype=torch.uint8).to(device)
    truth_box = torch.tensor([
        [
            [0.5, 0.5, 0.1, 0.1],
            [0.7, 0.7, 0.1, 0.1],
        ],
        [
            [0.6, 0.6, 0.1, 0.1],
            [0.2, 0.2, 0.1, 0.1],
        ],
    ]).to(device)
    truth_seg_map = torch.zeros(2, config.in_h, config.in_w, dtype=torch.uint8).to(device)
    for batch_i in range(truth_seg_map.size(0)):
        for detection_i in range(truth_classification.size(1)):
            truth_seg_map[batch_i, box_to_mask(truth_box[batch_i, detection_i], (config.in_h, config.in_w)).to(torch.bool)] = truth_classification[batch_i, detection_i]

    truth_belief = torch.zeros(2, 2, config.belief_depth, config.in_h, config.in_w).to(device)
    truth_belief[:, :, :, 0:config.in_h // 2, :] = 1
    truth_affinity = torch.zeros(2, 2, config.affinity_depth, config.in_h, config.in_w).to(device)

    truth = (truth_valid, truth_classification, truth_box, truth_seg_map, truth_belief, truth_affinity)

    model.train()

    prediction = model.forward(img)

    l = loss(prediction, truth, config)
    total_loss, _ = l

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for iteration_i in range(1000):
        optimizer.zero_grad()

        prediction = model.forward(img)

        l = loss(prediction, truth, config)
        total_loss, _ = l
        print(l)

        total_loss.backward()

        optimizer.step()

        if total_loss < 0.01:
            break

    pass


if __name__ == "__main__":
    main()