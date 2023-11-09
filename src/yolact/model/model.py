import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms

from yolact.model.config import Config
from yolact.model.weights import initialize_weights
from yolact.model.backbone import Resnet101Backbone
from yolact.model.feature_pyramid import FeaturePyramid
from yolact.model.masknet import Masknet
from yolact.model.prediction_head import PredictionHead
from yolact.model.anchors import get_anchor
from yolact.model.loss import loss
from yolact.model.boxes import box_to_mask


class Yolact(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self._config = config

        self._backbone = Resnet101Backbone(self._config)
        self._feature_pyramid = FeaturePyramid(self._backbone.depths, self._config)
        self._masknet = Masknet(self._config)
        self._prediction_heads = nn.ModuleList([PredictionHead(self._config) for _ in range(len(self._config.anchor_scales))])

    def forward(self, img: torch.Tensor) -> (torch.Tensor, ...):
        backbone_outputs = self._backbone(img)

        fpn_outputs = self._feature_pyramid(backbone_outputs)

        mask_prototype = self._masknet(fpn_outputs[0])

        classifications = []
        box_encodings = []
        mask_coeffs = []
        anchors = []

        for fpn_i, fpn_output in enumerate(fpn_outputs):
            classification, box_encoding, mask_coeff = self._prediction_heads[fpn_i](fpn_output)

            anchor = get_anchor(fpn_i, tuple(fpn_output.size()[2:4]), self._config).detach()
            anchor = anchor.to(box_encoding.device)

            classifications.append(classification)
            box_encodings.append(box_encoding)
            mask_coeffs.append(mask_coeff)
            anchors.append(anchor)

        classification = torch.cat(classifications, dim=1)
        box_encoding = torch.cat(box_encodings, dim=1)
        mask_coeff = torch.cat(mask_coeffs, dim=1)
        anchor = torch.cat(anchors, dim=1)

        return classification, box_encoding, mask_coeff, anchor, mask_prototype


def main():
    config = Config(
        in_w=960,
        in_h=480,
        feature_depth=256,
        n_classes=23,
        n_prototype_masks=32,
        n_masknet_layers_pre_upsample=1,
        n_masknet_layers_post_upsample=1,
        n_prediction_head_layers=1,
        n_fpn_downsample_layers=2,
        anchor_scales=(24, 48, 96, 192, 384),
        anchor_aspect_ratios=(1/2, 1, 2),
        iou_pos_threshold=0.5,
        iou_neg_threshold=0.4,
        negative_example_ratio=3,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Yolact(config).to(device)
    initialize_weights(model, [model._backbone])

    # img0 = Image.open("../../img/000000.left.jpg").convert("RGB")
    img0 = Image.open("../../img/cmu.png").convert("RGB")
    img0 = transforms.ToTensor()(img0).to(device)
    # img1 = Image.open("../../img/000001.left.jpg").convert("RGB")
    img1 = Image.open("../../img/cmu.png").convert("RGB")
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
            # truth_seg_map[batch_i, box_to_mask(truth_box[batch_i, detection_i], (config.in_h, config.in_w)).to(torch.bool)] = truth_classification[batch_i, detection_i]
            truth_seg_map[batch_i, box_to_mask(truth_box[batch_i, detection_i], (config.in_h, config.in_w)).to(torch.bool)] = detection_i

    truth = (truth_valid, truth_classification, truth_box, truth_seg_map)

    model.train()

    prediction = model.forward(img)

    l = loss(prediction, truth, config)
    total_loss, _ = l

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for iteration_i in range(1000):
        optimizer.zero_grad()

        prediction = model.forward(img)

        l = loss(prediction, truth, config)
        total_loss, _ = l
        print(l)

        total_loss.backward()

        optimizer.step()

    pass


if __name__ == "__main__":
    main()