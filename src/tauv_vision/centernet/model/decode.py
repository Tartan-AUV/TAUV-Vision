from typing import Tuple, Optional
from dataclasses import dataclass

import torch
from math import pi
import torch.nn.functional as F

from tauv_vision.centernet.model.centernet import Prediction, Truth
from tauv_vision.centernet.model.config import ModelConfig


@dataclass
class Detection:
    label: int
    score: float
    y: float
    x: float
    h: float
    w: float

    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

    depth: Optional[float] = None


def decode(prediction: Prediction, model_config: ModelConfig,
           n_detections: int, score_threshold: float) -> [[Detection]]:

    heatmap = F.sigmoid(prediction.heatmap)
    heatmap = heatmap_nms(heatmap, kernel_size=3)
    detected_index, detected_label, detected_score = heatmap_detect(heatmap, n_detections)

    batch_size = detected_index.shape[0]

    detections = []

    # TODO: Decode angles
    # TODO: Decode depth

    for sample_i in range(batch_size):
        sample_detections = []

        for detection_i in range(n_detections):
            if detected_score[sample_i, detection_i] < score_threshold:
                break

            detection = Detection(
                label=detected_label[sample_i, detection_i],
                score=detected_score[sample_i, detection_i],
                y=(model_config.downsample_ratio * float(detected_index[sample_i, detection_i, 0]) + float(prediction.offset[sample_i, detected_index[sample_i, detection_i, 0], detected_index[sample_i, detection_i, 1], 0])) / model_config.in_h,
                x=(model_config.downsample_ratio * float(detected_index[sample_i, detection_i, 1]) + float(prediction.offset[sample_i, detected_index[sample_i, detection_i, 0], detected_index[sample_i, detection_i, 1], 1])) / model_config.in_w,
                h=float(prediction.size[sample_i, detected_index[sample_i, detection_i, 0], detected_index[sample_i, detection_i, 1], 0]),
                w=float(prediction.size[sample_i, detected_index[sample_i, detection_i, 0], detected_index[sample_i, detection_i, 1], 1]),
            )
            
            # TODO: Fill in angles
            # TODO: Fill in depth

            sample_detections.append(detection)

        detections.append(sample_detections)

    return detections


def heatmap_nms(heatmap: torch.Tensor, kernel_size: int) -> torch.Tensor:
    # heatmap is [batch_size, n_heatmaps, feature_h, feature_w]
    # result is [batch_size, n_heatmaps, feature_h, feature_w]

    assert kernel_size >= 1 and kernel_size % 2 == 1

    heatmap_max = F.max_pool2d(
        heatmap,
        (kernel_size, kernel_size),
        stride=1,
        padding=(kernel_size - 1) // 2,
    )

    return (heatmap_max == heatmap).float() * heatmap


def heatmap_detect(heatmap: torch.Tensor, n_detections: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # heatmap is [batch_size, n_heatmaps, feature_h, feature_w]
    #
    # result is (detected_index, detected_label, detected_score)
    # index is [batch_size, n_detections, 2]
    #   index[:, :, 0] is y index in [0, feature_h)
    #   index[:, :, 1] is x index in [0, feature_w)
    # label is [batch_size, n_detections]
    # score is [batch_size, n_detections]

    batch_size, n_heatmaps, feature_h, feature_w = heatmap.shape

    scores = heatmap.reshape(batch_size, -1)

    selected_score, selected_index = torch.topk(scores, n_detections)

    selected_label = (selected_index / (feature_h * feature_w)).to(torch.long)
    selected_index = (selected_index % (feature_h * feature_w)).to(torch.long)

    selected_index = torch.stack((
        (selected_index / feature_w).to(torch.long),
        (selected_index % feature_w).to(torch.long),
    ), dim=-1)

    return selected_index, selected_label, selected_score


def angle_get_bins(bin_overlap: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # bins are ((bin_0_center, bin_0_min, bin_0_max), (bin_1_center, bin_1_min, bin_1_max))

    bin_0 = (pi / 2, -bin_overlap / 2, pi + bin_overlap / 2)
    bin_1 = (-pi / 2, -pi - bin_overlap / 2, bin_overlap / 2)

    return bin_0, bin_1


def angle_decode(predicted_bin: torch.Tensor, predicted_offset: torch.Tensor, theta_range: float, bin_overlap: float) -> torch.Tensor:
    # predicted_bin is [batch_size, n_detections, 4]
    #   predicted_bin[0:2] are [outside, inside] classifications for bin 0
    #   predicted_bin[2:4] are [outside, inside] classifications for bin 1
    # predicted_offset is [batch_size, n_detections, 4]
    #   predicted_offset[0:2] are [sin, cos] offsets for bin 0
    #   predicted_offset[2:4] are [sin, cos] offsets for bin 1
    #
    # result is [batch_size, n_detections]

    bins = angle_get_bins(bin_overlap)
    (bin_0_center, bin_0_min, bin_0_max), (bin_1_center, bin_1_min, bin_1_max) = bins

    classification_score_bin_0 = F.softmax(predicted_bin[:, :, 0:2], dim=-1)[:, :, 1]    # [batch_size, n_detections]
    classification_score_bin_1 = F.softmax(predicted_bin[:, :, 2:4], dim=-1)[:, :, 1]    # [batch_size, n_detections]

    use_bin_1 = classification_score_bin_1 > classification_score_bin_0    # [batch_size, n_detections]

    angle_bin_0 = bin_0_center + torch.atan2(predicted_offset[:, :, 0], predicted_offset[:, :, 1])    # [batch_size, n_detections]
    angle_bin_1 = bin_1_center + torch.atan2(predicted_offset[:, :, 2], predicted_offset[:, :, 3])    # [batch_size, n_detections]

    angle = torch.where(use_bin_1, angle_bin_1, angle_bin_0)   # [batch_size, n_detections]
    angle = angle % (2 * pi)
    angle = angle * (theta_range / (2 * pi))

    return angle


def depth_decode(prediction: torch.Tensor) -> torch.Tensor:
    # prediction is [size]
    #
    # result is [size]

    return (1 / F.sigmoid(prediction)) - 1


def main():
    from tauv_vision.centernet.model.loss import gaussian_splat

    heatmap = torch.cat((
        gaussian_splat(512, 512, 100, 100, 50).unsqueeze(0).unsqueeze(1),
        gaussian_splat(512, 512, 200, 200, 50).unsqueeze(0).unsqueeze(1),
    ), dim=1)

    heatmap = heatmap_nms(heatmap, 3)

    detected_index, detected_label, detected_score = heatmap_detect(heatmap, 100)

    assert detected_index[0, 0, 0] == 100 and detected_index[0, 0, 1] == 100


if __name__ == "__main__":
    main()
