from typing import List, Optional, Tuple

import torch
from math import pi
import torch.nn as nn
import torch.nn.functional as F


def gaussian_splat(h: float, w: float, cy: float, cx: float, sigma: int) -> torch.Tensor:
    y = torch.arange(0, h)
    x = torch.arange(0, w)

    y, x = torch.meshgrid(y, x, indexing="ij")

    out = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))

    return out


def focal_loss(prediction: torch.Tensor, truth: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # prediction is [batch_size, n_classes, h, w]
    # truth is [batch_size, n_classes, h, w]

    p = torch.isclose(truth, torch.Tensor([1]).to(truth.device))
    N = torch.sum(p)

    loss_p = ((1 - prediction) ** alpha) * torch.log(torch.clamp(prediction, min=1e-4)) * p.float()
    loss_n = ((1 - truth) ** beta) * (prediction ** alpha) * torch.log(torch.clamp(1 - prediction, min=1e-4)) * (1 - p.float())

    if N == 0:
        loss = -loss_p
    else:
        loss = -(loss_p + loss_n) / N

    return loss


def get_bins(bin_overlap: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # bins are ((bin_0_center, bin_0_min, bin_0_max), (bin_1_center, bin_1_min, bin_1_max))

    bin_0 = (pi / 2, -bin_overlap / 2, pi + bin_overlap / 2)
    bin_1 = (-pi / 2, -pi - bin_overlap / 2, bin_overlap / 2)

    return bin_0, bin_1


def angle_in_range(angles: torch.Tensor, range_min: float, range_max: float) -> torch.Tensor:
    # angles is [shape]
    #
    # result is [shape]

    range_min = range_min % (2 * pi)
    range_max = range_max % (2 * pi)
    angles = angles % (2 * pi)
    if range_min < range_max:
        return (range_min <= angles) & (angles <= range_max)
    else:
        return (range_min <= angles) | (angles <= range_max)


def angle_loss(predicted_bin: torch.Tensor, predicted_offset: torch.Tensor, truth: torch.Tensor, theta_range: float, bin_overlap: float) -> torch.Tensor:
    # All angles are taken to fall in range [0, theta_range)
    #
    # predicted_bin is [batch_size, n_detections, 4]
    #   predicted_bin[0:2] are [outside, inside] classifications for bin 0
    #   predicted_bin[2:4] are [outside, inside] classifications for bin 1
    # predicted_offset is [batch_size, n_detections, 4]
    #   predicted_offset[0:2] are [sin, cos] offsets for bin 0
    #   predicted_offset[2:4] are [sin, cos] offsets for bin 1
    #
    # truth is [batch_size, n_detections]
    #
    # result is [batch_size, n_detections]

    batch_size, n_detections = truth.shape

    truth = truth % theta_range              # angles from [0, theta_range)
    truth = truth * (2 * pi / theta_range)   # angles from [0, 2 * pi)

    bins = get_bins(bin_overlap)
    (bin_0_center, bin_0_min, bin_0_max), (bin_1_center, bin_1_min, bin_1_max) = bins

    inside_bin_0 = angle_in_range(truth, bin_0_min, bin_0_max).to(torch.long)  # [batch_size, n_detections]
    inside_bin_1 = angle_in_range(truth, bin_1_min, bin_1_max).to(torch.long)  # [batch_size, n_detections]

    offsets_bin_0 = torch.stack((torch.sin(truth - bin_0_center), torch.cos(truth - bin_0_center)), dim=-1) # [batch_size, n_detections, 2]
    offsets_bin_1 = torch.stack((torch.sin(truth - bin_1_center), torch.cos(truth - bin_1_center)), dim=-1) # [batch_size, n_detections, 2]

    classification_loss_bin_0 = F.cross_entropy(predicted_bin[:, :, 0:2].reshape(-1, 2), inside_bin_0.reshape(-1), reduction="none") # [batch_size x n_detections]
    classification_loss_bin_1 = F.cross_entropy(predicted_bin[:, :, 2:4].reshape(-1, 2), inside_bin_1.reshape(-1), reduction="none") # [batch_size x n_detections]

    offset_loss_bin_0 = F.l1_loss(predicted_offset[:, :, 0:2].reshape(-1, 2), offsets_bin_0.reshape(-1, 2), reduction="none").sum(dim=-1) # [batch_size x n_detections]
    offset_loss_bin_1 = F.l1_loss(predicted_offset[:, :, 2:4].reshape(-1, 2), offsets_bin_1.reshape(-1, 2), reduction="none").sum(dim=-1) # [batch_size x n_detections]

    result = (classification_loss_bin_0 + classification_loss_bin_1
              + inside_bin_0.reshape(-1).to(torch.float) * offset_loss_bin_0
              + inside_bin_1.reshape(-1).to(torch.float) * offset_loss_bin_1)   # [batch_size x n_detections]

    result = result.reshape(batch_size, n_detections)   # [batch_size, n_detections]

    return result


def angle_decode(predicted_bin: torch.Tensor, predicted_offset: torch.Tensor, theta_range: float, bin_overlap: float) -> torch.Tensor:
    # predicted_bin is [batch_size, n_detections, 4]
    #   predicted_bin[0:2] are [outside, inside] classifications for bin 0
    #   predicted_bin[2:4] are [outside, inside] classifications for bin 1
    # predicted_offset is [batch_size, n_detections, 4]
    #   predicted_offset[0:2] are [sin, cos] offsets for bin 0
    #   predicted_offset[2:4] are [sin, cos] offsets for bin 1
    #
    # result is [batch_size, n_detections]

    bins = get_bins(bin_overlap)
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


def depth_loss(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    # prediction is [batch_size, n_detections]
    # truth is [batch_size, n_detections]
    #
    # result is [batch_size, n_detections]

    batch_size, n_detections = truth.shape

    result = F.l1_loss(1 / F.sigmoid(prediction.reshape(-1)) - 1, truth.reshape(-1), reduction="none")
    result = result.reshape(batch_size, n_detections)

    return result

def depth_decode(prediction: torch.Tensor) -> torch.Tensor:
    # prediction is [size]
    #
    # result is [size]

    return (1 / F.sigmoid(prediction)) - 1

def main():
    # out = gaussian_splat(360, 640, 300, 300, 50)
    #
    # alpha = (90 * 160) / 1
    # beta = 4
    #
    # truth = torch.stack((
    #     gaussian_splat(90, 160, 50, 100, 10).unsqueeze(0),
    #     gaussian_splat(90, 160, 60, 130, 2).unsqueeze(0),
    # ), dim=1)
    # truth = torch.clamp(truth, min=1e-4)
    #
    # loss1 = focal_loss(torch.clamp(truth, min=0, max=1), truth, alpha, beta).sum()
    # loss2 = focal_loss(torch.clamp(truth + 1e-1 * torch.rand_like(truth), min=0, max=1), truth, alpha, beta).sum()
    # loss3 = focal_loss(torch.clamp(truth + 1e-2 * torch.rand_like(truth), min=0, max=1), truth, alpha, beta).sum()
    # loss4 = focal_loss(torch.rand_like(truth), truth, alpha, beta).sum()
    # loss5 = focal_loss(torch.full_like(truth, fill_value=1e-4), truth, alpha, beta).sum()
    # loss6 = focal_loss(torch.full_like(truth, fill_value=1-1e-4), truth, alpha, beta).sum()

    angles = torch.Tensor([0, pi / 4, pi / 2, 3 * pi / 4, pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4])
    bins = get_bins(pi / 3)
    print(angles)
    _, bin_0_min, bin_0_max = bins[0]
    _, bin_1_min, bin_1_max = bins[1]

    print(angle_in_range(angles, bin_0_min, bin_0_max))
    print(angle_in_range(angles, bin_1_min, bin_1_max))

    prediction = torch.rand(4, 3, 8)
    truth = torch.rand(4, 3)
    l = angle_loss(prediction, truth, pi / 2, pi / 3)
    a = angle_decode(prediction, pi / 2, pi / 3)

    prediction = torch.rand(4, 3)
    truth = torch.rand(4, 3)
    l = depth_loss(prediction, truth)

    pass

if __name__ == "__main__":
    main()
