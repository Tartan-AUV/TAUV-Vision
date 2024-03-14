import torch
from math import pi
import torch.nn.functional as F
from enum import Enum

from tauv_vision.centernet.model.centernet import Prediction, Truth
from tauv_vision.centernet.model.decode import angle_get_bins
from tauv_vision.centernet.model.config import ModelConfig, TrainConfig, ObjectConfig, ObjectConfigSet, AngleConfig


def gaussian_splat(h: int, w: int, cy: int, cx: int, sigma: float) -> torch.Tensor:
    y = torch.arange(0, h)
    x = torch.arange(0, w)

    y, x = torch.meshgrid(y, x, indexing="ij")

    out = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))

    return out


def out_index_for_position(position: torch.Tensor, model_config: ModelConfig) -> torch.Tensor:
    return (position / model_config.downsample_ratio).to(torch.long)


class Angle(Enum):
    Roll = 0
    Pitch = 1
    Yaw = 2


def angle_range(truth_label: torch.Tensor, object_config: ObjectConfigSet, angle: Angle) -> torch.Tensor:
    # truth_label is [batch_size, n_objects]
    #
    # result is [batch_size, n_objects]

    batch_size, n_objects = truth_label.shape
    device = truth_label.device

    result = torch.zeros_like(truth_label, dtype=torch.float32)

    for sample_i in range(batch_size):
        for object_i in range(n_objects):
            label = truth_label[sample_i, object_i]

            if angle == Angle.Roll:
                modulo = object_config.configs[label].roll.modulo
            elif angle == Angle.Pitch:
                modulo = object_config.configs[label].pitch.modulo
            elif angle == Angle.Yaw:
                modulo = object_config.configs[label].yaw.modulo

            if modulo is not None:
                result[sample_i, object_i] = modulo

    return result.to(device)


def loss(prediction: Prediction, truth: Truth, model_config: ModelConfig, train_config: TrainConfig, object_config: ObjectConfigSet) -> torch.Tensor:

    out_index = out_index_for_position(truth.center, model_config) # [batch_size, n_objects, 2]

    batch_size, n_objects, _ = out_index.shape
    device = out_index.device

    # TODO: Vectorize this
    prediction_size = torch.zeros((batch_size, n_objects, 2), dtype=torch.float32, device=device) # [batch_size, n_objects, 2]
    prediction_offset = torch.zeros((batch_size, n_objects, 2), dtype=torch.float32, device=device) # [batch_size, n_objects, 2]

    if prediction.roll_bin is not None:
        prediction_roll_bin = torch.zeros((batch_size, n_objects, 4), dtype=torch.float32, device=device)
        prediction_roll_offset = torch.zeros((batch_size, n_objects, 4), dtype=torch.float32, device=device)

    if prediction.pitch_bin is not None:
        prediction_pitch_bin = torch.zeros((batch_size, n_objects, 4), dtype=torch.float32, device=device)
        prediction_pitch_offset = torch.zeros((batch_size, n_objects, 4), dtype=torch.float32, device=device)

    if prediction.yaw_bin is not None:
        prediction_yaw_bin = torch.zeros((batch_size, n_objects, 4), dtype=torch.float32, device=device)
        prediction_yaw_offset = torch.zeros((batch_size, n_objects, 4), dtype=torch.float32, device=device)

    if prediction.depth is not None:
        prediction_depth = torch.zeros((batch_size, n_objects), dtype=torch.float32, device=device)

    for sample_i in range(batch_size):
        for object_i in range(n_objects):
            prediction_size[sample_i, object_i] = prediction.size[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]
            prediction_offset[sample_i, object_i] = prediction.offset[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]

            if prediction.roll_bin is not None:
                prediction_roll_bin[sample_i, object_i] = prediction.roll_bin[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]
                prediction_roll_offset[sample_i, object_i] = prediction.roll_offset[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]

            if prediction.pitch_bin is not None:
                prediction_pitch_bin[sample_i, object_i] = prediction.pitch_bin[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]
                prediction_pitch_offset[sample_i, object_i] = prediction.pitch_offset[sample_i, out_index[sample_i, object_i], out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]

            if prediction.yaw_bin is not None:
                prediction_yaw_bin[sample_i, object_i] = prediction.yaw_bin[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]
                prediction_yaw_offset[sample_i, object_i] = prediction.yaw_offset[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]

            if prediction.depth is not None:
                prediction_depth[sample_i, object_i] = prediction.depth[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]

    n_valid = truth.valid.to(torch.float32).sum()

    l_heatmap = focal_loss(F.sigmoid(prediction.heatmap), truth.heatmap, alpha=train_config.heatmap_focal_loss_a, beta=train_config.heatmap_focal_loss_b)

    l = l_heatmap.sum()

    l_size = F.smooth_l1_loss(prediction_size, truth.size, reduction="none")
    l += train_config.loss_lambda_size * (truth.valid * l_size).sum() / n_valid

    l_offset = F.smooth_l1_loss(prediction_offset, truth.center - model_config.downsample_ratio * (truth.center / model_config.downsample_ratio).to(torch.long), reduction="none")
    l += train_config.loss_lambda_offset * (truth.valid * l_offset).sum() / n_valid

    if prediction.roll_bin is not None:
        roll_theta_range = angle_range(truth.label, object_config, Angle.Roll)
        l_roll = angle_loss(prediction_roll_bin, prediction_roll_offset, truth.roll, roll_theta_range, model_config.angle_bin_overlap).sum()
        l += train_config.loss_lambda_angle * (truth.valid * l_roll).sum() / n_valid

    if prediction.pitch_bin is not None:
        pitch_theta_range = angle_range(truth.label, object_config, Angle.Pitch)
        l_pitch = angle_loss(prediction_pitch_bin, prediction_pitch_offset, truth.pitch, pitch_theta_range, model_config.angle_bin_overlap).sum()
        l += train_config.loss_lambda_angle * (truth.valid * l_pitch).sum() / n_valid

    if prediction.yaw_bin is not None:
        yaw_theta_range = angle_range(truth.label, object_config, Angle.Yaw)
        l_yaw = angle_loss(prediction_yaw_bin, prediction_yaw_offset, truth.yaw, yaw_theta_range, model_config.angle_bin_overlap).sum()
        l += train_config.loss_lambda_angle * (truth.valid * l_yaw).sum() / n_valid

    return l


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


def angle_loss(predicted_bin: torch.Tensor, predicted_offset: torch.Tensor, truth: torch.Tensor, theta_range: torch.Tensor, bin_overlap: float) -> torch.Tensor:
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
    # theta_range is [batch_size, n_detections]
    #
    # result is [batch_size, n_detections]

    batch_size, n_detections = truth.shape

    truth = truth % theta_range              # angles from [0, theta_range)
    truth = truth * (2 * pi / theta_range)   # angles from [0, 2 * pi)

    bins = angle_get_bins(bin_overlap)
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


def depth_loss(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    # prediction is [batch_size, n_detections]
    # truth is [batch_size, n_detections]
    #
    # result is [batch_size, n_detections]

    batch_size, n_detections = truth.shape

    result = F.l1_loss(1 / F.sigmoid(prediction.reshape(-1)) - 1, truth.reshape(-1), reduction="none")
    result = result.reshape(batch_size, n_detections)

    return result
