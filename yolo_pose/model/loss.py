import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from yolo_pose.model.boxes import box_decode, iou_matrix
from yolo_pose.model.config import Config


def loss(prediction: (torch.Tensor, ...), truth: (torch.Tensor, ...), config: Config) -> (torch.Tensor, (torch.Tensor, ...)):
    classification, box_encoding, mask_coeff, point_coeff, anchor, mask_prototype, point_prototype, direction_prototype = prediction
    truth_valid, truth_classification, truth_box, truth_mask, truth_point, truth_direction = truth

    n_batch = truth_classification.size(0)

    # What are the shapes of all of these things?

    box = box_decode(box_encoding, anchor)
    # truth_box_encoding = box_encode(truth_box, anchor)

    iou = iou_matrix(anchor, truth_box)

    match_iou, match_index = torch.max(iou * truth_valid.unsqueeze(1).float(), dim=2)

    positive_match = match_iou >= config.iou_pos_threshold
    negative_match = match_iou <= config.iou_neg_threshold
    neutral_match = ~(positive_match | negative_match)

    classification_losses = torch.zeros(n_batch)

    for batch_i in range(n_batch):
        match_classification = truth_classification[batch_i, match_index[batch_i]]
        match_classification[~positive_match[batch_i]] = 0

        classification_cross_entropy = F.cross_entropy(
            classification[batch_i],
            match_classification,
            reduction="none"
        )
        n_positive_match = positive_match[batch_i].sum()
        n_selected_negative_match = config.negative_example_ratio * n_positive_match

        _, selected_negative_match_index = torch.topk(
            torch.where(negative_match[batch_i], classification_cross_entropy, 0),
            k=n_selected_negative_match
        )
        selected_negative_match_index = selected_negative_match_index.detach()

        selected_match = torch.clone(positive_match[batch_i])
        selected_match[selected_negative_match_index] = True
        selected_match = selected_match.detach()

        print("positive", classification_cross_entropy[positive_match[batch_i]])
        print("selected_negative", classification_cross_entropy[selected_negative_match_index])

        classification_loss = (selected_match.float() * classification_cross_entropy).sum()

        classification_losses[batch_i] = classification_loss

    classification_loss = classification_losses.sum() / ((1 + config.negative_example_ratio) * positive_match.sum())

    box_losses = torch.zeros(n_batch)

    for batch_i in range(n_batch):
        # TODO: Check if loss should be over encoding or decoding
        box_loss = F.smooth_l1_loss(
            box[batch_i, positive_match[batch_i]],
            truth_box[batch_i, match_index[batch_i, positive_match[batch_i]]],
            reduction="none"
        ).sum()

        box_losses[batch_i] = box_loss

    box_loss = box_losses.sum() / positive_match.sum()

    total_loss = classification_loss + box_loss

    # Take smooth l1 loss for box encoding
    # Take cross-entropy loss for classification
    # Take binary cross-entropy loss for mask
    # Take binary cross-entropy loss for point
    # Take smooth l1 loss for direction

    return total_loss, (classification_loss, box_loss)