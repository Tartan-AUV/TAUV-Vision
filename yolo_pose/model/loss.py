import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from yolo_pose.model.boxes import box_decode, iou_matrix, box_to_mask
from yolo_pose.model.config import Config


def loss(prediction: (torch.Tensor, ...), truth: (torch.Tensor, ...), config: Config) -> (torch.Tensor, (torch.Tensor, ...)):
    classification, box_encoding, mask_coeff, point_coeff, anchor, mask_prototype, point_prototype, direction_prototype = prediction
    truth_valid, truth_classification, truth_box, truth_mask, truth_point, truth_direction = truth

    device = classification.device

    n_batch = truth_classification.size(0)

    # What are the shapes of all of these things?

    box = box_decode(box_encoding, anchor)
    # truth_box_encoding = box_encode(truth_box, anchor)

    iou = iou_matrix(anchor, truth_box)

    match_iou, match_index = torch.max(iou * truth_valid.unsqueeze(1).float(), dim=2)

    # TODO: Handle case where there are no positive matches
    positive_match = match_iou >= config.iou_pos_threshold
    negative_match = match_iou <= config.iou_neg_threshold
    neutral_match = ~(positive_match | negative_match)

    classification_losses = torch.zeros(n_batch, device=device)

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

        classification_loss = (selected_match.float() * classification_cross_entropy).sum()

        classification_losses[batch_i] = classification_loss

    classification_loss = classification_losses.sum() / ((1 + config.negative_example_ratio) * positive_match.sum())

    box_losses = torch.zeros(n_batch, device=device)

    for batch_i in range(n_batch):
        # TODO: Check if loss should be over encoding or decoding
        box_loss = F.smooth_l1_loss(
            box[batch_i, positive_match[batch_i]],
            truth_box[batch_i, match_index[batch_i, positive_match[batch_i]]],
            reduction="none"
        ).sum()

        box_losses[batch_i] = box_loss

    box_loss = box_losses.sum() / positive_match.sum()

    mask_losses = torch.zeros(n_batch, device=device)

    for batch_i in range(n_batch):
        mask_loss = torch.tensor(0, dtype=torch.float, device=device)

        for match_i in positive_match[batch_i].nonzero():
            match_i = int(match_i)
            mask = torch.sum(mask_coeff[batch_i, match_i].unsqueeze(1).unsqueeze(2) * mask_prototype[batch_i], dim=0)
            mask = F.sigmoid(mask)

            match_mask = truth_mask[batch_i, match_index[batch_i, match_i]].float()
            truth_mask_resized = F.interpolate(
                match_mask.unsqueeze(0).unsqueeze(0),
                mask.size(),
                mode="bilinear",
            ).squeeze(0).squeeze(0)

            mask_cross_entropy = F.binary_cross_entropy(
                mask,
                truth_mask_resized,
                reduction="none",
            )

            box_mask = box_to_mask(
                truth_box[batch_i, match_index[batch_i, match_i]],
                mask_cross_entropy.size()
            )

            mask_loss += (box_mask * mask_cross_entropy).sum() / box_mask.sum()

        mask_losses[batch_i] = mask_loss

    mask_loss = mask_losses.sum() / positive_match.sum()

    point_losses = torch.zeros(n_batch, device=device)

    for batch_i in range(n_batch):
        point_loss = torch.tensor(0, dtype=torch.float, device=device)

        for match_i in positive_match[batch_i].nonzero():
            match_i = int(match_i)

            predicted_point = point_prototype[batch_i, torch.argmax(point_coeff[batch_i, match_i], dim=-1)]

            match_point = truth_point[batch_i, match_index[batch_i, match_i]]
            match_point = F.interpolate(
                match_point.unsqueeze(0),
                predicted_point.size()[1:],
                mode="bilinear",
            ).squeeze(0)

            point_cross_entropy = F.binary_cross_entropy(
                predicted_point,
                match_point,
                reduction="none"
            )

            point_loss += point_cross_entropy.sum() / (match_point.size(1) * match_point.size(2))

        point_losses[batch_i] = point_loss

    point_loss = point_losses.sum() / positive_match.sum()

    direction_losses = torch.zeros(n_batch, device=device)

    for batch_i in range(n_batch):
        direction_loss = torch.tensor(0, dtype=torch.float, device=device)

        for match_i in positive_match[batch_i].nonzero():
            match_i = int(match_i)

            predicted_direction = direction_prototype[batch_i, torch.argmax(point_coeff[batch_i, match_i], dim=-1)[1:]]

            match_direction = truth_direction[batch_i, match_index[batch_i, match_i]]
            match_direction = F.interpolate(
                match_direction.unsqueeze(0),
                predicted_direction.size()[1:],
                mode="bilinear"
            ).squeeze(0)

            match_l1 = F.smooth_l1_loss(
                predicted_direction,
                match_direction,
                reduction="none"
            )

            direction_loss += match_l1.sum() / (match_direction.size(1) * match_direction.size(2))

        direction_losses[batch_i] = direction_loss

    direction_loss = direction_losses.sum() / positive_match.sum()

    total_loss = classification_loss + box_loss + mask_loss + point_loss + direction_loss

    return total_loss, (classification_loss, box_loss, mask_loss, point_loss, direction_loss)
