import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from yolo_pose.model.boxes import box_decode, iou_matrix, box_to_mask
from yolo_pose.model.config import Config


def loss(prediction: (torch.Tensor, ...), truth: (torch.Tensor, ...), config: Config) -> (torch.Tensor, (torch.Tensor, ...)):
    classification, box_encoding, mask_coeff, position_map_coeff, anchor, mask_prototype, position_map_prototype = prediction
    truth_valid, truth_classification, truth_box, truth_seg_map, truth_position_map = truth

    device = classification.device

    n_batch = truth_classification.size(0)

    box = box_decode(box_encoding, anchor)

    iou = iou_matrix(anchor, truth_box)

    match_iou, match_index = torch.max(iou * truth_valid.unsqueeze(1).float(), dim=2)

    # TODO: Handle case where there are no positive matches
    positive_match = match_iou >= config.iou_pos_threshold
    negative_match = match_iou <= config.iou_neg_threshold

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

        background_confidence = F.softmax(classification[batch_i], dim=-1)[:, 0]

        # Maybe change how I'm selecting these negative matches?
        _, selected_negative_match_index = torch.topk(
            torch.where(negative_match[batch_i], -background_confidence, -torch.inf),
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
            match_mask = torch.sum(mask_coeff[batch_i, match_i].unsqueeze(1).unsqueeze(2) * mask_prototype[batch_i], dim=0)
            match_mask = F.sigmoid(match_mask)
            match_mask = torch.clamp(match_mask, min=1e-4)

            with torch.no_grad():
                truth_match_mask = (truth_seg_map[batch_i] == truth_classification[batch_i, match_index[batch_i, match_i]]).float()
                truth_match_mask_resized = F.interpolate(
                    truth_match_mask.unsqueeze(0).unsqueeze(0),
                    match_mask.size(),
                    mode="bilinear",
                ).squeeze(0).squeeze(0)

            if truth_match_mask_resized.sum() == 0:
                continue

            mask_cross_entropy = F.binary_cross_entropy(
                match_mask.reshape(-1),
                truth_match_mask_resized.reshape(-1),
                reduction="none",
            )

            box_mask = box_to_mask(
                truth_box[batch_i, match_index[batch_i, match_i]],
                match_mask.size()
            )

            mask_loss += (box_mask.reshape(-1) * mask_cross_entropy).sum() / truth_match_mask_resized.sum()

        mask_losses[batch_i] = mask_loss

    mask_loss = mask_losses.sum() / positive_match.sum()

    position_map_losses = torch.zeros(n_batch, device=device)

    for batch_i in range(n_batch):
        position_map_loss = torch.tensor(0, dtype=torch.float, device=device)

        for match_i in positive_match[batch_i].nonzero():
            match_i = int(match_i)

            match_position_map = torch.sum(position_map_coeff[batch_i, match_i].unsqueeze(1).unsqueeze(2).unsqueeze(3) * position_map_prototype[batch_i], dim=0)

            with torch.no_grad():
                truth_match_mask = (truth_seg_map[batch_i] == truth_classification[
                    batch_i, match_index[batch_i, match_i]]).float()
                truth_match_mask_resized = F.interpolate(
                    truth_match_mask.unsqueeze(0).unsqueeze(0),
                    match_position_map.size()[1:3],
                    mode="nearest",
                ).squeeze(0).squeeze(0)
                truth_match_position_map = truth_position_map[batch_i]
                truth_match_position_map_resized = F.interpolate(
                    truth_match_position_map.unsqueeze(0),
                    match_position_map.size()[1:3],
                    mode="nearest",
                ).squeeze(0)
                truth_match_position_map_resized = Variable(truth_match_position_map_resized, requires_grad=False)

            if truth_match_mask_resized.sum() == 0:
                continue

            l1_loss = F.smooth_l1_loss(
                match_position_map.reshape(-1),
                truth_match_position_map_resized.reshape(-1),
                reduction="none"
            )
            l1_loss = l1_loss.reshape(match_position_map.size())

            position_map_loss += (truth_match_mask_resized.unsqueeze(0) * l1_loss).sum() / (3 * truth_match_mask_resized.sum())

        position_map_losses[batch_i] = position_map_loss

    position_map_loss = position_map_losses.sum() / positive_match.sum()

    total_loss = classification_loss + 10 * box_loss + mask_loss + position_map_loss

    return total_loss, (classification_loss, box_loss, mask_loss, position_map_loss)
