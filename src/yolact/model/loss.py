import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

from yolact.model.boxes import box_decode, box_encode, iou_matrix, box_to_mask
from yolact.model.config import Config


def loss(prediction: (torch.Tensor, ...), truth: (torch.Tensor, ...), config: Config) -> (torch.Tensor, (torch.Tensor, ...)):
    classification, box_encoding, mask_coeff, anchor, mask_prototype = prediction
    truth_valid, truth_classification, truth_box, truth_seg_map = truth

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

        # alpha = 0.25
        # gamma = 2
        #
        # log_p = F.log_softmax(classification[batch_i], dim=-1)
        # ce = F.nll_loss(log_p, match_classification)
        #
        # log_pt = log_p[torch.arange(log_p.size(0)), match_classification]
        # pt = torch.clamp(log_pt.exp(), min=1e-4)
        #
        # at = (1 - alpha) * (negative_match[batch_i].float()) + alpha * positive_match[batch_i].float()
        # focal_loss = at * torch.clamp((1 - pt) ** gamma, min=1e-4) * ce

        # bce_loss = F.binary_cross_entropy_with_logits(classification[batch_i], F.one_hot(match_classification, num_classes=config.n_classes + 1).to(torch.float32), reduction="none")
        #
        # gamma = 2
        # modulator = torch.exp(-gamma * F.one_hot(match_classification, num_classes=config.n_classes + 1) * classification[batch_i] - gamma * torch.log(torch.clamp(1 + torch.exp(-1 * classification[batch_i]), 1e-4)))
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
                truth_match_mask = (truth_seg_map[batch_i] == match_index[batch_i, match_i]).float()
                truth_match_mask_resized = F.interpolate(
                    truth_match_mask.unsqueeze(0).unsqueeze(0),
                    match_mask.size(),
                    mode="bilinear",
                ).squeeze(0).squeeze(0)

            if truth_match_mask_resized.sum() == 0:
                continue

            mask_cross_entropy = F.binary_cross_entropy(
                torch.clamp(match_mask.reshape(-1), 1e-4, 1-1e-4),
                torch.clamp(truth_match_mask_resized.reshape(-1), 1e-4, 1-1e-4),
                reduction="none",
            )

            box_mask = box_to_mask(
                truth_box[batch_i, match_index[batch_i, match_i]],
                match_mask.size()
            )

            mask_loss += (box_mask.reshape(-1) * mask_cross_entropy).sum() / truth_match_mask_resized.sum()

        mask_losses[batch_i] = mask_loss

    mask_loss = mask_losses.sum() / positive_match.sum()

    total_loss = classification_loss + 100 * box_loss + mask_loss
    # total_loss = classification_loss

    return total_loss, (classification_loss, box_loss, mask_loss)
