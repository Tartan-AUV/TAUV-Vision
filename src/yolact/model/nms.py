import torch


def class_nms(classification: torch.Tensor, box: torch.Tensor):
    classification_max = torch.argmax(classification, dim=-1).squeeze(0)
    detection = classification_max.nonzero().squeeze(-1)

    return detection
