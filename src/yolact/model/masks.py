import torch
import torch.nn.functional as F

from yolact.model.boxes import box_to_mask


def assemble_mask(mask_prototype: torch.Tensor, mask_coeff: torch.Tensor, box: torch.Tensor) -> torch.Tensor:

    mask = torch.zeros(mask_coeff.size(0), mask_prototype.size(1), mask_prototype.size(2), device=mask_prototype.device)

    for i in range(mask_coeff.size(0)):
        mask[i] = torch.sum(mask_coeff[i].unsqueeze(1).unsqueeze(2) * mask_prototype, dim=0)

        box_mask = box_to_mask(box[i], mask[i].size())

        mask[i] *= box_mask

    mask = F.sigmoid(mask)

    return mask
