import torch
import torch.nn.functional as F


def assemble_mask(mask_prototype: torch.Tensor, mask_coeff: torch.Tensor) -> torch.Tensor:

    mask = torch.zeros(mask_coeff.size(0), mask_prototype.size(1), mask_prototype.size(2), device=mask_prototype.device)

    for i in range(mask_coeff.size(0)):
        mask[i] = torch.sum(mask_coeff[i].unsqueeze(1).unsqueeze(2) * mask_prototype, dim=0)

    mask = F.sigmoid(mask)

    return mask
