import torch


def box_to_corners(box: torch.Tensor) -> torch.Tensor:
    # box is y, x, h, w
    # size of box is n_batch, n, 4
    # corners is min_y, min_x, max_y, max_x
    # size of corners is n_batch, n, 4
    pass


def corners_to_box(box: torch.Tensor) -> torch.Tensor:
    # box is y, x, h, w
    # size of box is n_batch, n, 4
    # corners is min_y, min_x, max_y, max_x
    # size of corners is n_batch, n, 4
    pass
