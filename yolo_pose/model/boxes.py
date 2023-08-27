import torch


def box_to_corners(box: torch.Tensor) -> torch.Tensor:
    # box is y, x, h, w
    # size of box is n_batch, n, 4
    # corners is min_y, min_x, max_y, max_x
    # size of corners is n_batch, n, 4

    corners = torch.stack((
        box[:, :, 0] - (box[:, :, 2] / 2),
        box[:, :, 1] - (box[:, :, 3] / 2),
        box[:, :, 0] + (box[:, :, 2] / 2),
        box[:, :, 1] + (box[:, :, 3] / 2),
    ), dim=-1)
    return corners


def corners_to_box(corners: torch.Tensor) -> torch.Tensor:
    # box is y, x, h, w
    # size of box is n_batch, n, 4
    # corners is min_y, min_x, max_y, max_x
    # size of corners is n_batch, n, 4

    box = torch.stack((
        (corners[:, :, 0] + corners[:, :, 2]) / 2,
        (corners[:, :, 1] + corners[:, :, 3]) / 2,
        corners[:, :, 2] - corners[:, :, 0],
        corners[:, :, 3] - corners[:, :, 1],
    ), dim=-1)
    return box


def box_encode(box: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    box_encoding = torch.cat((
        box[:, :, 0:2] - anchor[:, :, 0:2],
        torch.log(box[:, :, 2:4] / anchor[:, :, 2:4]),
    ), dim=-1)
    return box_encoding


def box_decode(box_encoding: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    box = torch.cat((
        box_encoding[:, :, 0:2] + anchor[:, :, 0:2],
        torch.exp(box_encoding[:, :, 2:4]) * anchor[:, :, 2:4],
    ), dim=-1)
    return box


def iou_matrix(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    corners_a = box_to_corners(box_a)
    corners_b = box_to_corners(box_b)

    intersect_y_min = torch.max(corners_a[:, :, 0].unsqueeze(2), corners_b[:, :, 0].unsqueeze(1))
    intersect_x_min = torch.max(corners_a[:, :, 1].unsqueeze(2), corners_b[:, :, 1].unsqueeze(1))
    intersect_y_max = torch.min(corners_a[:, :, 2].unsqueeze(2), corners_b[:, :, 2].unsqueeze(1))
    intersect_x_max = torch.min(corners_a[:, :, 3].unsqueeze(2), corners_b[:, :, 3].unsqueeze(1))

    intersect_h = torch.clamp(intersect_y_max - intersect_y_min, min=0)
    intersect_w = torch.clamp(intersect_x_max - intersect_x_min, min=0)

    intersect_area = intersect_h * intersect_w

    area_a = box_a[:, :, 2] * box_a[:, :, 3]
    area_b = box_b[:, :, 2] * box_b[:, :, 3]

    union_area = (area_a.unsqueeze(2) + area_b.unsqueeze(1)) - intersect_area

    iou = intersect_area / union_area

    return iou


def box_to_mask(box, img_size):
    y_grid = torch.arange(0, img_size[0], dtype=torch.float, device=box.device)
    x_grid = torch.arange(0, img_size[1], dtype=torch.float, device=box.device)
    x_coords, y_coords = torch.meshgrid(x_grid, y_grid)

    box = box * torch.tensor([img_size[0], img_size[1], img_size[0], img_size[1]], device=box.device)

    left = box[1] - box[3] / 2
    right = box[1] + box[3] / 2
    top = box[0] - box[2] / 2
    bottom = box[0] + box[2] / 2

    mask = (x_coords >= left) & (x_coords <= right) & (y_coords >= top) & (y_coords <= bottom)
    mask = mask.float()

    return mask


def main():
    box = torch.rand((1, 1, 4))
    anchor = torch.rand((1, 1, 4))

    assert torch.allclose(box, corners_to_box(box_to_corners(box)))

    assert torch.allclose(box, box_decode(box_encode(box, anchor), anchor))

    box_a = torch.rand((1, 1, 4))
    box_b = torch.rand((1, 1, 4))

    iou = iou_matrix(box_a, box_b)


if __name__ == "__main__":
    main()