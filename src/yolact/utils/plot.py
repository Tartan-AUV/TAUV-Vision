import torch
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from math import sqrt, ceil
from typing import Optional
import torch.nn.functional as F
from typing import Optional
from yolact.model.boxes import box_to_corners
import pathlib


def save_plot(fig: plt.Figure, save_dir: Optional[pathlib.Path], name: str):
    if save_dir is not None:
        save_path = save_dir / name
        fig.savefig(save_path)
    else:
        fig.show()


def plot_prototype(prototype: torch.Tensor) -> plt.Figure:
    # protoype is n_prototypes x h x w
    fig = plt.figure()

    depth = prototype.size(0)
    nrows = int(ceil(sqrt(depth)))

    grid = ImageGrid(fig,111, nrows_ncols=(nrows, nrows), share_all=True, cbar_mode="single", axes_pad=0.2, cbar_pad=0.5)

    for i in range(depth):
        img = grid[i].imshow(prototype[i].detach().cpu())

    grid.cbar_axes[0].colorbar(img)

    return fig


def plot_detection(img: torch.Tensor,
                   classification: torch.Tensor, box: torch.Tensor,
                   truth_valid: Optional[torch.Tensor], truth_classification: Optional[torch.Tensor], truth_box: Optional[torch.Tensor]) -> plt.Figure:
    fig = plt.figure()

    plt.imshow(img.permute(1, 2, 0).detach().cpu())

    size = torch.Tensor([img.size(1), img.size(2), img.size(1), img.size(2)]).to(box.device)
    corners = box_to_corners(box.unsqueeze(0)).squeeze(0) * size

    cmap = matplotlib.colormaps.get_cmap("tab10")

    for i in range(corners.size(0)):
        y0, x0, y1, x1 = corners[i].detach().cpu()
        rect = plt.Rectangle(
            (x0, y0), (x1 - x0), (y1 - y0),
            linewidth=1,
            linestyle="solid",
            edgecolor=cmap(int(classification[i])),
            facecolor="none"
        )
        plt.gca().add_patch(rect)

    if truth_valid is not None:
        truth_corners = box_to_corners(truth_box.unsqueeze(0)).squeeze(0) * size

        for i in range(truth_corners.size(0)):
            if not truth_valid[i]:
                continue

            y0, x0, y1, x1 = truth_corners[i].detach().cpu()
            rect = plt.Rectangle(
                (x0, y0), (x1 - x0), (y1 - y0),
                linewidth=1,
                linestyle="dashed",
                edgecolor=cmap(int(truth_classification[i])),
                facecolor="none"
            )
            plt.gca().add_patch(rect)

    return fig


def plot_mask(img: Optional[torch.Tensor], mask: torch.Tensor, opacity: float = 0.1) -> plt.Figure:
    mask_depth = mask.size(0)
    mask_size = mask.size()[1:3]

    if img is not None:
        img_resized = F.interpolate(img.unsqueeze(0), mask_size, mode="bilinear").squeeze(0)
        overlay = (mask.unsqueeze(1) * img_resized.unsqueeze(0)) + opacity * img_resized.unsqueeze(0)

    else:
        overlay = mask

    overlay = torch.clamp(overlay, 0, 1)

    fig = plt.figure()

    nrows = int(ceil(sqrt(mask_depth)))
    grid = ImageGrid(fig,111, nrows_ncols=(nrows, nrows), share_all=True, cbar_mode="single", axes_pad=0.2, cbar_pad=0.5)

    for i in range(mask_depth):
        if len(overlay.size()) == 4:
            im = grid[i].imshow(overlay[i].permute(1, 2, 0).detach().cpu())
        else:
            im = grid[i].imshow(overlay[i].detach().cpu())

    grid.cbar_axes[0].colorbar(im)

    return fig


if __name__ == "__main__":
    prototype_fig = plot_prototype(torch.rand((52, 32, 32)))
    prototype_fig.set_size_inches(10, 10)
    plt.show()

    detection_fig = plot_detection(
        torch.rand((3, 720, 1280)),
        torch.Tensor([1, 2, 3]),
        torch.Tensor([
            [0.5, 0.5, 0.5, 0.5],
            [0.3, 0.6, 0.2, 0.2],
            [0.6, 0.3, 0.3, 0.2],
        ]),
        torch.Tensor([1]),
        torch.Tensor([
            [0.5, 0.5, 0.52, 0.35],
        ]),
    )
    plt.show()

    mask_fig_1 = plot_mask(
        torch.rand((3, 720, 1280)),
        torch.rand((10, 360, 640)),
    )
    plt.show()

    mask_fig_2 = plot_mask(
        None,
        torch.rand((10, 360, 640)),
    )
    plt.show()


