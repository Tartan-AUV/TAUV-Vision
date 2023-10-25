import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from math import sqrt, ceil


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


if __name__ == "__main__":
    prototype_fig = plot_prototype(torch.rand((52, 32, 32)))
    plt.show()