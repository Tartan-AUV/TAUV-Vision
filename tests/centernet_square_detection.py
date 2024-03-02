from typing import Tuple
import random
from math import pi, floor, sin, cos

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.transforms as tfs
import matplotlib.pyplot as plt

from tauv_vision.centernet.model.centernet import Centernet, initialize_weights
from tauv_vision.centernet.model.backbones.dla import DLABackbone
from tauv_vision.centernet.model.loss import gaussian_splat, focal_loss, angle_loss, angle_decode

torch.autograd.set_detect_anomaly(True)


def generate_sample(img_h: int, img_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # img = torch.rand((3, img_h, img_w))

    w = random.randint(50, 150)
    h = w
    # h = random.randint(50, 150)
    # w = 100
    # h = 100

    y = random.randint(h, img_h - h - 1)
    x = random.randint(w, img_w - w - 1)
    # y = img_h // 2
    # x = img_w // 2

    theta = random.uniform(0, pi / 2)
    # theta = 0

    thickness = floor(0.2 * (w + h) / 2)

    img_np = (10 * np.random.rand(img_h, img_w, 3)).astype(np.uint8)
    rot_matrix = np.array([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
    ])
    corners = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ])
    rotated_corners = (corners @ np.transpose(rot_matrix)) + np.array([[x, y]])
    # box = cv2.boxPoints(((x, y), (w, h), np.).astype(np.intp)
    cv2.drawContours(img_np, [rotated_corners.astype(np.intp)], 0, (255, 0, 0), thickness)

    img = torch.from_numpy(img_np).permute(2, 0, 1).to(torch.float)

    # img[:, y:y + h + 1, x:x + w + 1] = torch.Tensor([1, 0, 0]).unsqueeze(1).unsqueeze(2)

    sigma = 0.05 * (w + h) / 2
    # truth_heatmap = gaussian_splat(img_h // 4, img_w // 4, y // 4, x // 4, sigma)
    truth_heatmap = gaussian_splat(img_h // 2, img_w // 2, int(y / 2), int(x / 2), sigma)

    return img, truth_heatmap, w, h, theta


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    heights = [2, 2, 2, 2]
    channels = [32, 32, 32, 32, 32]
    out_channels = [1, 2, 4, 4]

    n_batches = 1000
    batch_size = 32

    in_h = 512
    in_w = 512

    alpha = 2
    beta = 4

    dla_backbone = DLABackbone(heights, channels)
    centernet = Centernet(dla_backbone, out_channels).to(device)
    centernet.train()
    initialize_weights(centernet, [])

    optimizer = torch.optim.Adam(centernet.parameters(), lr=1e-3)

    for batch_i in range(n_batches):
        img = torch.zeros((batch_size, 3, in_h, in_w), device=device)
        truth_heatmap = torch.zeros(batch_size, 1, in_h // 2, in_w // 2, device=device)
        w = torch.zeros(batch_size, device=device)
        h = torch.zeros(batch_size, device=device)
        theta = torch.zeros(batch_size, device=device)

        for sample_i in range(batch_size):
            sample_img, sample_truth_heatmap, sample_w, sample_h, sample_theta = generate_sample(in_h, in_w)
            sample_img = tfs.Normalize((0.5, 0.5, 0.5), (1, 1, 1))(sample_img)
            img[sample_i] = sample_img
            truth_heatmap[sample_i] = sample_truth_heatmap.unsqueeze(0)
            w[sample_i] = sample_w
            h[sample_i] = sample_h
            theta[sample_i] = sample_theta

        optimizer.zero_grad()

        prediction = centernet(img)
        predicted_heatmap = prediction[0]
        predicted_h = prediction[1][:, 0]
        predicted_w = prediction[1][:, 1]
        predicted_theta_bin = prediction[2].permute(0, 2, 3, 1)[truth_heatmap.squeeze(1) == 1].unsqueeze(1)
        predicted_theta_offset = prediction[3].permute(0, 2, 3, 1)[truth_heatmap.squeeze(1) == 1].unsqueeze(1)
        predicted_heatmap = F.sigmoid(predicted_heatmap)

        fl = focal_loss(predicted_heatmap, truth_heatmap, alpha, beta)
        sl = F.l1_loss(predicted_h[truth_heatmap.squeeze(1) == 1], h, reduction="none") + F.l1_loss(predicted_w[truth_heatmap.squeeze(1) == 1], w, reduction="none")
        al = angle_loss(predicted_theta_bin, predicted_theta_offset, theta.unsqueeze(1), pi / 2, pi / 3)

        print(f"truth: {w}")
        print(f"predicted: {predicted_w[truth_heatmap.squeeze(1) == 1]}")
        print(f"error: {(torch.abs(w - predicted_w[truth_heatmap.squeeze(1) == 1]))}")

        print(f"truth: {theta}")
        print(f"predicted: {angle_decode(predicted_theta_bin, predicted_theta_offset, pi / 2, pi / 3).squeeze(1)}")
        print(f"error: {(torch.abs(theta - angle_decode(predicted_theta_bin, predicted_theta_offset, pi / 2, pi / 3).squeeze(1))) % (2 * pi)}")

        # loss = fl.sum() + 0.1 * sl.sum() + 0.1 * al.sum()
        loss = fl.sum() + 0.1 * sl.mean() + al.mean()
        # loss = fl.sum() + 0.1 * sl.sum()
        # loss = fl.sum() + 0.1 * (sl.sum() + al.sum())
        # loss = fl.sum()

        print(f"focal loss: {float(fl.sum())}, size loss: {float(sl.sum())}, angle loss: {float(al.sum())} total loss: {float(loss)}")
        # loss = F.mse_loss(predicted_heatmap, truth_heatmap, reduction="none")
        if batch_i % 10 == 0:
            plt.figure()
            plt.imshow(predicted_heatmap[0, 0].detach().cpu())
            plt.show()

        loss.sum().backward()

        optimizer.step()


if __name__ == "__main__":
    main()