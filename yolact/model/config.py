from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Config:
    in_w: int
    in_h: int

    feature_depth: int

    n_classes: int
    n_prototype_masks: int

    n_masknet_layers_pre_upsample: int
    n_masknet_layers_post_upsample: int
    n_prediction_head_layers: int
    n_fpn_downsample_layers: int

    anchor_scales: (int, ...)
    anchor_aspect_ratios: (int, ...)

    iou_pos_threshold: float
    iou_neg_threshold: float

    negative_example_ratio: int
