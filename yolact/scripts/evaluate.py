import torch
import torchvision.transforms.v2 as T
from PIL import Image
import pathlib
import matplotlib.pyplot as plt

from yolact.model.config import Config
from yolact.model.model import Yolact
from yolact.utils.plot import save_plot, plot_prototype, plot_mask, plot_detection
from yolact.model.boxes import box_decode
from yolact.model.masks import assemble_mask


config = Config(
    in_w=960, # TODO: THESE ARE WRONG!
    in_h=480,
    feature_depth=256,
    n_classes=3,
    n_prototype_masks=32,
    n_masknet_layers_pre_upsample=1,
    n_masknet_layers_post_upsample=1,
    n_prediction_head_layers=1,
    n_fpn_downsample_layers=2,
    anchor_scales=(24, 48, 96, 192, 384),
    anchor_aspect_ratios=(1 / 2, 1, 2),
    iou_pos_threshold=0.5,
    iou_neg_threshold=0.4,
    negative_example_ratio=3,
)

img_mean = (0.485, 0.456, 0.406)
img_stddev = (0.229, 0.224, 0.225)

img_path = pathlib.Path("~/Documents/yolo_pose/img/torpedo-target-frc-2.png").expanduser()
weights_path = pathlib.Path("~/Documents/yolo_pose/weights/10.pt").expanduser()

def main():
    model = Yolact(config)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))

    model.eval()

    img_pil = Image.open(img_path).convert("RGB")

    img_raw = T.ToTensor()(img_pil).unsqueeze(0)
    img = T.Normalize(mean=img_mean, std=img_stddev)(img_raw)

    prediction = model(img)
    classification, box_encoding, mask_coeff, anchor, mask_prototype = prediction

    box = box_decode(box_encoding, anchor)

    # TODO: Implement NMS for detections here

    for sample_i in range(img.size(0)):
        classification_max = torch.argmax(classification[sample_i], dim=-1).squeeze(0)
        detection = classification_max.nonzero().squeeze(-1)

        plot_prototype(mask_prototype[sample_i])

        mask = assemble_mask(mask_prototype[sample_i], mask_coeff[sample_i, detection])
        plot_mask(None, mask)
        plot_mask(img_raw[sample_i], mask)

        plot_detection(
            img_raw[sample_i],
            classification_max[detection],
            box[sample_i, detection],
            None,
            None,
            None
        )

    plt.show()

if __name__ == "__main__":
    main()
