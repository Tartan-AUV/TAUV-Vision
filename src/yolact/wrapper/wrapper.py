import rospy
import numpy as np
import torch
import torchvision.transforms as T
from spatialmath import SE3, SO3
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros as tf2
from typing import Dict
from message_filters import ApproximateTimeSynchronizer, Subscriber
from functools import partial
from transform_client import TransformClient
import torch.nn.functional as F
import cv2
import io
import matplotlib.pyplot as plt

from tauv_util.cameras import CameraIntrinsics
from tauv_util.spatialmath import ros_transform_to_se3, r3_to_ros_vector3, r3_to_ros_point
from tauv_msgs.msg import FeatureDetection, FeatureDetections

from yolact.model.model import Yolact
from yolact.model.config import Config
from yolact.model.boxes import box_decode
from yolact.model.masks import assemble_mask
from yolact.model.nms import class_nms
from yolact.utils.plot import plot_detection


config = Config(
    in_w=1280,
    in_h=720,
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

classification_names = {0: "torpedo_22", 1: "torpedo_22_circle", 2: "torpedo_22_trapezoid"}

img_mean = (0.485, 0.456, 0.406)
img_stddev = (0.229, 0.224, 0.225)


def fig_to_np(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    img_cv2 = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), -1)
    return img_cv2


class Wrapper:

    def __init__(self):
        self._load_config()

        self._tf_client: TransformClient = TransformClient()

        self._cv_bridge: CvBridge = CvBridge()

        self._device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Yolact = Yolact(config).to(self._device)

        rospy.loginfo(f"Loading weights from {self._weight_path}...")
        self._model.load_state_dict(torch.load(self._weight_path, map_location=self._device))
        rospy.loginfo("Done loading weights.")

        self._camera_infos: Dict[str, CameraInfo] = {}
        self._intrinsics: Dict[str, CameraIntrinsics] = {}
        self._synchronizers: Dict[str, ApproximateTimeSynchronizer] = {}

        for frame_id in self._frame_ids:
            self._camera_infos[frame_id] = rospy.wait_for_message(f"vehicle/{frame_id}/depth/camera_info", CameraInfo, 60)
            self._intrinsics[frame_id] = CameraIntrinsics.from_matrix(np.array(self._camera_infos[frame_id].K))

            color_sub = Subscriber(f"vehicle/{frame_id}/color/image_raw", Image)
            depth_sub = Subscriber(f"vehicle/{frame_id}/depth/image_raw", Image)

            synchronizer = ApproximateTimeSynchronizer(
                [color_sub, depth_sub],
                queue_size=10,
                slop=0.5,
            )
            synchronizer.registerCallback(partial(self._handle_imgs, frame_id=frame_id))

            self._synchronizers[frame_id] = synchronizer

        rospy.loginfo("Set up subscribers")

        self._detections_image_pub: rospy.Publisher = rospy.Publisher("detections_image", Image, queue_size=10)

        self._detections_pub: rospy.Publisher = \
            rospy.Publisher("global_map/feature_detections", FeatureDetections, queue_size=10)

        rospy.loginfo("Set up publishers")

    def start(self):
        rospy.spin()

    def _handle_imgs(self, color_msg: Image, depth_msg: Image, frame_id: str):
        rospy.loginfo("Yolact got images")

        color_np = self._cv_bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
        depth = self._cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="mono16")
        depth = depth.astype(float) / 1000

        img_raw = T.ToTensor()(color_np)
        img = T.Normalize(mean=img_mean, std=img_stddev)(img_raw.unsqueeze(0)).to(self._device)

        rospy.loginfo("Calculating prediction...")

        prediction = self._model(img)

        rospy.loginfo("Got prediction")

        classification, box_encoding, mask_coeff, anchor, mask_prototype = prediction
        box = box_decode(box_encoding, anchor)
        classification_max = torch.argmax(classification[0], dim=-1).squeeze(0)
        detections = class_nms(classification, box)
        if len(detections) == 0:
            rospy.loginfo("No detections")
            return
        mask = assemble_mask(mask_prototype[0], mask_coeff[0, detections], box[0, detections])
        mask = F.interpolate(mask.unsqueeze(0), (img.size(2), img.size(3))).squeeze(0)

        detection_fig = plot_detection(
            img_raw,
            classification_max[detections],
            box[0, detections],
            None,
            None,
            None
        )

        detection_fig_np = fig_to_np(detection_fig)
        plt.close(detection_fig)

        detection_fig_msg = self._cv_bridge.cv2_to_imgmsg(detection_fig_np, encoding="rgba8")
        self._detections_image_pub.publish(detection_fig_msg)

        world_frame = f"{self._tf_namespace}/odom"
        camera_frame = f"{self._tf_namespace}/{frame_id}"

        rospy.loginfo(f"{world_frame} to {camera_frame}")

        world_t_cam = None
        while world_t_cam is None:
            try:
                world_t_cam = self._tf_client.get_a_to_b(world_frame, camera_frame, color_msg.header.stamp)
            except Exception as e:
                rospy.logwarn(e)
                rospy.logwarn("Failed to get transform")

        rospy.loginfo("Got transforms")

        detection_array_msg = FeatureDetections()
        detection_array_msg.detector_tag = "yolact"

        for detection_i, detection in enumerate(detections):
            cam_t_detection = SE3()
            world_t_detection = world_t_cam * cam_t_detection
            classification_name = classification_names[int(classification_max[detection])]

            mask_np = mask[detection_i].detach().cpu().numpy()

            mean_depth = np.nanmean(np.where(mask_np > 0.5, depth, np.nan))

            print(f"mean depth: {mean_depth}")

            detection_msg = FeatureDetection()
            detection_msg.confidence = 1
            detection_msg.tag = classification_name
            detection_msg.SE2 = False
            detection_msg.position = r3_to_ros_point(world_t_detection.t)
            rpy = world_t_detection.rpy()
            rpy[0:2] = 0
            detection_msg.orientation = r3_to_ros_point(rpy)

            detection_array_msg.detections.append(detection_msg)

        rospy.loginfo("Publishing detections")

        self._detections_pub.publish(detection_array_msg)

    def _load_config(self):
        self._frame_ids: [str] = rospy.get_param("~frame_ids")
        self._tf_namespace: str = rospy.get_param("tf_namespace")

        self._weight_path: str = rospy.get_param("~weight_path")


def main():
    rospy.init_node('yolact')
    n = Wrapper()
    n.start()


if __name__ == "__main__":
    main()