import rospy
import numpy as np
import torch
from spatialmath import SE3, SO3
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from functools import partial
from typing import Dict
import pathlib
from math import pi
import torchvision.transforms as T
import cv2

from transform_client import TransformClient
from tauv_util.cameras import CameraIntrinsics
from tauv_util.spatialmath import ros_transform_to_se3, r3_to_ros_vector3, r3_to_ros_point
from tauv_msgs.msg import FeatureDetection, FeatureDetections
from tauv_vision.centernet.model.centernet import Centernet
from tauv_vision.centernet.model.backbones.dla import DLABackbone
from tauv_vision.centernet.model.config import ObjectConfig, ObjectConfigSet, AngleConfig, ModelConfig, TrainConfig
from tauv_vision.centernet.model.decode import decode_keypoints


model_config = ModelConfig(
    in_h=360,
    in_w=640,
    backbone_heights=[2, 2, 2, 2, 2, 2],
    backbone_channels=[128, 128, 128, 128, 128, 128, 128],
    downsamples=1,
    angle_bin_overlap=pi / 3,
)

object_config = ObjectConfigSet(
    configs=[
        ObjectConfig(
            id="torpedo_22_trapezoid",
            yaw=AngleConfig(
                train=False,
                modulo=2 * pi,
            ),
            pitch=AngleConfig(
                train=False,
                modulo=2 * pi,
            ),
            roll=AngleConfig(
                train=False,
                modulo=2 * pi,
            ),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0.0, 0.095, 0.105),
                (0.0, 0.095, -0.105),
                (0.0, -0.12, -0.06),
                (0.0, -0.12, 0.06),
                (0.0, 0.509, 0.432),
                (0.0, 0.223, 0.337),
                (0.0, 0.398, 0.207),
                (0.0, 0.334, 0.063),
                (0.0, -0.112, 0.278),
                (0.0, 0.269, -0.062),
            ],
        ),
    ]
)

object_t_detections: Dict[str, SE3] = {
    "torpedo_22_trapezoid": SE3(SO3.TwoVectors(x="-x", z="-y")),
}

weights_path = pathlib.Path("/shared/weights/ancient-frost-119_10.pt").expanduser()


class CenternetNode:

    def __init__(self):
        self._load_config()

        self._tf_client: TransformClient = TransformClient()

        self._cv_bridge: CvBridge = CvBridge()

        self._device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dla_backbone = DLABackbone(model_config.backbone_heights, model_config.backbone_channels,
                                   model_config.downsamples)
        self._centernet = Centernet(dla_backbone, object_config).to(self._device)
        self._centernet.load_state_dict(torch.load(weights_path, map_location=self._device))
        self._centernet.eval()

        self._centernet.forward(torch.rand(1, 3, model_config.in_h, model_config.in_w, device=self._device))

        self._camera_infos: Dict[str, CameraInfo] = {}
        self._intrinsics: Dict[str, CameraIntrinsics] = {}
        self._color_subs: Dict[str, rospy.Subscriber] = {}
        self._debug_pubs: Dict[str, rospy.Publisher] = {}

        for frame_id in self._frame_ids:
            self._camera_infos[frame_id] = rospy.wait_for_message(f"vehicle/{frame_id}/depth/camera_info", CameraInfo, 60)
            self._intrinsics[frame_id] = CameraIntrinsics.from_matrix(np.array(self._camera_infos[frame_id].K))

            self._color_subs[frame_id] = rospy.Subscriber(f"vehicle/{frame_id}/color/image_raw", Image, partial(self._handle_img, frame_id=frame_id))

            self._debug_pubs[frame_id] = rospy.Publisher(f"centernet/{frame_id}/debug", Image, queue_size=10)

        self._detections_pub: rospy.Publisher = rospy.Publisher("global_map/feature_detections", FeatureDetections, queue_size=10)

    def start(self):
        rospy.spin()

    def _handle_img(self, color_msg: Image, frame_id: str):
        color_np = self._cv_bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")

        img_raw = T.ToTensor()(color_np)
        img = T.Resize((model_config.in_h, model_config.in_w))(img_raw.unsqueeze(0))
        img = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.3, 0.3, 0.3))(img).to(self._device)

        prediction = self._centernet.forward(img)

        intrinsics = self._intrinsics[frame_id]

        # print(intrinsics)

        M_projection = np.array([
            [intrinsics.f_x / 2, 0, intrinsics.c_x / 2],
            [0, intrinsics.f_y / 2, intrinsics.c_y / 2],
            [0, 0, 0],
        ])

        detections = decode_keypoints(
            prediction,
            model_config,
            object_config,
            M_projection,
            n_detections=10,
            keypoint_n_detections=10,
            score_threshold=0.3,
            keypoint_score_threshold=0.3,
            keypoint_angle_threshold=0.5,
        )[0]

        world_frame = f"{self._tf_namespace}/odom"
        camera_frame = f"{self._tf_namespace}/{frame_id}"

        rospy.logdebug(f"{world_frame} to {camera_frame}")

        world_t_cam = None
        while world_t_cam is None:
            try:
                world_t_cam = self._tf_client.get_a_to_b(world_frame, camera_frame, color_msg.header.stamp)
            except Exception as e:
                rospy.logwarn(e)
                rospy.logwarn("Failed to get transform")

        rospy.logdebug("Got transforms")

        detection_debug_np = color_np.copy()

        detection_array_msg = FeatureDetections()
        detection_array_msg.detector_tag = "centernet"

        for detection_i, detection in enumerate(detections):
            if detection.cam_t_object is None:
                continue

            detection_id = object_config.configs[detection.label].id
            cam_t_detection = detection.cam_t_object * object_t_detections[detection_id]

            world_t_detection = world_t_cam * cam_t_detection
            # cam_t_object = detection.cam_t_object
            # cam_t_object = SE3.Rt(SO3.TwoVectors(x="z", y="x"), cam_t_object.t)

            # world_t_object = world_t_cam * cam_t_object
            # world_t_detection = world_t_object * object_t_detections[detection_id]
            # world_t_detection = world_t_object

            # self._tf_client.set_a_to_b('kf/odom', 'raw_buoy', world_t_object)
            self._tf_client.set_a_to_b('kf/odom', 'adjusted_buoy', world_t_detection)

            detection_msg = FeatureDetection()
            detection_msg.confidence = 1
            detection_msg.tag = detection_id
            detection_msg.SE2 = False
            detection_msg.position = r3_to_ros_point(world_t_detection.t)
            rpy = world_t_detection.rpy()
            detection_msg.orientation = r3_to_ros_point(rpy)

            detection_array_msg.detections.append(detection_msg)

            rvec, _ = cv2.Rodrigues(detection.cam_t_object.R)
            tvec = detection.cam_t_object.t

            cv2.drawFrameAxes(detection_debug_np, M_projection, None, rvec, tvec, 0.1, 3)

        self._detections_pub.publish(detection_array_msg)

        detection_debug_msg = self._cv_bridge.cv2_to_imgmsg(np.flip(detection_debug_np, axis=-1), encoding="bgr8")
        self._debug_pubs[frame_id].publish(detection_debug_msg)

    def _load_config(self):
        self._frame_ids: [str] = rospy.get_param("~frame_ids")
        self._tf_namespace: str = rospy.get_param("tf_namespace")


def main():
    rospy.init_node('centernet')
    n = CenternetNode()
    n.start()


if __name__ == "__main__":
    main()