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
from threading import Lock

from transform_client import TransformClient
from tauv_util.cameras import CameraIntrinsics
from tauv_util.spatialmath import ros_transform_to_se3, r3_to_ros_vector3, r3_to_ros_point
from tauv_msgs.msg import FeatureDetection, FeatureDetections
from tauv_vision.centernet.model.backbones.centerpoint_dla import CenterpointDLA34
from tauv_vision.centernet.model.config import ObjectConfig, ObjectConfigSet, AngleConfig, ModelConfig, TrainConfig
from tauv_vision.centernet.model.decode import decode_keypoints


model_config = ModelConfig(
    in_h=360,
    in_w=640,
    backbone_heights=[2, 2, 2, 2, 2],
    backbone_channels=[128, 128, 128, 128, 128, 128],
    downsamples=2,
    angle_bin_overlap=pi / 3,
)


object_config = ObjectConfigSet(
    configs=[
        ObjectConfig(
            id="sample_24_coral",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0)
            ]
        ),
        ObjectConfig(
            id="sample_24_nautilus",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0)
            ]
        ),
        ObjectConfig(
            id="torpedo_24",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0),
                # (0.6096, 0.6096, 0),
                # (0.6096, -0.6096, 0),
                # (-0.6096, -0.6096, 0),
                # (-0.6096, 0.6096, 0),
                # (-0.3, 0, 0),
                # (0.3, 0, 0),
                # (0, -0.3, 0),
                # (0, 0.3, 0),
            ],
        ),
        ObjectConfig(
            id="torpedo_24_octagon",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0),
            ],
        )
    ]
)

# object_config = ObjectConfigSet(
#     configs=[
#         ObjectConfig(
#             id="torpedo_24",
#             yaw=AngleConfig(train=False, modulo=2 * pi),
#             pitch=AngleConfig(train=False, modulo=2 * pi),
#             roll=AngleConfig(train=False, modulo=2 * pi),
#             train_depth=True,
#             train_keypoints=True,
#             keypoints=[
#                 (0, 0, 0),
#                 (0.6096, 0.6096, 0),
#                 (0.6096, -0.6096, 0),
#                 (-0.6096, -0.6096, 0),
#                 (-0.6096, 0.6096, 0),
#                 (-0.3, 0, 0),
#                 (0.3, 0, 0),
#                 (0, -0.3, 0),
#                 (0, 0.3, 0),
#             ],
#         ),
#         ObjectConfig(
#             id="torpedo_24_octagon",
#             yaw=AngleConfig(train=False, modulo=2 * pi),
#             pitch=AngleConfig(train=False, modulo=2 * pi),
#             roll=AngleConfig(train=False, modulo=2 * pi),
#             train_depth=True,
#             train_keypoints=True,
#             keypoints=[
#                 (0, 0, 0),
#             ],
#         )
#     ]
# )

object_t_detections: Dict[str, SE3] = {
    "torpedo_24": SE3(SO3.TwoVectors(x="-z", y="x")),
    "torpedo_24_octagon": SE3(SO3.TwoVectors(x="-z", y="x")),
}

# weights_path = pathlib.Path("/shared/weights/dauntless-disco-272-latest.pt").expanduser()
weights_path = pathlib.Path("/shared/weights/polished-salad-301_50.pt").expanduser()


class CenternetNode:

    def __init__(self):
        self._load_config()

        self._tf_client: TransformClient = TransformClient()

        self._cv_bridge: CvBridge = CvBridge()

        self._device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._centernet = CenterpointDLA34(object_config).to(self._device)
        self._centernet.load_state_dict(torch.load(weights_path, map_location=self._device))
        self._centernet.eval()

        self._centernet.forward(torch.rand(1, 3, model_config.in_h, model_config.in_w, device=self._device))

        self._camera_infos: Dict[str, CameraInfo] = {}
        self._intrinsics: Dict[str, CameraIntrinsics] = {}
        self._color_subs: Dict[str, rospy.Subscriber] = {}
        self._depth_subs: Dict[str, rospy.Subscriber] = {}
        self._debug_pubs: Dict[str, rospy.Publisher] = {}

        for frame_id in self._frame_ids:
            self._camera_infos[frame_id] = rospy.wait_for_message(f"vehicle/{frame_id}/depth/camera_info", CameraInfo, 60)
            self._intrinsics[frame_id] = CameraIntrinsics.from_matrix(np.array(self._camera_infos[frame_id].K))

            self._color_subs[frame_id] = rospy.Subscriber(f"vehicle/{frame_id}/color/image_raw", Image, partial(self._handle_img, frame_id=frame_id))
            self._depth_subs[frame_id] = rospy.Subscriber(f'vehicle/{frame_id}/depth/image_raw', Image, self._handle_depth, callback_args=frame_id)

            self._debug_pubs[frame_id] = rospy.Publisher(f"centernet/{frame_id}/debug", Image, queue_size=10)

        self._detections_pub: rospy.Publisher = rospy.Publisher("global_map/feature_detections", FeatureDetections, queue_size=10)

        self._depth = {}

    def start(self):
        rospy.spin()

    def _handle_depth(self, msg, frame_id):
        self._depth[frame_id] = msg

    def _handle_img(self, color_msg: Image, frame_id: str):
        depth_msg = self._depth.get(frame_id)
        if depth_msg is None:
            return

        color_np = self._cv_bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
        depth = self._cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='mono16')
        depth = depth.astype(float) / 1000

        depth_debug = (depth * (255 / depth.max())).astype(np.uint8)
        detection_debug_msg = self._cv_bridge.cv2_to_imgmsg(cv2.applyColorMap(depth_debug, cv2.COLORMAP_JET), encoding="bgr8")
        self._debug_pubs[frame_id].publish(detection_debug_msg)

        img_raw = T.ToTensor()(color_np)
        img = T.Resize((model_config.in_h, model_config.in_w))(img_raw.unsqueeze(0))
        img = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(img).to(self._device)

        prediction = self._centernet.forward(img)

        intrinsics = self._intrinsics[frame_id]

        M_projection = np.array([
            [intrinsics.f_x / 2, 0, intrinsics.c_x / 2],
            [0, intrinsics.f_y / 2, intrinsics.c_y / 2],
            [0, 0, 0],
        ])
        M_projection[0, 0] *= 1.33
        M_projection[1, 1] *= 1.33

        detections = decode_keypoints(
            prediction,
            model_config,
            object_config,
            M_projection,
            n_detections=10,
            keypoint_n_detections=50,
            score_threshold=0.6,
            keypoint_score_threshold=0.3,
            keypoint_angle_threshold=0.3,
        )[0]

        world_frame = f"{self._tf_namespace}/odom"
        camera_frame = f"{self._tf_namespace}/{frame_id}"

        rospy.logdebug(f"{world_frame} to {camera_frame}")

        world_t_cam = None
        # while world_t_cam is None:
        try:
            world_t_cam = self._tf_client.get_a_to_b(world_frame, camera_frame, color_msg.header.stamp)
        except Exception as e:
            rospy.logwarn(e)
            rospy.logwarn("Failed to get transform")
            return

        rospy.logdebug("Got transforms")

        detection_debug_np = color_np.copy()

        detection_array_msg = FeatureDetections()
        detection_array_msg.detector_tag = "centernet"

        for detection_i, detection in enumerate(detections):
            print(detection)

            cv2.circle(detection_debug_np, (int(detection.x * 640), int(detection.y * 360)), 3, (255, 0, 0), -1)

            e_x = detection.x * 640
            e_y = detection.y * 360
            w = detection.w * 640
            h = detection.h * 360

            depth_mask = np.zeros(depth.shape, dtype=np.uint8)

            cv2.rectangle(
                depth_mask,
                (int(e_x - 0.4 * w), int(e_y - 0.4 * h)),
                (int(e_x + 0.4 * w), int(e_y + 0.4 * h)),
                255,
                -1
            )

            cv2.rectangle(
                detection_debug_np,
                (int(e_x - 0.4 * w), int(e_y - 0.4 * h)),
                (int(e_x + 0.4 * w), int(e_y + 0.4 * h)),
                (0, 0, 255),
                1
            )

            cv2.putText(detection_debug_np, f"{detection.score:02f}", (int(e_x - 0.4 * w), int(e_y - 0.5 * h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            if np.sum(depth[(depth_mask > 0) & (depth > 0)]) < 10:
                continue

            z = np.mean(depth[(depth_mask > 0) & (depth > 0)])

            if z < 1:
                continue

            x = (e_x - M_projection[0, 2]) * (z / M_projection[0, 0])
            y = (e_y - M_projection[1, 2]) * (z / M_projection[1, 1])

            cam_t_detection = SE3.Rt(SO3.TwoVectors(x="z", y="x"), np.array([x, y, z]))

            detection_id = object_config.configs[detection.label].id
            # cam_t_detection = cam_t_object * object_t_detections[detection_id]

            world_t_detection = world_t_cam * cam_t_detection
            # cam_t_object = detection.cam_t_object
            # cam_t_object = SE3.Rt(SO3.TwoVectors(x="z", y="x"), cam_t_object.t)

            # world_t_object = world_t_cam * cam_t_object
            # world_t_detection = world_t_object * object_t_detections[detection_id]
            # world_t_detection = world_t_object

            # self._tf_client.set_a_to_b('kf/odom', 'raw_buoy', world_t_object)
            # self._tf_client.set_a_to_b('kf/odom', 'adjusted_buoy', world_t_detection)

            detection_msg = FeatureDetection()
            detection_msg.confidence = 1
            detection_msg.tag = detection_id
            detection_msg.SE2 = False
            detection_msg.position = r3_to_ros_point(world_t_detection.t)
            rpy = world_t_detection.rpy()
            detection_msg.orientation = r3_to_ros_point(rpy)

            detection_array_msg.detections.append(detection_msg)

        self._detections_pub.publish(detection_array_msg)

        # detection_debug_msg = self._cv_bridge.cv2_to_imgmsg(np.flip(detection_debug_np, axis=-1), encoding="bgr8")

    def _load_config(self):
        self._frame_ids: [str] = rospy.get_param("~frame_ids")
        self._tf_namespace: str = rospy.get_param("tf_namespace")


def main():
    rospy.init_node('centernet')
    n = CenternetNode()
    n.start()


if __name__ == "__main__":
    main()