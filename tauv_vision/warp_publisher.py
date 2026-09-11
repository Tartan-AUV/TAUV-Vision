import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros

import warp as wp

from tauv_vision.bounds_localization import tf_stamped_to_transformation_matrix

class WarpPublisher(Node):
    def __init__(self):
        super().__init__('crop_surface')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('rgb_topic', 'oak/rgb/image_raw'),
                ('depth_topic', 'oak/stereo/image_raw'),
                ('camera_info_topic', 'oak/rgb/camera_info'),
                ('odom_frame', 'odom'),
                ('robot_frame', 'os/base_link'),

                # Absolute z location at or near the surface of the water, used to reject pixels by reprojected height
                ('surface_height', -0.05),
                ('tf_timeout', 0.2),

                # Visualizes intermediate images and results
                ('debug_mode', True)
            ]
        )

        self.surface_height = self.get_parameter('surface_height').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value

        self.debug_mode = self.get_parameter('debug_mode').value

        self.bridge = CvBridge()
        self.camera_info = None
        self.device = wp.get_device()

        # ---- TF ----
        # spin_thread=True gives the listener its own executor, so the blocking
        # lookup below can wait for a transform without deadlocking this node's
        # callback thread.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=True)
        self.tf_timeout = Duration(
            seconds=self.get_parameter('tf_timeout').value)

        image_sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            Image, self.get_parameter('depth_topic').value,
            self.depth_image_callback,
            image_sensor_qos)

        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value,
            self.info_callback, image_sensor_qos)

        # Debug visualization publishers, keyed by topic name. Only created when
        # debug_mode is set, so publish_normalized_image is a no-op otherwise.
        self.debug_publishers = {}
        if self.debug_mode:
            self.debug_publishers['debug_xyz'] = self.create_publisher(Image, 'debug_xyz', image_sensor_qos)
            self.debug_publishers['debug_odom_xyz'] = self.create_publisher(Image, 'debug_odom_xyz', image_sensor_qos)

        self.get_logger().info('crop_surface node ready.')

    def depth_image_callback(self, depth_msg):
        if self.camera_info is None:
            self.get_logger().warn('No camera_info yet; skipping frame.')
            return

        fx, fy, cx, cy = self.intrinsics()
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        height, width = depth.shape[:2]
        num_pixels = height * width

        # The kernel is 1D over pixels and reads float32 millimetres, so flatten
        # the row-major image and widen in place. Casting straight to float32
        # keeps 16UC1 and 16SC1 sources equivalent. Invalid pixels stay 0 and
        # deproject to the origin.
        depth_mm = np.ascontiguousarray(depth, dtype=np.float32).reshape(-1)

        depth_wp = wp.array(depth_mm, dtype=wp.float32, device=self.device)
        xyz_wp = wp.empty(num_pixels, dtype=wp.vec3, device=self.device)

        wp.launch(
            deproject_to_cam_frame_mm,
            dim=num_pixels,
            inputs=[depth_wp, xyz_wp, width, fx, fy, cx, cy],
            device=self.device,
        )

        # (height, width, 3) of camera-frame millimetres.
        xyz = xyz_wp.numpy().reshape(height, width, 3)

        # Runs after the debug publish so a TF dropout doesn't take the
        # camera-frame visualization down with it.
        xyz_odom = self.deproject_depth_to_frame(
            depth_wp, height, width, depth_msg.header.frame_id, self.odom_frame)
        if xyz_odom is None:
            return

        if self.debug_mode:
            self.publish_normalized_image(xyz, 'debug_xyz', -5000.0, 5000.0, depth_msg.header)
            self.publish_normalized_image(xyz_odom, 'debug_odom_xyz', -1, 1, depth_msg.header)

    def deproject_depth_to_frame(self, depth_wp, height, width, base_frame, target_frame):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, base_frame, Time(), timeout=self.tf_timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF unavailable ({e}); skipping frame.')
            return None

        # Row-major flatten matches Warp's mat44 element order.
        cam_to_target = tf_stamped_to_transformation_matrix(transform)
        cam_to_target_wp = wp.mat44(*cam_to_target.astype(np.float32).flatten())

        fx, fy, cx, cy = self.intrinsics()
        num_pixels = height * width
        xyz_wp = wp.empty(num_pixels, dtype=wp.vec3, device=self.device)

        wp.launch(
            deproject_to_target_frame,
            dim=num_pixels,
            inputs=[depth_wp, xyz_wp, width, fx, fy, cx, cy, cam_to_target_wp],
            device=self.device,
        )

        return xyz_wp.numpy().reshape(height, width, 3)

    def publish_normalized_image(self, image, topic, min_value, max_value, header=None):
        publisher = self.debug_publishers.get(topic)
        if publisher is None:
            return

        normalized = np.clip(
            (image - min_value) / (max_value - min_value), 0.0, 1.0) * 255.0
        debug_msg = self.bridge.cv2_to_imgmsg(
            normalized.astype(np.uint8), encoding='rgb8')
        if header is not None:
            debug_msg.header = header

        publisher.publish(debug_msg)

    def intrinsics(self):
        k = self.camera_info.k
        return k[0], k[4], k[2], k[5]
    
    def info_callback(self, msg):
        self.camera_info = msg

# Sentinel written for pixels the stereo pair returned no depth for. NaN
# compares false against everything, so bad pixels self-reject downstream
# without needing a separate validity mask.
NAN = wp.constant(float('nan'))

MM_TO_M = wp.constant(0.001)

@wp.func
def deproject_pixel_mm(depth: wp.array(dtype=wp.float32), tid: wp.int32, width: wp.int32, fx: wp.float32, fy: wp.float32, cx: wp.float32, cy: wp.float32) -> wp.vec3:
    """Camera-frame millimetres for one pixel of a flattened, row-major depth image."""
    u = tid % width
    v = tid // width

    z = depth[tid]
    x = (wp.float32(u) - cx) * z / fx
    y = (wp.float32(v) - cy) * z / fy

    return wp.vec3(x, y, z)

@wp.kernel
def deproject_to_cam_frame_mm(depth: wp.array(dtype=wp.float32), outXyz: wp.array(dtype=wp.vec3), width: wp.int32, fx: wp.float32, fy: wp.float32, cx: wp.float32, cy: wp.float32):
    tid = wp.tid()
    outXyz[tid] = deproject_pixel_mm(depth, tid, width, fx, fy, cx, cy)

@wp.kernel
def deproject_to_target_frame(depth: wp.array(dtype=wp.float32), outXyz: wp.array(dtype=wp.vec3), width: wp.int32, fx: wp.float32, fy: wp.float32, cx: wp.float32, cy: wp.float32, camToTarget: wp.mat44):
    """Deproject to metres and apply an arbitrary 4x4 transform.

    camToTarget is opaque to the kernel: whichever frame it maps into is the
    frame outXyz lands in.
    """
    tid = wp.tid()

    # A zero here means "no return", not a point at the sensor. Left alone it
    # would deproject to the camera origin and transform into a perfectly
    # plausible-looking coordinate in the target frame.
    if depth[tid] == 0.0:
        outXyz[tid] = wp.vec3(NAN, NAN, NAN)
        return

    p_m = deproject_pixel_mm(depth, tid, width, fx, fy, cx, cy) * MM_TO_M
    outXyz[tid] = wp.transform_point(camToTarget, p_m)

def main(args=None):
    rclpy.init(args=args)
    wp.init()
    node = WarpPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()