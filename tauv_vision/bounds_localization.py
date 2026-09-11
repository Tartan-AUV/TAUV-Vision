import os

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from math import pi

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import warnings

import tf2_ros

def tf_stamped_to_Rt(tf_stamped):
    """Convert a geometry_msgs/TransformStamped into a (3x3 rotation, 3 translation)."""
    q = tf_stamped.transform.rotation
    t = tf_stamped.transform.translation

    x, y, z, w = q.x, q.y, q.z, q.w
    # Quaternion -> rotation matrix
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)

    translation = np.array([t.x, t.y, t.z], dtype=np.float64)
    return R, translation

def tf_stamped_to_transformation_matrix(tf_stamped):
    """Convert a geometry_msgs/TransformStamped into a 4x4 transformation matrix."""
    q = tf_stamped.transform.rotation
    t = tf_stamped.transform.translation

    x, y, z, w = q.x, q.y, q.z, q.w
    # Quaternion:translation -> transformation matrix
    T = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),     t.x ],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),     t.y ],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y), t.z ],
        [0,                       0,                       0,                       1   ]
    ], dtype=np.float64)

    return T

class BoundsLocalization(Node):
    def __init__(self):
        super().__init__('bounds_localization')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('rgb_topic', 'oak/rgb/image_raw'),
                ('depth_topic', 'oak/stereo/image_raw'),
                ('camera_info_topic', 'oak/rgb/camera_info'),
                ('odom_frame', 'odom'),
                ('robot_frame', 'os/base_link'),

                # Fraction of pixels in the stereo image needed before a frame is considered
                ('stereo_nonzero_threshold', 0.7),

                # Absolute z location at or near the surface of the water, used to reject pixels by reprojected height
                ('surface_height', -0.05),
                ('tf_timeout', 0.2),

                # Visualizes intermediate images and results
                ('debug_mode', True)
            ]
        )

        # ---- load parameters ----
        self.stereo_nonzero_threshold = self.get_parameter('stereo_nonzero_threshold').value
        self.surface_height = self.get_parameter('surface_height').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value

        self.debug_mode = self.get_parameter('debug_mode').value

        # ---- state ----
        self.bridge = CvBridge()
        self.camera_info = None

        # ---- TF ----
        # spin_thread=True gives the listener its own executor, so the blocking
        # lookup below can wait for a transform without deadlocking this node's
        # callback thread.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=True)
        self.tf_timeout = Duration(
            seconds=self.get_parameter('tf_timeout').value)

        # ---- pubs / subs ----
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

        # ---- debug mode pubs ----
        if self.debug_mode:
            self.pub_depth_blurred = self.create_publisher(Image, 'depth_blurred', 10)
            self.surface_mask = self.create_publisher(Image, 'surface_mask', 10)

        self.get_logger().info('bounds_localization node ready.')


    # ------------------------------------------------------------------
    # Main update step. Any early `return` just skips this frame.
    # ------------------------------------------------------------------
    def depth_image_callback(self, depth_msg):
        if self.camera_info is None:
            self.get_logger().warn('No camera_info yet; skipping frame.')
            return

        depth = self.bridge.imgmsg_to_cv2(depth_msg)

        if(cv2.countNonZero(depth)/depth.size < self.stereo_nonzero_threshold):
            self.get_logger().warn('Not enough depth for wall detection')
            return # Not enough stereo pixels to be reliable

        # downscale
        depth = self.downscale_median_nonzero(depth, 2)

        # filter out points near or above the water surface
        xyz_odom = self.deproject_depth_to_frame(depth, depth_msg.header.frame_id, self.odom_frame)
        if(xyz_odom is None):
            self.get_logger().warn("TF issue")
            return

        surface_mask = (xyz_odom[:, :, 2] < self.surface_height).astype(np.uint16)
        depth_surface_mask = depth * surface_mask

        depth_blurred = self.gaussian_blur_ignore_zeros(depth_surface_mask, (3,3), 1)
        normals, points, valid = self.fit_block_planes(depth_blurred, depth_msg.header.frame_id, self.odom_frame)

        if (normals is None) or (points is None) or (valid is None):
            self.get_logger().warn("plane fitting issue")
            return

        robot_pos = None
        try:
            transform_robot = self.tf_buffer.lookup_transform(self.odom_frame, self.robot_frame, Time(), timeout=self.tf_timeout)
            _, robot_pos = tf_stamped_to_Rt(transform_robot)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'odom->os/base_link unavailable ({e}); skipping frame.')
            return

        expected_normals = np.array([
            [1, 0, 0]
        ], dtype=np.float32)
        merge_results = self.merge_wall_planes_axis_aligned(normals, points, valid, robot_pos, expected_normals)

        if (merge_results is None):
            self.get_logger().warn("plane merging issue")
            return

        self.get_logger().info(f"Normal: {merge_results[0]['normal']}")
        self.get_logger().info(f"Closest to origin: {merge_results[0]['closest_point']}")

        if self.debug_mode:
            self.get_logger().info("publishing debug")
            self.pub_depth_blurred.publish(self.bridge.cv2_to_imgmsg(depth_blurred, depth_msg.encoding, depth_msg.header))
            self.surface_mask.publish(self.bridge.cv2_to_imgmsg(surface_mask, "16UC1", depth_msg.header))

    def merge_wall_planes_axis_aligned(self, normals, points, valid, camera_pos,
                                     expected_normals,
                                     normal_angle_thresh_rads=(pi/20),
                                     dist_thresh=0.1):
        """
        Find plane(s) whose normal matches one of a small set of expected
        directions.

        Returns a list of dicts (one per matched direction), sorted by inlier
        count descending, each with keys:
            normal, closest_point, mean_range, inlier_mask, expected_idx
        """
        idx = np.argwhere(valid)
        if idx.shape[0] < 3:
            return []

        ns = normals[valid]  # (M, 3)
        ps = points[valid]   # (M, 3)

        expected_normals = np.asarray(expected_normals, dtype=np.float32)
        expected_normals = expected_normals / np.linalg.norm(expected_normals, axis=1, keepdims=True)
        cos_thresh = np.cos(normal_angle_thresh_rads)

        # (M, K): cos angle between every block normal and every expected normal
        cos_angles = ns @ expected_normals.T

        results = []
        for k, e in enumerate(expected_normals):
            cand_idx = np.nonzero(cos_angles[:, k] > cos_thresh)[0]
            if cand_idx.size < 3:
                continue

            # Distance of each candidate along direction e -- now a scalar
            # clustering problem instead of a full plane-fit consensus problem.
            d_vals = ps[cand_idx] @ e
            order = np.argsort(d_vals)
            d_sorted = d_vals[order]

            # Widest run of points within dist_thresh of each other, via a sorted
            # sliding window -- O(C) after the O(C log C) sort, not O(C^2).
            best_l, best_r, best_count = 0, 0, 0
            left = 0
            for right in range(len(d_sorted)):
                while d_sorted[right] - d_sorted[left] > dist_thresh:
                    left += 1
                if right - left + 1 > best_count:
                    best_count, best_l, best_r = right - left + 1, left, right

            if best_count < 3:
                continue

            inlier_idx = cand_idx[order[best_l:best_r + 1]]
            inlier_pts = ps[inlier_idx]

            centroid = inlier_pts.mean(axis=0)
            centered = inlier_pts - centroid
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            normal = vt[-1]
            normal /= np.linalg.norm(normal)

            # Orient toward the robot/camera. Safe as a stable convention only
            # because the vehicle never crosses to the other side of the wall
            if np.dot(normal, camera_pos - centroid) < 0:
                normal = -normal

            d = np.dot(normal, centroid)
            closest_point = (d * normal).astype(np.float32)

            mean_range = float(np.linalg.norm(inlier_pts - camera_pos, axis=1).mean())

            inlier_mask = np.zeros(valid.shape, dtype=bool)
            full_idx = idx[inlier_idx]
            inlier_mask[full_idx[:, 0], full_idx[:, 1]] = True

            results.append({
                'normal': normal.astype(np.float32),
                'closest_point': closest_point,
                'mean_range': mean_range,
                'inlier_mask': inlier_mask,
                'expected_idx': k,
                'inlier_count': best_count,
            })

        results.sort(key=lambda r: r['inlier_count'], reverse=True)
        return results

    def fit_block_planes(self, depth, base_frame, target_frame,
                      block_size=8, num_samples=8, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        H, W = depth.shape
        if H % block_size != 0 or W % block_size != 0:
            raise ValueError(f'depth shape {(H, W)} not divisible by block_size={block_size}')

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, base_frame, Time(), timeout=self.tf_timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF unavailable ({e}); skipping frame.')
            return None, None, None

        T_cam_to_target = tf_stamped_to_transformation_matrix(transform)
        camera_pos = T_cam_to_target[:3, 3]  # camera origin, in target_frame

        # Deproject once; per-block sampling then just indexes into this.
        cam_xyz = self.deproject_depth_to_cam_frame_mm(depth) / 1000.0  # (H, W, 3), meters

        blocks_h, blocks_w = H // block_size, W // block_size
        normals = np.zeros((blocks_h, blocks_w, 3), dtype=np.float32)
        points = np.zeros((blocks_h, blocks_w, 3), dtype=np.float32)
        valid = np.zeros((blocks_h, blocks_w), dtype=bool)

        for by in range(blocks_h):
            for bx in range(blocks_w):
                y0, y1 = by * block_size, (by + 1) * block_size
                x0, x1 = bx * block_size, (bx + 1) * block_size

                nz_rows, nz_cols = np.nonzero(depth[y0:y1, x0:x1])
                if nz_rows.size < 3:
                    continue  # can't fit a plane to fewer than 3 points

                n = min(num_samples, nz_rows.size)
                idx = rng.choice(nz_rows.size, size=n, replace=False)
                rows = nz_rows[idx] + y0
                cols = nz_cols[idx] + x0

                pts_cam = cam_xyz[rows, cols, :]  # (n, 3), camera frame
                pts_cam_h = np.hstack([pts_cam, np.ones((n, 1), dtype=np.float32)])
                pts_target = (pts_cam_h @ T_cam_to_target.T)[:, :3]  # (n, 3), target frame

                centroid = pts_target.mean(axis=0)
                centered = pts_target - centroid
                # Total-least-squares plane fit: normal is the right singular vector
                # with the smallest singular value (direction of least point spread).
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                normal = vt[-1]
                normal /= np.linalg.norm(normal)

                # SVD gives the normal's axis, not its sign — orient it to face the
                # camera so results are consistent block-to-block.
                if np.dot(normal, camera_pos - centroid) < 0:
                    normal = -normal

                normals[by, bx] = normal
                points[by, bx] = centroid
                valid[by, bx] = True

        return normals, points, valid

    def deproject_depth_to_frame(self, depth, base_frame, target_frame):
        # get relative to camera in mm, then convert to m
        cam_xyz = self.deproject_depth_to_cam_frame_mm(depth) / 1000.0

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, base_frame, Time(), timeout=self.tf_timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF unavailable ({e}); skipping frame.')
            return

        pts_cam = cam_xyz.reshape(-1, 3)  # flatten to (H*W, 3)
        pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)]) # convert to homogeneous
        T_cam_to_target = tf_stamped_to_transformation_matrix(transform)
        pts_target_h = pts_cam_h @ T_cam_to_target.T

        # convert out of homogeneous coordinates
        pts_target = pts_target_h[:, :3]

        H, W = depth.shape
        return pts_target.reshape(H, W, 3).astype(np.float32)


    def deproject_depth_to_cam_frame_mm(self, depth):
        fx, fy, cx, cy = self.intrinsics()

        # scale if needed
        scale = depth.shape[1] / self.camera_info.width
        fx, fy, cx, cy = fx * scale, fy * scale, cx * scale, cy * scale

        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = depth.astype(np.float32)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
        return xyz

    def gaussian_blur_ignore_zeros(self, img, ksize=(3, 3), sigma=0):
        img_f = img.astype(np.float32)
        mask = (img_f != 0).astype(np.float32)  # 1 where valid, 0 where zero

        # Blur the image (zeros contribute 0 to the sum) and the mask separately
        blurred_img = cv2.GaussianBlur(img_f * mask, ksize, sigma)
        blurred_mask = cv2.GaussianBlur(mask, ksize, sigma)

        # Avoid divide-by-zero where the whole neighborhood was zero
        with np.errstate(invalid="ignore", divide="ignore"):
            result = blurred_img / blurred_mask

        result[blurred_mask == 0] = 0  # regions with no valid data stay 0
        return result.astype(img.dtype)

    
    def downscale_median_nonzero(self, img, N):
        """Downscale a mono image Nx (with N aligned to power of 2) by taking the median of nonzero pixels
        in each NxN block. Blocks with no nonzero pixels become 0."""
        H, W = img.shape
        H, W = H - H % N, W - W % N   # crop to even dims if needed
        img = img[:H, :W]

        # Group into (H/N, W/N, N^2) blocks of the NxN neighborhoods
        blocks = (img.reshape(H // N, N, W // N, N)
                    .transpose(0, 2, 1, 3)
                    .reshape(H // N, W // N, N*N)
                    .astype(np.float32))

        blocks[blocks == 0] = np.nan  # exclude zeros from the median

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN blocks
            med = np.nanmedian(blocks, axis=2)

        med = np.nan_to_num(med, nan=0.0)
        return med.astype(img.dtype)

    def intrinsics(self):
        k = self.camera_info.k
        return k[0], k[4], k[2], k[5]

    def info_callback(self, msg):
        self.camera_info = msg

def main(args=None):
    rclpy.init(args=args)
    node = BoundsLocalization()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()