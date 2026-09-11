#!/usr/bin/env python3
"""Crop a PointCloud2 to an axis-aligned box and republish it.
 
Written to filter rtabmap's assembled ``/rtabmap/cloud_map`` by height, so the
map you look at in Foxglove drops points above (or below) a chosen level.
 
Why this works without any TF: ``cloud_map`` is published in the map frame, so
a plain z crop on it *is* a world-frame height cut. For a cloud that lives in a
moving frame (e.g. a raw sensor cloud off the OAK-D), set ``target_frame`` and
the cloud is transformed into that frame with TF2 before cropping.
 
The box is a real CropBox: x/y/z min/max plus an ``invert`` flag (keep points
outside the box instead of inside). x and y default wide open, so by default
this is purely a z filter. All parameters are read every message, so you can
retune the box live with ``ros2 param set`` while a bag plays.
"""
 
import numpy as np
 
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
 
from sensor_msgs.msg import PointCloud2, PointField
 
 
# sensor_msgs/PointField datatype enum -> numpy scalar type.
_ROS_TO_NP = {
    PointField.INT8: np.int8,
    PointField.UINT8: np.uint8,
    PointField.INT16: np.int16,
    PointField.UINT16: np.uint16,
    PointField.INT32: np.int32,
    PointField.UINT32: np.uint32,
    PointField.FLOAT32: np.float32,
    PointField.FLOAT64: np.float64,
}
 
_OPEN = 1.0e9  # effectively +/-infinity for an "unset" bound
 
 
def cloud_to_struct(msg: PointCloud2) -> np.ndarray:
    """View a PointCloud2's raw bytes as a structured numpy array.
 
    The dtype places each field at its real byte offset and pads the itemsize
    out to ``point_step``. That means every field we don't touch -- rgb,
    normals, whatever -- is carried through byte-for-byte when we later drop
    rows, so colour survives the crop untouched.
    """
    names, formats, offsets = [], [], []
    for f in msg.fields:
        base = _ROS_TO_NP[f.datatype]
        names.append(f.name)
        formats.append(base if f.count == 1 else (base, (f.count,)))
        offsets.append(f.offset)
 
    dtype = np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": msg.point_step,
    })
 
    n = msg.width * msg.height
    return np.frombuffer(bytes(msg.data), dtype=dtype, count=n)
 
 
class CloudCropBox(Node):
    def __init__(self):
        super().__init__("cloud_crop_box")
 
        self.declare_parameter("input_topic", "/rtabmap/cloud_map")
        self.declare_parameter("output_topic", "/rtabmap/cloud_map_cropped")
        # z is the bound that matters for a height cut. x/y default wide open.
        self.declare_parameter("min_z", -_OPEN)
        self.declare_parameter("max_z", _OPEN)
        self.declare_parameter("min_x", -_OPEN)
        self.declare_parameter("max_x", _OPEN)
        self.declare_parameter("min_y", -_OPEN)
        self.declare_parameter("max_y", _OPEN)
        # Keep points OUTSIDE the box instead of inside (PCL setNegative).
        self.declare_parameter("invert", False)
        # If non-empty, transform the cloud into this frame before cropping.
        # Leave empty for cloud_map, which is already in the map frame.
        self.declare_parameter("target_frame", "")
 
        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.target_frame = self.get_parameter("target_frame").value
 
        self._tf_buffer = None
        if self.target_frame:
            # Lazy import: the default (no-TF) path pulls in no tf2 packages.
            from tf2_ros.buffer import Buffer
            from tf2_ros.transform_listener import TransformListener
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            self.get_logger().info(
                'Transforming clouds into "%s" before cropping.' % self.target_frame)
 
        # cloud_map is published reliable; the default subscription QoS matches.
        # The output is latched (transient-local, depth 1) so a Foxglove client
        # that connects after the last map update still receives the current
        # cropped cloud rather than waiting for the next one.
        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(PointCloud2, self.output_topic, latched)
        self.sub = self.create_subscription(
            PointCloud2, self.input_topic, self.on_cloud, 10)
 
        self._warned_fields = False
        self.get_logger().info(
            "Cropping %s -> %s" % (self.input_topic, self.output_topic))
 
    def on_cloud(self, msg: PointCloud2):
        # Re-read every callback so `ros2 param set` retunes the box live.
        min_x = self.get_parameter("min_x").value
        max_x = self.get_parameter("max_x").value
        min_y = self.get_parameter("min_y").value
        max_y = self.get_parameter("max_y").value
        min_z = self.get_parameter("min_z").value
        max_z = self.get_parameter("max_z").value
        invert = self.get_parameter("invert").value
 
        field_names = {f.name for f in msg.fields}
        if not {"x", "y", "z"} <= field_names:
            if not self._warned_fields:
                self.get_logger().error(
                    "Cloud has no x/y/z fields (%s); passing through unchanged."
                    % sorted(field_names))
                self._warned_fields = True
            self.pub.publish(msg)
            return
 
        if self.target_frame and self.target_frame != msg.header.frame_id:
            msg = self._transform(msg)
            if msg is None:
                return
 
        pts = cloud_to_struct(msg)
        if pts.size == 0:
            self.pub.publish(msg)
            return
 
        x = pts["x"].astype(np.float64)
        y = pts["y"].astype(np.float64)
        z = pts["z"].astype(np.float64)
 
        # Drop non-finite points regardless of invert, so the output is dense.
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        in_box = (
            (x >= min_x) & (x <= max_x)
            & (y >= min_y) & (y <= max_y)
            & (z >= min_z) & (z <= max_z)
        )
        keep = finite & (~in_box if invert else in_box)
 
        self.pub.publish(self._rebuild(msg, pts[keep]))
 
    def _transform(self, msg: PointCloud2):
        # do_transform_cloud has moved around between distros; try both spots.
        try:
            from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
        except ImportError:
            from tf2_sensor_msgs import do_transform_cloud
        try:
            tf = self._tf_buffer.lookup_transform(
                self.target_frame, msg.header.frame_id, msg.header.stamp)
        except Exception as exc:  # TF not ready yet / extrapolation into future
            self.get_logger().warn(
                "TF %s -> %s failed: %s" % (
                    msg.header.frame_id, self.target_frame, exc),
                throttle_duration_sec=2.0)
            return None
        return do_transform_cloud(msg, tf)
 
    @staticmethod
    def _rebuild(template: PointCloud2, kept: np.ndarray) -> PointCloud2:
        out = PointCloud2()
        out.header = template.header
        out.height = 1
        out.width = int(kept.shape[0])
        out.fields = template.fields
        out.is_bigendian = template.is_bigendian
        out.point_step = template.point_step
        out.row_step = template.point_step * out.width
        out.is_dense = True
        out.data = kept.tobytes()
        return out
 
 
def main():
    rclpy.init()
    node = CloudCropBox()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
 
 
if __name__ == "__main__":
    main()
