import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import yaml
import os

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from cv2 import aruco

class ArucoPoseEstimator(Node):
    def __init__(self):
        super().__init__('aruco_pose_node')

        self.declare_parameter('image_topic', '/zed/right/image_rect')
        self.declare_parameter('marker_map_path', 'markers.yaml')
        self.declare_parameter('marker_size', 0.05)  # in meters

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.marker_map_path = self.get_parameter('marker_map_path').get_parameter_value().string_value
        self.marker_size = self.get_parameter('marker_size').get_parameter_value().double_value

        self.bridge = CvBridge()
        self.camera_matrix = np.array([
            [1070.0, 0.0, 960.0,
            0.0, 1070.0, 540.0,
            0.0, 0.0, 1.0]
        ])

        self.marker_map = self.load_marker_map(self.marker_map_path)

        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Image, self.image_topic, self.image_callback, 10)

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()

        self.get_logger().info("Aruco pose estimator initialized.")

    def load_marker_map(self, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return {int(k): np.array(v, dtype=np.float32) for k, v in data['markers'].items()}

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        corners, ids, _ = aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            return

        ids = ids.flatten()
        for i, marker_id in enumerate(ids):
            if marker_id in self.marker_map:
                image_corners = corners[i][0]
                obj_points = np.array([
                    [-self.marker_size/2,  self.marker_size/2, 0],
                    [ self.marker_size/2,  self.marker_size/2, 0],
                    [ self.marker_size/2, -self.marker_size/2, 0],
                    [-self.marker_size/2, -self.marker_size/2, 0]
                ], dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    image_corners,
                    self.camera_matrix,
                )

                if success:
                    marker_world_pos = self.marker_map[marker_id]
                    # tvec is the marker pose in the camera frame
                    self.get_logger().info(f"Marker {marker_id}: Camera-relative position: {tvec.flatten()}")
            else:
                self.get_logger().warn(f"Marker ID {marker_id} not in map")
