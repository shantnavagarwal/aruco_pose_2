import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
import sys
from sensor_msgs.msg import CameraInfo
import numpy as np
from sensor_msgs.msg import CameraInfo

class DummyCameraPublisher(Node):
    def __init__(self, image_folder, publish_topic="/camera/image_raw", publish_rate=10.0):
        super().__init__('dummy_camera_publisher')
        self.publisher_ = self.create_publisher(Image, publish_topic, 10)
        self.bridge = CvBridge()
        self.image_folder = image_folder
        self.image_files = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        if not self.image_files:
            self.get_logger().error(f"No images found in {image_folder}")
            rclpy.shutdown()
            return
        self.index = 0
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)\
        # CameraInfo publisher setup
        self.camera_info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", 10)
        # Camera intrinsic matrix (K)
        self.K = [2800.0, 0.0, 2016.0,
              0.0, 2800.0, 1512.0,
              0.0, 0.0, 1.0]
        # Set image dimensions (example: 4032x3024, adjust as needed)
        self.width = 4032
        self.height = 3024

    def timer_callback(self):
        img_path = self.image_files[self.index]
        img = cv2.imread(img_path)
        if img is None:
            self.get_logger().warn(f"Failed to read image: {img_path}")
        else:
            msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(msg)
            self.get_logger().info(f"Published {img_path}")
        self.index = (self.index + 1) % len(self.image_files)
    
        cam_msg = CameraInfo()
        cam_msg.header.stamp = msg.header.stamp
        cam_msg.header.frame_id = "camera"
        cam_msg.width = self.width
        cam_msg.height = self.height
        cam_msg.k = self.K
        cam_msg.d = []
        cam_msg.r = [1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0]
        cam_msg.p = [self.K[0], self.K[1], self.K[2], 0.0,
             self.K[3], self.K[4], self.K[5], 0.0,
             self.K[6], self.K[7], self.K[8], 0.0]
        cam_msg.distortion_model = "plumb_bob"
        self.camera_info_pub.publish(msg)

def main():
    rclpy.init()
    image_folder = '/home/ws/data/iphone'
    publish_topic = "/camera/image_raw"
    publish_rate = 1.0
    node = DummyCameraPublisher(image_folder, publish_topic, publish_rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()