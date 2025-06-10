#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import os
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
from rclpy.qos import QoSProfile, ReliabilityPolicy

class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver_node')
        
        # Declare parameters
        self.declare_parameter('save_path', os.path.expanduser('/home/ws/data/'))
        self.declare_parameter('image_topic', '/zed/right/image_rect')
        self.declare_parameter('file_prefix', 'image_')
        
        # Get parameters
        self.save_path = self.get_parameter('save_path').value
        self.image_topic = self.get_parameter('image_topic').value
        self.file_prefix = self.get_parameter('file_prefix').value
        
        # Create save directory if needed
        os.makedirs(self.save_path, exist_ok=True)
        self.get_logger().info(f"Saving images to: {self.save_path}")
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        self.latest_image = None
        self.lock = threading.Lock()
        
        # Setup subscriber with QoS profile (typically BEST_EFFORT for sensors)
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        
        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile
        )
        
        # Setup timer (1Hz)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info(f"Image saver initialized. Saving at 1 Hz from topic: {self.image_topic}")

    def image_callback(self, msg):
        with self.lock:
            self.latest_image = msg

    def save_image_message(self, img_msg, path):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            cv2.imwrite(path, cv_image)
            self.get_logger().info(f"Saved image: {path}")
            return True
        except Exception as e:
            self.get_logger().error(f"Error saving image: {str(e)}")
            return False

    def timer_callback(self):
        with self.lock:
            if self.latest_image is None:
                self.get_logger().warn("No image received yet")
                return
                
            # Generate filename with timestamp
            timestamp = self.get_clock().now().nanoseconds
            filename = f"{self.file_prefix}{timestamp}.jpg"
            file_path = os.path.join(self.save_path, filename)
            
            # Save the image
            self.save_image_message(self.latest_image, file_path)

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaverNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()