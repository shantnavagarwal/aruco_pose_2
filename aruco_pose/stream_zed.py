#!/usr/bin/env python3

import numpy as np
import os
import configparser
import sys
import cv2
import wget
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def download_calibration_file(serial_number) :
    if os.name == 'nt' :
        hidden_path = os.getenv('APPDATA') + '\\Stereolabs\\settings\\'
    else :
        hidden_path = '/home/ws/data/zed/'
    calibration_file = hidden_path + 'SN' + str(serial_number) + '.conf'

    if os.path.isfile(calibration_file) == False:
        url = 'http://calib.stereolabs.com/?SN='
        filename = wget.download(url=url+str(serial_number), out=calibration_file)

        if os.path.isfile(calibration_file) == False:
            print('Invalid Calibration File')
            return ""

    return calibration_file

def init_calibration(calibration_file, image_size) :

    cameraMarix_left = cameraMatrix_right = map_left_y = map_left_x = map_right_y = map_right_x = np.array([])

    config = configparser.ConfigParser()
    config.read(calibration_file)

    check_data = True
    resolution_str = ''
    if image_size.width == 2208 :
        resolution_str = '2K'
    elif image_size.width == 1920 :
        resolution_str = 'FHD'
    elif image_size.width == 1280 :
        resolution_str = 'HD'
    elif image_size.width == 672 :
        resolution_str = 'VGA'
    else:
        resolution_str = 'HD'
        check_data = False

    T_ = np.array([-float(config['STEREO']['Baseline'] if 'Baseline' in config['STEREO'] else 0),
                   float(config['STEREO']['TY_'+resolution_str] if 'TY_'+resolution_str in config['STEREO'] else 0),
                   float(config['STEREO']['TZ_'+resolution_str] if 'TZ_'+resolution_str in config['STEREO'] else 0)])


    left_cam_cx = float(config['LEFT_CAM_'+resolution_str]['cx'] if 'cx' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_cy = float(config['LEFT_CAM_'+resolution_str]['cy'] if 'cy' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_fx = float(config['LEFT_CAM_'+resolution_str]['fx'] if 'fx' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_fy = float(config['LEFT_CAM_'+resolution_str]['fy'] if 'fy' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k1 = float(config['LEFT_CAM_'+resolution_str]['k1'] if 'k1' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k2 = float(config['LEFT_CAM_'+resolution_str]['k2'] if 'k2' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p1 = float(config['LEFT_CAM_'+resolution_str]['p1'] if 'p1' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p2 = float(config['LEFT_CAM_'+resolution_str]['p2'] if 'p2' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p3 = float(config['LEFT_CAM_'+resolution_str]['p3'] if 'p3' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k3 = float(config['LEFT_CAM_'+resolution_str]['k3'] if 'k3' in config['LEFT_CAM_'+resolution_str] else 0)


    right_cam_cx = float(config['RIGHT_CAM_'+resolution_str]['cx'] if 'cx' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_cy = float(config['RIGHT_CAM_'+resolution_str]['cy'] if 'cy' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_fx = float(config['RIGHT_CAM_'+resolution_str]['fx'] if 'fx' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_fy = float(config['RIGHT_CAM_'+resolution_str]['fy'] if 'fy' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k1 = float(config['RIGHT_CAM_'+resolution_str]['k1'] if 'k1' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k2 = float(config['RIGHT_CAM_'+resolution_str]['k2'] if 'k2' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p1 = float(config['RIGHT_CAM_'+resolution_str]['p1'] if 'p1' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p2 = float(config['RIGHT_CAM_'+resolution_str]['p2'] if 'p2' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p3 = float(config['RIGHT_CAM_'+resolution_str]['p3'] if 'p3' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k3 = float(config['RIGHT_CAM_'+resolution_str]['k3'] if 'k3' in config['RIGHT_CAM_'+resolution_str] else 0)

    R_zed = np.array([float(config['STEREO']['RX_'+resolution_str] if 'RX_' + resolution_str in config['STEREO'] else 0),
                      float(config['STEREO']['CV_'+resolution_str] if 'CV_' + resolution_str in config['STEREO'] else 0),
                      float(config['STEREO']['RZ_'+resolution_str] if 'RZ_' + resolution_str in config['STEREO'] else 0)])

    R, _ = cv2.Rodrigues(R_zed)
    cameraMatrix_left = np.array([[left_cam_fx, 0, left_cam_cx],
                         [0, left_cam_fy, left_cam_cy],
                         [0, 0, 1]])

    cameraMatrix_right = np.array([[right_cam_fx, 0, right_cam_cx],
                          [0, right_cam_fy, right_cam_cy],
                          [0, 0, 1]])

    distCoeffs_left = np.array([[left_cam_k1], [left_cam_k2], [left_cam_p1], [left_cam_p2], [left_cam_k3]])

    distCoeffs_right = np.array([[right_cam_k1], [right_cam_k2], [right_cam_p1], [right_cam_p2], [right_cam_k3]])

    T = np.array([[T_[0]], [T_[1]], [T_[2]]])
    R1 = R2 = P1 = P2 = np.array([])

    R1, R2, P1, P2 = cv2.stereoRectify(cameraMatrix1=cameraMatrix_left,
                                       cameraMatrix2=cameraMatrix_right,
                                       distCoeffs1=distCoeffs_left,
                                       distCoeffs2=distCoeffs_right,
                                       R=R, T=T,
                                       flags=cv2.CALIB_ZERO_DISPARITY,
                                       alpha=0,
                                       imageSize=(image_size.width, image_size.height),
                                       newImageSize=(image_size.width, image_size.height))[0:4]

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, (image_size.width, image_size.height), cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, (image_size.width, image_size.height), cv2.CV_32FC1)

    cameraMatrix_left = P1
    cameraMatrix_right = P2

    return cameraMatrix_left, cameraMatrix_right, map_left_x, map_left_y, map_right_x, map_right_y

class Resolution :
    width = 1920
    height = 1080


class ZedStereoPublisher(Node):
    def __init__(self):
        super().__init__('zed_stereo_publisher')
        
        # Create publishers for left and right images
        self.left_pub = self.create_publisher(Image, '/zed/left/image_rect', 10)
        self.right_pub = self.create_publisher(Image, '/zed/right/image_rect', 10)
        self.left_raw_pub = self.create_publisher(Image, '/zed/left/image_raw', 10)
        self.right_raw_pub = self.create_publisher(Image, '/zed/right/image_raw', 10)
        
        # Create CV bridge
        self.bridge = CvBridge()

        if len(sys.argv) == 1:
            self.get_logger().error('Please provide ZED serial number')
            sys.exit(1)

        # Open the ZED camera
        self.cap = cv2.VideoCapture('/dev/video2')
        if not self.cap.isOpened():
            self.get_logger().error('Failed to open ZED camera')
            sys.exit(-1)

        self.image_size = Resolution()
        # self.image_size.width = 1280
        # self.image_size.height = 720

        # Set the video resolution to HD720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size.width*2)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size.height)

        serial_number = int(sys.argv[1])
        calibration_file = download_calibration_file(serial_number)
        if calibration_file == "":
            sys.exit(1)
        self.get_logger().info("Calibration file found. Loading...")

        (self.camera_matrix_left, self.camera_matrix_right, 
         self.map_left_x, self.map_left_y, 
         self.map_right_x, self.map_right_y) = init_calibration(calibration_file, self.image_size)

        # Create a timer for publishing images at 30Hz
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)

    def timer_callback(self):
        # Get a new frame from camera
        retval, frame = self.cap.read()
        if not retval:
            self.get_logger().warn("Failed to capture frame")
            return
            
        # Extract left and right images from side-by-side
        left_right_image = np.split(frame, 2, axis=1)
        
        # Publish raw images
        try:
            left_raw_msg = self.bridge.cv2_to_imgmsg(left_right_image[0], encoding="bgr8")
            right_raw_msg = self.bridge.cv2_to_imgmsg(left_right_image[1], encoding="bgr8")
            left_raw_msg.header.stamp = self.get_clock().now().to_msg()
            right_raw_msg.header.stamp = left_raw_msg.header.stamp
            self.left_raw_pub.publish(left_raw_msg)
            self.right_raw_pub.publish(right_raw_msg)
        except Exception as e:
            self.get_logger().error(f"Error converting raw images: {str(e)}")

        # Rectify images
        left_rect = cv2.remap(left_right_image[0], self.map_left_x, self.map_left_y, interpolation=cv2.INTER_LINEAR)
        right_rect = cv2.remap(left_right_image[1], self.map_right_x, self.map_right_y, interpolation=cv2.INTER_LINEAR)

        # Publish rectified images
        try:
            left_msg = self.bridge.cv2_to_imgmsg(left_rect, encoding="bgr8")
            right_msg = self.bridge.cv2_to_imgmsg(right_rect, encoding="bgr8")
            left_msg.header.stamp = self.get_clock().now().to_msg()
            right_msg.header.stamp = left_msg.header.stamp
            self.left_pub.publish(left_msg)
            self.right_pub.publish(right_msg)
        except Exception as e:
            self.get_logger().error(f"Error converting rectified images: {str(e)}")

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


def main(args=None):
    rclpy.init(args=args)
    zed_publisher = ZedStereoPublisher()
    
    try:
        rclpy.spin(zed_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        zed_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()