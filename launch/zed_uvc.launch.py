from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            output='screen',
            parameters=[{
                'video_device': '/dev/video2',          # Confirm device path
                'image_width': 1280,                    # Max UVC width for ZED 2i
                'image_height': 720,                    # Max UVC height
                'framerate': 30.0,                       # FPS
                'auto_focus': False,
                'pixel_format': 'yuyv',                 # Common UVC format
                'camera_name': 'zed2i',                 # Camera frame ID
                'brightness': 50,                       # Adjust as needed (0-100)
            }]
        )
    ])