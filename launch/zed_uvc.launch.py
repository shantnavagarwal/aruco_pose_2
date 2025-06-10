from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='apriltag_ros',
            executable='apriltag_node',
            name='apriltag_detector',
            output='screen',
            parameters=[{
                'image_transport': 'raw',
                'family': '36h11',
                'size': 1.2,
                'max_hamming': 0,

                'detector.threads': 4,
                'detector.decimate': 2.0,
                'detector.blur': 0.0,
                'detector.refine': True,
                'detector.sharpening': 0.25,
                'detector.debug': False,

                'pose_estimation_method': 'pnp',
            }],
            remappings=[
                ('image_rect', '/camera/image_raw'),
                ('camera_info', '/camera/camera_info'),
            ]
        )
    ])
