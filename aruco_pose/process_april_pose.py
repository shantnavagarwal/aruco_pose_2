import rclpy
from rclpy.node import Node
import yaml
import numpy as np
from ament_index_python.packages import get_package_share_directory

from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseStamped

from scipy.linalg import svd

def quest_algorithm(world_points, camera_points):
    """Compute rotation and translation using the QUEST algorithm."""
    assert len(world_points) == len(camera_points), "Point sets must match"

    H = np.zeros((3, 3))
    for w, c in zip(world_points, camera_points):
        H += np.outer(c, w)

    U, S, Vt = svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1

    t = np.mean(camera_points, axis=0) - R @ np.mean(world_points, axis=0)
    return R, t


def pose_msg_to_matrix(pose):
    """Convert geometry_msgs/Pose to 4x4 transformation matrix."""
    import tf_transformations as tf
    trans = tf.translation_matrix([pose.position.x, pose.position.y, pose.position.z])
    rot = tf.quaternion_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    return tf.concatenate_matrices(trans, rot)

def matrix_to_pose_msg(matrix):
    """Convert 4x4 matrix to geometry_msgs/Pose."""
    from geometry_msgs.msg import Pose
    import tf_transformations as tf
    pose = Pose()
    trans = tf.translation_from_matrix(matrix)
    quat = tf.quaternion_from_matrix(matrix)
    pose.position.x, pose.position.y, pose.position.z = trans
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
    return pose

class AprilTagLocalizer(Node):
    def __init__(self):
        super().__init__('april_tag_localizer')
        self.declare_parameter('config_file', 'config/tag_poses.yaml')
        self.config_file = self.get_parameter('config_file').get_parameter_value().string_value

        self.known_tag_poses = self.load_config(self.config_file)

        self.subscription = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.publisher = self.create_publisher(PoseStamped, '/unknown_tag_poses', 10)
        self.get_logger().info("AprilTag Localizer Node Started")

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        tag_poses = {}
        for tag_id, pose in data.items():
            tag_id = int(tag_id)
            T = np.eye(4)
            T[:3, 3] = pose['position']
            import tf_transformations as tf
            T[:3, :3] = tf.quaternion_matrix(pose['orientation'])[:3, :3]
            tag_poses[tag_id] = T
        return tag_poses

    def detection_callback(self, msg):
        world_points = []
        camera_points = []
        unknown_tags = []

        for detection in msg.detections:
            tag_id = detection.id[0]
            T_cam_tag = pose_msg_to_matrix(detection.pose.pose.pose)

            T_tag_cam = np.linalg.inv(T_cam_tag)

            if tag_id in self.known_tag_poses:
                T_world_tag = self.known_tag_poses[tag_id]
                p_world = T_world_tag[:3, 3]
                p_camera = T_tag_cam[:3, 3]
                world_points.append(p_world)
                camera_points.append(p_camera)
            else:
                unknown_tags.append((tag_id, T_tag_cam))

        if len(world_points) < 3:
            self.get_logger().warn("Not enough known tags detected for pose estimation.")
            return

        R_wc, t_wc = quest_algorithm(np.array(world_points), np.array(camera_points))
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = R_wc
        T_world_cam[:3, 3] = t_wc

        for tag_id, T_tag_cam in unknown_tags:
            T_world_tag = T_world_cam @ T_tag_cam
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "world"
            pose_msg.pose = matrix_to_pose_msg(T_world_tag)
            self.publisher.publish(pose_msg)
            self.get_logger().info(f"Published pose for unknown tag {tag_id}")

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagLocalizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
