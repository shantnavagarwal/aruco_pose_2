import rclpy
from rclpy.node import Node
import yaml
import numpy as np
from ament_index_python.packages import get_package_share_directory
from rclpy.duration import Duration

from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseStamped, Pose
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

from scipy.linalg import svd
from scipy.spatial.transform import Rotation

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



class MultiTfPoseListener(Node):
    def __init__(self, parent_frame: str, child_frames: list):
        super().__init__('multi_tf_pose_listener')

        self.declare_parameter('config_file', 'config/tag_poses.yaml')
        self.config_file = self.get_parameter('config_file').get_parameter_value().string_value

        self.known_tag_poses = self.load_config(self.config_file)
        self.known_tag_detect = {}
        
        self.parent_frame = 'zed_ceiling'
        self.child_frames = ['tag36h11:' + str(k) for k in self.known_tag_poses.keys()]

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0, self.timer_callback)
        
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        tag_poses = {}
        for tag_id, pose in data.items():
            tag_id = int(tag_id)
            T = np.eye(4)
            T[:3, 3] = pose['position']
            T[:3, :3] = Rotation.from_quat(pose['orientation']).as_matrix()
            tag_poses[tag_id] = T
        return tag_poses
    
    def get_pose_child(self, child_frame):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self.parent_frame,
                child_frame,
                now,
                timeout=Duration(seconds=.1)
            )

            pose = PoseStamped()
            pose.header = trans.header
            pose.pose.position.x = trans.transform.translation.x
            pose.pose.position.y = trans.transform.translation.y
            pose.pose.position.z = trans.transform.translation.z
            pose.pose.orientation = trans.transform.rotation

            return pose

        except Exception as e:
            self.get_logger().warn(f"Transform from '{self.parent_frame}' to '{child_frame}' not available: {e}")
            return None
        
    def publish_camera_frame(self):
        P = np.zeros((len(self.known_tag_detect), 3))
        Q = np.zeros((len(self.known_tag_detect), 3))
        i = 0
        for k, p in self.known_tag_detect.items():
            q = self.known_tag_poses[k]
            P[i] = p
            Q[i] = q
            i += 1
        R, t, rms = self.kabsch_numpy(P, Q)
        
        return
    
    @staticmethod
    def kabsch_numpy(P, Q):
        """
        https://hunterheidenreich.com/posts/kabsch_algorithm/
        Computes the optimal rotation and translation to align two sets of points (P -> Q),
        and their RMSD.

        :param P: A Nx3 matrix of points
        :param Q: A Nx3 matrix of points
        :return: A tuple containing the optimal rotation matrix, the optimal
                translation vector, and the RMSD.
        """
        assert P.shape == Q.shape, "Matrix dimensions must match"

        # Compute centroids
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)

        # Optimal translation
        t = centroid_Q - centroid_P

        # Center the points
        p = P - centroid_P
        q = Q - centroid_Q

        # Compute the covariance matrix
        H = np.dot(p.T, q)

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # Validate right-handed coordinate system
        if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
            Vt[-1, :] *= -1.0

        # Optimal rotation
        R = np.dot(Vt.T, U.T)

        # RMSD
        rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / P.shape[0])

        return R, t, rmsd

    def timer_callback(self):
        self.known_tag_detect = {}
        for child_frame in self.child_frames:
            child_pose = self.get_pose_child(child_frame)
            if child_pose is not None:
                self.known_tag_detect[int(child_frame[-1])] = child_pose
        
        
def main(args=None):
    rclpy.init(args=args)

    parent_frame = 'world'
    child_frames = ['camera_link', 'lidar_link', 'gripper_link']  # Replace with actual frame names

    node = MultiTfPoseListener(parent_frame, child_frames)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

