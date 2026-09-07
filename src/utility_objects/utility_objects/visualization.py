# This is the visualization.py file for the utility_objects package.
#!/usr/bin/env python3

"""
Trajectory Visualization Module for RViz2

This module provides functions to visualize drone trajectories in RViz2 using various message types:
- Path messages for trajectory paths
- PoseArray messages for waypoints with orientation
- MarkerArray messages for custom trajectory visualization with colors and styles
- PointCloud2 messages for high-density trajectory points

Usage:
    from .visualization import TrajectoryVisualizer
    
    # In your ROS2 node
    visualizer = TrajectoryVisualizer(node)
    visualizer.publish_trajectory_path(trajectory_data)
    visualizer.publish_trajectory_poses(trajectory_data)
    visualizer.publish_trajectory_markers(trajectory_data)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Point, Quaternion, Vector3, TransformStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, PointField
from builtin_interfaces.msg import Time
from tf2_ros import TransformBroadcaster
import struct


class TrajectoryVisualizer:
    
    def __init__(self, node: Node, frame_id: str = "map"):
        self.node = node
        self.frame_id = frame_id
        
        # Create publishers for different visualization types
        self.path_publisher = node.create_publisher(Path, '/trajectory_path', 10)
        self.actual_path_publisher = node.create_publisher(Path, '/actual_path', 10)
        self.mpc_plan_publisher = node.create_publisher(Path, '/mpc_plan', 10)
        self.marker_publisher = node.create_publisher(MarkerArray, '/trajectory_markers', 10)
        
        # Create transform broadcaster instead of pose publishers
        self.tf_broadcaster = TransformBroadcaster(node)
        
        # Store actual path history
        self.actual_path_history = []
        
        self.node.get_logger().info("TrajectoryVisualizer initialized with TF broadcasting")
    
    def _create_header(self) -> Header:
        """Create a header with current timestamp and frame_id"""
        header = Header()
        header.frame_id = self.frame_id
        now = self.node.get_clock().now()
        header.stamp = now.to_msg()
        return header
    
    def _trajectory_to_poses(self, trajectory: np.ndarray) -> list:
        poses = []
        num_points = trajectory.shape[1]
        
        for i in range(num_points):
            pose = Pose()
            
            # Position
            pose.position.x = float(trajectory[0, i])
            pose.position.y = float(trajectory[1, i])
            pose.position.z = float(trajectory[2, i])
            
            # Orientation (quaternion)
            pose.orientation.w = float(trajectory[3, i])
            pose.orientation.x = float(trajectory[4, i])
            pose.orientation.y = float(trajectory[5, i])
            pose.orientation.z = float(trajectory[6, i])
            
            poses.append(pose)
        
        return poses
    
    def publish_trajectory_path(self, trajectory: np.ndarray, color: tuple = (1.0, 0.0, 0.0, 1.0)):
        try:
            path_msg = Path()
            path_msg.header = self._create_header()
            
            poses = self._trajectory_to_poses(trajectory)
            
            for pose in poses:
                pose_stamped = PoseStamped()
                pose_stamped.header = self._create_header()
                pose_stamped.pose = pose
                path_msg.poses.append(pose_stamped)
            
            self.path_publisher.publish(path_msg)
            self.node.get_logger().debug(f"Published trajectory path with {len(poses)} points")
            
        except Exception as e:
            self.node.get_logger().error(f"Error publishing trajectory path: {e}")
    
   
   
    def publish_transform_frame(self, pose_data: np.ndarray, frame_name: str = "drone_mocap"):
        try:
            if pose_data is None or len(pose_data) < 7:
                return
                
            transform = TransformStamped()
            transform.header = self._create_header()
            transform.child_frame_id = frame_name
            
            # Translation
            transform.transform.translation.x = float(pose_data[0])
            transform.transform.translation.y = float(pose_data[1])
            transform.transform.translation.z = float(pose_data[2])
            
            # Rotation (quaternion)
            transform.transform.rotation.w = float(pose_data[3])
            transform.transform.rotation.x = float(pose_data[4])
            transform.transform.rotation.y = float(pose_data[5])
            transform.transform.rotation.z = float(pose_data[6])
            
            # Broadcast the transform
            self.tf_broadcaster.sendTransform(transform)
                
        except Exception as e:
            self.node.get_logger().error(f"Error publishing transform for {frame_name}: {e}")
    
    def publish_actual_path(self, current_pose: np.ndarray):
        try:
            if current_pose is None or len(current_pose) < 7:
                return
            
            # Add current pose to history (limit to last 1000 points to avoid memory issues)
            pose_stamped = PoseStamped()
            pose_stamped.header = self._create_header()
            pose_stamped.pose.position.x = float(current_pose[0])
            pose_stamped.pose.position.y = float(current_pose[1])
            pose_stamped.pose.position.z = float(current_pose[2])
            pose_stamped.pose.orientation.w = float(current_pose[3])
            pose_stamped.pose.orientation.x = float(current_pose[4])
            pose_stamped.pose.orientation.y = float(current_pose[5])
            pose_stamped.pose.orientation.z = float(current_pose[6])
            
            self.actual_path_history.append(pose_stamped)
            
            # Keep only last 200 points (shorter trail)
            if len(self.actual_path_history) > 200:
                self.actual_path_history.pop(0)
            
            # Publish actual path
            path_msg = Path()
            path_msg.header = self._create_header()
            path_msg.poses = self.actual_path_history.copy()
            
            self.actual_path_publisher.publish(path_msg)
            
            # Publish markers for dotted line effect
            self._publish_dotted_path_markers(self.actual_path_history, "actual_path", 
                                            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0))
            
        except Exception as e:
            self.node.get_logger().error(f"Error publishing actual path: {e}")
    
    def publish_mpc_plan(self, mpc_trajectory: np.ndarray):
        try:
            if mpc_trajectory is None or mpc_trajectory.shape[1] < 2:
                return
            
            path_msg = Path()
            path_msg.header = self._create_header()
            
            poses = self._trajectory_to_poses(mpc_trajectory)
            
            for pose in poses:
                pose_stamped = PoseStamped()
                pose_stamped.header = self._create_header()
                pose_stamped.pose = pose
                path_msg.poses.append(pose_stamped)
            
            self.mpc_plan_publisher.publish(path_msg)
            
            # Publish markers for solid purple line
            self._publish_solid_path_markers(path_msg.poses, "mpc_plan", 
                                           ColorRGBA(r=0.5, g=0.0, b=1.0, a=1.0))
            
        except Exception as e:
            self.node.get_logger().error(f"Error publishing MPC plan: {e}")
    
    def _publish_dotted_path_markers(self, poses, namespace, color):
        try:
            marker_array = MarkerArray()
            
            for i, pose in enumerate(poses[::5]):  # Every 5th point for dotted effect
                marker = Marker()
                marker.header = self._create_header()
                marker.ns = namespace
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose = pose.pose
                marker.scale.x = 0.02
                marker.scale.y = 0.02
                marker.scale.z = 0.02
                marker.color = color
                marker_array.markers.append(marker)
            
            self.marker_publisher.publish(marker_array)
            
        except Exception as e:
            self.node.get_logger().error(f"Error publishing dotted markers: {e}")
    
    def _publish_solid_path_markers(self, poses, namespace, color):
        try:
            marker_array = MarkerArray()
            
            if len(poses) < 2:
                return
            
            marker = Marker()
            marker.header = self._create_header()
            marker.ns = namespace
            marker.id = 0
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02  # Line width
            marker.color = color
            
            for pose in poses:
                point = Point()
                point.x = pose.pose.position.x
                point.y = pose.pose.position.y
                point.z = pose.pose.position.z
                marker.points.append(point)
            
            marker_array.markers.append(marker)
            self.marker_publisher.publish(marker_array)
            
        except Exception as e:
            self.node.get_logger().error(f"Error publishing solid markers: {e}")

    def publish_all_visualizations(self, trajectory: np.ndarray, **kwargs):
        self.publish_trajectory_path(trajectory)