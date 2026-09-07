#!/usr/bin/env python3

import numpy as np

# Patch np.float to avoid issues with deprecated alias
if not hasattr(np, 'float'):
    np.float = float

import rclpy
from rclpy.node import Node
from actuator_msgs.msg import Actuators
from geometry_msgs.msg import Twist, PoseArray, Pose, PoseStamped
from tf_transformations import euler_from_quaternion, quaternion_multiply, quaternion_inverse, quaternion_matrix
from builtin_interfaces.msg import Time
import signal
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
from interfaces.msg import MotionCaptureState 

class PentaVerify(Node):
    def __init__(self):
        super().__init__('penta_verify')
        self.worldPoseSub_ = self.create_subscription(PoseArray, '/model/x3/pose', self.worldPoseCallback, 10)
        self.publisher = self.create_publisher(MotionCaptureState, '/orb_slam_state', 10)
        self.pose_publisher = self.create_publisher(PoseStamped, '/rviz_pose', 10)

        self.last_pose = None
        self.last_orientation = None
        self.last_time = None

        self.databuffer = []

        self.dt = 1.0 /30.0
        self.timer = self.create_timer(self.dt, self.orbslam_state_publisher_callback)

        self.current_position = Pose().position
        self.current_orientation = Pose().orientation


        self.delay_buffer_length = 6
        
        # Initialize delay buffer
        self.delay_buffer = []

    def normalize_quaternion_positive_w(self, x, y, z, w):
        """Normalize quaternion and ensure w is positive."""
        # If w is negative, negate the quaternion
        if w < 0:
            return -x, -y, -z, -w
        return x, y, z, w

    def worldPoseCallback(self, msg):
        print("START")
        # Extract current pose and time # 9 for omnicopter, 5 for quadcopter
        # Add noise to the current position
        noise_position = np.random.normal(0, 0.0001, 3)  # Mean 0, standard deviation 0.01
        self.current_position = msg.poses[5].position
        self.current_position.x += noise_position[0]
        self.current_position.y += noise_position[1]
        self.current_position.z += noise_position[2]

        # Add noise to the current orientation
        noise_orientation = np.random.normal(0, 0.00001, 4)  # Mean 0, standard deviation 0.001
        self.current_orientation = msg.poses[5].orientation
        self.current_orientation.x += noise_orientation[0]
        self.current_orientation.y += noise_orientation[1]
        self.current_orientation.z += noise_orientation[2]
        self.current_orientation.w += noise_orientation[3]

        # Normalize the quaternion to ensure it remains valid
        norm = math.sqrt(
            self.current_orientation.x**2 +
            self.current_orientation.y**2 +
            self.current_orientation.z**2 +
            self.current_orientation.w**2
        )
        self.current_orientation.x /= norm
        self.current_orientation.y /= norm
        self.current_orientation.z /= norm
        self.current_orientation.w /= norm
       
       


    def orbslam_state_publisher_callback(self):

    
        # Ensure w is positive
        self.current_orientation.x, self.current_orientation.y, self.current_orientation.z, self.current_orientation.w = self.normalize_quaternion_positive_w(
            self.current_orientation.x, self.current_orientation.y, self.current_orientation.z, self.current_orientation.w
        )
        
        current_time = self.get_clock().now().to_msg()

        if self.last_pose is None:
            # Initialize the last pose, orientation, and time
            self.last_pose = self.current_position
            self.last_orientation = self.current_orientation
            self.last_time = current_time
            return

        # Calculate time difference (dt)
        dt = (current_time.sec + current_time.nanosec * 1e-9) - (self.last_time.sec + self.last_time.nanosec * 1e-9)

        print(f"Time difference (dt): {dt:.6f} seconds")

        if dt <= 0:

            print("Time difference is non-positive, skipping callback.")
            # Skip this callback if the time difference is non-positive
            self.last_pose = self.current_position
            self.last_orientation = self.current_orientation
            self.last_time = current_time
            return


        # Calculate linear velocity in the world frame
        dx = self.current_position.x - self.last_pose.x
        dy = self.current_position.y - self.last_pose.y
        dz = self.current_position.z - self.last_pose.z
        linear_velocity_world = np.array([dx / dt, dy / dt, dz / dt])

        # Calculate angular velocity
        q1 = [self.last_orientation.x, self.last_orientation.y, self.last_orientation.z, self.last_orientation.w]
        q2 = [self.current_orientation.x, self.current_orientation.y, self.current_orientation.z, self.current_orientation.w]
        q_relative = quaternion_multiply(q2, quaternion_inverse(q1))  # Relative rotation
        angular_velocity = 2 * np.array([q_relative[0], q_relative[1], q_relative[2]]) / dt  # Angular velocity
        q_current = [self.current_orientation.x, self.current_orientation.y, self.current_orientation.z, self.current_orientation.w]
        rotation_matrix = quaternion_matrix(q1)[:3, :3]  # Extract 3x3 rotation part

        # Transform  velocity from world frame to body frame
        angular_velocity_body = np.dot(rotation_matrix.T, angular_velocity)

        # Create and manage delay buffer
        if len(self.delay_buffer) >= self.delay_buffer_length:
            # Publish the oldest message in the buffer
            oldest_mcs = self.delay_buffer.pop(0)
            self.publisher.publish(oldest_mcs)

        # Create a new MotionCaptureState message
        mcs = MotionCaptureState()
        mcs.pose = Pose()
        mcs.pose.position.x = float(self.current_position.x)
        mcs.pose.position.y = float(self.current_position.y)
        mcs.pose.position.z = float(self.current_position.z)
        mcs.pose.orientation.x = float(self.current_orientation.x)
        mcs.pose.orientation.y = float(self.current_orientation.y)
        mcs.pose.orientation.z = float(self.current_orientation.z)
        mcs.pose.orientation.w = float(self.current_orientation.w)

        mcs.twist = Twist()
        mcs.twist.linear.x = float(linear_velocity_world[0])
        mcs.twist.linear.y = float(linear_velocity_world[1])
        mcs.twist.linear.z = float(linear_velocity_world[2])
        mcs.twist.angular.x = float(angular_velocity_body[0])
        mcs.twist.angular.y = float(angular_velocity_body[1])
        mcs.twist.angular.z = float(angular_velocity_body[2])

        # Append the new message to the delay buffer
        self.delay_buffer.append(mcs)

        # Update last pose, orientation, and time
        self.last_pose = self.current_position
        self.last_orientation = self.current_orientation
        self.last_time = current_time





def main(args=None):
    rclpy.init(args=args)
    pentaVerify = PentaVerify()
    rclpy.spin(pentaVerify)
    pentaVerify.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
