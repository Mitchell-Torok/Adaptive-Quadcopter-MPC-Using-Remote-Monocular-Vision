import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Header
from interfaces.msg import MotionCaptureState  # Import the new message type
from geometry_msgs.msg import Twist, PoseArray, Pose, PoseStamped
from tf_transformations import quaternion_multiply, quaternion_inverse, quaternion_matrix
import socket
from dataclasses import dataclass
from typing import Tuple, Optional
import sys
import time
import numpy as np
from collections import deque  # Import deque for the rolling average filter
import csv
import scipy.signal  # Import scipy.signal for Butterworth filter
import re 

@dataclass
class ObjectData:
    id: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]
    velocity: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]

class ParseData():
    def __init__(self):
        self.last_pose = np.array([0, 0, 0])
        self.last_orientation = np.array([0, 0, 0, 1])
        self.last_time = time.time()

        '''self.last_pen_pose = np.array([0, 0, 0])
        self.last_pen_orientation = np.array([0, 0, 0, 1])
        self.last_pen_time = time.time()'''
        # Rolling average buffers for position, orientation, velocity, and angular velocity

        self.velocity_buffers = {
            'x': deque(maxlen=8),
            'y': deque(maxlen=8),
            'z': deque(maxlen=8)
        }
        self.angular_velocity_buffers = {
            'x': deque(maxlen=8),
            'y': deque(maxlen=8),
            'z': deque(maxlen=8)
        }

        # Rolling average buffers for velocitykpz, kiz, kdz = 15.0, 10.0, 10.0 
        self.rolling_velocity_buffers = {
            'x': deque(maxlen=8),
            'y': deque(maxlen=8),
            'z': deque(maxlen=8)
        }

        # Low-pass filter buffers for velocity
        self.low_pass_velocity_buffers = {
            'x': deque(maxlen=8),
            'y': deque(maxlen=8),
            'z': deque(maxlen=8)
        }

        # Butterworth filter buffers for velocity
        self.butter_velocity_buffers = {
            'x': deque(maxlen=8),
            'y': deque(maxlen=8),
            'z': deque(maxlen=8)
        }

        # Rolling average buffers for angular velocity
        self.rolling_angular_velocity_buffers = {
            'x': deque(maxlen=8),
            'y': deque(maxlen=8),
            'z': deque(maxlen=8)
        }

        # Low-pass filter buffers for angular velocity
        self.low_pass_angular_velocity_buffers = {
            'x': deque(maxlen=8),
            'y': deque(maxlen=8),
            'z': deque(maxlen=8)
        }

        # Butterworth filter buffers for angular velocity
        self.butter_angular_velocity_buffers = {
            'x': deque(maxlen=8),
            'y': deque(maxlen=8),
            'z': deque(maxlen=8)
        }

        # Low-pass filter parameters
        self.alpha = 0.2  # Smoothing factor (0 < alpha <= 1)


        # Butterworth filter parameters
        self.butter_cutoff = 0.1  # Cutoff frequency (normalized, 0 < butter_cutoff < 0.5)
        self.butter_order = 2    # Order of the Butterworth filter


        # Precompute Butterworth filter coefficients
        self.butter_b, self.butter_a = scipy.signal.butter(
            self.butter_order, self.butter_cutoff, btype='low', analog=False
        )

        # Initialize filtered values for velocity and angular velocity
        self.filtered_velocity = np.array([0.0, 0.0, 0.0])
        self.filtered_angular_velocity = np.array([0.0, 0.0, 0.0])
       


    '''def clean_message(self, message: str) -> str:
        message = message.replace('-(', '|').replace(')-', '|')
        message = message.replace('(', '').replace(')', '')
        message = message.strip()
        message = message.replace('||', '|')
        return message'''
    
    def normalize_quaternion_positive_w(self, x, y, z, w):
        """Normalize quaternion and ensure w is positive."""
        print(f"Normalizing quaternion: {x}, {y}, {z}, {w}")
        if w < 0:
            print(f"Quaternion w is negative, negating all components.")
            return -x, -y, -z, -w
        print(f"Quaternion w is positive, no change needed.")
        return x, y, z, w

    def apply_rolling_average(self, buffers, *values):
        """Apply a rolling average filter to the given components."""
        for key, value in zip(buffers.keys(), values):
            buffers[key].append(value)

        averages = tuple(sum(buffers[key]) / len(buffers[key]) for key in buffers.keys())
        return averages

    def apply_low_pass_filter(self, buffers, *values):
        """Apply a low-pass filter to the given components."""
        filtered_values = []
        for key, value in zip(buffers.keys(), values):
            if len(buffers[key]) == 0:
                # Initialize the buffer with the first value
                buffers[key].append(value)
            else:
                # Apply low-pass filter
                filtered_value = self.alpha * value + (1 - self.alpha) * buffers[key][-1]
                buffers[key].append(filtered_value)
            filtered_values.append(buffers[key][-1])
        return tuple(filtered_values)

    def apply_butterworth_filter(self, buffers, *values):
        """Apply a Butterworth filter to the given components."""
        filtered_values = []
        for key, value in zip(buffers.keys(), values):
            buffers[key].append(value)
            if len(buffers[key]) < len(self.butter_b):
                # Not enough data to apply the filter yet
                filtered_values.append(value)
            else:
                # Apply the Butterworth filter
                filtered_value = scipy.signal.lfilter(
                    self.butter_b, self.butter_a, list(buffers[key])
                )[-1]
                filtered_values.append(filtered_value)
        return tuple(filtered_values)

    def parse_packet(self, data: str) -> Optional[ObjectData]:
        try:
            if not data or '|' not in data:
                print("Invalid data format, skipping packet.")
                return None

            parts = [p.strip() for p in data.split('|') if p.strip()]
            if len(parts) != 3:
                print("Invalid parts length.")
                return None

            obj_id, pos_str, rot_str = parts

            
            # Parse position
            pos_parts = [p.strip() for p in pos_str.split(',')]
            if len(pos_parts) != 3:
                print("Invalid position format.")
                return None
            r_x, r_y, r_z = map(float, pos_parts)
            
            # Apply rolling average filter to position
            #x, y, z = self.apply_rolling_average(self.position_buffers, x, y, z)
            
            # Parse orientation
            rot_parts = [p.strip() for p in rot_str.split(',')]
            if len(rot_parts) != 4:
                print("Invalid rotation format.")
                return None
            r_qx, r_qy, r_qz, r_qw = map(float, rot_parts)
            
            # Apply rolling average filter to orientation
            #qx, qy, qz, qw = self.apply_rolling_average(self.orientation_buffers, qx, qy, qz, qw)

            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            

            if dt <= 0:
                self.last_pose = np.array([r_x, r_y, r_z])
                self.last_orientation = np.array([r_qx, r_qy, r_qz, r_qw])
                self.last_time = current_time
                return None

            # Calculate linear velocity in the world frame
            dx, dy, dz = r_x - self.last_pose[0], r_y - self.last_pose[1], r_z - self.last_pose[2]
            linear_velocity_world = np.array([dx / dt, dy / dt, dz / dt])


            # Apply low-pass filter to velocity
            low_pass_vx, low_pass_vy, low_pass_vz = self.apply_low_pass_filter(
                self.low_pass_velocity_buffers, *linear_velocity_world
            )


            # Calculate angular velocity
            q1 = self.last_orientation
            q2 = np.array([r_qx, r_qy,r_qz, r_qw])
            q_relative = quaternion_multiply(q2, quaternion_inverse(q1))  # Relative rotation
            angular_velocity = 2 * np.array([q_relative[0], q_relative[1], q_relative[2]]) / dt  # Angular velocity
            rotation_matrix = quaternion_matrix(q2)[:3, :3]  # Extract 3x3 rotation part

            # Transform velocity from world frame to body frame
            angular_velocity_body = np.dot(rotation_matrix.T, angular_velocity)

        
            low_pass_wx, low_pass_wy, low_pass_wz = self.apply_low_pass_filter(
                self.low_pass_angular_velocity_buffers, *angular_velocity_body
            )


            # Update last pose, orientation, and time
            self.last_pose = np.array([r_x, r_y, r_z])
            self.last_orientation = np.array([r_qx, r_qy, r_qz, r_qw])


            # Round angular velocity (rad/s) to 1 decimal place
            low_pass_wx = round(low_pass_wx, 1)
            low_pass_wy = round(low_pass_wy, 1)
            low_pass_wz = round(low_pass_wz, 1)

            # Convert angular velocity from rad/s to deg/s for logging
            deg_wx = np.degrees(low_pass_wx)
            deg_wy = np.degrees(low_pass_wy)
            deg_wz = np.degrees(low_pass_wz)
            # Format with constant width, including sign and padding
            print(
                f"Angular velocity (body frame): "
                f"wx={low_pass_wx:+08.4f} rad/s ({deg_wx:+08.2f}°/s), "
                f"wy={low_pass_wy:+08.4f} rad/s ({deg_wy:+08.2f}°/s), "
                f"wz={low_pass_wz:+08.4f} rad/s ({deg_wz:+08.2f}°/s)"
            )




            
            return ObjectData(
                id=obj_id,
                position=(r_x, r_y, r_z),
                rotation=(r_qw, r_qx, r_qy, r_qz),
                velocity=(low_pass_vx, low_pass_vy, low_pass_vz),  # Use Butterworth-filtered velocity
                angular_velocity=(low_pass_wx, low_pass_wy, low_pass_wz)  # Use Butterworth-filtered angular velocity
            )
        except Exception:
            return None
        
    

class MotionCapturePublisher(Node):
    def __init__(self):
        super().__init__('udp_to_pose_node')
        
        # ROS2 Publisher
        self.publisher = self.create_publisher(MotionCaptureState, 'motion_capture_state', 10)
        self.pose_publisher = self.create_publisher(PoseStamped, '/rviz_pose', 10)
        self.pen_publisher = self.create_publisher(MotionCaptureState, '/pendulum_state_publisher', 10)
        # UDP Setup
        #self.HOST = "192.168.1.105"
        #self.PORT = 1511
        self.HOST = "192.168.0.87" #"192.168.1.105"
        self.PORT = 1511
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.HOST, self.PORT))
        
        self.get_logger().info(f'Listening for UDP on {self.HOST}:{self.PORT}')

        

    def clean_message(self, message: str) -> str:
        message = message.replace('-(', '|').replace(')-', '|')
        message = message.replace('(', '').replace(')', '')
        message = message.strip()
        message = message.replace('||', '|')
        return message
    
    def create_motion_capture_state_msg(self, obj_data: ObjectData) -> MotionCaptureState:
        msg = MotionCaptureState()
        
        # Set header
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = obj_data.id
        
        # Set pose
        msg.pose = Pose()
        msg.pose.position.x = obj_data.position[0]
        msg.pose.position.y = obj_data.position[1]
        msg.pose.position.z = obj_data.position[2]
        msg.pose.orientation.w = obj_data.rotation[0]
        msg.pose.orientation.x = obj_data.rotation[1]
        msg.pose.orientation.y = obj_data.rotation[2]
        msg.pose.orientation.z = obj_data.rotation[3]
        
        # Set twist (velocity)
        msg.twist = Twist()
        msg.twist.linear.x = obj_data.velocity[0]
        msg.twist.linear.y = obj_data.velocity[1]
        msg.twist.linear.z = obj_data.velocity[2]
        msg.twist.angular.x = obj_data.angular_velocity[0]
        msg.twist.angular.y = obj_data.angular_velocity[1]
        msg.twist.angular.z = obj_data.angular_velocity[2]
        
        return msg


    def run(self):
        self.penParseData = ParseData()
        self.quadParseData = ParseData()
        try:
            while rclpy.ok():
                data, _ = self.sock.recvfrom(255)
                message = data.decode().strip()
                
                data = self.clean_message(message)
                try:
                    if not data or '|' not in data:
                        print("Invalid data format, skipping packet.")
                        return None

                    parts = [p.strip() for p in data.split('|') if p.strip()]
                    if len(parts) != 3:
                        print("Invalid parts length.")
                        return None

                    obj_id, pos_str, rot_str = parts
                except Exception:
                    print("Error")
                    continue

                if re.search("8", obj_id):
                    # use pen parse data object
                    obj_data = self.penParseData.parse_packet(data)
                    pendulum_msg = self.create_motion_capture_state_msg(obj_data)
                    self.pen_publisher.publish(pendulum_msg)
                    
                else:
                    #use quad parse data object
                    obj_data = self.quadParseData.parse_packet(data)
                    motion_capture_msg = self.create_motion_capture_state_msg(obj_data)
                    self.publisher.publish(motion_capture_msg)

                
        except KeyboardInterrupt:
            self.get_logger().info('Shutting down...')
        finally:
            self.sock.close()

def main(args=None):
    rclpy.init(args=args)
    node = MotionCapturePublisher()
    
    try:
        node.run()
    except Exception as e:
        node.get_logger().error(f'Error: {str(e)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()