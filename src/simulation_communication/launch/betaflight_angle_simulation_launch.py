from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # ROS2 pose bridge
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='pose_bridge',
            arguments=['/world/quadcopter/dynamic_pose/info@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V']
        ),
        # ROS2 control bridge
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='control_bridge',
            arguments=['/X3/gazebo/command/motor_speed@actuator_msgs/msg/Actuators]ignition.msgs.Actuators']
        ),
        # ROS2 camera bridge
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='camera_bridge',
            arguments=['/world/quadcopter/model/x3/link/X3/base_link/sensor/camera_sensor/image@sensor_msgs/msg/Image[ignition.msgs.Image']
        ),
        # ROS2 camera info bridge
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='camera_info_bridge',
            arguments=['/world/quadcopter/model/x3/link/X3/base_link/sensor/camera_sensor/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo']
        ),
        Node(
            package='image_transport',
            executable='republish',
            name='image_republish',
            arguments=['raw', 'raw'],
            remappings=[
                ('in',  '/world/quadcopter/model/x3/link/X3/base_link/sensor/camera_sensor/image'),
                ('out', '/image_raw'),
            ],
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='pose_bridgey2',
            arguments=['/model/x3/pose@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V']
        ),
        # Motion Capture Emulator
        Node(
            package='simulation_communication',
            executable='motion_capture_emulator',
            name='motion_capture_emulator',
            parameters=[{'target_object_id': 5}]  # Index of the quadcopter base link in /model/x3/pose
        ),

        Node(
            package='simulation_communication',
            executable='angle_betaflight_communication',
            name='angle_betaflight_communication'
        ),
    ])

