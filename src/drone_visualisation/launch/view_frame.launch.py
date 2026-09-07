from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('drone_visualisation')
    urdf_path = os.path.join(pkg_share, 'urdf', 'frame.urdf')
    rviz_config_path = os.path.join(pkg_share, 'rviz', 'default.rviz')

    # Read URDF into robot_description param (no deprecated fallback)
    with open(urdf_path, 'r') as f:
        base_robot_description = f.read()

    # Create multiple robot configurations with different colors
    robots = [
        {
            'name': 'mocap_drone',
            'tf_prefix': 'drone_mocap',
            'robot_description_topic': 'robot_description_mocap',
            'color': {'r': 1.0, 'g': 0.0, 'b': 0.0, 'a': 0.7}  # Red
        },
        {
            'name': 'orbslam_drone', 
            'tf_prefix': 'drone_orbslam',
            'robot_description_topic': 'robot_description_orbslam',
            'color': {'r': 0.0, 'g': 0.0, 'b': 1.0, 'a': 0.7}  # Blue with transparency
        }
    ]

    nodes = []
    
    # Create robot state publisher for each robot
    for robot in robots:
        # Modify URDF to use the specific tf_prefix and color
        robot_description = base_robot_description.replace(
            'name="base_link"', 
            f'name="{robot["tf_prefix"]}/base_link"'
        )
        
        # Replace the material color with the robot-specific color
        color = robot['color']
        robot_description = robot_description.replace(
            '<color rgba="0.15 0.15 0.15 1.0"/>',
            f'<color rgba="{color["r"]} {color["g"]} {color["b"]} {color["a"]}"/>'
        )
        
        # Add robot state publisher node
        nodes.append(
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name=f'robot_state_publisher_{robot["name"]}',
                namespace=robot['name'],
                output='screen',
                parameters=[{
                    'robot_description': robot_description,
                    'tf_prefix': robot['tf_prefix']
                }],
                remappings=[
                    ('robot_description', robot['robot_description_topic'])
                ]
            )
        )
        
        # Add static transform from motion capture frame to robot base_link
        nodes.append(
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name=f'static_tf_{robot["name"]}',
                arguments=['0', '0', '0', '0', '0', '0', robot['tf_prefix'], f'{robot["tf_prefix"]}/base_link']
            )
        )

    # Add RViz node
    nodes.append(
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path]
        )
    )

    return LaunchDescription(nodes)
