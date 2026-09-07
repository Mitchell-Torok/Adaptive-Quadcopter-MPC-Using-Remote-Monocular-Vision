from setuptools import setup
from glob import glob
import os

package_name = 'drone_communication'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='A ROS 2 package for quadcopter communication.',
    license='GPL-3.0',
    entry_points={
        'console_scripts': [
            'elrs_interface = drone_communication.elrs_interface:main',
            'video_interface = drone_communication.video_interface:main',
            'motion_capture_publisher_node = drone_communication.motion_capture_publisher_node:main'
        ],
    },
)
