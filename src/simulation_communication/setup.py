from setuptools import find_packages, setup

package_name = 'simulation_communication'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        ('share/' + package_name + '/launch', ['launch/betaflight_angle_simulation_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mitchell',
    maintainer_email='mitch.torok@gmail.com',
    description='TODO: Package description',
    license='GPL-3.0',
    entry_points={
        'console_scripts': [
            'motion_capture_emulator = simulation_communication.motion_capture_emulator:main',
            'angle_betaflight_communication = simulation_communication.angle_betaflight_communication:main',
            'orbslam_emulator = simulation_communication.orbslam_emulator:main',
        ],
    },
)
