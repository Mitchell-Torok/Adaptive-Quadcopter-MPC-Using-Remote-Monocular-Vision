from setuptools import find_packages, setup

package_name = 'controller_angle_ukf'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mitchell',
    maintainer_email='mitch.torok@gmail.com',
    description='TODO: Package description',
    license='GPL-3.0',
    entry_points={
        'console_scripts': [
            'main = controller_angle_ukf.main:main',
            'delay_estimator_node = controller_angle_ukf.delay_estimator_node:main'
        ],
    },
)
