from setuptools import setup

package_name = 'utility_objects'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml'])],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Utility objects for drone controllers',
    license='GPL-3.0',
    tests_require=['pytest'],
)
