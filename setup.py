from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'autonomous_state_machine'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'scipy', 'numpy'],
    zip_safe=True,
    maintainer='yonder',
    maintainer_email='arsureshkumar@ucsd.edu',
    description='Autonomous state machine ROS2 node.',
    license='BSD-3-Clause',
    extras_require={
        'terrain': ['rasterio>=1.3,<2'],
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'autonomous_state_machine = autonomous_state_machine.autonomous_state_machine:main',
        ],
    },
)
