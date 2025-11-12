from setuptools import setup
import os
from glob import glob

package_name = 'hri_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    
    # This is the crucial part for finding launch/config files
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install all files in the 'launch' directory
        (os.path.join('share', package_name, 'launch'), 
            glob(os.path.join('launch', '*.launch.py'))),
        # Install all files in the 'config' directory
        (os.path.join('share', package_name, 'config'), 
            glob(os.path.join('config', '*'))),
    ],
    
    # This tells pip what to install
    install_requires=[
        'setuptools',
        'numpy',
        'gymnasium',
        'stable-baselines3[extra]',
        'rclpy'
    ],
    zip_safe=True,
    maintainer='vboxuser_anagha',
    maintainer_email='vboxuser_anagha@todo.todo',
    description='HRI project with RL',
    license='Apache-2.0',
    tests_require=['pytest'],
    
    # This creates the 'ros2 run' executables
    entry_points={
        'console_scripts': [
            # Phase 1 test (old)
            'test_joint_move = hri_control.test_joint_move:main',
            # Phase 2
            'hri_env = hri_control.hri_env:main',
            'train = hri_control.train:main',
            # Phase 3
            'hand_simulator = hri_control.hand_simulator_node:main',
            'hri_env_final = hri_control.hri_env_final:main',
            'train_final = hri_control.train_final:main',
        ],
    },
)
