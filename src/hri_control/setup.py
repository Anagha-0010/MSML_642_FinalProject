from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'hri_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        (os.path.join('share', package_name, 'launch'), 
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
            
        (os.path.join('share', package_name, 'models'), 
            ['models/shadow_hand_right.urdf']), 

        (os.path.join('share', package_name, 'models/meshes/visual'), 
            glob('models/meshes/visual/*')),

        (os.path.join('share', package_name, 'models/meshes/collision'), 
            glob('models/meshes/collision/*')),
            
        # --- THIS IS THE NEW LINE YOU NEED TO ADD ---
        (os.path.join('share', package_name, 'config'), glob('config/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anagha',
    maintainer_email='vboxuser_anagha@todo',
    description='HRI RL Control Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = hri_control.train:main',
            'test = hri_control.test:main',
            'hand_simulator = hri_control.hand_simulator:main',
        ],
    },
)
