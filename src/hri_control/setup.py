from setuptools import find_packages, setup
import os           # <-- MAKE SURE THIS IS HERE
from glob import glob     # <-- MAKE SURE THIS IS HERE

package_name = 'hri_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # --- MAKE SURE THIS BLOCK IS HERE ---
        (os.path.join('share', package_name, 'launch'), 
            glob(os.path.join('launch', '*.launch.py'))),
        # --- END OF BLOCK ---
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vboxuser_anagha',
    maintainer_email='vboxuser_anagha@todo.todo',
    description='Package for HRI project with RL',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = hri_control.train:main',
            'hand_simulator = hri_control.hand_simulator:main',
            'test = hri_control.test:main',
        ],
    },
)
