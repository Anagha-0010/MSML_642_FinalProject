from setuptools import find_packages, setup

package_name = 'hri_control'

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
    maintainer='vboxuser_anagha',
    maintainer_email='vboxuser_anagha@todo.todo',
    description='Package for HRI project with RL',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = hri_control.train:main',  # <-- THIS IS THE NEW LINE
            # 'hri_node = hri_control.hri_node:main', # <-- We removed the old node
        ],
    },
)
