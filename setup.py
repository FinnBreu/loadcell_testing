from glob import glob
from setuptools import find_packages
from setuptools import setup

package_name = 'loadcell_testing'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/docs', glob('docs/*.tex')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'matplotlib', 'numpy', 'requests', 'scipy', 'PyYAML'],
    zip_safe=True,
    maintainer='Finn Breuer',
    maintainer_email='finn.breuer@tuhh.de',
    description='LAN-XI loadcell publisher and independent thruster test tools.',
    license='GPLv2',
    entry_points={
        'console_scripts': [
            'lanxi_loadcell_publisher = loadcell_testing.lanxi_loadcell_publisher:main',
            'single_thruster_sweep = loadcell_testing.single_thruster_sweep:main',
            'analyze_thruster_run = loadcell_testing.analyze_thruster_run:main',
        ],
    },
)
