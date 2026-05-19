from setuptools import setup

package_name = 'tb4_arrow_follower'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rahul',
    maintainer_email='ravivk.rahul@gmail.com',
    description='ENPM673 Task 1: arrow following with phone camera + ResNet18 angle regression',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arrow_follower = tb4_arrow_follower.arrow_follower_node_classical:main',
            'arrow_follower_test = tb4_arrow_follower.test:main',
        ],
    },
)
