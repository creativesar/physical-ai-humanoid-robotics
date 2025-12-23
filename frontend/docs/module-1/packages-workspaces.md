---
sidebar_position: 4
title: "Building ROS 2 Packages with Python"
---

# Building ROS 2 Packages with Python

## Introduction to ROS 2 Packages

A ROS 2 package is the basic unit of organization for ROS 2 software. It contains nodes, libraries, configuration files, and other resources needed for a specific functionality. Packages provide a way to organize, distribute, and reuse ROS 2 software.

## Package Structure

A typical ROS 2 package has the following structure:

```
my_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml             # Package metadata
├── setup.py                # Python package setup
├── setup.cfg               # Installation configuration
├── my_package/             # Python module
│   ├── __init__.py
│   └── my_module.py
├── launch/                 # Launch files
│   └── my_launch_file.py
├── config/                 # Configuration files
│   └── params.yaml
├── test/                   # Test files
│   └── test_my_module.py
└── resource/               # Resource files
    └── my_package
```

## Creating a Python Package

### 1. Using colcon to create a package

```bash
# Create a new workspace
mkdir -p ~/ros2_workspace/src
cd ~/ros2_workspace/src

# Create a new Python package
ros2 pkg create --build-type ament_python my_robot_package --dependencies rclpy std_msgs geometry_msgs
```

### 2. Package.xml Configuration

The `package.xml` file contains metadata about your package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Example robot package</description>
  <maintainer email="user@example.com">User Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### 3. setup.py Configuration

The `setup.py` file defines how your Python package should be installed:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User Name',
    maintainer_email='user@example.com',
    description='Example robot package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = my_robot_package.my_node:main',
            'another_node = my_robot_package.another_node:main',
        ],
    },
)
```

## Python Node Implementation

### Basic Node Structure

```python
#!/usr/bin/env python3
"""
Example ROS 2 node implementation
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class MyRobotNode(Node):
    def __init__(self):
        super().__init__('my_robot_node')

        # Create publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_publisher = self.create_publisher(String, '/robot_status', 10)

        # Create subscribers
        self.cmd_subscriber = self.create_subscription(
            String,
            '/command',
            self.command_callback,
            10
        )

        # Create timers
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

        # Declare parameters
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('max_velocity', 1.0)

        self.get_logger().info('MyRobotNode initialized')

    def command_callback(self, msg):
        """Handle incoming commands"""
        self.get_logger().info(f'Received command: {msg.data}')
        # Process command logic here

    def timer_callback(self):
        """Timer callback - runs at 10 Hz"""
        # Publish robot status
        status_msg = String()
        status_msg.data = 'Operating normally'
        self.status_publisher.publish(status_msg)

        # Other periodic tasks


def main(args=None):
    rclpy.init(args=args)

    node = MyRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Workspace Management

### Creating a Workspace

```bash
# Create workspace directory
mkdir -p ~/my_robot_ws/src

# Navigate to source directory
cd ~/my_robot_ws/src

# Create or copy packages to src directory
# (packages can be created with ros2 pkg create or copied from elsewhere)

# Build the workspace
cd ~/my_robot_ws
colcon build --packages-select my_robot_package

# Source the workspace
source install/setup.bash
```

### Building with colcon

```bash
# Build all packages in workspace
colcon build

# Build specific package
colcon build --packages-select my_robot_package

# Build with additional options
colcon build --packages-select my_robot_package --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

## Advanced Package Features

### 1. Parameter Management

```python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('wheel_diameter', 0.1)
        self.declare_parameter('max_speed', 1.0)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.wheel_diameter = self.get_parameter('wheel_diameter').value
        self.max_speed = self.get_parameter('max_speed').value

        # Add parameter callback for dynamic reconfiguration
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        """Handle parameter changes"""
        for param in params:
            if param.name == 'max_speed' and param.value > 5.0:
                return SetParametersResult(successful=False, reason='Max speed too high')
        return SetParametersResult(successful=True)
```

### 2. Lifecycle Nodes

```python
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn


class LifecycleRobotNode(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_robot_node')

    def on_configure(self, state):
        """Called when transitioning to CONFIGURING state"""
        self.get_logger().info('Configuring lifecycle node')
        # Initialize resources
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """Called when transitioning to ACTIVATING state"""
        self.get_logger().info('Activating lifecycle node')
        # Activate publishers/subscribers
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        """Called when transitioning to DEACTIVATING state"""
        self.get_logger().info('Deactivating lifecycle node')
        # Deactivate publishers/subscribers
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        """Called when transitioning to CLEANINGUP state"""
        self.get_logger().info('Cleaning up lifecycle node')
        # Clean up resources
        return TransitionCallbackReturn.SUCCESS
```

## Testing Your Package

### Unit Tests

```python
# test/test_my_robot_package.py
import unittest
import rclpy
from my_robot_package.my_node import MyRobotNode


class TestMyRobotNode(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = MyRobotNode()

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_node_initialization(self):
        self.assertEqual(self.node.get_name(), 'my_robot_node')

    def test_parameter_defaults(self):
        # Test that parameters have expected default values
        pass


if __name__ == '__main__':
    unittest.main()
```

### Launch Files

```python
# launch/my_launch_file.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('my_robot_package'),
        'config',
        'params.yaml'
    )

    my_robot_node = Node(
        package='my_robot_package',
        executable='my_node',
        name='my_robot_node',
        parameters=[config],
        output='screen'
    )

    return LaunchDescription([
        my_robot_node
    ])
```

## Configuration Files

### YAML Parameter Files

```yaml
# config/params.yaml
my_robot_node:
  ros__parameters:
    robot_name: "my_custom_robot"
    wheel_diameter: 0.15
    max_speed: 2.0
    sensors:
      lidar_enabled: true
      camera_enabled: true
    navigation:
      planner_frequency: 5.0
      controller_frequency: 10.0
```

## Best Practices

1. **Package Naming**: Use descriptive names that reflect the package's purpose
2. **Dependencies**: Only declare necessary dependencies in package.xml
3. **Documentation**: Include README files and proper docstrings
4. **Testing**: Write unit tests for your nodes and functions
5. **Error Handling**: Implement proper error handling and logging
6. **Resource Management**: Properly clean up resources in node destruction
7. **Code Style**: Follow PEP 8 and ROS 2 coding standards

## Summary

Building ROS 2 packages with Python involves understanding the package structure, configuration files, and node implementation patterns. Key concepts include:

- Proper package.xml metadata
- Correct setup.py configuration
- Well-structured Python nodes
- Parameter management
- Testing and launch files

In the next section, we'll explore launch files and parameter management in detail.