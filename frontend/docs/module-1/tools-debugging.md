---
sidebar_position: 5
title: "Launch Files and Parameter Management"
---

# Launch Files and Parameter Management

## Launch Files

Launch files in ROS 2 allow you to start multiple nodes with a single command and configure them with specific parameters. They provide a convenient way to manage complex robotic systems.

### Python Launch Files

Python launch files are the recommended approach in ROS 2. They offer more flexibility and are easier to maintain than XML launch files.

#### Basic Launch File Structure

```python
# launch/my_robot_system.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')

    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Name of the robot'
    )

    # Get configuration file path
    config = os.path.join(
        get_package_share_directory('my_robot_package'),
        'config',
        'robot_config.yaml'
    )

    # Define nodes
    robot_driver = Node(
        package='my_robot_package',
        executable='robot_driver',
        name='robot_driver',
        parameters=[
            config,
            {'robot_name': robot_name},
            {'use_sim_time': use_sim_time}
        ],
        output='screen',
        respawn=True,  # Restart if node dies
        respawn_delay=2  # Wait 2 seconds before restarting
    )

    robot_controller = Node(
        package='my_robot_package',
        executable='robot_controller',
        name='robot_controller',
        parameters=[config],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        robot_name_arg,
        robot_driver,
        robot_controller
    ])
```

### Advanced Launch File Features

#### Conditional Launch

```python
from launch import LaunchDescription, LaunchCondition
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch argument to enable/disable a node
    enable_camera = LaunchConfiguration('enable_camera')

    camera_node = Node(
        package='camera_package',
        executable='camera_node',
        name='camera_node',
        condition=LaunchCondition(
            expression=['"', enable_camera, '" == "true"']
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'enable_camera',
            default_value='false',
            description='Enable camera node'
        ),
        camera_node
    ])
```

#### Including Other Launch Files

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Include another launch file
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            )
        ]),
        launch_arguments={
            'use_sim_time': 'true'
        }.items()
    )

    return LaunchDescription([
        navigation_launch
    ])
```

## Parameter Management

### Parameter Declaration and Usage

```python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType


class ParameterExampleNode(Node):
    def __init__(self):
        super().__init__('parameter_example_node')

        # Declare parameters with descriptors
        self.declare_parameter(
            'robot_name',
            'default_robot',
            ParameterDescriptor(
                description='Name of the robot',
                type=ParameterType.PARAMETER_STRING
            )
        )

        self.declare_parameter(
            'max_velocity',
            1.0,
            ParameterDescriptor(
                description='Maximum velocity of the robot',
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[ParameterDescriptor(
                    from_value=0.0,
                    to_value=10.0,
                    step=0.1
                )]
            )
        )

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value

        # Add parameter callback
        self.add_on_set_parameters_callback(self.parameters_callback)

    def parameters_callback(self, parameters):
        """Callback for parameter changes"""
        from rcl_interfaces.msg import SetParametersResult

        for param in parameters:
            if param.name == 'max_velocity':
                if param.value > 5.0:
                    self.get_logger().warn('Max velocity is quite high!')
                elif param.value <= 0.0:
                    return SetParametersResult(successful=False, reason='Max velocity must be positive')

        return SetParametersResult(successful=True)
```

### YAML Parameter Files

YAML parameter files provide a clean way to organize and manage parameters:

```yaml
# config/robot_params.yaml
/**:  # Global namespace
  ros__parameters:
    use_sim_time: false
    log_level: "info"

robot_driver:
  ros__parameters:
    robot_name: "my_robot"
    max_velocity: 1.0
    wheel_diameter: 0.15
    encoder_resolution: 4096
    control_frequency: 50.0

robot_controller:
  ros__parameters:
    kp: 1.0
    ki: 0.1
    kd: 0.05
    max_integral: 10.0
    max_output: 100.0

navigation:
  ros__parameters:
    planner_frequency: 5.0
    controller_frequency: 20.0
    recovery_enabled: true
    clear_costmap_timeout: 2.0
    oscillation_timeout: 0.0
    oscillation_distance: 0.5

sensors:
  ros__parameters:
    lidar_topic: "/scan"
    camera_topic: "/camera/image_raw"
    imu_topic: "/imu/data"
    lidar_enabled: true
    camera_enabled: true
```

### Loading Parameters in Launch Files

```python
# launch/parameter_example.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get parameter file path
    param_file = os.path.join(
        get_package_share_directory('my_robot_package'),
        'config',
        'robot_params.yaml'
    )

    robot_node = Node(
        package='my_robot_package',
        executable='robot_node',
        name='robot_node',
        parameters=[param_file],
        output='screen'
    )

    # Or load parameters with overrides
    robot_node_with_override = Node(
        package='my_robot_package',
        executable='robot_node',
        name='robot_node_with_override',
        parameters=[
            param_file,
            {'robot_name': 'overridden_robot'},  # Override specific parameter
            {'max_velocity': 2.5}
        ],
        output='screen'
    )

    return LaunchDescription([
        robot_node,
        robot_node_with_override
    ])
```

## ROS 2 Tools for Debugging

### ros2 topic

```bash
# List all topics
ros2 topic list

# Get information about a specific topic
ros2 topic info /cmd_vel

# Echo messages from a topic
ros2 topic echo /scan

# Echo with specific number of messages
ros2 topic echo /cmd_vel --field linear.x -n 10

# Publish to a topic
ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 1.0}, angular: {z: 0.5}}'

# Show topic statistics
ros2 topic hz /camera/image_raw
```

### ros2 service

```bash
# List all services
ros2 service list

# Call a service
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts '{a: 1, b: 2}'

# Get service type
ros2 service type /set_parameters
```

### ros2 action

```bash
# List all actions
ros2 action list

# Send a goal to an action
ros2 action send_goal /fibonacci example_interfaces/action/Fibonacci '{order: 5}'
```

### ros2 node

```bash
# List all nodes
ros2 node list

# Get information about a specific node
ros2 node info /robot_driver

# List parameters of a node
ros2 param list /robot_driver

# Get parameter value
ros2 param get /robot_driver robot_name

# Set parameter value
ros2 param set /robot_driver max_velocity 2.0
```

### rqt Tools

rqt is a Qt-based framework for GUI plugins in ROS 2:

```bash
# Start rqt
rqt

# Start specific rqt plugins
rqt_graph          # Shows node graph
rqt_plot           # Plot numeric values
rqt_console        # Shows log messages
rqt_bag            # Record and play back data
rqt_publisher      # Publish messages manually
rqt_subscriber     # Monitor topics
```

## Advanced Debugging Techniques

### Logging

```python
import rclpy
from rclpy.node import Node
import logging


class LoggingExampleNode(Node):
    def __init__(self):
        super().__init__('logging_example_node')

        # Different log levels
        self.get_logger().debug('Debug message')
        self.get_logger().info('Info message')
        self.get_logger().warn('Warning message')
        self.get_logger().error('Error message')
        self.get_logger().fatal('Fatal message')

        # Log with parameters
        robot_name = 'my_robot'
        self.get_logger().info(f'Robot {robot_name} initialized')

        # Set log level programmatically
        self.get_logger().set_level(logging.DEBUG)
```

### Performance Monitoring

```python
import time
from rclpy.node import Node


class PerformanceNode(Node):
    def __init__(self):
        super().__init__('performance_node')
        self.timer = self.create_timer(0.1, self.performance_callback)

    def performance_callback(self):
        start_time = time.time()

        # Your processing code here
        self.process_data()

        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds

        if execution_time > 50:  # Warn if execution takes more than 50ms
            self.get_logger().warn(f'Processing took {execution_time:.2f}ms')
        else:
            self.get_logger().info(f'Processing took {execution_time:.2f}ms')

    def process_data(self):
        # Simulate processing
        time.sleep(0.01)  # Simulate 10ms of processing
```

### Memory Management

```python
import psutil
import os
from rclpy.node import Node


class MemoryMonitoringNode(Node):
    def __init__(self):
        super().__init__('memory_monitoring_node')
        self.process = psutil.Process(os.getpid())
        self.memory_timer = self.create_timer(5.0, self.memory_callback)

    def memory_callback(self):
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        self.get_logger().info(
            f'Memory usage: RSS={memory_info.rss / 1024 / 1024:.2f}MB, '
            f'Percent={memory_percent:.2f}%'
        )
```

## Common Debugging Scenarios

### Topic Connection Issues

```bash
# Check if nodes are publishing/subscribing to topics
ros2 topic info /topic_name

# Check if nodes are alive
ros2 node list

# Verify message types match
ros2 topic type /topic_name
ros2 interface show msg_type
```

### Parameter Issues

```bash
# Check current parameter values
ros2 param list node_name
ros2 param get node_name param_name

# Set parameters at runtime
ros2 param set node_name param_name value
```

### Performance Issues

```bash
# Monitor topic frequency
ros2 topic hz /topic_name

# Monitor CPU usage of nodes
top -p $(pgrep -f node_name)

# Check for memory leaks
watch -n 1 'ps aux | grep node_name'
```

## Summary

Launch files and parameter management are crucial for effectively managing ROS 2 systems:

- Launch files provide a way to start multiple nodes with proper configuration
- Parameters allow runtime configuration of nodes
- ROS 2 provides various tools for debugging and monitoring
- Proper logging and performance monitoring help maintain system health

In the next section, we'll explore bridging Python agents to ROS controllers using rclpy.