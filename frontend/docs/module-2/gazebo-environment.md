---
sidebar_position: 3
title: "Gazebo Simulation Environment Setup"
---

# Gazebo Simulation Environment Setup

## Introduction to Gazebo

Gazebo is a powerful, open-source robotics simulator that provides high-fidelity physics simulation, realistic rendering, and convenient programmatic interfaces. It's widely used in the robotics community for testing algorithms, robot design, and training AI systems.

Gazebo features:
- Multiple physics engines (ODE, Bullet, SimBody)
- High-quality graphics rendering
- Extensive robot models and environments
- ROS/ROS 2 integration
- Sensor simulation (cameras, LiDAR, IMU, etc.)
- Plugins system for custom functionality

## Installing Gazebo

### Gazebo Garden (Latest Version)

For ROS 2 Humble and newer distributions, Gazebo Garden is recommended:

```bash
# Update package list
sudo apt update

# Install Gazebo Garden
sudo apt install gazebo-garden

# Install ROS 2 Gazebo packages
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-gazebo-ros2-control
sudo apt install ros-humble-gazebo-ros2-control-demos
```

### Alternative: Ignition Gazebo

For more recent versions:

```bash
# Add OSRF APT repository
sudo apt update && sudo apt install wget
sudo sh -c 'echo "deb [arch=amd64] http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt update

# Install Ignition Gazebo
sudo apt install ignition-harmonic
```

## Basic Gazebo Setup and Configuration

### Environment Variables

Set up environment variables for Gazebo:

```bash
# Add to ~/.bashrc
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/.gazebo/models
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:~/.gazebo/models
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:~/.gazebo/plugins
```

### Gazebo Configuration

Create a basic Gazebo configuration in your ROS 2 workspace:

```xml
<!-- config/gazebo_config.xml -->
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="default">
    <!-- Include standard world -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
  </world>
</sdf>
```

## Launching Gazebo with ROS 2

### Basic Launch File

```python
# launch/gazebo_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')

    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true'
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=[PathJoinSubstitution([FindPackageShare('my_robot_gazebo'), 'worlds', 'my_world.sdf'])],
        description='SDF world file'
    )

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py'])
        ]),
        launch_arguments={
            'world': world,
            'use_sim_time': use_sim_time
        }.items()
    )

    return LaunchDescription([
        use_sim_time_arg,
        world_arg,
        gazebo
    ])
```

### World File Example

```xml
<!-- worlds/simple_world.sdf -->
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="simple_world">
    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add a simple box obstacle -->
    <model name="box_obstacle">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
            <specular>1 0 0 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1667</iyy>
            <iyz>0</iyz>
            <izz>0.1667</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Physics configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
  </world>
</sdf>
```

## Integrating Robots with Gazebo

### Robot Spawn Launch File

```python
# launch/spawn_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')
    z_pose = LaunchConfiguration('z_pose', default='0.0')

    # Robot description
    robot_description_content = Command([
        'xacro ',
        FindPackageShare('my_robot_description'),
        '/urdf/my_robot.urdf.xacro'
    ])

    robot_description = {'robot_description': robot_description_content}

    # Spawn robot node
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', robot_name,
            '-x', x_pose,
            '-y', y_pose,
            '-z', z_pose
        ],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    return LaunchDescription([
        robot_state_publisher,
        spawn_entity
    ])
```

## Gazebo Plugins for ROS 2

### Joint State Publisher Plugin

```xml
<!-- Include in robot URDF/XACRO -->
<gazebo>
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <ros>
      <namespace>/my_robot</namespace>
      <remapping>~/out:=joint_states</remapping>
    </ros>
    <update_rate>30</update_rate>
    <joint_name>joint1</joint_name>
    <joint_name>joint2</joint_name>
  </plugin>
</gazebo>
```

### Diff Drive Controller Plugin

```xml
<gazebo>
  <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
    <ros>
      <namespace>/my_robot</namespace>
      <remapping>cmd_vel:=cmd_vel</remapping>
      <remapping>odom:=odom</remapping>
      <remapping>tf:=tf</remapping>
    </ros>
    <update_rate>30</update_rate>
    <!-- Wheel Information -->
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.3</wheel_separation>
    <wheel_diameter>0.1</wheel_diameter>

    <!-- Limits -->
    <max_wheel_torque>20</max_wheel_torque>
    <max_wheel_acceleration>1.0</max_wheel_acceleration>

    <!-- Output -->
    <odometry_frame>odom</odometry_frame>
    <robot_base_frame>base_link</robot_base_frame>
    <publish_odom>true</publish_odom>
    <publish_odom_tf>true</publish_odom_tf>
    <publish_wheel_tf>true</publish_wheel_tf>
  </plugin>
</gazebo>
```

## Advanced Gazebo Configuration

### Physics Engine Tuning

```xml
<physics type="ode">
  <!-- Time stepping -->
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>

  <!-- Gravity -->
  <gravity>0 0 -9.8</gravity>

  <!-- ODE-specific parameters -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Rendering Configuration

```xml
<!-- Adjust rendering quality -->
<scene>
  <ambient>0.4 0.4 0.4 1.0</ambient>
  <background>0.7 0.7 0.7 1.0</background>
  <shadows>true</shadows>
</scene>

<!-- Camera configuration -->
<sensor name="camera" type="camera">
  <camera name="head">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>800</width>
      <height>600</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
</sensor>
```

## Working with Gazebo Models

### Creating Custom Models

Create a directory structure for your custom model:

```
~/.gazebo/models/my_robot_model/
├── model.config
└── model.sdf
```

**model.config**:
```xml
<?xml version="1.0"?>
<model>
  <name>My Robot Model</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>A custom robot model for simulation</description>
</model>
```

**model.sdf**:
```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="my_robot_model">
    <link name="chassis">
      <pose>0 0 0.1 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
      </visual>
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.2</iyy>
          <iyz>0</iyz>
          <izz>0.15</izz>
        </inertia>
      </inertial>
    </link>
  </model>
</sdf>
```

## Gazebo Commands and Tools

### Common Gazebo Commands

```bash
# Launch Gazebo with a specific world
gazebo --verbose worlds/willow.world

# Launch empty world
gazebo

# Launch with specific configuration
gazebo my_world.sdf

# Run without GUI (headless)
gazebo -s libgazebo_ros_init.so -s libgazebo_ros_factory.so my_world.sdf
```

### Gazebo Services and Topics

```bash
# List all Gazebo services
rosservice list | grep gazebo

# Get model states
rosservice call /gazebo/get_model_state "model_name: 'my_robot' relative_entity_name: 'world'"

# Set model state
rosservice call /gazebo/set_model_state "model_state:
  model_name: 'my_robot'
  pose:
    position: {x: 1.0, y: 0.0, z: 0.0}
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  twist:
    linear: {x: 0.0, y: 0.0, z: 0.0}
    angular: {x: 0.0, y: 0.0, z: 0.0}
  reference_frame: 'world'"

# Apply force to a link
rosservice call /gazebo/apply_body_wrench "body_name: 'my_robot::chassis'
reference_frame: 'world'
reference_point: {x: 0, y: 0, z: 0}
force: {x: 10.0, y: 0.0, z: 0.0}
torque: {x: 0.0, y: 0.0, z: 0.0}
duration: {sec: 1, nanosec: 0}"
```

## Performance Optimization

### Reducing Computational Load

1. **Adjust Physics Update Rate**: Lower `max_step_size` and `real_time_update_rate` for less demanding simulations
2. **Simplify Models**: Use simpler collision and visual geometries
3. **Limit Sensor Data**: Reduce sensor update rates and resolutions
4. **Use Threading**: Enable multi-threaded physics simulation

### Multi-threaded Physics

```xml
<physics type="ode">
  <max_step_size>0.01</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>100.0</real_time_update_rate>
  <threads>4</threads>  <!-- Enable multi-threading -->
</physics>
```

## Troubleshooting Common Issues

### 1. Robot Not Spawning

```bash
# Check if robot description is valid
check_urdf /path/to/robot.urdf

# Verify that spawn_entity.py is working
ros2 run gazebo_ros spawn_entity.py -entity my_robot -file /path/to/robot.urdf
```

### 2. Performance Issues

```bash
# Monitor Gazebo performance
gz stats

# Check ROS 2 topics for high frequency
ros2 topic hz /joint_states
```

### 3. Sensor Data Issues

- Verify sensor plugins are properly configured
- Check topic connections: `ros2 topic info /camera/image_raw`
- Confirm sensor parameters match expectations

## Best Practices

1. **Start Simple**: Begin with basic models and gradually add complexity
2. **Validate Physics**: Test physical properties in isolation
3. **Use Standard Models**: Leverage existing models when possible
4. **Document Configurations**: Keep track of working parameters
5. **Version Control**: Store world files and configurations in version control
6. **Performance Testing**: Regularly test simulation performance

## Summary

Gazebo provides a powerful platform for robotics simulation with extensive customization options. Proper setup involves configuring physics parameters, integrating with ROS 2, and creating appropriate models and worlds for your specific applications.

In the next section, we'll explore URDF and SDF formats in detail, which are essential for describing robots and environments in Gazebo.