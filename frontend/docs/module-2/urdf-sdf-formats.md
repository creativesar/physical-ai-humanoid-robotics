---
sidebar_position: 4
title: "URDF and SDF Robot Description Formats"
---

# URDF and SDF Robot Description Formats

## Introduction

In robotics simulation, accurately describing both robots and environments is crucial for realistic and effective simulation. Two primary formats are used in the ROS ecosystem: URDF (Unified Robot Description Format) for robots and SDF (Simulation Description Format) for environments and complete simulation scenarios.

## URDF vs SDF: Key Differences

### URDF (Unified Robot Description Format)
- **Purpose**: Describes robot models and their kinematic structure
- **Scope**: Individual robots and their components
- **Format**: XML-based
- **Usage**: ROS ecosystem for robot modeling
- **Limitations**: Not suitable for complete simulation environments

### SDF (Simulation Description Format)
- **Purpose**: Describes complete simulation scenarios including robots, environments, and physics
- **Scope**: Entire simulation worlds
- **Format**: XML-based (based on SDFormat)
- **Usage**: Gazebo and other simulators
- **Advantages**: Supports multiple robots, complex environments, and physics configurations

## URDF in Detail

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.25 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>
</robot>
```

### URDF Elements

#### Links
Links represent rigid bodies in the robot:

```xml
<link name="link_name">
  <!-- Visual properties for rendering -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Can be box, cylinder, sphere, or mesh -->
      <box size="1 1 1"/>
    </geometry>
    <material name="color_name">
      <color rgba="1 0 0 1"/> <!-- Red -->
    </material>
  </visual>

  <!-- Collision properties for physics -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="1 1 1"/>
    </geometry>
  </collision>

  <!-- Physical properties for dynamics -->
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
  </inertial>
</link>
```

#### Joints
Joints define connections between links:

```xml
<!-- Fixed joint (no movement) -->
<joint name="fixed_joint" type="fixed">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>

<!-- Revolute joint (rotational) -->
<joint name="revolute_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/> <!-- Rotation around Z-axis -->
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>

<!-- Continuous joint (unlimited rotation) -->
<joint name="continuous_joint" type="continuous">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
</joint>

<!-- Prismatic joint (linear motion) -->
<joint name="prismatic_joint" type="prismatic">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="0" upper="0.5" effort="100" velocity="1"/>
</joint>
```

### URDF with Transmissions

For ROS control integration:

```xml
<transmission name="wheel_transmission">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
  <actuator name="wheel_motor">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## SDF in Detail

### Basic SDF Structure

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="my_world">
    <!-- Models in the world -->
    <model name="my_robot">
      <!-- Robot definition (similar to URDF structure) -->
      <link name="base_link">
        <pose>0 0 0.5 0 0 0</pose>
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
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1</iyy>
            <iyz>0</iyz>
            <izz>0.1</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Physics configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
  </world>
</sdf>
```

### SDF World Elements

#### Models
Models in SDF can contain the same link and joint structure as URDF:

```xml
<model name="simple_robot">
  <static>false</static> <!-- Whether model moves -->
  <pose>0 0 0.5 0 0 0</pose> <!-- Position and orientation -->

  <!-- Links as defined above -->
  <link name="chassis">
    <!-- ... link definition ... -->
  </link>

  <!-- Joints connecting links -->
  <joint name="chassis_wheel" type="revolute">
    <parent>chassis</parent>
    <child>wheel</child>
    <axis>
      <xyz>0 1 0</xyz>
      <limit>
        <lower>-1.57</lower>
        <upper>1.57</upper>
        <effort>100</effort>
        <velocity>1</velocity>
      </limit>
    </axis>
  </joint>
</model>
```

#### World Elements
SDF worlds can include:

```xml
<!-- Terrain -->
<terrain name="ground_terrain">
  <pose>0 0 0 0 0 0</pose>
  <geometry>
    <heightmap>
      <uri>file://path/to/heightmap.png</uri>
      <size>100 100 10</size>
    </heightmap>
  </geometry>
</terrain>

<!-- Actors (animated models) -->
<actor name="walking_person">
  <pose>2 2 0 0 0 0</pose>
  <skin>
    <filename>walking.dae</filename>
    <scale>1.0</scale>
  </skin>
  <animation name="walking">
    <filename>walking.dae</filename>
    <scale>1.0</scale>
    <interpolate_x>true</interpolate_x>
  </animation>
  <script>
    <loop>true</loop>
    <delay_start>0</delay_start>
    <trajectory id="0" type="square">
      <waypoint>
        <time>0</time>
        <pose>2 2 0 0 0 0</pose>
      </waypoint>
      <waypoint>
        <time>5</time>
        <pose>2 -2 0 0 0 0</pose>
      </waypoint>
    </trajectory>
  </script>
</actor>
```

### SDF Sensors

SDF allows detailed sensor definitions:

```xml
<model name="sensor_robot">
  <link name="sensor_mount">
    <sensor name="camera_sensor" type="camera">
      <pose>0.1 0 0 0 0 0</pose>
      <camera name="head_camera">
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
      <always_on>1</always_on>
      <update_rate>30</update_rate>
      <visualize>true</visualize>
    </sensor>

    <sensor name="lidar_sensor" type="ray">
      <pose>0.1 0 0.1 0 0 0</pose>
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>30</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <always_on>1</always_on>
      <update_rate>10</update_rate>
      <visualize>false</visualize>
    </sensor>
  </link>
</model>
```

## Converting Between URDF and SDF

### URDF to SDF
Gazebo can automatically convert URDF to SDF when spawning models:

```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf

# Or use the spawn tool which handles conversion automatically
ros2 run gazebo_ros spawn_entity.py -entity my_robot -file robot.urdf
```

### Xacro for Complex Models
Xacro (XML Macros) helps manage complex URDFs:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="complex_robot">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />

  <!-- Macros for repeated elements -->
  <xacro:macro name="wheel" params="prefix parent xyz rpy">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
    </joint>

    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.02"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Robot definition using macros -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.15"/>
    </inertial>
  </link>

  <!-- Use the macro to create wheels -->
  <xacro:wheel prefix="front_left" parent="base_link" xyz="0.2 0.15 -0.1" rpy="0 0 0"/>
  <xacro:wheel prefix="front_right" parent="base_link" xyz="0.2 -0.15 -0.1" rpy="0 0 0"/>
  <xacro:wheel prefix="rear_left" parent="base_link" xyz="-0.2 0.15 -0.1" rpy="0 0 0"/>
  <xacro:wheel prefix="rear_right" parent="base_link" xyz="-0.2 -0.15 -0.1" rpy="0 0 0"/>
</robot>
```

## Advanced SDF Features

### Plugins in SDF

SDF supports plugins for custom functionality:

```xml
<model name="robot_with_plugins">
  <!-- ROS 2 control plugin -->
  <plugin name="ros_control" filename="libgazebo_ros_control.so">
    <parameters>$(find my_robot_description)/config/robot_controllers.yaml</parameters>
  </plugin>

  <!-- IMU sensor plugin -->
  <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
    <ros>
      <namespace>/my_robot</namespace>
      <remapping>~/out:=imu/data</remapping>
    </ros>
    <update_rate>100</update_rate>
    <body_name>imu_link</body_name>
    <frame_name>imu_link</frame_name>
    <topic_name>imu/data</topic_name>
    <gaussian_noise>0.001</gaussian_noise>
  </plugin>

  <!-- Camera plugin -->
  <plugin name="camera_plugin" filename="libgazebo_ros_camera.so">
    <ros>
      <namespace>/my_robot</namespace>
      <remapping>image_raw:=camera/image_raw</remapping>
      <remapping>camera_info:=camera/camera_info</remapping>
    </ros>
    <camera_name>camera</camera_name>
    <image_topic_name>image_raw</image_topic_name>
    <camera_info_topic_name>camera_info</camera_info_topic_name>
    <frame_name>camera_link</frame_name>
  </plugin>
</model>
```

### Physics Configuration

SDF allows detailed physics configuration:

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

  <!-- Enable physics engine threading -->
  <threads>4</threads>
</physics>
```

## Best Practices for URDF/SDF

### URDF Best Practices

1. **Consistent Naming**: Use clear, consistent naming conventions
2. **Proper Inertial Values**: Calculate accurate mass and inertia properties
3. **Collision vs Visual**: Use simplified collision geometries for performance
4. **Joint Limits**: Set realistic joint limits based on hardware constraints
5. **Xacro Macros**: Use macros for repetitive elements

### SDF Best Practices

1. **World Organization**: Structure worlds logically with clear hierarchies
2. **Performance**: Optimize models and environments for simulation speed
3. **Realism**: Include realistic sensor noise and environmental factors
4. **Documentation**: Comment complex SDF files for maintainability
5. **Validation**: Test SDF files for syntax and logical errors

## Tools for Working with URDF/SDF

### Validation Tools

```bash
# Validate URDF
check_urdf /path/to/robot.urdf

# Convert and view URDF as graph
urdf_to_graphiz /path/to/robot.urdf

# Validate SDF
gz sdf -k world.sdf  # Check syntax
gz sdf -p world.sdf  # Parse and print
```

### Visualization Tools

```bash
# Visualize URDF in RViz
ros2 run rviz2 rviz2

# Launch robot state publisher for visualization
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:=$(cat robot.urdf)

# Visualize SDF in Gazebo
gazebo world.sdf
```

## Common Issues and Solutions

### 1. URDF/SDF Parsing Errors
- Check XML syntax and proper closing tags
- Verify all referenced files exist
- Ensure all required elements are present

### 2. Physics Instability
- Verify inertial properties are realistic
- Check joint limits and dynamics parameters
- Adjust physics engine parameters

### 3. Performance Issues
- Simplify collision geometries
- Reduce sensor update rates
- Use efficient mesh formats

### 4. Transform Issues
- Verify all transforms are properly defined
- Check for disconnected kinematic chains
- Ensure consistent coordinate frames

## Integration with ROS 2

### Robot State Publishing

```python
# Example of integrating URDF with ROS 2
import rclpy
from rclpy.node import Node
from urdf_parser_py.urdf import URDF
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)

        # Create transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Parse robot description parameter
        self.declare_parameter('robot_description', '')
        robot_desc = self.get_parameter('robot_description').value

        # Parse URDF
        self.robot = URDF.from_xml_string(robot_desc)

        self.get_logger().info('Robot State Publisher initialized')

    def joint_callback(self, msg):
        """Process joint state messages and publish transforms"""
        # Process joint states and publish corresponding transforms
        pass
```

## Summary

URDF and SDF are fundamental formats for robotics simulation:

- **URDF** is ideal for describing individual robots with their kinematic structure
- **SDF** provides comprehensive simulation scenarios including environments and physics
- Both formats use XML syntax but serve different purposes
- Xacro helps manage complex URDFs with macros and properties
- Proper validation and optimization are essential for effective simulation

Understanding these formats is crucial for creating realistic and efficient robotics simulations in Gazebo and other simulators. In the next section, we'll explore physics simulation and dynamics in detail.