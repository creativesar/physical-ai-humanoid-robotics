---
sidebar_position: 3
title: "Robot Modeling in Simulation"
---

# Robot Modeling in Simulation

## Introduction to Robot Modeling

Creating accurate robot models is crucial for successful simulation-to-reality transfer in humanoid robotics. A well-designed robot model serves as the foundation for all simulation activities, from basic kinematic validation to complex AI training. In this module, we'll explore the principles and techniques for creating effective robot models for humanoid robotics simulation.

## Components of a Robot Model

### 1. Kinematic Model

The kinematic model defines the robot's structure and movement capabilities:

#### Link Properties
- **Geometry**: Visual and collision shapes
- **Inertial properties**: Mass, center of mass, and inertia tensor
- **Material properties**: Color, texture, and physical characteristics

#### Joint Properties
- **Joint type**: Revolute, prismatic, fixed, etc.
- **Joint limits**: Position, velocity, and effort constraints
- **Dynamics**: Damping and friction coefficients

### 2. Dynamic Model

The dynamic model captures the robot's behavior under forces and torques:

#### Mass Distribution
- Accurate mass properties for each link
- Center of mass location
- Inertia tensor (3x3 matrix for rotational inertia)

#### Actuator Modeling
- Motor characteristics (torque-speed curves)
- Gear ratios and transmission efficiency
- Joint compliance and damping

## URDF (Unified Robot Description Format)

### URDF Structure for Humanoid Robots

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Include common definitions -->
  <xacro:include filename="$(find humanoid_description)/urdf/materials.urdf.xacro"/>
  <xacro:include filename="$(find humanoid_description)/urdf/transmissions.urdf.xacro"/>

  <!-- Base link (torso) -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="package://humanoid_description/meshes/torso.dae"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="package://humanoid_description/meshes/torso_collision.stl"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="15.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.8" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="2.0"/>
    <dynamics damping="0.1" friction="0.01"/>
  </joint>

  <link name="head_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="package://humanoid_description/meshes/head.dae"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="package://humanoid_description/meshes/head_collision.stl"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_pitch" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0.2 0.7" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="1.0" effort="50" velocity="2.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <mesh filename="package://humanoid_description/meshes/upper_arm.dae"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <mesh filename="package://humanoid_description/meshes/upper_arm_collision.stl"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <mass value="3.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Continue with other joints and links... -->
</robot>
```

### Xacro Macros for Complex Humanoid Models

Xacro (XML Macros) helps manage complex humanoid models:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Define constants -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_mass" value="15.0" />
  <xacro:property name="upper_arm_mass" value="3.0" />
  <xacro:property name="lower_arm_mass" value="2.0" />

  <!-- Macro for creating a humanoid joint -->
  <xacro:macro name="humanoid_joint" params="name type parent child xyz axis lower upper effort velocity damping friction">
    <joint name="${name}" type="${type}">
      <parent link="${parent}"/>
      <child link="${child}"/>
      <origin xyz="${xyz}" rpy="0 0 0"/>
      <axis xyz="${axis}"/>
      <limit lower="${lower}" upper="${upper}" effort="${effort}" velocity="${velocity}"/>
      <dynamics damping="${damping}" friction="${friction}"/>
    </joint>
  </xacro:macro>

  <!-- Macro for creating a humanoid link -->
  <xacro:macro name="humanoid_link" params="name mesh mass ixx iyy izz xyz">
    <link name="${name}">
      <visual>
        <origin xyz="${xyz}" rpy="0 0 0"/>
        <geometry>
          <mesh filename="package://humanoid_description/meshes/${mesh}.dae"/>
        </geometry>
        <material name="light_grey"/>
      </visual>
      <collision>
        <origin xyz="${xyz}" rpy="0 0 0"/>
        <geometry>
          <mesh filename="package://humanoid_description/meshes/${mesh}_collision.stl"/>
        </geometry>
      </collision>
      <inertial>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <mass value="${mass}"/>
        <inertia ixx="${ixx}" ixy="0.0" ixz="0.0" iyy="${iyy}" iyz="0.0" izz="${izz}"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Use macros to create the robot -->
  <xacro:humanoid_link name="base_link" mesh="torso" mass="${torso_mass}"
                       ixx="0.2" iyy="0.3" izz="0.4" xyz="0 0 0"/>

  <!-- Define joints using macro -->
  <xacro:humanoid_joint name="left_hip_yaw" type="revolute"
                        parent="base_link" child="left_thigh"
                        xyz="0 -0.1 -0.1" axis="0 0 1"
                        lower="${-M_PI/3}" upper="${M_PI/3}"
                        effort="100" velocity="3.0"
                        damping="1.0" friction="0.1"/>
</robot>
```

## Inertial Properties and Dynamics

### Calculating Inertial Properties

For accurate simulation, inertial properties must be carefully calculated:

#### Simple Geometric Shapes
```xml
<!-- Box -->
<inertial>
  <mass value="1.0"/>
  <inertia ixx="0.083" ixy="0.0" ixz="0.0"
           iyy="0.083" iyz="0.0" izz="0.083"/>
</inertial>

<!-- For a box: Ixx = Iyy = Izz = (1/12) * m * (w² + h²) -->
<!-- Where w and h are dimensions perpendicular to the axis -->
```

#### Complex Shapes
For complex shapes, use CAD software to calculate inertial properties:
- SolidWorks: Tools → Mass Properties
- Fusion 360: Inspect → Measure
- Blender: Can calculate if density is specified

### Dynamics Parameters

#### Damping and Friction
```xml
<!-- Joint dynamics in URDF -->
<dynamics damping="0.5" friction="0.1"/>

<!-- In Gazebo plugin -->
<gazebo reference="joint_name">
  <joint>
    <dynamics>
      <damping>0.5</damping>
      <friction>0.1</friction>
      <spring_reference>0</spring_reference>
      <spring_stiffness>0</spring_stiffness>
    </dynamics>
  </joint>
</gazebo>
```

## Collision and Visual Geometry

### Collision Geometry Strategies

#### 1. Convex Hull Decomposition
For complex meshes, decompose into convex hulls:

```xml
<!-- Multiple collision elements for complex shapes -->
<link name="complex_link">
  <collision>
    <geometry>
      <mesh filename="package://humanoid_description/meshes/complex_shape_convex_1.stl"/>
    </geometry>
  </collision>
  <collision>
    <geometry>
      <mesh filename="package://humanoid_description/meshes/complex_shape_convex_2.stl"/>
    </geometry>
  </collision>
  <!-- Add more convex pieces as needed -->
</link>
```

#### 2. Primitive Approximation
Use simple primitives for collision:

```xml
<!-- Approximate complex shape with multiple primitives -->
<link name="arm_link">
  <collision>
    <geometry>
      <cylinder length="0.3" radius="0.05"/>
    </geometry>
  </collision>
  <visual>
    <geometry>
      <mesh filename="package://humanoid_description/meshes/arm.dae"/>
    </geometry>
  </visual>
</link>
```

### Visual vs Collision Geometry

```xml
<link name="humanoid_link">
  <!-- Detailed visual geometry for rendering -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <mesh filename="package://humanoid_description/meshes/detailed_visual.dae"/>
    </geometry>
    <material name="robot_material"/>
  </visual>

  <!-- Simplified collision geometry for physics -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <mesh filename="package://humanoid_description/meshes/simple_collision.stl"/>
    </geometry>
  </collision>
</link>
```

## Humanoid-Specific Modeling Considerations

### 1. Balance and Stability

For humanoid robots, center of mass is critical:

```xml
<!-- Torso link with carefully positioned center of mass -->
<link name="torso">
  <inertial>
    <!-- CoM positioned for optimal balance -->
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <mass value="15.0"/>
    <!-- Adjust inertia tensor for realistic balance -->
    <inertia ixx="0.2" ixy="0.0" ixz="0.0"
             iyy="0.3" iyz="0.0" izz="0.4"/>
  </inertial>
</link>
```

### 2. Anthropomorphic Design

Model joints to match human-like ranges of motion:

```xml
<!-- Human-like joint limits -->
<joint name="left_shoulder_pitch" type="revolute">
  <parent link="torso"/>
  <child link="left_upper_arm"/>
  <axis xyz="1 0 0"/>
  <!-- Human shoulder range: -2.0 to 1.0 radians -->
  <limit lower="-2.0" upper="1.0" effort="50" velocity="2.0"/>
</joint>

<joint name="left_elbow_pitch" type="revolute">
  <parent link="left_upper_arm"/>
  <child link="left_lower_arm"/>
  <axis xyz="1 0 0"/>
  <!-- Human elbow range: -2.5 to 0.0 radians -->
  <limit lower="-2.5" upper="0.0" effort="30" velocity="3.0"/>
</joint>
```

### 3. Foot and Hand Modeling

For locomotion and manipulation:

```xml
<!-- Multi-DOF foot for balance -->
<joint name="left_ankle_roll" type="revolute">
  <parent link="left_shin"/>
  <child link="left_foot"/>
  <axis xyz="0 1 0"/>
  <limit lower="-0.5" upper="0.5" effort="20" velocity="2.0"/>
</joint>

<joint name="left_ankle_pitch" type="revolute">
  <axis xyz="1 0 0"/>
  <limit lower="-0.5" upper="0.5" effort="20" velocity="2.0"/>
</joint>

<!-- Simplified hand model -->
<link name="left_hand">
  <visual>
    <geometry>
      <mesh filename="package://humanoid_description/meshes/hand.dae"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <sphere radius="0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.5"/>
    <inertia ixx="0.001" iyy="0.001" izz="0.001"/>
  </inertial>
</link>
```

## Sensor Integration in Models

### IMU Placement

```xml
<!-- IMU sensor in the torso for balance -->
<link name="imu_link">
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
  <visual>
    <geometry>
      <box size="0.01 0.01 0.01"/>
    </geometry>
    <material name="red"/>
  </visual>
  <collision>
    <geometry>
      <box size="0.01 0.01 0.01"/>
    </geometry>
  </collision>
</link>

<joint name="imu_joint" type="fixed">
  <parent link="base_link"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>

<!-- Gazebo plugin for IMU -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
  </sensor>
</gazebo>
```

### Camera Integration

```xml
<!-- Head-mounted camera -->
<link name="camera_link">
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
  </inertial>
</link>

<joint name="camera_joint" type="fixed">
  <parent link="head_link"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
</joint>

<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <camera>
      <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
    </camera>
  </sensor>
</gazebo>
```

## Model Validation Techniques

### 1. Static Balance Test

Verify the model can maintain static balance:

```xml
<!-- Use Gazebo's static balance test -->
<world name="balance_test">
  <!-- Physics with appropriate parameters -->
  <physics type="ode">
    <gravity>0 0 -9.8</gravity>
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1.0</real_time_factor>
  </physics>

  <!-- Ground plane -->
  <include>
    <uri>model://ground_plane</uri>
  </include>

  <!-- Your humanoid model -->
  <model name="test_humanoid">
    <!-- Model content -->
  </model>
</world>
```

### 2. Kinematic Validation

Check joint limits and ranges:

```bash
# Use ROS 2 tools to validate kinematics
ros2 run robot_state_publisher robot_state_publisher --ros-args --param robot_description:=$(cat robot.urdf)

# Visualize in RViz
ros2 run rviz2 rviz2
```

### 3. Dynamic Simulation Test

Test with simple controllers:

```cpp
// Simple gravity compensation test
class GravityCompensationTest : public rclcpp::Node
{
public:
    GravityCompensationTest() : Node("gravity_test")
    {
        joint_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/effort_controllers/commands", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&GravityCompensationTest::publishGravityCompensation, this));
    }

private:
    void publishGravityCompensation()
    {
        // Publish gravity compensation torques
        std_msgs::msg::Float64MultiArray msg;
        // Calculate gravity compensation based on joint angles
        msg.data = calculateGravityCompensation();
        joint_pub_->publish(msg);
    }

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};
```

## Performance Optimization

### 1. Mesh Simplification

For real-time simulation, optimize meshes:

#### Visual Meshes
- Keep high detail for visualization
- Use texture mapping instead of geometric detail where possible

#### Collision Meshes
- Use simplified convex hulls
- Reduce polygon count significantly
- Use primitive shapes where appropriate

### 2. Link and Joint Optimization

#### Combine Small Links
```xml
<!-- Instead of multiple small links, combine when possible -->
<!-- Before: 5 small links connected by joints -->
<!-- After: 1 link with appropriate inertial properties -->
<link name="combined_sensor_mount">
  <inertial>
    <mass value="0.2"/>
    <inertia ixx="0.001" ixy="0.0" ixz="0.0"
             iyy="0.002" iyz="0.0" izz="0.001"/>
  </inertial>
  <!-- Mount multiple sensors to this link -->
</link>
```

### 3. Transmission Optimization

```xml
<!-- Use efficient transmission types -->
<transmission name="simple_transmission">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="joint_name">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="motor_name">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Advanced Modeling Techniques

### 1. Flexible Joint Modeling

For more realistic actuator behavior:

```xml
<!-- Spring-damper model for flexible joints -->
<gazebo reference="joint_name">
  <joint>
    <dynamics>
      <spring_reference>0.0</spring_reference>
      <spring_stiffness>1000.0</spring_stiffness>
      <damping>10.0</damping>
      <friction>5.0</friction>
    </dynamics>
  </joint>
</gazebo>
```

### 2. Cable Routing and Constraints

Model physical constraints like cables:

```xml
<!-- Use additional fixed joints to model cable routing -->
<joint name="cable_route_1" type="fixed">
  <parent link="torso"/>
  <child link="cable_point_1"/>
  <origin xyz="0.05 0.05 0.3" rpy="0 0 0"/>
</joint>
```

### 3. Wearable Sensor Integration

Model external sensors and equipment:

```xml
<!-- Model a backpack computer or external sensors -->
<joint name="backpack_joint" type="fixed">
  <parent link="base_link"/>
  <child link="backpack"/>
  <origin xyz="0 0 0.2" rpy="0 0 0"/>
</joint>

<link name="backpack">
  <visual>
    <geometry>
      <box size="0.2 0.3 0.1"/>
    </geometry>
    <material name="black"/>
  </visual>
  <collision>
    <geometry>
      <box size="0.2 0.3 0.1"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="2.0"/>
    <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.08"/>
  </inertial>
</link>
```

## Model Testing and Verification

### 1. Zero Torque Test

Verify model behavior with zero applied torques:

```cpp
// Test with zero torques to check for stability
class ZeroTorqueTest : public rclcpp::Node
{
public:
    ZeroTorqueTest() : Node("zero_torque_test")
    {
        cmd_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/position_controllers/commands", 10);

        // Publish zero commands
        auto timer = this->create_wall_timer(
            std::chrono::milliseconds(10),
            [this]() {
                std_msgs::msg::Float64MultiArray zero_cmd;
                zero_cmd.data = std::vector<double>(28, 0.0); // 28 joints
                cmd_pub_->publish(zero_cmd);
            });
    }

private:
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_pub_;
};
```

### 2. Range of Motion Test

Verify all joints move within expected ranges:

```bash
# Use joint state publisher to test ranges
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```

### 3. Balance Test

Test static and dynamic balance capabilities:

```bash
# Run simulation and observe balance
gz sim -r --verbose
```

## Best Practices

### 1. Iterative Development
- Start with simple models and gradually add complexity
- Test each component individually
- Validate at each step

### 2. Documentation
- Document all modeling assumptions
- Record inertial property calculations
- Maintain model revision history

### 3. Modularity
- Create reusable components
- Use xacro macros for common patterns
- Separate model files by function

### 4. Validation
- Compare simulation results with physical tests when possible
- Validate against CAD models
- Test with multiple physics engines if available

## Next Steps

In the next section, we'll explore Unity integration for humanoid robotics simulation, learning how to leverage Unity's advanced graphics and physics capabilities for creating highly realistic digital twins of humanoid robots.