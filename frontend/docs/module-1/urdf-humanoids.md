---
sidebar_position: 7
title: "Understanding URDF for Humanoids"
---

# Understanding URDF for Humanoids

## Introduction to URDF

Unified Robot Description Format (URDF) is an XML format used in ROS to describe robot models. It defines the physical and visual properties of a robot, including its links, joints, and the relationships between them. For humanoid robots, URDF is essential for simulation, visualization, and control.

## URDF Structure

A URDF file contains several key elements:

- **Links**: Rigid bodies that make up the robot
- **Joints**: Connections between links with specific degrees of freedom
- **Visual**: How the link appears in visualization
- **Collision**: Collision properties for physics simulation
- **Inertial**: Mass, center of mass, and inertia properties

## Basic URDF for Humanoid Robot

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.15 0.15 0.4"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.15 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Joint connecting base to torso -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.08"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.08"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.28" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1"/>
  </joint>
</robot>
```

## Humanoid-Specific URDF Elements

### Links for Humanoid Robots

Humanoid robots typically have the following link structure:

```xml
<!-- Complete humanoid skeleton -->
<robot name="humanoid_robot">
  <!-- Core links -->
  <link name="base_link"/>
  <link name="pelvis"/>
  <link name="torso"/>
  <link name="head"/>
  <link name="neck"/>

  <!-- Left arm -->
  <link name="left_shoulder"/>
  <link name="left_upper_arm"/>
  <link name="left_lower_arm"/>
  <link name="left_hand"/>

  <!-- Right arm -->
  <link name="right_shoulder"/>
  <link name="right_upper_arm"/>
  <link name="right_lower_arm"/>
  <link name="right_hand"/>

  <!-- Left leg -->
  <link name="left_hip"/>
  <link name="left_upper_leg"/>
  <link name="left_lower_leg"/>
  <link name="left_foot"/>

  <!-- Right leg -->
  <link name="right_hip"/>
  <link name="right_upper_leg"/>
  <link name="right_lower_leg"/>
  <link name="right_foot"/>
</robot>
```

### Joint Definitions for Humanoid Movement

```xml
<!-- Example joints for humanoid robot -->
<joint name="torso_to_head" type="revolute">
  <parent link="torso"/>
  <child link="neck"/>
  <origin xyz="0 0 0.4" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>  <!-- Y-axis for head nodding -->
  <limit lower="-0.5" upper="0.5" effort="50" velocity="2"/>
  <dynamics damping="0.5" friction="0.1"/>
</joint>

<joint name="left_shoulder_joint" type="continuous">
  <parent link="torso"/>
  <child link="left_shoulder"/>
  <origin xyz="0.1 0.1 0.3" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>  <!-- Z-axis for shoulder rotation -->
</joint>

<joint name="left_elbow_joint" type="revolute">
  <parent link="left_upper_arm"/>
  <child link="left_lower_arm"/>
  <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>  <!-- Y-axis for elbow flexion -->
  <limit lower="0" upper="2.5" effort="50" velocity="2"/>
</joint>

<joint name="left_knee_joint" type="revolute">
  <parent link="left_upper_leg"/>
  <child link="left_lower_leg"/>
  <origin xyz="0 0 -0.4" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>  <!-- Y-axis for knee flexion -->
  <limit lower="0" upper="2.0" effort="100" velocity="1.5"/>
</joint>
```

## Advanced URDF Features for Humanoids

### Transmission Elements

For control purposes, you need to define transmissions:

```xml
<!-- Transmission for joint control -->
<transmission name="left_elbow_transmission">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_elbow_joint">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_elbow_motor">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>

<transmission name="right_knee_transmission">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="right_knee_joint">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="right_knee_actuator">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>100</mechanicalReduction>
  </actuator>
</transmission>
```

### Gazebo-Specific Elements

For simulation in Gazebo:

```xml
<!-- Gazebo-specific properties -->
<gazebo reference="left_foot">
  <mu1>0.8</mu1>
  <mu2>0.8</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
  <material>Gazebo/Blue</material>
  <turnGravityOff>false</turnGravityOff>
</gazebo>

<!-- Gazebo plugins -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/humanoid</robotNamespace>
    <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
  </plugin>
</gazebo>
```

## Complete Humanoid URDF Example

Here's a more complete humanoid robot URDF:

```xml
<?xml version="1.0"?>
<robot name="complete_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Pelvis -->
  <link name="pelvis">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.25 0.15"/>
      </geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.25 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="5.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <joint name="base_to_pelvis" type="fixed">
    <parent link="base_link"/>
    <child link="pelvis"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.18 0.25 0.5"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.18 0.25 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <mass value="8.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="pelvis_to_torso" type="revolute">
    <parent link="pelvis"/>
    <child link="torso"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="200" velocity="1"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <origin xyz="0 0 0.08" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.08"/>
      </geometry>
      <material name="skin"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.08" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.08"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.08" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="torso_to_head" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.8" upper="0.8" effort="50" velocity="2"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_shoulder">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="blue"/>
      <origin xyz="0 0 0" rpy="0 1.5708 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 1.5708 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="torso_to_left_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="left_shoulder"/>
    <origin xyz="0.1 0.1 0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="blue"/>
      <origin xyz="0 0 -0.15" rpy="1.5708 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <origin xyz="0 0 -0.15" rpy="1.5708 0 0"/>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_to_upper_arm" type="revolute">
    <parent link="left_shoulder"/>
    <child link="left_upper_arm"/>
    <origin xyz="0 0 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="100" velocity="2"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder radius="0.03" length="0.25"/>
      </geometry>
      <material name="blue"/>
      <origin xyz="0 0 -0.125" rpy="1.5708 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.03" length="0.25"/>
      </geometry>
      <origin xyz="0 0 -0.125" rpy="1.5708 0 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_upper_arm_to_lower_arm" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="3.14" effort="80" velocity="2"/>
  </joint>

  <link name="left_hand">
    <visual>
      <geometry>
        <box size="0.1 0.08 0.05"/>
      </geometry>
      <material name="skin"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_lower_arm_to_hand" type="fixed">
    <parent link="left_lower_arm"/>
    <child link="left_hand"/>
    <origin xyz="0 0 -0.25" rpy="0 0 0"/>
  </joint>

  <!-- Right Arm (similar to left, mirrored) -->
  <link name="right_shoulder">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="blue"/>
      <origin xyz="0 0 0" rpy="0 1.5708 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 1.5708 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="torso_to_right_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="right_shoulder"/>
    <origin xyz="0.1 -0.1 0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
  </joint>

  <!-- Additional links for right arm would follow similar pattern -->

  <!-- Left Leg -->
  <link name="left_hip">
    <visual>
      <geometry>
        <cylinder radius="0.06" length="0.1"/>
      </geometry>
      <material name="red"/>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.06" length="0.1"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="pelvis_to_left_hip" type="revolute">
    <parent link="pelvis"/>
    <child link="left_hip"/>
    <origin xyz="0 0.08 -0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="150" velocity="1.5"/>
  </joint>

  <link name="left_upper_leg">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="red"/>
      <origin xyz="0 0 -0.2" rpy="1.5708 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <origin xyz="0 0 -0.2" rpy="1.5708 0 0"/>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_hip_to_upper_leg" type="revolute">
    <parent link="left_hip"/>
    <child link="left_upper_leg"/>
    <origin xyz="0 0 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.5" effort="200" velocity="1.5"/>
  </joint>

  <link name="left_lower_leg">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.4"/>
      </geometry>
      <material name="red"/>
      <origin xyz="0 0 -0.2" rpy="1.5708 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.4"/>
      </geometry>
      <origin xyz="0 0 -0.2" rpy="1.5708 0 0"/>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_upper_leg_to_lower_leg" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.0" effort="180" velocity="1.5"/>
  </joint>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.15 0.08 0.06"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.08 0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_lower_leg_to_foot" type="fixed">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
  </joint>

  <!-- Similar structure for right leg -->
</robot>
```

## Xacro for Complex Humanoid URDFs

For complex humanoid robots, Xacro (XML Macros) is often used to simplify URDF creation:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_xacro">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_height" value="0.5" />
  <xacro:property name="upper_arm_length" value="0.3" />
  <xacro:property name="lower_arm_length" value="0.25" />

  <!-- Macro for arm links -->
  <xacro:macro name="arm_chain" params="side parent_link position">
    <!-- Shoulder -->
    <link name="${side}_shoulder">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.1"/>
        </geometry>
        <material name="blue"/>
        <origin xyz="0 0 0" rpy="0 1.5708 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.05" length="0.1"/>
        </geometry>
        <origin xyz="0 0 0" rpy="0 1.5708 0"/>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="torso_to_${side}_shoulder" type="revolute">
      <parent link="${parent_link}"/>
      <child link="${side}_shoulder"/>
      <origin xyz="0.1 ${position} 0.3" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
    </joint>

    <!-- Upper arm -->
    <link name="${side}_upper_arm">
      <visual>
        <geometry>
          <cylinder radius="0.04" length="${upper_arm_length}"/>
        </geometry>
        <material name="blue"/>
        <origin xyz="0 0 -${upper_arm_length/2}" rpy="1.5708 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.04" length="${upper_arm_length}"/>
        </geometry>
        <origin xyz="0 0 -${upper_arm_length/2}" rpy="1.5708 0 0"/>
      </collision>
      <inertial>
        <mass value="1.5"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_shoulder_to_upper_arm" type="revolute">
      <parent link="${side}_shoulder"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-2.0" upper="2.0" effort="100" velocity="2"/>
    </joint>

    <!-- Lower arm -->
    <link name="${side}_lower_arm">
      <visual>
        <geometry>
          <cylinder radius="0.03" length="${lower_arm_length}"/>
        </geometry>
        <material name="blue"/>
        <origin xyz="0 0 -${lower_arm_length/2}" rpy="1.5708 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.03" length="${lower_arm_length}"/>
        </geometry>
        <origin xyz="0 0 -${lower_arm_length/2}" rpy="1.5708 0 0"/>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_upper_arm_to_lower_arm" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_lower_arm"/>
      <origin xyz="0 0 -${upper_arm_length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="0" upper="3.14" effort="80" velocity="2"/>
    </joint>
  </xacro:macro>

  <!-- Base links -->
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <link name="pelvis">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.25 0.15"/>
      </geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.25 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="5.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <joint name="base_to_pelvis" type="fixed">
    <parent link="base_link"/>
    <child link="pelvis"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <visual>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.18 0.25 0.5"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.18 0.25 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <mass value="8.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="pelvis_to_torso" type="revolute">
    <parent link="pelvis"/>
    <child link="torso"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="200" velocity="1"/>
  </joint>

  <!-- Instantiate arms using macro -->
  <xacro:arm_chain side="left" parent_link="torso" position="0.1"/>
  <xacro:arm_chain side="right" parent_link="torso" position="-0.1"/>
</robot>
```

## Validation and Tools

### URDF Validation

```bash
# Check URDF for errors
check_urdf /path/to/robot.urdf

# Parse URDF and show joint information
urdf_to_graphiz /path/to/robot.urdf
```

### Visualizing URDF

```bash
# Launch robot state publisher
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:=$(cat robot.urdf)

# Or with parameters
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:=$(ros2 param load /path/to/robot.urdf)
```

## Best Practices for Humanoid URDF

1. **Proper Inertial Properties**: Accurate mass and inertia values are crucial for physics simulation
2. **Realistic Joint Limits**: Set appropriate limits based on actual robot capabilities
3. **Collision vs Visual**: Use simplified geometries for collision to improve performance
4. **Consistent Naming**: Use consistent naming conventions for easy identification
5. **Modular Design**: Use Xacro macros for repetitive elements
6. **Documentation**: Comment your URDF for maintainability

## Common Issues and Solutions

### Self-Collision Issues
- Ensure proper spacing between links
- Define appropriate collision geometries
- Use collision filtering when necessary

### Simulation Instability
- Verify inertial properties
- Check joint limits and dynamics
- Adjust solver parameters in Gazebo

### Performance Issues
- Simplify collision meshes
- Reduce unnecessary visual elements
- Use appropriate mesh resolutions

## Summary

URDF is fundamental for humanoid robotics as it defines the robot's physical structure. For humanoid robots, special attention must be paid to:

- Creating a proper kinematic chain that mimics human anatomy
- Setting realistic joint limits and dynamics
- Accurate inertial properties for stable simulation
- Using Xacro for complex, repetitive structures

In the next section, we'll explore how ROS 2 is specifically applied to humanoid robotics.