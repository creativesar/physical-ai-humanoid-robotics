---
sidebar_position: 4
title: "Packages and Workspaces"
---

# Packages and Workspaces

## Understanding ROS 2 Packages

Packages are the fundamental unit of organization in ROS 2. They contain source code, configuration files, launch files, and other resources needed for a specific functionality. In humanoid robotics, packages typically represent individual robot components, algorithms, or complete subsystems.

## Package Structure

A typical ROS 2 package has the following structure:

```
my_robot_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml             # Package metadata and dependencies
├── src/                    # Source code files
├── include/                # Header files (C++)
├── launch/                 # Launch files
├── config/                 # Configuration files
├── params/                 # Parameter files
├── test/                   # Test files
└── scripts/                # Python scripts
```

### package.xml

The `package.xml` file contains metadata about the package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>humanoid_controller</name>
  <version>0.1.0</version>
  <description>Humanoid robot controller package</description>
  <maintainer email="maintainer@example.com">Maintainer Name</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>control_msgs</depend>
  <depend>geometry_msgs</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### CMakeLists.txt

The `CMakeLists.txt` file defines how to build the package:

```cmake
cmake_minimum_required(VERSION 3.8)
project(humanoid_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Create executable
add_executable(humanoid_controller_node
  src/humanoid_controller.cpp)

# Link libraries
ament_target_dependencies(humanoid_controller_node
  rclcpp
  std_msgs
  sensor_msgs
  control_msgs
  geometry_msgs)

# Install targets
install(TARGETS
  humanoid_controller_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
```

## Creating a New Package

### Using colcon

The recommended way to create a new package is using the `ros2 pkg create` command:

```bash
# Create a new package with C++ support
ros2 pkg create --build-type ament_cmake --dependencies rclcpp std_msgs sensor_msgs humanoid_bringup

# Create a new package with Python support
ros2 pkg create --build-type ament_python --dependencies rclpy std_msgs sensor_msgs humanoid_bringup_py
```

### Manual Package Creation

You can also create packages manually by following the structure and creating the required files.

## Workspaces

### What is a Workspace?

A workspace is a directory where you modify and build ROS 2 code. It typically contains multiple packages that work together to form a complete robot system.

### Workspace Structure

```
~/ros2_ws/                 # Workspace root
├── src/                   # Source directory (contains packages)
│   ├── humanoid_controller/
│   ├── humanoid_description/
│   ├── humanoid_bringup/
│   └── humanoid_perception/
├── build/                 # Build artifacts (created during build)
├── install/               # Installation directory (created after build)
└── log/                   # Build logs
```

## Building Packages

### Using colcon

The `colcon` build system is used to build ROS 2 packages:

```bash
# Build all packages in the workspace
cd ~/ros2_ws
colcon build

# Build specific packages
colcon build --packages-select humanoid_controller humanoid_description

# Build with additional options
colcon build --packages-select humanoid_controller --cmake-args -DCMAKE_BUILD_TYPE=Release

# Build and install to a specific location
colcon build --install-base /opt/my_robot
```

### Sourcing the Workspace

After building, you need to source the workspace to use the packages:

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Add to your bashrc to source automatically
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Package Dependencies

### Build Dependencies

Dependencies required to build the package:

```xml
<buildtool_depend>ament_cmake</buildtool_depend>
<build_depend>geometry_msgs</build_depend>
<build_depend>sensor_msgs</build_depend>
```

### Execution Dependencies

Dependencies required to run the package:

```xml
<exec_depend>rclcpp</exec_depend>
<exec_depend>std_msgs</exec_depend>
```

### Test Dependencies

Dependencies required for testing:

```xml
<test_depend>ament_lint_auto</test_depend>
<test_depend>ament_cmake_gtest</test_depend>
```

## Package Management Best Practices

### 1. Naming Conventions

- Use lowercase names with underscores as separators
- Use descriptive names that clearly indicate package purpose
- Use prefixes to group related packages (e.g., `humanoid_`)

### 2. Dependency Management

- Only depend on packages that are actually used
- Use specific version dependencies when necessary
- Group related packages in the same repository when appropriate

### 3. Versioning

- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Update versions when making breaking changes
- Use version tags for releases

## Humanoid Robotics Package Organization

### Common Package Categories

1. **Description Packages**: Robot URDF, meshes, and visual assets
2. **Controller Packages**: Joint controllers and robot control
3. **Perception Packages**: Vision, sensors, and perception algorithms
4. **Navigation Packages**: Path planning and navigation
5. **Simulation Packages**: Gazebo plugins and simulation utilities
6. **Interface Packages**: Human-robot interaction and teleoperation

### Example Package Structure

```
humanoid_robot_ws/
├── src/
│   ├── humanoid_description/     # Robot URDF and meshes
│   ├── humanoid_control/         # Controllers and hardware interfaces
│   ├── humanoid_perception/      # Vision and sensor processing
│   ├── humanoid_navigation/      # Path planning and localization
│   ├── humanoid_bringup/         # Launch files and configurations
│   ├── humanoid_msgs/            # Custom message definitions
│   ├── humanoid_sim/             # Gazebo plugins and simulation
│   └── humanoid_apps/            # High-level applications
```

## Creating Custom Message Types

Sometimes you need custom message types for your humanoid robot:

```bash
# Create a new message definition
mkdir -p humanoid_msgs/msg
```

Create `humanoid_msgs/msg/JointCommand.msg`:
```
# Custom joint command message
string joint_name
float64 position
float64 velocity
float64 effort
float64 k_p
float64 k_i
float64 k_d
```

Update `package.xml`:
```xml
<depend>builtin_interfaces</depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

Update `CMakeLists.txt`:
```cmake
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/JointCommand.msg"
)
```

## Launch Files

Launch files allow you to start multiple nodes with a single command:

### XML Launch File Example

```xml
<launch>
  <!-- Arguments -->
  <arg name="robot_name" default="atlas"/>
  <arg name="use_sim_time" default="false"/>

  <!-- Robot state publisher -->
  <node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
    <param name="robot_description" value="$(find-pkg-share humanoid_description)/urdf/humanoid.urdf"/>
    <param name="use_sim_time" value="$(var use_sim_time)"/>
  </node>

  <!-- Joint state publisher -->
  <node pkg="joint_state_publisher" exec="joint_state_publisher" name="joint_state_publisher">
    <param name="use_sim_time" value="$(var use_sim_time)"/>
  </node>

  <!-- Humanoid controller -->
  <node pkg="humanoid_control" exec="humanoid_controller_node" name="humanoid_controller" output="screen">
    <param name="robot_name" value="$(var robot_name)"/>
    <param name="control_frequency" value="100"/>
  </node>
</launch>
```

### Python Launch File Example

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_name',
            default_value='atlas',
            description='Name of the robot'
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': Command(['xacro ', FindPackageShare('humanoid_description'), '/urdf/humanoid.urdf.xacro'])
            }]
        ),

        Node(
            package='humanoid_control',
            executable='humanoid_controller_node',
            name='humanoid_controller',
            parameters=[{
                'robot_name': LaunchConfiguration('robot_name'),
                'control_frequency': 100
            }]
        )
    ])
```

## Managing Multiple Workspaces

### Overlaying Workspaces

You can overlay workspaces to extend functionality:

```bash
# Source base workspace
source /opt/ros/humble/setup.bash

# Source your custom workspace
source ~/ros2_ws/install/setup.bash

# Build your workspace with base ROS 2 packages available
colcon build
```

## Testing and Quality Assurance

### Unit Testing

Create tests for your packages:

```cpp
// test/test_humanoid_controller.cpp
#include <gtest/gtest.h>
#include "humanoid_controller/humanoid_controller.hpp"

TEST(HumanoidControllerTest, InitializationTest) {
    // Test initialization logic
    EXPECT_TRUE(true); // Replace with actual test
}
```

Update `CMakeLists.txt`:
```cmake
find_package(ament_cmake_gtest REQUIRED)

ament_add_gtest(test_humanoid_controller
  test/test_humanoid_controller.cpp)
target_link_libraries(test_humanoid_controller
  humanoid_controller_node)
```

## Next Steps

In the next section, we'll explore ROS 2 tools and debugging techniques that are essential for developing and maintaining humanoid robotics systems.