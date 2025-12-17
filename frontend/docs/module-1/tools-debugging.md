---
sidebar_position: 5
title: "ROS 2 Tools and Debugging"
---

# ROS 2 Tools and Debugging

## Essential ROS 2 Command Line Tools

ROS 2 provides a comprehensive set of command-line tools for inspecting, debugging, and managing your robotic systems. These tools are crucial for developing and maintaining humanoid robotics applications.

## Core Command Line Tools

### 1. Node Management

#### Listing Nodes
```bash
# List all active nodes
ros2 node list

# Get information about a specific node
ros2 node info /humanoid_controller
```

#### Node Execution
```bash
# Run a node directly
ros2 run package_name executable_name

# Run with parameters
ros2 run package_name executable_name --ros-args --param-file params.yaml

# Run with remappings
ros2 run package_name executable_name --ros-args --remap old_topic:=new_topic
```

### 2. Topic Management

#### Inspecting Topics
```bash
# List all topics
ros2 topic list

# Get information about a specific topic
ros2 topic info /joint_states

# Echo topic data (view messages in real-time)
ros2 topic echo /joint_states

# Echo with specific number of messages
ros2 topic echo /joint_states --field position --field velocity -n 5

# Show topic statistics
ros2 topic hz /camera/image_raw
```

#### Publishing to Topics
```bash
# Publish a single message
ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 1.0}, angular: {z: 0.5}}'

# Publish with a specific rate
ros2 topic pub -r 10 /cmd_vel geometry_msgs/Twist '{linear: {x: 1.0}}'
```

### 3. Service Management

#### Service Calls
```bash
# List all services
ros2 service list

# Call a service
ros2 service call /reset_positions std_srvs/srv/Empty

# Call with parameters
ros2 service call /set_parameters rcl_interfaces/srv/SetParameters '{parameters: [{name: "control_frequency", value: {type: 3, double_value: 100.0}}]}'
```

### 4. Action Management

#### Action Commands
```bash
# List all actions
ros2 action list

# Send a goal to an action server
ros2 action send_goal /move_joint control_msgs/action/FollowJointTrajectory trajectory.yaml

# Get action info
ros2 action info /move_joint
```

### 5. Parameter Management

#### Parameter Operations
```bash
# List parameters of a node
ros2 param list /humanoid_controller

# Get parameter value
ros2 param get /humanoid_controller control_frequency

# Set parameter value
ros2 param set /humanoid_controller control_frequency 200

# Load parameters from file
ros2 param load /humanoid_controller params.yaml
```

## Graph Visualization

### ros2 graph
Visualize the communication graph between nodes:
```bash
# Install graphviz first
sudo apt install graphviz

# Generate graph
ros2 graph --output-format png --show-arguments

# Generate graph for specific nodes
ros2 graph --include-hidden-nodes
```

## Advanced Debugging Techniques

### 1. Logging and Diagnostics

#### RCLCPP Logging
```cpp
#include "rclcpp/rclcpp.hpp"

class DebugNode : public rclcpp::Node
{
public:
    DebugNode() : Node("debug_node")
    {
        // Different log levels
        RCLCPP_INFO(this->get_logger(), "This is an info message");
        RCLCPP_WARN(this->get_logger(), "This is a warning");
        RCLCPP_ERROR(this->get_logger(), "This is an error");
        RCLCPP_DEBUG(this->get_logger(), "This is a debug message");

        // Conditional logging
        RCLCPP_INFO_EXPRESSION(
            this->get_logger(),
            (some_condition == true),
            "Conditional log message"
        );
    }
};
```

#### Custom Logger Configuration
```cpp
// Configure logger settings
auto logger = rclcpp::get_logger("custom_logger");
RCLCPP_LOG_SEVERITY_THRESHOLD(logger, RCLCPP_DEBUG);
```

### 2. Memory and Performance Monitoring

#### Using ros2 doctor
Check the health of your ROS 2 system:
```bash
# Basic health check
ros2 doctor

# Detailed check
ros2 doctor --report
```

#### Performance Analysis
```bash
# Monitor CPU and memory usage
htop

# Monitor network usage (for distributed systems)
nethogs

# Monitor ROS 2 communication
ros2 topic hz /topic_name
```

### 3. Profiling Tools

#### Using tracetools
For detailed performance analysis:
```bash
# Install tracetools
sudo apt install ros-humble-tracetools

# Record trace data
ros2 trace my_trace_directory

# Launch your application
ros2 launch my_package my_launch.py

# Stop tracing with Ctrl+C
```

## Debugging Humanoid Robotics Systems

### 1. Sensor Data Validation

#### Validating Joint States
```bash
# Monitor joint states for consistency
ros2 topic echo /joint_states --field name --field position | head -n 20

# Check for NaN values in sensor data
ros2 topic echo /imu/data --field orientation_covariance | grep -E 'nan|inf'
```

#### Visualizing Sensor Data
```bash
# Use rviz2 for visualization
ros2 run rviz2 rviz2

# Add displays for joint states, IMU data, etc.
```

### 2. Control Loop Debugging

#### Monitoring Control Frequencies
```bash
# Check the frequency of control commands
ros2 topic hz /joint_commands

# Monitor feedback loops
ros2 topic hz /feedback_topic
```

### 3. Synchronization Issues

#### Time Synchronization
```cpp
// Use synchronized time in simulation
this->declare_parameter("use_sim_time", false);
bool use_sim_time = this->get_parameter("use_sim_time").as_bool();

// For simulation environments
if (use_sim_time) {
    // Use simulation time instead of system time
    rclcpp::Time current_time = this->now();
}
```

## Debugging Tools and Techniques

### 1. Interactive Debugging

#### Using rqt tools
```bash
# Install rqt tools
sudo apt install ros-humble-rqt ros-humble-rqt-common-plugins

# Launch rqt
rqt

# Use various plugins:
# - rqt_graph: Visualize node connections
# - rqt_plot: Plot numeric values
# - rqt_console: View log messages
# - rqt_bag: Play and record bag files
```

### 2. Bag Files for Data Recording

#### Recording Data
```bash
# Record all topics
ros2 bag record -a

# Record specific topics
ros2 bag record /joint_states /imu/data /camera/image_raw

# Record with compression
ros2 bag record --compression-mode file --compression-format zstd /joint_states

# Record with duration limit
ros2 bag record -d 60 /joint_states  # Record for 60 seconds
```

#### Playing Back Data
```bash
# Play back a bag file
ros2 bag play my_bag_file

# Play with playback rate
ros2 bag play my_bag_file --rate 0.5  # Play at half speed

# Play specific topics
ros2 bag play my_bag_file --topics /joint_states
```

### 3. Remote Debugging

#### SSH for Remote Systems
```bash
# Set up ROS 2 environment on remote robot
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp

# Connect to remote robot from development machine
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp
```

## Common Debugging Scenarios in Humanoid Robotics

### 1. Joint Control Issues

#### Troubleshooting Joint Commands
```bash
# Check if joint command topics are being published
ros2 topic echo /position_commands

# Verify joint state feedback
ros2 topic echo /joint_states --field position --field velocity

# Check controller status
ros2 service call /controller_manager/list_controllers controller_manager_msgs/srv/ListControllers
```

### 2. Balance and Stability Problems

#### IMU and Sensor Validation
```bash
# Monitor IMU data for drift
ros2 topic echo /imu/data --field angular_velocity --field linear_acceleration

# Check orientation stability
ros2 topic echo /imu/data --field orientation
```

### 3. Communication Latency

#### Measuring Latency
```bash
# Use timestamp comparison
ros2 topic echo /topic_name --field header.stamp

# Monitor for dropped messages
ros2 topic hz /critical_topic --window 100
```

## Advanced Debugging Setup

### 1. Custom Diagnostic Nodes

```cpp
#include "diagnostic_updater/diagnostic_updater.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"

class HumanoidDiagnostics : public rclcpp::Node
{
public:
    HumanoidDiagnostics() : Node("humanoid_diagnostics")
    {
        updater_.setHardwareID("humanoid_robot_v1.0");
        updater_.add("Joint Controller Status", this, &HumanoidDiagnostics::check_joints);
        updater_.add("Sensor Status", this, &HumanoidDiagnostics::check_sensors);
        updater_.add("Communication Status", this, &HumanoidDiagnostics::check_communication);

        // Timer for periodic updates
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&HumanoidDiagnostics::update_diagnostics, this));
    }

private:
    void check_joints(diagnostic_updater::DiagnosticStatusWrapper & stat)
    {
        // Custom diagnostic logic
        stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "All joints operational");
        stat.add("Joint Count", 28);
        stat.add("Position Errors", 0);
    }

    void check_sensors(diagnostic_updater::DiagnosticStatusWrapper & stat)
    {
        // Sensor diagnostic logic
        stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "All sensors operational");
    }

    void check_communication(diagnostic_updater::DiagnosticStatusWrapper & stat)
    {
        // Communication diagnostic logic
        stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "Communication nominal");
    }

    void update_diagnostics()
    {
        updater_.update();
    }

    diagnostic_updater::Updater updater_;
    rclcpp::TimerBase::SharedPtr timer_;
};
```

### 2. Logging Best Practices

#### Structured Logging
```cpp
// Use structured logging for better analysis
RCLCPP_INFO(
    this->get_logger(),
    "Joint %s position: %.3f, velocity: %.3f, effort: %.3f",
    joint_name.c_str(),
    position,
    velocity,
    effort
);

// Log with context
RCLCPP_INFO_STREAM(
    this->get_logger(),
    "Control cycle " << cycle_count <<
    " - Position error: " << position_error <<
    " - Velocity error: " << velocity_error
);
```

## Troubleshooting Common Issues

### 1. Node Discovery Problems
```bash
# Check if nodes can discover each other
export ROS_DOMAIN_ID=0  # Ensure same domain ID
export RMW_IMPLEMENTATION=  # Use same DDS implementation

# Check network configuration
ifconfig  # Verify network interfaces
netstat -tulpn | grep -i ros  # Check ROS ports
```

### 2. Permission Issues
```bash
# Check ROS log directory permissions
ls -la ~/.ros/log/
sudo chown -R $USER:$USER ~/.ros/
```

### 3. Memory Issues
```bash
# Monitor memory usage
ros2 run top top  # Or use system tools like htop
# Set memory limits in launch files if needed
```

## Development Workflow Integration

### 1. IDE Integration

Most modern IDEs support ROS 2 development:
- **VS Code**: With ROS extension
- **CLion**: With ROS plugins
- **Qt Creator**: For C++ development

### 2. Continuous Integration

Example GitHub Actions workflow:
```yaml
name: ROS 2 CI
on: [push, pull_request]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup ROS 2
        uses: ros-tooling/setup-ros@v0.7
        with:
          required-ros-distributions: humble
      - name: Build and Test
        uses: ros-tooling/action-ros-ci@v0.3
        with:
          package-name: humanoid_controller
          target-ros2-distro: humble
```

## Next Steps

In the next section, we'll explore how ROS 2 is specifically applied to humanoid robotics, covering topics like joint control, balance algorithms, and human-robot interaction patterns.