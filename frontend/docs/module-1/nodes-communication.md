---
sidebar_position: 3
title: "Nodes and Communication"
---

# Nodes and Communication

## Understanding Nodes in ROS 2

Nodes are the fundamental building blocks of any ROS 2 system. In the context of humanoid robotics, nodes represent individual components of the robot system such as sensor drivers, control algorithms, perception modules, and user interfaces.

## Creating Nodes

### Node Structure

Every ROS 2 node typically includes:

1. **Node Class Definition**: Inherits from `rclcpp::Node` or uses the Python equivalent
2. **Initialization**: Constructor that sets up the node name and other parameters
3. **Communication Interfaces**: Publishers, subscribers, services, and actions
4. **Execution Loop**: Spin loop that processes callbacks

### Example Node Implementation

```cpp
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

class HumanoidController : public rclcpp::Node
{
public:
    HumanoidController() : Node("humanoid_controller")
    {
        // Create subscriber for joint states
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", 10,
            std::bind(&HumanoidController::jointStateCallback, this, std::placeholders::_1));

        // Create publisher for joint commands
        joint_command_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "joint_commands", 10);

        RCLCPP_INFO(this->get_logger(), "Humanoid Controller Node Initialized");
    }

private:
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received %d joint states", msg->name.size());
        // Process joint states and generate commands
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_command_pub_;
};
```

## Communication Patterns

### 1. Topics (Publish-Subscribe)

Topics are used for streaming data between nodes and are ideal for sensor data, robot state information, and other continuous data streams.

#### Key Characteristics:
- Unidirectional data flow
- Many-to-many communication
- Asynchronous communication
- Data is sent regardless of subscriber presence

#### Use Cases in Humanoid Robotics:
- Sensor data (IMU, cameras, joint encoders)
- Robot state information
- Trajectory commands
- Perception results

### 2. Services (Request-Response)

Services provide synchronous communication where a client sends a request and waits for a response.

#### Key Characteristics:
- Synchronous communication
- One-to-one communication
- Request-response pattern
- Blocking until response received

#### Use Cases in Humanoid Robotics:
- Calibration procedures
- Configuration changes
- Diagnostic requests
- Emergency stop commands

### 3. Actions (Goal-Feedback-Result)

Actions are designed for long-running tasks that require feedback and the ability to be canceled.

#### Key Characteristics:
- Asynchronous with feedback
- Goal-feedback-result pattern
- Cancellation capability
- Ideal for tasks with duration

#### Use Cases in Humanoid Robotics:
- Navigation to a location
- Manipulation tasks
- Walking gaits
- Complex motion sequences

## Advanced Communication Features

### Quality of Service (QoS) Configuration

QoS settings allow fine-tuning of communication behavior:

```cpp
// Configure QoS for sensor data (high frequency, best effort)
rclcpp::QoS sensor_qos(10);
sensor_qos.best_effort().keep_last(5);

// Configure QoS for critical commands (reliable, transient)
rclcpp::QoS command_qos(5);
command_qos.reliable().transient_local().keep_all();

auto sensor_pub = this->create_publisher<sensor_msgs::msg::JointState>(
    "sensor_data", sensor_qos);
auto command_pub = this->create_publisher<control_msgs::msg::JointTrajectoryControllerCommand>(
    "critical_commands", command_qos);
```

### Parameters

Parameters provide a way to configure nodes at runtime:

```cpp
// Declare parameters with default values
this->declare_parameter("control_frequency", 100);
this->declare_parameter("max_velocity", 1.0);
this->declare_parameter("robot_name", "atlas");

// Get parameter values
double control_freq = this->get_parameter("control_frequency").as_double();
std::string robot_name = this->get_parameter("robot_name").as_string();

// Parameter callback for dynamic reconfiguration
auto param_change_callback = [this](const std::vector<rclcpp::Parameter> & parameters) {
    for (const auto & param : parameters) {
        if (param.get_name() == "control_frequency") {
            RCLCPP_INFO(this->get_logger(), "Control frequency changed to: %f",
                       param.as_double());
        }
    }
};
this->set_on_parameters_set_callback(param_change_callback);
```

## Inter-Process Communication

### Client Libraries

ROS 2 supports multiple client libraries that can communicate with each other:

- **C++ (rclcpp)**: High-performance applications
- **Python (rclpy)**: Rapid prototyping and scripting
- **Other languages**: Java, C, etc.

### Cross-Language Communication Example

A Python node can subscribe to topics published by a C++ node:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateMonitor(Node):
    def __init__(self):
        super().__init__('joint_state_monitor')
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.listener_callback,
            10)

    def listener_callback(self, msg):
        self.get_logger().info(f'Received joint states for {len(msg.name)} joints')
```

## Best Practices for Humanoid Robotics

### 1. Modular Design
- Separate perception, planning, and control into different nodes
- Use composition for performance-critical components
- Design nodes with single responsibilities

### 2. Error Handling
- Implement proper exception handling
- Use latching for critical state topics
- Monitor node health and connections

### 3. Performance Optimization
- Use intra-process communication when appropriate
- Configure QoS settings based on data requirements
- Implement proper message throttling

### 4. Security Considerations
- Implement proper namespace isolation
- Use ROS 2 security features for sensitive data
- Validate all incoming messages

## Debugging and Monitoring

### ROS 2 Tools
- `ros2 topic list`: List available topics
- `ros2 topic echo <topic>`: View topic data
- `ros2 node list`: List active nodes
- `ros2 run <pkg> <exec> __params:=<file>`: Launch with parameter file

### Custom Monitoring
```cpp
// Add custom diagnostics
#include "diagnostic_updater/diagnostic_updater.hpp"

class DiagnosticsNode : public rclcpp::Node
{
public:
    DiagnosticsNode() : Node("diagnostics_node")
    {
        updater_.setHardwareID("humanoid_robot");
        updater_.add("Joint Controller Status", this, &DiagnosticsNode::check_joints);
    }

private:
    void check_joints(diagnostic_updater::DiagnosticStatusWrapper & stat)
    {
        // Custom diagnostic logic
        stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "All joints operational");
        stat.add("Joint Count", 28); // Example for humanoid robot
    }

    diagnostic_updater::Updater updater_;
};
```

## Next Steps

In the next section, we'll explore how to create and manage ROS 2 packages and workspaces, which are essential for organizing your humanoid robotics codebase.