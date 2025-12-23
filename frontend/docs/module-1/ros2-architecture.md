---
sidebar_position: 2
title: "ROS 2 Architecture and Core Concepts"
---

# ROS 2 Architecture and Core Concepts

## Introduction to ROS 2

Robot Operating System 2 (ROS 2) is not an actual operating system, but rather a flexible framework for writing robotic software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robotic applications.

ROS 2 was designed to address the limitations of ROS 1, particularly in areas of real-time support, security, and multi-platform compatibility. It provides a distributed computing framework where multiple processes can interact seamlessly, regardless of the programming language they're written in.

## Key Concepts

### Nodes
Nodes are the fundamental building blocks of ROS 2. A node is a process that performs computation and communicates with other nodes through messages. Each node should perform a specific task and communicate with other nodes to achieve complex robotic behaviors.

### Packages
Packages are the basic unit of organization in ROS 2. A package contains nodes, libraries, configuration files, and other resources needed for a specific functionality. Packages provide a way to organize and distribute ROS 2 software.

### Communication Primitives
ROS 2 provides three main communication mechanisms:
- **Topics**: Unidirectional, asynchronous communication using a publish/subscribe model
- **Services**: Bidirectional, synchronous communication using a request/response model
- **Actions**: Bidirectional, asynchronous communication for long-running tasks with feedback

## Architecture Layers

### 1. Client Library Layer
The client library layer provides the API that developers use to interact with ROS 2. The most common client libraries are:
- **rclcpp**: C++ client library
- **rclpy**: Python client library
- **rcl**: Common client library layer that provides the foundation for other client libraries

### 2. Middleware Layer
The middleware layer handles the actual communication between nodes. ROS 2 uses the Data Distribution Service (DDS) as its default middleware. DDS implementations include:
- Fast DDS (default in recent ROS 2 versions)
- Cyclone DDS
- RTI Connext DDS

### 3. Operating System Layer
ROS 2 can run on various operating systems including Linux, Windows, and macOS, with real-time operating systems like Real-Time ROS 2 for time-critical applications.

## Quality of Service (QoS) Settings

ROS 2 introduces Quality of Service settings that allow fine-tuning of communication behavior:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Example QoS profile for sensor data
sensor_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST
)

# Example QoS profile for real-time control
control_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST
)
```

## Execution Models

ROS 2 provides different execution models to manage node execution:

### Single-threaded Executor
Executes all nodes sequentially in a single thread.

### Multi-threaded Executor
Distributes nodes across multiple threads for concurrent execution.

## Namespaces and Naming

ROS 2 uses a hierarchical naming system with namespaces to organize nodes and topics:

```
/namespace/node_name
/namespace/topic_name
```

## Parameters

Parameters in ROS 2 provide a way to configure nodes at runtime:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')

        # Declare parameters with default values
        self.declare_parameter('param_name', 'default_value')

        # Get parameter value
        param_value = self.get_parameter('param_name').value
```

## Lifecycle Nodes

ROS 2 introduces lifecycle nodes that have a well-defined state machine, allowing for better system management and coordination.

## Security

ROS 2 includes built-in security features including:
- Authentication
- Authorization
- Encryption

## Summary

ROS 2 provides a robust framework for developing complex robotic systems with features like:
- Distributed computing capabilities
- Language independence
- Quality of Service settings
- Security features
- Lifecycle management
- Real-time support

In the next section, we'll explore the communication primitives in detail, focusing on nodes, topics, services, and actions.