---
sidebar_position: 2
title: "ROS 2 Architecture"
---

# ROS 2 Architecture

## Understanding the ROS 2 Architecture

ROS 2 uses a distributed architecture based on the Data Distribution Service (DDS) standard. This architecture provides several key advantages over ROS 1, including improved real-time support, enhanced security features, and better cross-platform compatibility.

## DDS: The Foundation of ROS 2

### What is DDS?

The Data Distribution Service (DDS) is an Object Management Group (OMG) standard that defines a middleware protocol and API for device-to-device, application-to-application, and service-to-service communication. In ROS 2, DDS serves as the communication layer that handles:

- Message passing between nodes
- Discovery of nodes and services
- Quality of Service (QoS) policies
- Security features
- Data serialization

### DDS Implementation Options

ROS 2 is designed to be DDS-implementation agnostic, meaning you can choose from several DDS implementations:

- **Fast DDS (formerly Fast RTPS)**: Developed by eProsima, commonly used
- **Cyclone DDS**: Developed by ADLINK and Eclipse Foundation
- **RTI Connext DDS**: Commercial implementation by RTI
- **OpenSplice DDS**: Open-source implementation by ADLINK

## Core Architecture Components

### 1. Nodes

Nodes are the fundamental execution units in ROS 2. Each node:
- Encapsulates a specific functionality
- Communicates with other nodes through topics, services, or actions
- Runs in its own process
- Can be written in different programming languages

### 2. Communication Primitives

#### Topics
- Unidirectional, many-to-many communication
- Uses publisher-subscriber pattern
- Asynchronous communication
- Ideal for streaming data like sensor readings

#### Services
- Bidirectional, request-response communication
- Synchronous communication
- One-to-one communication
- Suitable for operations that return a result

#### Actions
- Bidirectional communication for long-running tasks
- Supports goal requests, feedback, and result responses
- Can be preempted
- Perfect for navigation or manipulation tasks

### 3. Parameters

Parameters provide a way to configure nodes:
- Key-value pairs that can be set at runtime
- Hierarchical parameter namespacing
- Parameter change callbacks
- Automatic parameter persistence

## Quality of Service (QoS) Policies

QoS policies allow you to fine-tune communication behavior:

### Reliability Policy
- **Reliable**: All messages are guaranteed to be delivered
- **Best Effort**: Messages may be lost, but faster delivery

### Durability Policy
- **Transient Local**: Late-joining subscribers receive recent messages
- **Volatile**: Only future messages are delivered

### History Policy
- **Keep Last**: Maintain only the most recent messages
- **Keep All**: Maintain all messages (limited by memory)

### Deadline Policy
- Specifies the maximum time between consecutive messages

### Lifespan Policy
- Specifies how long a message remains valid

## Launch System

The launch system in ROS 2 provides a way to start multiple nodes with a single command:

### Launch Files
- XML or Python files that define node configurations
- Parameter settings for multiple nodes
- Remapping of topics and services
- Conditional startup logic

### Composition
- Run multiple nodes in a single process
- Reduce inter-process communication overhead
- Improve performance for tightly coupled components

## Security in ROS 2

ROS 2 includes built-in security features:

### Authentication
- Verify identity of nodes
- Prevent unauthorized nodes from joining the system

### Access Control
- Control which nodes can communicate with each other
- Define permissions for topics, services, and parameters

### Encryption
- Encrypt data in transit
- Protect sensitive information

## Client Libraries

ROS 2 supports multiple client libraries:

### rclcpp
- C++ client library
- Object-oriented design
- Direct access to underlying DDS implementation

### rclpy
- Python client library
- Pythonic interface
- Easy to use for rapid prototyping

### Other Languages
- rcl (C core library)
- rclnodejs (Node.js)
- rclc (C for microcontrollers)

## Practical Example: Creating a Simple Publisher-Subscriber System

```cpp
// Publisher Example
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher()
    : Node("minimal_publisher"), count_(0)
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        timer_ = this->create_wall_timer(
            500ms, std::bind(&MinimalPublisher::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello, world! " + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
};
```

## Next Steps

In the next section, we'll explore nodes and communication patterns in detail, learning how to implement the publisher-subscriber model, services, and actions in ROS 2.