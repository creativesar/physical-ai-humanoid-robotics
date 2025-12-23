---
sidebar_position: 3
title: "Nodes, Topics, Services, and Actions"
---

# Nodes, Topics, Services, and Actions

## Nodes

Nodes are the fundamental building blocks of ROS 2. Each node is a process that performs computation and communicates with other nodes through messages. Let's look at how to create and structure nodes in ROS 2.

### Creating a Basic Node in Python

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Subscriber Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Topics

Topics provide unidirectional, asynchronous communication using a publish/subscribe model. Multiple nodes can publish to the same topic, and multiple nodes can subscribe to the same topic.

### Topic Communication Characteristics
- **Asynchronous**: Publishers don't wait for subscribers
- **Unidirectional**: Data flows from publisher to subscriber
- **Many-to-many**: Multiple publishers and subscribers can use the same topic

### Creating Custom Message Types

To create custom message types, define them in `.msg` files:

```
# In msg/CustomMessage.msg
string name
int32 id
float64 value
bool active
```

## Services

Services provide bidirectional, synchronous communication using a request/response model. A service client sends a request and waits for a response from the service server.

### Creating a Service Server

```python
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Service Client

```python
import sys
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main():
    rclpy.init()
    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    minimal_client.get_logger().info(
        'Result of add_two_ints: for %d + %d = %d' %
        (int(sys.argv[1]), int(sys.argv[2]), response.sum))
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Actions

Actions provide bidirectional, asynchronous communication for long-running tasks with feedback. They're ideal for tasks that take time to complete and need to provide feedback during execution.

### Action States
- **Goal**: Request to start a long-running task
- **Feedback**: Continuous updates during task execution
- **Result**: Final outcome of the task

### Creating an Action Server

```python
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            self.get_logger().info('Publishing feedback: {feedback_msg.sequence}')
            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

### Creating an Action Client

```python
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {result.sequence}')
```

## Communication Patterns

### Publisher-Subscriber Pattern
Used for broadcasting information that multiple nodes might need, such as sensor data or robot state.

### Client-Server Pattern
Used for specific requests that require a response, such as calculating a value or performing a specific action.

### Action Pattern
Used for long-running tasks that require feedback, such as navigation or manipulation tasks.

## Best Practices

1. **Topic Naming**: Use descriptive, consistent names following ROS naming conventions
2. **Message Design**: Keep messages simple and focused on specific data
3. **Node Design**: Each node should have a single responsibility
4. **Error Handling**: Implement proper error handling for all communication
5. **Resource Management**: Properly clean up resources when nodes are destroyed

## Summary

Nodes, topics, services, and actions form the core communication infrastructure of ROS 2. Understanding these concepts is crucial for developing complex robotic systems:

- **Nodes** provide the execution context for your code
- **Topics** enable asynchronous, one-way communication
- **Services** enable synchronous, request-response communication
- **Actions** enable asynchronous, goal-oriented communication with feedback

In the next section, we'll explore how to build ROS 2 packages with Python.