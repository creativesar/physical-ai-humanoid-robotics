---
sidebar_position: 6
title: "Bridging Python Agents to ROS Controllers"
---

# Bridging Python Agents to ROS Controllers

## Introduction

In modern robotics, it's common to have AI agents implemented in Python that need to interact with ROS-based controllers. This bridging allows for high-level decision making, machine learning integration, and advanced planning to be seamlessly integrated with the low-level control systems of a robot.

## Architecture Overview

The bridge between Python agents and ROS controllers typically involves:

1. **High-level Python Agent**: Implements decision making, planning, and AI algorithms
2. **ROS Interface Layer**: Translates between agent actions and ROS messages/services
3. **ROS Controllers**: Low-level controllers that directly interface with robot hardware

## Implementing the Bridge

### Basic Bridge Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import cv2
from cv2 import cv2
from typing import Dict, Any, Optional
import json


class AgentBridgeNode(Node):
    def __init__(self):
        super().__init__('agent_bridge_node')

        # QoS profile for sensor data (best effort for real-time)
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # QoS profile for control commands (reliable)
        control_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Publishers for control commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', control_qos)
        self.joint_cmd_pub = self.create_publisher(Float64, '/joint_command', control_qos)

        # Subscribers for sensor data
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, sensor_qos)
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, sensor_qos)

        # Agent state variables
        self.laser_data = None
        self.camera_data = None
        self.agent_action = None

        # Timer for agent updates
        self.agent_timer = self.create_timer(0.1, self.agent_update_callback)  # 10 Hz

        # Initialize the Python agent
        self.agent = PythonRobotAgent()

        self.get_logger().info('Agent Bridge Node initialized')

    def laser_callback(self, msg: LaserScan):
        """Process laser scan data"""
        self.laser_data = {
            'ranges': list(msg.ranges),
            'intensities': list(msg.intensities),
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment,
            'time_increment': msg.time_increment,
            'scan_time': msg.scan_time,
            'range_min': msg.range_min,
            'range_max': msg.range_max
        }

    def camera_callback(self, msg: Image):
        """Process camera image data"""
        # Convert ROS Image message to OpenCV format
        try:
            # Convert to numpy array
            height = msg.height
            width = msg.width
            encoding = msg.encoding

            if encoding == 'rgb8':
                # Reshape the data into image format
                image_array = np.frombuffer(msg.data, dtype=np.uint8)
                image_array = image_array.reshape((height, width, 3))

                self.camera_data = {
                    'image': image_array,
                    'height': height,
                    'width': width,
                    'encoding': encoding
                }
        except Exception as e:
            self.get_logger().error(f'Error processing camera data: {e}')

    def agent_update_callback(self):
        """Update the agent and execute actions"""
        if self.laser_data is not None and self.camera_data is not None:
            # Prepare observation for the agent
            observation = {
                'laser_scan': self.laser_data,
                'camera_image': self.camera_data,
                'timestamp': self.get_clock().now().nanoseconds
            }

            # Get action from the agent
            action = self.agent.get_action(observation)

            # Execute the action
            self.execute_agent_action(action)

    def execute_agent_action(self, action: Dict[str, Any]):
        """Execute the action returned by the agent"""
        if 'cmd_vel' in action:
            # Send velocity command
            twist_msg = Twist()
            twist_msg.linear.x = action['cmd_vel'].get('linear_x', 0.0)
            twist_msg.linear.y = action['cmd_vel'].get('linear_y', 0.0)
            twist_msg.linear.z = action['cmd_vel'].get('linear_z', 0.0)
            twist_msg.angular.x = action['cmd_vel'].get('angular_x', 0.0)
            twist_msg.angular.y = action['cmd_vel'].get('angular_y', 0.0)
            twist_msg.angular.z = action['cmd_vel'].get('angular_z', 0.0)

            self.cmd_vel_pub.publish(twist_msg)

        if 'joint_cmd' in action:
            # Send joint command
            joint_msg = Float64()
            joint_msg.data = action['joint_cmd']
            self.joint_cmd_pub.publish(joint_msg)


class PythonRobotAgent:
    """High-level Python agent for robot control"""

    def __init__(self):
        self.state_history = []
        self.action_history = []
        self.episode_step = 0

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Process observation and return action"""
        # Process sensor data
        laser_ranges = observation['laser_scan']['ranges']
        camera_image = observation['camera_image']['image']

        # Simple obstacle avoidance based on laser data
        min_distance = min([r for r in laser_ranges if r > 0 and not np.isinf(r)], default=float('inf'))

        action = {}

        if min_distance < 0.5:  # Obstacle detected within 0.5m
            # Turn away from obstacle
            action['cmd_vel'] = {
                'linear_x': 0.0,
                'angular_z': 0.5  # Turn left
            }
        else:
            # Move forward
            action['cmd_vel'] = {
                'linear_x': 0.5,
                'angular_z': 0.0
            }

        # Store for history
        self.state_history.append(observation)
        self.action_history.append(action)
        self.episode_step += 1

        return action
```

## Advanced Agent Integration

### Reinforcement Learning Agent Bridge

```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random


class RLAgentBridgeNode(AgentBridgeNode):
    """ROS node that bridges with a reinforcement learning agent"""

    def __init__(self):
        super().__init__()

        # RL agent
        self.rl_agent = DQNAgent(state_size=360, action_size=5)  # 360 laser readings, 5 actions
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Training parameters
        self.batch_size = 32
        self.gamma = 0.95  # Discount factor
        self.update_target_freq = 100  # Update target network every 100 steps
        self.step_count = 0

        # Reward calculation
        self.previous_distance = 0.0

    def agent_update_callback(self):
        """Update the RL agent and execute actions"""
        if self.laser_data is not None and self.camera_data is not None:
            # Prepare state from sensor data
            state = self.process_sensor_data(self.laser_data, self.camera_data)

            # Get action from RL agent
            action = self.rl_agent.act(state, self.epsilon)

            # Execute action
            self.execute_action(action)

            # Calculate reward (simplified example)
            reward = self.calculate_reward(self.laser_data)

            # Store experience for training
            if hasattr(self, 'previous_state'):
                experience = (self.previous_state, self.previous_action, reward, state, False)
                self.memory.append(experience)

            # Train the agent
            if len(self.memory) > self.batch_size:
                self.train_agent()

            # Update for next iteration
            self.previous_state = state
            self.previous_action = action

            # Decay exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.step_count += 1

    def process_sensor_data(self, laser_data, camera_data):
        """Process sensor data into state representation for RL agent"""
        # Process laser data (simplified - take every 10th reading for 360-degree scan)
        laser_ranges = laser_data['ranges']
        processed_laser = []
        for i in range(0, len(laser_ranges), len(laser_ranges)//36):  # Reduce to 36 readings
            processed_laser.append(min(laser_ranges[i:i+len(laser_ranges)//36] or [10.0]))

        # Normalize laser readings
        processed_laser = [min(r/10.0, 1.0) for r in processed_laser]  # Normalize to 0-1

        return np.array(processed_laser)

    def calculate_reward(self, laser_data):
        """Calculate reward based on current state"""
        # Simplified reward function
        laser_ranges = laser_data['ranges']
        min_distance = min([r for r in laser_ranges if r > 0 and not np.isinf(r)], default=10.0)

        # Reward for moving forward without collision
        reward = 0
        if min_distance > 0.8:
            reward += 1.0  # Safe distance
        elif min_distance > 0.5:
            reward += 0.5  # Medium distance
        else:
            reward -= 10.0  # Collision risk

        # Small penalty for not moving forward
        reward -= 0.01

        return reward

    def execute_action(self, action):
        """Execute the action determined by the RL agent"""
        # Map discrete action to continuous command
        action_map = {
            0: {'linear_x': 0.5, 'angular_z': 0.0},    # Move forward
            1: {'linear_x': 0.3, 'angular_z': 0.5},    # Turn left
            2: {'linear_x': 0.3, 'angular_z': -0.5},   # Turn right
            3: {'linear_x': 0.0, 'angular_z': 0.8},    # Spin left
            4: {'linear_x': 0.0, 'angular_z': -0.8}    # Spin right
        }

        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = action_map[action]['linear_x']
        cmd_vel_msg.angular.z = action_map[action]['angular_z']

        self.cmd_vel_pub.publish(cmd_vel_msg)

    def train_agent(self):
        """Train the RL agent with experiences from memory"""
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        # Train the agent
        self.rl_agent.replay(states, actions, rewards, next_states, dones, self.batch_size)

        # Update target network periodically
        if self.step_count % self.update_target_freq == 0:
            self.rl_agent.update_target_network()


class DQNAgent:
    """Deep Q-Network agent for robotic control"""

    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # Neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Training parameters
        self.gamma = 0.95  # Discount factor

    def _build_model(self):
        """Build the neural network model"""
        class DQNModel(nn.Module):
            def __init__(self, state_size, action_size):
                super(DQNModel, self).__init__()
                self.fc1 = nn.Linear(state_size, 128)
                self.fc2 = nn.Linear(128, 128)
                self.fc3 = nn.Linear(128, 64)
                self.fc4 = nn.Linear(64, action_size)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.relu(self.fc3(x))
                return self.fc4(x)

        return DQNModel(self.state_size, self.action_size)

    def act(self, state, epsilon):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, states, actions, rewards, next_states, dones, batch_size):
        """Train the model on a batch of experiences"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Update the target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
```

## Communication Patterns

### Service-Based Agent Interface

```python
from rclpy.action import ActionServer, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
from std_msgs.msg import Bool


class ServiceBasedAgentBridge(AgentBridgeNode):
    """Agent bridge using services for communication"""

    def __init__(self):
        super().__init__()

        # Service for requesting agent actions
        self.agent_action_service = self.create_service(
            AgentAction,
            'request_agent_action',
            self.handle_agent_action_request
        )

        # Service for updating agent state
        self.update_agent_service = self.create_service(
            UpdateAgentState,
            'update_agent_state',
            self.handle_agent_state_update
        )

        # Publisher for agent status
        self.agent_status_pub = self.create_publisher(Bool, 'agent_active', 10)

        # Agent status
        self.agent_active = True

    def handle_agent_action_request(self, request, response):
        """Handle request for agent action"""
        if not self.agent_active:
            response.success = False
            response.message = "Agent is not active"
            return response

        try:
            # Prepare observation from request
            observation = self.parse_observation_request(request)

            # Get action from agent
            action = self.agent.get_action(observation)

            # Convert action to response
            response.action = self.convert_action_to_response(action)
            response.success = True
            response.message = "Action computed successfully"

        except Exception as e:
            response.success = False
            response.message = f"Error computing action: {str(e)}"

        return response

    def handle_agent_state_update(self, request, response):
        """Handle request to update agent state"""
        try:
            # Update agent with new state
            self.agent.update_state(request.state_data)
            response.success = True
            response.message = "Agent state updated successfully"
        except Exception as e:
            response.success = False
            response.message = f"Error updating agent state: {str(e)}"

        return response
```

### Action-Based Agent Interface

```python
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from example_interfaces.action import Fibonacci  # Using Fibonacci as example


class ActionBasedAgentBridge(AgentBridgeNode):
    """Agent bridge using actions for complex tasks"""

    def __init__(self):
        super().__init__()

        # Action server for complex agent tasks
        self.agent_task_server = ActionServer(
            self,
            AgentTask,  # Custom action type
            'execute_agent_task',
            self.execute_agent_task,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

        # Track active tasks
        self.active_tasks = {}

    def goal_callback(self, goal_request):
        """Handle incoming goal requests"""
        self.get_logger().info(f'Received agent task goal: {goal_request.task_type}')

        # Accept all goals for now
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle goal cancellation requests"""
        self.get_logger().info('Received agent task cancellation')
        return CancelResponse.ACCEPT

    def execute_agent_task(self, goal_handle):
        """Execute the agent task"""
        self.get_logger().info('Executing agent task')

        # Create feedback message
        feedback_msg = AgentTask.Feedback()

        # Initialize task in agent
        task_result = self.agent.execute_complex_task(
            goal_handle.request.task_type,
            goal_handle.request.task_parameters
        )

        # Process task with feedback
        for progress in task_result:
            feedback_msg.progress = progress['progress']
            feedback_msg.status = progress['status']
            goal_handle.publish_feedback(feedback_msg)

            # Check if task was canceled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = AgentTask.Result()
                result.success = False
                result.message = 'Task was canceled'
                return result

        # Task completed successfully
        goal_handle.succeed()
        result = AgentTask.Result()
        result.success = True
        result.message = 'Task completed successfully'
        result.result_data = task_result.get_final_result()

        return result
```

## Integration with ROS Controllers

### Joint Trajectory Controller Bridge

```python
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState


class JointControllerBridge(AgentBridgeNode):
    """Bridge between Python agents and joint trajectory controllers"""

    def __init__(self):
        super().__init__()

        # Publishers for joint trajectory commands
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)

        # Subscribers for joint state feedback
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # Controller state feedback
        self.controller_state_sub = self.create_subscription(
            JointTrajectoryControllerState,
            '/joint_trajectory_controller/state',
            self.controller_state_callback, 10)

        # Store current joint positions
        self.current_joint_positions = {}
        self.controller_state = None

    def joint_state_callback(self, msg: JointState):
        """Update current joint positions"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

    def controller_state_callback(self, msg):
        """Update controller state"""
        self.controller_state = msg

    def send_joint_trajectory(self, joint_names, positions, velocities=None,
                            accelerations=None, time_from_start=1.0):
        """Send joint trajectory command"""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions

        if velocities is not None:
            point.velocities = velocities
        if accelerations is not None:
            point.accelerations = accelerations

        # Set time from start
        point.time_from_start.sec = int(time_from_start)
        point.time_from_start.nanosec = int((time_from_start - int(time_from_start)) * 1e9)

        traj_msg.points = [point]

        self.joint_trajectory_pub.publish(traj_msg)

    def execute_agent_manipulation_action(self, action):
        """Execute manipulation action from agent"""
        if action['type'] == 'move_to_pose':
            # Convert pose to joint positions using inverse kinematics
            joint_positions = self.inverse_kinematics(action['target_pose'])
            joint_names = self.get_robot_joint_names()

            self.send_joint_trajectory(joint_names, joint_positions)

        elif action['type'] == 'gripper_control':
            # Control gripper
            self.control_gripper(action['gripper_position'])
```

## Performance Considerations

### Threading and Concurrency

```python
import threading
from rclpy.qos import QoSProfile
from std_msgs.msg import String


class ThreadedAgentBridge(AgentBridgeNode):
    """Agent bridge with proper threading support"""

    def __init__(self):
        super().__init__()

        # Lock for thread-safe access to shared data
        self.data_lock = threading.RLock()

        # Agent thread
        self.agent_thread = None
        self.agent_running = False

        # Start agent thread
        self.start_agent_thread()

    def start_agent_thread(self):
        """Start the agent processing thread"""
        self.agent_running = True
        self.agent_thread = threading.Thread(target=self.agent_worker, daemon=True)
        self.agent_thread.start()

    def agent_worker(self):
        """Agent processing loop running in separate thread"""
        while self.agent_running:
            with self.data_lock:
                if self.laser_data is not None and self.camera_data is not None:
                    observation = {
                        'laser_scan': self.laser_data,
                        'camera_image': self.camera_data,
                        'timestamp': self.get_clock().now().nanoseconds
                    }

                    action = self.agent.get_action(observation)

                    # Publish action to main thread for ROS communication
                    action_msg = String()
                    action_msg.data = json.dumps(action)
                    self.agent_action_pub.publish(action_msg)

            # Sleep to control update rate
            time.sleep(0.05)  # 20 Hz

    def stop_agent_thread(self):
        """Stop the agent processing thread"""
        self.agent_running = False
        if self.agent_thread:
            self.agent_thread.join()
```

## Error Handling and Safety

### Safety Monitor

```python
class SafetyMonitor:
    """Safety monitor for agent actions"""

    def __init__(self, node):
        self.node = node
        self.emergency_stop = False
        self.safety_limits = {
            'max_velocity': 1.0,
            'max_angular_velocity': 1.0,
            'min_distance': 0.3,
            'max_joint_velocity': 2.0
        }

    def check_action_safety(self, action):
        """Check if an action is safe to execute"""
        if self.emergency_stop:
            return False, "Emergency stop active"

        if 'cmd_vel' in action:
            cmd_vel = action['cmd_vel']

            # Check velocity limits
            if abs(cmd_vel.get('linear_x', 0)) > self.safety_limits['max_velocity']:
                return False, f"Linear velocity exceeds limit: {cmd_vel.get('linear_x', 0)}"

            if abs(cmd_vel.get('angular_z', 0)) > self.safety_limits['max_angular_velocity']:
                return False, f"Angular velocity exceeds limit: {cmd_vel.get('angular_z', 0)}"

        # If all checks pass
        return True, "Action is safe"

    def emergency_stop_callback(self):
        """Activate emergency stop"""
        self.emergency_stop = True
        self.node.get_logger().warn('EMERGENCY STOP ACTIVATED')
```

## Summary

Bridging Python agents to ROS controllers involves creating a communication layer that translates between high-level agent decisions and low-level ROS messages. Key aspects include:

- Proper ROS node structure with publishers/subscribers
- State representation and action mapping
- Integration with different ROS communication patterns (topics, services, actions)
- Performance optimization with threading
- Safety monitoring and error handling

This bridge enables advanced AI and machine learning algorithms to control robotic systems through the robust ROS framework.

In the next section, we'll explore URDF (Unified Robot Description Format) for humanoid robot description.