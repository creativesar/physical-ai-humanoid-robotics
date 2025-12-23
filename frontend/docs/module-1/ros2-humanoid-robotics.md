---
sidebar_position: 8
title: "ROS 2 for Humanoid Robotics"
---

# ROS 2 for Humanoid Robotics

## Introduction

Humanoid robotics represents one of the most complex and challenging areas in robotics, requiring sophisticated control systems, perception, planning, and coordination. ROS 2 provides the ideal framework for developing humanoid robots, offering the necessary tools for distributed computing, real-time performance, and system integration.

## Challenges in Humanoid Robotics

Humanoid robots face unique challenges that require specialized approaches:

### 1. Balance and Locomotion
- Maintaining balance with bipedal locomotion
- Real-time control for stability
- Dynamic walking patterns
- Recovery from disturbances

### 2. Coordination and Control
- Synchronizing multiple degrees of freedom
- Coordinated movement of arms, legs, and torso
- Smooth transitions between behaviors
- Redundancy resolution in high-DOF systems

### 3. Perception and Interaction
- 3D perception for environment understanding
- Human-robot interaction
- Object manipulation
- Social robotics aspects

## ROS 2 Architecture for Humanoid Systems

### Multi-Node Architecture

A typical humanoid robot system in ROS 2 consists of multiple interconnected nodes:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Perception      │    │ Planning        │    │ Control         │
│ Nodes           │◄──►│ Nodes           │◄──►│ Nodes           │
│ - Vision        │    │ - Path Planner  │    │ - Joint Control │
│ - IMU Processing│    │ - Motion Planner│    │ - Balance Ctrl  │
│ - Sensor Fusion │    │ - Behavior Tree │    │ - Trajectory Gen│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Humanoid Middleware Layer                    │
│              - State Estimation                               │
│              - Sensor Integration                             │
│              - Safety Monitoring                              │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Hardware        │
│ Interface       │
│ - Joint Drivers │
│ - Sensor Read   │
│ - Safety Sys    │
└─────────────────┘
```

### Example Humanoid System Launch File

```python
# launch/humanoid_system.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_description_path = LaunchConfiguration('robot_description_path')

    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    robot_description_arg = DeclareLaunchArgument(
        'robot_description_path',
        default_value=os.path.join(
            get_package_share_directory('humanoid_description'),
            'urdf',
            'humanoid.urdf'
        ),
        description='Path to robot description file'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': open(robot_description_path).read()}
        ]
    )

    # Joint state publisher (for simulation)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Perception nodes
    vision_node = Node(
        package='humanoid_perception',
        executable='vision_processor',
        name='vision_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    imu_processor = Node(
        package='humanoid_perception',
        executable='imu_processor',
        name='imu_processor',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Control nodes
    balance_controller = Node(
        package='humanoid_control',
        executable='balance_controller',
        name='balance_controller',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
        respawn=True
    )

    joint_trajectory_controller = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            os.path.join(
                get_package_share_directory('humanoid_control'),
                'config',
                'controllers.yaml'
            )
        ],
        output='screen'
    )

    # Planning nodes
    motion_planner = Node(
        package='humanoid_planning',
        executable='motion_planner',
        name='motion_planner',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    behavior_tree = Node(
        package='humanoid_behavior',
        executable='behavior_tree',
        name='behavior_tree',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Safety monitor
    safety_monitor = Node(
        package='humanoid_safety',
        executable='safety_monitor',
        name='safety_monitor',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
        respawn=True
    )

    return LaunchDescription([
        use_sim_time_arg,
        robot_description_arg,
        robot_state_publisher,
        joint_state_publisher,
        TimerAction(
            period=1.0,
            actions=[
                vision_node,
                imu_processor,
                balance_controller,
                joint_trajectory_controller,
                motion_planner,
                behavior_tree,
                safety_monitor
            ]
        )
    ])
```

## Balance Control Systems

### Center of Mass (CoM) Control

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Vector3, Point
from std_msgs.msg import Float64
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np
from scipy import signal


class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller')

        # Publishers and subscribers
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.com_publisher = self.create_publisher(
            Point, '/center_of_mass', 10)
        self.zmp_publisher = self.create_publisher(
            Point, '/zero_moment_point', 10)

        # Joint command publishers
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)

        # Balance control parameters
        self.com_position = np.array([0.0, 0.0, 0.0])
        self.com_velocity = np.array([0.0, 0.0, 0.0])
        self.com_acceleration = np.array([0.0, 0.0, 0.0])

        self.support_polygon = []  # Define support polygon based on feet positions
        self.zmp_reference = np.array([0.0, 0.0])

        # Control gains
        self.balance_gains = {
            'kp': 50.0,  # Proportional gain
            'kd': 10.0,  # Derivative gain
            'ki': 1.0    # Integral gain
        }

        # Low-pass filter for sensor data
        self.filter_b, self.filter_a = signal.butter(2, 0.1, 'low')

        # Timer for balance control loop
        self.balance_timer = self.create_timer(0.01, self.balance_control_loop)  # 100 Hz

        self.get_logger().info('Balance Controller initialized')

    def imu_callback(self, msg: Imu):
        """Process IMU data for balance estimation"""
        # Extract orientation and angular velocity
        orientation = msg.orientation
        angular_velocity = msg.angular_velocity
        linear_acceleration = msg.linear_acceleration

        # Calculate roll, pitch, yaw from quaternion
        # (simplified - in practice, use tf2 for quaternion operations)
        # This is a basic implementation; real systems use more sophisticated filtering

    def joint_state_callback(self, msg: JointState):
        """Process joint states for CoM calculation"""
        # Calculate current center of mass based on joint positions and link masses
        # This is a simplified example - real CoM calculation involves complex kinematics
        pass

    def balance_control_loop(self):
        """Main balance control loop"""
        # Calculate current ZMP (Zero Moment Point)
        current_zmp = self.calculate_zmp()

        # Calculate error from reference
        zmp_error = self.zmp_reference - current_zmp

        # Simple PD control for balance
        balance_correction = (
            self.balance_gains['kp'] * zmp_error +
            self.balance_gains['kd'] * self.get_angular_velocity_error()
        )

        # Generate corrective joint commands
        corrective_commands = self.generate_balance_commands(balance_correction)

        # Publish commands
        self.publish_joint_commands(corrective_commands)

    def calculate_zmp(self):
        """Calculate Zero Moment Point"""
        # ZMP = [x, y] where moments around x and y axes are zero
        # Simplified calculation - real implementation requires full dynamics model
        zmp_x = self.com_position[0] - (self.com_position[2] * self.com_acceleration[0]) / 9.81
        zmp_y = self.com_position[1] - (self.com_position[2] * self.com_acceleration[1]) / 9.81

        return np.array([zmp_x, zmp_y])

    def generate_balance_commands(self, correction):
        """Generate joint commands for balance correction"""
        # This would implement inverse kinematics to achieve desired balance
        # For now, return a simplified example
        commands = {}

        # Adjust hip joints for lateral balance
        commands['left_hip_roll'] = correction[1] * 0.1  # Proportional to lateral error
        commands['right_hip_roll'] = -correction[1] * 0.1

        # Adjust ankle joints for forward/backward balance
        commands['left_ankle_pitch'] = correction[0] * 0.1
        commands['right_ankle_pitch'] = correction[0] * 0.1

        return commands

    def publish_joint_commands(self, commands):
        """Publish joint trajectory commands"""
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        traj_msg = JointTrajectory()
        traj_msg.joint_names = list(commands.keys())

        point = JointTrajectoryPoint()
        point.positions = list(commands.values())
        point.time_from_start = Duration(sec=0, nanosec=50000000)  # 50ms

        traj_msg.points = [point]

        self.joint_trajectory_pub.publish(traj_msg)
```

## Walking Pattern Generation

### Inverse Kinematics for Walking

```python
import numpy as np
from math import sin, cos, sqrt, atan2, acos, pi


class WalkingPatternGenerator:
    def __init__(self, robot_params):
        # Robot-specific parameters
        self.hip_offset = robot_params['hip_offset']
        self.leg_length = robot_params['leg_length']
        self.torso_height = robot_params['torso_height']

        # Walking parameters
        self.step_length = 0.2  # meters
        self.step_height = 0.05  # meters
        self.step_duration = 1.0  # seconds
        self.dsp_ratio = 0.6  # Double support phase ratio

    def calculate_foot_trajectory(self, time, support_leg, step_params):
        """
        Calculate foot trajectory for walking
        support_leg: 'left' or 'right'
        step_params: {'step_length', 'step_height', 'phase'}
        """
        phase = step_params['phase']
        t = time % self.step_duration

        # Normalize time within step cycle
        t_norm = t / self.step_duration

        # Calculate foot position based on phase
        if phase < self.dsp_ratio:
            # Double support phase - feet stay in place
            x = 0
            y = 0 if support_leg == 'left' else -2 * self.hip_offset
            z = 0
        else:
            # Single support phase - swing foot moves
            swing_leg = 'right' if support_leg == 'left' else 'left'

            # Calculate swing trajectory
            phase_in_ssp = (t_norm - self.dsp_ratio) / (1 - self.dsp_ratio)

            # X trajectory (forward movement)
            x = step_params['step_length'] * phase_in_ssp

            # Z trajectory (foot lift)
            if phase_in_ssp < 0.5:
                z = step_params['step_height'] * sin(pi * phase_in_ssp)
            else:
                z = step_params['step_height'] * sin(pi * phase_in_ssp)

            # Y trajectory (lateral movement during step)
            y = 0 if support_leg == 'left' else -2 * self.hip_offset

        return np.array([x, y, z])

    def inverse_kinematics_leg(self, target_position, leg_side):
        """
        Calculate joint angles for leg to reach target position
        target_position: [x, y, z] in leg coordinate frame
        leg_side: 'left' or 'right'
        """
        x, y, z = target_position

        # Calculate hip angles
        hip_yaw = atan2(y, sqrt(x*x + z*z))
        hip_roll = atan2(-x, z)  # Simplified

        # Calculate knee angle using law of cosines
        foot_distance = sqrt(x*x + y*y + z*z)

        if foot_distance > 2 * self.leg_length:
            # Position unreachable, return joint limits
            return self.get_joint_limits()

        # Law of cosines for knee
        cos_knee = (self.leg_length**2 + self.leg_length**2 - foot_distance**2) / (2 * self.leg_length**2)
        cos_knee = np.clip(cos_knee, -1, 1)  # Ensure valid range
        knee_angle = acos(cos_knee)

        # Calculate ankle angles
        ankle_pitch = atan2(-x, z)
        ankle_roll = atan2(y, sqrt(x*x + z*z))

        # Adjust for robot-specific kinematics
        joint_angles = {
            'hip_yaw': hip_yaw,
            'hip_roll': hip_roll,
            'hip_pitch': 0,  # Simplified
            'knee': knee_angle,
            'ankle_pitch': ankle_pitch,
            'ankle_roll': ankle_roll
        }

        return joint_angles
```

## Humanoid-Specific ROS 2 Packages

### Navigation for Humanoid Robots

```python
# humanoid_nav2_config/params/humanoid_nav2_params.yaml
amcl:
  ros__parameters:
    use_sim_time: False
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_link"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.5
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.2
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

bt_navigator:
  ros__parameters:
    use_sim_time: False
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    # Humanoid-specific behavior tree
    default_nav_to_pose_bt_xml: "humanoid_navigate_to_pose_w_replanning_and_recovery.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_truncate_path_local_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node
    - nav2_controller_cancel_bt_node
    - nav2_path_longer_on_approach_bt_node
    - nav2_wait_cancel_bt_node
    - nav2_spin_cancel_bt_node
    - nav2_back_up_cancel_bt_node
    - nav2_assisted_teleop_cancel_bt_node
    - nav2_follow_path_cancel_bt_node
    - nav2_is_battery_charging_condition_bt_node
```

### Manipulation for Humanoid Robots

```python
# humanoid_manipulation/manipulation_controller.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
from scipy.spatial.transform import Rotation as R


class HumanoidManipulationController(Node):
    def __init__(self):
        super().__init__('humanoid_manipulation_controller')

        # Publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.left_arm_command_pub = self.create_publisher(
            JointTrajectory, '/left_arm_controller/joint_trajectory', 10)
        self.right_arm_command_pub = self.create_publisher(
            JointTrajectory, '/right_arm_controller/joint_trajectory', 10)

        # Manipulation command subscriber
        self.manip_command_sub = self.create_subscription(
            String, '/manipulation_command', self.manip_command_callback, 10)

        # Current joint positions
        self.current_joint_positions = {}
        self.left_arm_joints = ['left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
                               'left_elbow_pitch', 'left_wrist_yaw', 'left_wrist_pitch']
        self.right_arm_joints = ['right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
                                'right_elbow_pitch', 'right_wrist_yaw', 'right_wrist_pitch']

        # Forward and inverse kinematics solvers
        self.left_arm_ik_solver = HumanoidArmIK('left')
        self.right_arm_ik_solver = HumanoidArmIK('right')

        self.get_logger().info('Humanoid Manipulation Controller initialized')

    def joint_state_callback(self, msg: JointState):
        """Update current joint positions"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

    def manip_command_callback(self, msg: String):
        """Process manipulation commands"""
        command = msg.data.split()

        if command[0] == 'reach':
            # Reach to a specific pose
            if len(command) >= 5:  # reach x y z [left|right]
                target_x = float(command[1])
                target_y = float(command[2])
                target_z = float(command[3])
                arm_side = command[4] if len(command) > 4 else 'right'

                target_pose = Pose()
                target_pose.position = Point(x=target_x, y=target_y, z=target_z)
                target_pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)  # Default orientation

                self.reach_to_pose(target_pose, arm_side)

        elif command[0] == 'grasp':
            # Grasp an object
            self.execute_grasp(command[1] if len(command) > 1 else 'default')

        elif command[0] == 'wave':
            # Wave gesture
            self.execute_wave(command[1] if len(command) > 1 else 'right')

    def reach_to_pose(self, target_pose, arm_side):
        """Move arm to reach target pose"""
        if arm_side == 'left':
            ik_solution = self.left_arm_ik_solver.inverse_kinematics(target_pose)
            self.publish_joint_trajectory(self.left_arm_joints, ik_solution,
                                        self.left_arm_command_pub)
        else:  # right arm
            ik_solution = self.right_arm_ik_solver.inverse_kinematics(target_pose)
            self.publish_joint_trajectory(self.right_arm_joints, ik_solution,
                                        self.right_arm_command_pub)

    def execute_grasp(self, grasp_type):
        """Execute grasping motion"""
        # Simplified grasp execution
        # In practice, this would involve complex hand/robot hand coordination
        pass

    def execute_wave(self, arm_side):
        """Execute waving gesture"""
        if arm_side == 'left':
            # Define wave trajectory for left arm
            wave_trajectory = self.generate_wave_trajectory(self.left_arm_joints)
            self.publish_joint_trajectory(self.left_arm_joints, wave_trajectory,
                                        self.left_arm_command_pub)
        else:
            # Define wave trajectory for right arm
            wave_trajectory = self.generate_wave_trajectory(self.right_arm_joints)
            self.publish_joint_trajectory(self.right_arm_joints, wave_trajectory,
                                        self.right_arm_command_pub)

    def publish_joint_trajectory(self, joint_names, positions, publisher):
        """Publish joint trajectory command"""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=0, nanosec=100000000)  # 100ms

        traj_msg.points = [point]
        publisher.publish(traj_msg)


class HumanoidArmIK:
    """Inverse kinematics solver for humanoid arm"""

    def __init__(self, arm_side):
        self.arm_side = arm_side
        self.upper_arm_length = 0.3  # meters
        self.lower_arm_length = 0.25  # meters
        self.shoulder_offset = 0.15  # from torso center

    def inverse_kinematics(self, target_pose):
        """Solve inverse kinematics for target pose"""
        # Convert target pose to arm coordinate frame
        target_pos = np.array([
            target_pose.position.x,
            target_pose.position.y,
            target_pose.position.z
        ])

        # Simplified 3DOF arm IK (shoulder, elbow, wrist)
        # Calculate distance from shoulder to target
        dist_2d = np.sqrt((target_pos[0]**2) + (target_pos[1]**2))

        # Calculate elbow angle using law of cosines
        total_length = self.upper_arm_length + self.lower_arm_length
        if dist_2d > total_length:
            # Target out of reach, extend fully
            shoulder_angle = np.arctan2(target_pos[1], target_pos[0])
            elbow_angle = 0
        else:
            # Calculate arm configuration
            elbow_angle = self.calculate_elbow_angle(target_pos)
            shoulder_angle = self.calculate_shoulder_angle(target_pos, elbow_angle)

        # Return joint angles (simplified)
        return [shoulder_angle, 0, 0, elbow_angle, 0, 0]  # [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_yaw, wrist_pitch]

    def calculate_elbow_angle(self, target_pos):
        """Calculate elbow joint angle"""
        # Simplified calculation
        dist = np.linalg.norm(target_pos)
        if dist > self.upper_arm_length + self.lower_arm_length:
            return 0  # Fully extended
        else:
            # Law of cosines
            cos_angle = (self.upper_arm_length**2 + self.lower_arm_length**2 - dist**2) / (2 * self.upper_arm_length * self.lower_arm_length)
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.arccos(cos_angle)

    def calculate_shoulder_angle(self, target_pos, elbow_angle):
        """Calculate shoulder joint angles"""
        # Simplified calculation
        return np.arctan2(target_pos[1], target_pos[0])
```

## Safety and Emergency Systems

### Safety Monitor Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist
import numpy as np


class HumanoidSafetyMonitor(Node):
    def __init__(self):
        super().__init__('humanoid_safety_monitor')

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Publishers
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        self.safety_status_pub = self.create_publisher(String, '/safety_status', 10)

        # Safety parameters
        self.joint_limits = self.initialize_joint_limits()
        self.imu_thresholds = {
            'angular_velocity': 5.0,  # rad/s
            'linear_acceleration': 20.0  # m/s^2
        }
        self.safety_zones = self.define_safety_zones()

        # Current states
        self.current_joint_positions = {}
        self.current_joint_velocities = {}
        self.current_imu_data = None
        self.last_cmd_vel_time = self.get_clock().now()

        # Emergency stop status
        self.emergency_stop_active = False

        # Timer for safety checks
        self.safety_timer = self.create_timer(0.01, self.safety_check)  # 100 Hz

        self.get_logger().info('Humanoid Safety Monitor initialized')

    def initialize_joint_limits(self):
        """Initialize joint limits for safety checking"""
        return {
            'left_hip_pitch': (-1.57, 1.57),
            'left_hip_roll': (-0.5, 0.5),
            'left_hip_yaw': (-0.5, 0.5),
            'left_knee': (0, 2.5),
            'left_ankle_pitch': (-0.5, 0.5),
            'left_ankle_roll': (-0.5, 0.5),
            # Add other joints...
        }

    def joint_state_callback(self, msg: JointState):
        """Update current joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]

    def imu_callback(self, msg: Imu):
        """Update IMU data"""
        self.current_imu_data = msg

    def cmd_vel_callback(self, msg: Twist):
        """Update last command velocity time"""
        self.last_cmd_vel_time = self.get_clock().now()

    def safety_check(self):
        """Perform safety checks"""
        if self.emergency_stop_active:
            return

        # Check joint limits
        joint_violation = self.check_joint_limits()
        if joint_violation:
            self.trigger_emergency_stop(f"Joint limit violation: {joint_violation}")
            return

        # Check IMU data
        if self.current_imu_data:
            imu_violation = self.check_imu_data()
            if imu_violation:
                self.trigger_emergency_stop(f"IMU safety violation: {imu_violation}")
                return

        # Check for excessive command timeout
        time_since_cmd = (self.get_clock().now() - self.last_cmd_vel_time).nanoseconds / 1e9
        if time_since_cmd > 5.0:  # 5 seconds without command
            self.get_logger().warn("No command received for 5 seconds")

        # If all checks pass
        status_msg = String()
        status_msg.data = "SAFE"
        self.safety_status_pub.publish(status_msg)

    def check_joint_limits(self):
        """Check if any joints are outside safe limits"""
        for joint_name, position in self.current_joint_positions.items():
            if joint_name in self.joint_limits:
                lower_limit, upper_limit = self.joint_limits[joint_name]
                if position < lower_limit or position > upper_limit:
                    return f"{joint_name}={position:.3f}, limits=[{lower_limit:.3f}, {upper_limit:.3f}]"
        return None

    def check_imu_data(self):
        """Check IMU data for safety violations"""
        ang_vel = self.current_imu_data.angular_velocity
        lin_acc = self.current_imu_data.linear_acceleration

        ang_vel_mag = np.sqrt(ang_vel.x**2 + ang_vel.y**2 + ang_vel.z**2)
        lin_acc_mag = np.sqrt(lin_acc.x**2 + lin_acc.y**2 + lin_acc.z**2)

        if ang_vel_mag > self.imu_thresholds['angular_velocity']:
            return f"Excessive angular velocity: {ang_vel_mag:.3f} > {self.imu_thresholds['angular_velocity']}"

        if lin_acc_mag > self.imu_thresholds['linear_acceleration']:
            return f"Excessive linear acceleration: {lin_acc_mag:.3f} > {self.imu_thresholds['linear_acceleration']}"

        return None

    def trigger_emergency_stop(self, reason):
        """Trigger emergency stop"""
        self.emergency_stop_active = True
        self.get_logger().error(f"EMERGENCY STOP TRIGGERED: {reason}")

        # Publish emergency stop command
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)

        # Publish safety status
        status_msg = String()
        status_msg.data = f"EMERGENCY_STOP: {reason}"
        self.safety_status_pub.publish(status_msg)
```

## Performance Optimization

### Real-time Considerations

For humanoid robots, real-time performance is critical. Here are key considerations:

1. **Control Loop Timing**: Maintain consistent control frequencies (typically 100-1000 Hz)
2. **Communication Latency**: Minimize message passing delays between nodes
3. **Computation Efficiency**: Optimize algorithms for real-time execution
4. **Resource Management**: Properly configure CPU and memory allocation

### Quality of Service (QoS) Configuration

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# For safety-critical messages
safety_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST
)

# For sensor data (best effort for real-time)
sensor_qos = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST
)

# For control commands
control_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST
)
```

## Integration with Hardware

### ros2_control Configuration

```yaml
# config/humanoid_ros2_control.yaml
controller_manager:
  ros__parameters:
    update_rate: 500  # Hz

    # Joint trajectory controllers
    left_leg_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    right_leg_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    left_arm_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    right_arm_controller:
      type: joint_trajectory_controller/JointTrajectoryController

left_leg_controller:
  ros__parameters:
    joints:
      - left_hip_pitch
      - left_hip_roll
      - left_hip_yaw
      - left_knee
      - left_ankle_pitch
      - left_ankle_roll
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity

# Similar configuration for other controllers...
```

## Testing and Validation

### Unit Testing for Humanoid Nodes

```python
# test/test_balance_controller.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from humanoid_control.balance_controller import BalanceController
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
import time


class TestBalanceController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = BalanceController()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()

    def test_imu_subscription(self):
        """Test that IMU data is properly received"""
        # Publish test IMU message
        imu_pub = self.node.create_publisher(Imu, '/imu/data', 10)

        test_imu = Imu()
        test_imu.header.stamp = self.node.get_clock().now().to_msg()

        imu_pub.publish(test_imu)

        # Allow time for processing
        start_time = time.time()
        while time.time() - start_time < 1.0:
            self.executor.spin_once(timeout_sec=0.1)

        # Test would check internal state changes
        self.assertTrue(True)  # Placeholder for actual test

    def test_balance_control_output(self):
        """Test balance control output generation"""
        # This would test the balance control loop output
        self.assertTrue(True)  # Placeholder for actual test


if __name__ == '__main__':
    unittest.main()
```

## Summary

ROS 2 provides a comprehensive framework for humanoid robotics with:

- **Distributed Architecture**: Multiple nodes for perception, planning, and control
- **Real-time Capabilities**: Quality of Service settings for time-critical operations
- **Safety Systems**: Built-in safety monitoring and emergency stop capabilities
- **Standardized Interfaces**: Consistent APIs for hardware integration
- **Simulation Support**: Integration with Gazebo for testing and development
- **Community Support**: Extensive libraries and tools for humanoid development

The combination of ROS 2's robust middleware, real-time capabilities, and extensive tooling makes it the ideal platform for developing sophisticated humanoid robotic systems. Success in humanoid robotics with ROS 2 requires careful attention to real-time performance, safety systems, and proper system architecture.