---
sidebar_position: 8
title: "Nav2: Path Planning for Humanoid Movement"
---

# Nav2: Path Planning for Humanoid Movement

## Introduction to Navigation 2 (Nav2)

Navigation 2 (Nav2) is the next-generation navigation framework for ROS 2, designed to provide advanced path planning, navigation, and obstacle avoidance capabilities. For humanoid robotics, Nav2 offers specialized features that address the unique challenges of bipedal locomotion, including complex kinematics, balance requirements, and anthropomorphic motion constraints.

Nav2 provides a complete navigation stack including:
- **Global Path Planning**: Long-term route planning from start to goal
- **Local Path Planning**: Short-term obstacle avoidance and trajectory generation
- **Controller Integration**: Integration with robot controllers for execution
- **Recovery Behaviors**: Strategies for handling navigation failures
- **Behavior Trees**: Flexible navigation behavior orchestration

## Architecture of Nav2

### Nav2 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Nav2 Architecture                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │   Global        │  │   Local         │  │   Controller    │            │
│  │   Planner       │  │   Planner       │  │   Server        │            │
│  │   - A*          │  │   - DWA         │  │   - FollowPath  │            │
│  │   - NavFn       │  │   - MPC         │  │   - Rotate      │            │
│  │   - SmacPlanner │  │   - RPP         │  │   - Wait        │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                              │              │              │               │
│                              ▼              ▼              ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Behavior Tree Executor                          │   │
│  │  - NavigateToPose                                                │   │
│  │  - NavigateThroughPoses                                          │   │
│  │  - ComputePathToPose                                             │   │
│  │  - FollowPath                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Recovery System                                  │   │
│  │  - BackUp                                                        │   │
│  │  - Spin                                                          │   │
│  │  - Wait                                                          │   │
│  │  - ClearCostmap                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │   Costmap       │  │   Sensors       │  │   TF2           │            │
│  │   Server        │  │   Server        │  │   Server        │            │
│  │   - Global      │  │   - LaserScan   │  │   - Transform   │            │
│  │   - Local       │  │   - PointCloud  │  │   Management    │            │
│  │   - Updates     │  │   - Image       │  │                 │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Navigation Server
- Coordinates navigation execution
- Manages navigation lifecycle
- Handles action requests

#### 2. Global Planner
- Generates long-term path
- Considers global costmap
- Produces optimal route

#### 3. Local Planner
- Performs short-term planning
- Handles obstacle avoidance
- Generates executable trajectories

#### 4. Controller Server
- Executes navigation commands
- Tracks planned paths
- Interfaces with robot drivers

## Nav2 for Humanoid Robotics

### Unique Challenges for Humanoid Navigation

Humanoid robots present specific navigation challenges that require specialized approaches:

#### 1. Bipedal Kinematics
- Complex inverse kinematics for walking
- Balance constraints during motion
- Step planning for stable locomotion

#### 2. Anthropomorphic Motion
- Human-like movement patterns
- Social navigation considerations
- Interaction with human environments

#### 3. Stability Requirements
- Maintaining center of mass
- Avoiding falls during navigation
- Dynamic balance during motion

### Humanoid-Specific Nav2 Configuration

```yaml
# humanoid_nav2_params.yaml
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

controller_server:
  ros__parameters:
    use_sim_time: False
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific controller
    FollowPath:
      plugin: "nav2_mppi_controller::MppiController"
      time_steps: 20
      control_horizon: 5
      trajectory_dt: 0.2
      discretization: 0.5
      penalty_scaling: 1.0
      obstacle_weight: 1.0
      goal_weight: 1.0
      reference_weight: 1.0
      curvature_weight: 0.1
      rate_limits_enabled: true
      vx_max: 0.5
      vx_min: -0.2
      vy_max: 0.3
      vy_min: -0.3
      wz_max: 0.6
      wz_min: -0.6
      ax_max: 2.5
      ay_max: 2.5
      wz_max_acc: 3.2
      progress_checker:
        plugin: "nav2_controller::SimpleProgressChecker"
        required_movement_radius: 0.5
        movement_time_allowance: 10.0
      goal_checker:
        plugin: "nav2_controller::SimpleGoalChecker"
        xy_goal_tolerance: 0.25
        yaw_goal_tolerance: 0.25
        stateful: True

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: False
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      robot_radius: 0.35  # Humanoid-specific radius
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: False
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: False
      robot_radius: 0.35  # Humanoid-specific radius
      resolution: 0.05
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: False
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
      visualize_potential: false

smoother_server:
  ros__parameters:
    use_sim_time: False
    smoother_plugins: ["simple_smoother"]
    simple_smoother:
      plugin: "nav2_smoother::SimpleSmoother"
      tolerance: 1.0e-10
      max_its: 1000
      do_refinement: True

behavior_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    behavior_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_behaviors::Spin"
      spin_dist: 1.57
    backup:
      plugin: "nav2_behaviors::BackUp"
      backup_dist: 0.15
      backup_speed: 0.025
    wait:
      plugin: "nav2_behaviors::Wait"
      wait_duration: 1.0

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: true
      waypoint_pause_duration: 200
```

## Global Path Planning for Humanoid Robots

### Global Planners

#### Navfn Planner
The Navfn planner uses the Dijkstra algorithm to find optimal paths:

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.srv import ComputePathToPose
from sensor_msgs.msg import LaserScan
import numpy as np

class HumanoidGlobalPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_global_planner')

        # Service client for path computation
        self.path_client = self.create_client(
            ComputePathToPose, 'compute_path_to_pose')

        # Publishers
        self.global_plan_pub = self.create_publisher(Path, '/humanoid/global_plan', 10)

        # Parameters
        self.declare_parameter('robot_radius', 0.35)
        self.declare_parameter('step_height', 0.15)  # Maximum step height for humanoid
        self.declare_parameter('max_slope', 0.3)     # Maximum slope angle

        self.robot_radius = self.get_parameter('robot_radius').value
        self.max_step_height = self.get_parameter('step_height').value
        self.max_slope = self.get_parameter('max_slope').value

        self.get_logger().info('Humanoid Global Planner initialized')

    def compute_path_to_pose(self, start_pose, goal_pose):
        """Compute global path considering humanoid constraints"""
        # Wait for service
        while not self.path_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for compute_path_to_pose service...')

        # Create request
        request = ComputePathToPose.Request()
        request.start = start_pose
        request.goal = goal_pose

        # Call service
        future = self.path_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            path = future.result().path

            # Apply humanoid-specific constraints
            constrained_path = self.apply_humanoid_constraints(path)

            # Publish global plan
            self.global_plan_pub.publish(constrained_path)

            return constrained_path
        else:
            self.get_logger().error('Failed to compute path')
            return None

    def apply_humanoid_constraints(self, path):
        """Apply humanoid-specific constraints to path"""
        # Check for step height constraints
        constrained_poses = []
        for i, pose in enumerate(path.poses):
            if i == 0:
                constrained_poses.append(pose)
                continue

            # Calculate height difference between consecutive poses
            prev_pose = path.poses[i-1]
            height_diff = abs(pose.pose.position.z - prev_pose.pose.position.z)

            # Check if step height is acceptable
            if height_diff <= self.max_step_height:
                constrained_poses.append(pose)
            else:
                # Need to find alternative path around obstacle
                self.get_logger().warn(
                    f'Step height constraint violated at pose {i}, height diff: {height_diff}'
                )
                # In practice, this would trigger replanning or alternative pathfinding

        # Create new path with constrained poses
        constrained_path = Path()
        constrained_path.header = path.header
        constrained_path.poses = constrained_poses

        return constrained_path

    def validate_path_feasibility(self, path):
        """Validate path feasibility for humanoid locomotion"""
        for i in range(len(path.poses) - 1):
            current_pose = path.poses[i]
            next_pose = path.poses[i + 1]

            # Calculate distance between poses
            dx = next_pose.pose.position.x - current_pose.pose.position.x
            dy = next_pose.pose.position.y - current_pose.pose.position.y
            dz = next_pose.pose.position.z - current_pose.pose.position.z
            distance = np.sqrt(dx*dx + dy*dy)

            # Check slope constraints
            if distance > 0:
                slope = abs(dz) / distance
                if slope > self.max_slope:
                    self.get_logger().warn(
                        f'Slope constraint violated between poses {i} and {i+1}, slope: {slope}'
                    )
                    return False

            # Check for obstacles in path
            if self.check_path_for_obstacles(current_pose, next_pose):
                self.get_logger().warn(f'Obstacle detected in path between poses {i} and {i+1}')
                return False

        return True

    def check_path_for_obstacles(self, start_pose, end_pose):
        """Check if path segment has obstacles"""
        # This would check local costmap for obstacles along the path
        # Implementation would depend on costmap access
        return False  # Simplified for example
```

#### Smac Planner
For more complex path planning, the SMAC (Search with Motion Primitives) planner can be used:

```yaml
# smac_planner_config.yaml
planner_server:
  ros__parameters:
    use_sim_time: False
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_smac_planner::SmacPlanner"
      tolerance: 0.5
      downsample_costmap: false
      downsampling_factor: 1
      allow_unknown: true
      max_iterations: 1000000
      motion_model_for_search: "DUBIN"
      cost_penalty: 1.5
      reverse_penalty: 2.0
      change_penalty: 0.5
      non_straight_penalty: 1.2
      heuristic_scale: 1.0
      cache_obstacle_heuristic: true
```

## Local Path Planning and Control

### Local Planner Configuration

```yaml
# local_planner_config.yaml
controller_server:
  ros__parameters:
    use_sim_time: False
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # MPPI Controller for humanoid robots
    FollowPath:
      plugin: "nav2_mppi_controller::MppiController"
      time_steps: 15
      control_horizon: 5
      trajectory_dt: 0.1
      discretization: 0.3
      penalty_scaling: 1.0
      obstacle_weight: 1.5
      goal_weight: 1.0
      reference_weight: 0.5
      curvature_weight: 0.2
      rate_limits_enabled: true
      vx_max: 0.3    # Slower for humanoid stability
      vx_min: -0.15
      vy_max: 0.2
      vy_min: -0.2
      wz_max: 0.4    # Slower turning for balance
      wz_min: -0.4
      ax_max: 1.0    # Softer acceleration for balance
      ay_max: 1.0
      wz_max_acc: 1.5
      progress_checker:
        plugin: "nav2_controller::SimpleProgressChecker"
        required_movement_radius: 0.3
        movement_time_allowance: 15.0
      goal_checker:
        plugin: "nav2_controller::SimpleGoalChecker"
        xy_goal_tolerance: 0.3  # Larger for humanoid
        yaw_goal_tolerance: 0.3
        stateful: True
```

### Humanoid-Specific Local Planner

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np

class HumanoidLocalPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_local_planner')

        # Subscribers
        self.global_plan_sub = self.create_subscription(
            Path, '/humanoid/global_plan', self.global_plan_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.local_plan_pub = self.create_publisher(Path, '/humanoid/local_plan', 10)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Parameters
        self.declare_parameter('local_plan_lookahead', 1.0)
        self.declare_parameter('obstacle_threshold', 0.5)
        self.declare_parameter('humanoid_width', 0.35)
        self.declare_parameter('balance_margin', 0.1)

        self.local_plan_lookahead = self.get_parameter('local_plan_lookahead').value
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value
        self.humanoid_width = self.get_parameter('humanoid_width').value
        self.balance_margin = self.get_parameter('balance_margin').value

        # State variables
        self.global_plan = None
        self.latest_scan = None
        self.current_pose = None

        # Timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        self.get_logger().info('Humanoid Local Planner initialized')

    def global_plan_callback(self, msg):
        """Receive global plan"""
        self.global_plan = msg

    def laser_callback(self, msg):
        """Receive laser scan data"""
        self.latest_scan = msg

    def control_loop(self):
        """Main control loop for humanoid navigation"""
        if self.global_plan is None or self.latest_scan is None:
            return

        # Get current robot pose
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            self.current_pose = transform
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform: {ex}')
            return

        # Generate local plan considering obstacles
        local_plan = self.generate_local_plan(self.global_plan)

        # Check for obstacles in local plan
        if self.has_obstacles_in_local_plan(local_plan):
            # Trigger obstacle avoidance
            cmd_vel = self.avoid_obstacles()
        else:
            # Follow local plan
            cmd_vel = self.follow_local_plan(local_plan)

        # Publish velocity command
        self.cmd_vel_pub.publish(cmd_vel)

        # Publish local plan for visualization
        self.local_plan_pub.publish(local_plan)

    def generate_local_plan(self, global_plan):
        """Generate local plan from global plan"""
        if not global_plan.poses:
            return Path()

        # Find closest point on global plan to current pose
        closest_idx = self.find_closest_pose(global_plan)
        if closest_idx is None:
            return Path()

        # Extract local segment of plan
        local_plan = Path()
        local_plan.header = global_plan.header

        start_idx = closest_idx
        end_idx = min(
            start_idx + int(self.local_plan_lookahead / 0.1),  # Assume 0.1m resolution
            len(global_plan.poses)
        )

        local_plan.poses = global_plan.poses[start_idx:end_idx]

        return local_plan

    def find_closest_pose(self, plan):
        """Find closest pose on plan to current robot position"""
        if not plan.poses or self.current_pose is None:
            return None

        current_pos = self.current_pose.transform.translation
        min_dist = float('inf')
        closest_idx = 0

        for i, pose in enumerate(plan.poses):
            dist = np.sqrt(
                (pose.pose.position.x - current_pos.x) ** 2 +
                (pose.pose.position.y - current_pos.y) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx

    def has_obstacles_in_local_plan(self, local_plan):
        """Check if there are obstacles in the local plan path"""
        if not local_plan.poses or self.latest_scan is None:
            return False

        # Convert laser scan to points in robot frame
        laser_points = self.laser_scan_to_points(self.latest_scan)

        # Check each segment of local plan
        for i in range(len(local_plan.poses) - 1):
            start = local_plan.poses[i].pose.position
            end = local_plan.poses[i + 1].pose.position

            # Check for obstacles along this segment
            for point in laser_points:
                if self.is_point_in_path_segment(point, start, end, self.humanoid_width):
                    if np.sqrt(point.x**2 + point.y**2) < self.obstacle_threshold:
                        return True

        return False

    def laser_scan_to_points(self, scan):
        """Convert laser scan to points in robot frame"""
        points = []
        angle = scan.angle_min

        for range_val in scan.ranges:
            if scan.range_min <= range_val <= scan.range_max:
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                points.append(Point(x=x, y=y, z=0.0))
            angle += scan.angle_increment

        return points

    def is_point_in_path_segment(self, point, start, end, width):
        """Check if a point is near a path segment within a certain width"""
        # Calculate distance from point to line segment
        # This is a simplified version - in practice, you'd use more sophisticated geometry
        line_vec = np.array([end.x - start.x, end.y - start.y])
        point_vec = np.array([point.x - start.x, point.y - start.y])

        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq == 0:
            # Line segment is actually a point
            dist_sq = np.dot(point_vec, point_vec)
        else:
            # Calculate projection of point onto line
            t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
            projection = start + t * line_vec
            dist_sq = np.sum((np.array([point.x, point.y]) - np.array([projection.x, projection.y])) ** 2)

        return np.sqrt(dist_sq) <= width

    def avoid_obstacles(self):
        """Generate velocity commands to avoid obstacles"""
        cmd_vel = Twist()

        # Simple obstacle avoidance strategy
        # In practice, this would use more sophisticated local planning
        if self.latest_scan is not None:
            # Find the safest direction based on scan data
            left_clear = self.check_direction_clear(self.latest_scan, -1.0, 0.5)  # Left
            right_clear = self.check_direction_clear(self.latest_scan, 0.5, 1.0)  # Right

            if left_clear and not right_clear:
                cmd_vel.angular.z = 0.3  # Turn left
                cmd_vel.linear.x = 0.1   # Move forward slowly
            elif right_clear and not left_clear:
                cmd_vel.angular.z = -0.3  # Turn right
                cmd_vel.linear.x = 0.1    # Move forward slowly
            else:
                # Both sides clear, turn towards the wider opening
                left_avg = self.average_range_in_sector(self.latest_scan, -1.0, 0.0)
                right_avg = self.average_range_in_sector(self.latest_scan, 0.0, 1.0)

                if left_avg > right_avg:
                    cmd_vel.angular.z = 0.2
                else:
                    cmd_vel.angular.z = -0.2

                cmd_vel.linear.x = 0.1

        return cmd_vel

    def check_direction_clear(self, scan, min_angle, max_angle):
        """Check if a direction is clear of obstacles"""
        min_idx = int((min_angle - scan.angle_min) / scan.angle_increment)
        max_idx = int((max_angle - scan.angle_min) / scan.angle_increment)

        min_idx = max(0, min_idx)
        max_idx = min(len(scan.ranges) - 1, max_idx)

        for i in range(min_idx, max_idx + 1):
            if scan.ranges[i] < self.obstacle_threshold:
                return False

        return True

    def average_range_in_sector(self, scan, min_angle, max_angle):
        """Calculate average range in a sector"""
        min_idx = int((min_angle - scan.angle_min) / scan.angle_increment)
        max_idx = int((max_angle - scan.angle_min) / scan.angle_increment)

        min_idx = max(0, min_idx)
        max_idx = min(len(scan.ranges) - 1, max_idx)

        valid_ranges = [r for r in scan.ranges[min_idx:max_idx+1]
                       if scan.range_min <= r <= scan.range_max]

        return sum(valid_ranges) / len(valid_ranges) if valid_ranges else 0.0

    def follow_local_plan(self, local_plan):
        """Generate velocity commands to follow local plan"""
        cmd_vel = Twist()

        if not local_plan.poses:
            return cmd_vel

        # Calculate desired direction to next waypoint
        target = local_plan.poses[0].pose.position
        target_angle = np.arctan2(target.y, target.x)  # In robot frame

        # PID-like control for direction
        cmd_vel.angular.z = 2.0 * target_angle  # Proportional control

        # Move forward if facing approximately the right direction
        if abs(target_angle) < 0.5:  # Within 28 degrees
            cmd_vel.linear.x = 0.2  # Move forward
        else:
            cmd_vel.linear.x = 0.05  # Move slowly when turning

        # Apply humanoid-specific constraints
        cmd_vel.linear.x = min(cmd_vel.linear.x, 0.3)  # Max speed for humanoid stability
        cmd_vel.angular.z = max(min(cmd_vel.angular.z, 0.4), -0.4)  # Max turning rate

        return cmd_vel
```

## Behavior Trees for Humanoid Navigation

### Custom Behavior Tree Nodes

```xml
<!-- humanoid_navigate_to_pose_w_replanning_and_recovery.xml -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <SequenceStar name="NavigateToPose">
            <RecoveryNode number_of_retries="6" name="NavigateRecovery">
                <PipelineSequence name="NavigateWithReplanning">
                    <RateController hz="1.0" name="NavigationRate">
                        <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
                    </RateController>
                    <FollowPath path="{path}" controller_id="FollowPath"/>
                </PipelineSequence>
                <ReactiveFallback name="RecoveryFallback">
                    <GoalUpdated/>
                    <ClearEntireCostmap name="ClearLocalCostmap" service_name="local_costmap/clear_entirely_local_costmap"/>
                    <ClearEntireCostmap name="ClearGlobalCostmap" service_name="global_costmap/clear_entirely_global_costmap"/>
                    <RecoveryNode number_of_retries="2" name="LocalRecovery">
                        <ReactiveFallback name="LocalFallback">
                            <Spin recovery_behavior_enabled="{local_spin_enabled}"/>
                            <Backup recovery_behavior_enabled="{local_backup_enabled}"/>
                        </ReactiveFallback>
                    </RecoveryNode>
                </ReactiveFallback>
            </RecoveryNode>
        </SequenceStar>
    </BehaviorTree>
</root>
```

### Custom Humanoid Behavior Tree Nodes

```python
import rclpy
from rclpy.node import Node
from py_trees_ros.trees import BehaviourTree
from py_trees_ros.interfaces import BlackBox
from py_trees.common import Status
import py_trees

class HumanoidStepPlanningNode(py_trees.behaviour.Behaviour):
    """
    Custom behavior tree node for humanoid step planning
    """
    def __init__(self, name="HumanoidStepPlanning"):
        super(HumanoidStepPlanningNode, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name=name)
        self.blackboard.register_key("step_plan", access=py_trees.common.Access.WRITE)

    def setup(self, **kwargs):
        """Setup the behavior node"""
        try:
            self.node = kwargs['node']
        except KeyError:
            raise KeyError('A ROS node wasn\'t passed into the setup function')
        self.logger.debug(f"Created node: {self.name}")

    def update(self):
        """Update the behavior node"""
        # Generate step plan for humanoid locomotion
        step_plan = self.generate_step_plan()
        self.blackboard.set("step_plan", step_plan)

        if step_plan is not None:
            return Status.SUCCESS
        else:
            return Status.FAILURE

    def generate_step_plan(self):
        """Generate a plan for humanoid steps"""
        # This would implement humanoid-specific step planning
        # considering balance, step constraints, and terrain
        return {
            'left_foot_steps': [],
            'right_foot_steps': [],
            'balance_checks': [],
            'timing': []
        }

class HumanoidBalanceCheckNode(py_trees.behaviour.Behaviour):
    """
    Custom behavior tree node for humanoid balance checking
    """
    def __init__(self, name="HumanoidBalanceCheck"):
        super(HumanoidBalanceCheckNode, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name=name)
        self.blackboard.register_key("balance_status", access=py_trees.common.Access.READ)

    def setup(self, **kwargs):
        """Setup the behavior node"""
        try:
            self.node = kwargs['node']
        except KeyError:
            raise KeyError('A ROS node wasn\'t passed into the setup function')

    def update(self):
        """Update the behavior node"""
        balance_ok = self.check_balance()

        if balance_ok:
            return Status.SUCCESS
        else:
            return Status.FAILURE

    def check_balance(self):
        """Check if humanoid is balanced"""
        # This would check balance based on IMU, ZMP, or other sensors
        # For now, return True as a placeholder
        return True
```

## Humanoid-Specific Navigation Behaviors

### Social Navigation

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from people_msgs.msg import People
from visualization_msgs.msg import MarkerArray
import numpy as np

class HumanoidSocialNavigation(Node):
    def __init__(self):
        super().__init__('humanoid_social_navigation')

        # Subscribers
        self.people_sub = self.create_subscription(
            People, '/people', self.people_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)

        # Publishers
        self.social_markers_pub = self.create_publisher(
            MarkerArray, '/social_navigation_markers', 10)

        # Parameters
        self.declare_parameter('personal_space_radius', 1.0)
        self.declare_parameter('social_zone_radius', 2.0)
        self.declare_parameter('comfortable_distance', 1.5)

        self.personal_space_radius = self.get_parameter('personal_space_radius').value
        self.social_zone_radius = self.get_parameter('social_zone_radius').value
        self.comfortable_distance = self.get_parameter('comfortable_distance').value

        # State
        self.people_positions = []
        self.current_goal = None

        self.get_logger().info('Humanoid Social Navigation initialized')

    def people_callback(self, msg):
        """Process people detection messages"""
        self.people_positions = [
            person.position for person in msg.people
        ]

    def goal_callback(self, msg):
        """Process goal messages"""
        self.current_goal = msg.pose

    def adjust_path_for_social_navigation(self, original_path):
        """Adjust navigation path considering social zones"""
        if not self.people_positions:
            return original_path

        adjusted_path = []
        for i, pose in enumerate(original_path.poses):
            # Calculate distances to all people
            min_distance = float('inf')
            closest_person = None

            for person_pos in self.people_positions:
                dist = np.sqrt(
                    (pose.pose.position.x - person_pos.x) ** 2 +
                    (pose.pose.position.y - person_pos.y) ** 2
                )
                if dist < min_distance:
                    min_distance = dist
                    closest_person = person_pos

            # If too close to a person, adjust the pose
            if min_distance < self.comfortable_distance:
                # Calculate vector away from person
                direction_vector = np.array([
                    pose.pose.position.x - closest_person.x,
                    pose.pose.position.y - closest_person.y
                ])
                direction_unit = direction_vector / np.linalg.norm(direction_vector)

                # Move the pose away from the person
                adjustment_distance = self.comfortable_distance - min_distance + 0.1
                new_x = pose.pose.position.x + direction_unit[0] * adjustment_distance
                new_y = pose.pose.position.y + direction_unit[1] * adjustment_distance

                # Create new pose with adjusted position
                adjusted_pose = PoseStamped()
                adjusted_pose.header = pose.header
                adjusted_pose.pose.position.x = new_x
                adjusted_pose.pose.position.y = new_y
                adjusted_pose.pose.position.z = pose.pose.position.z

                # Maintain original orientation
                adjusted_pose.pose.orientation = pose.pose.orientation

                adjusted_path.append(adjusted_pose)
            else:
                adjusted_path.append(pose)

        # Create new path with adjusted poses
        social_path = original_path
        social_path.poses = adjusted_path

        return social_path

    def create_social_markers(self):
        """Create visualization markers for social navigation"""
        markers = MarkerArray()
        marker_id = 0

        # Personal space circles around people
        for i, person_pos in enumerate(self.people_positions):
            marker = self.create_circle_marker(
                person_pos, self.personal_space_radius,
                [1.0, 0.0, 0.0, 0.3], marker_id
            )
            marker_id += 1
            markers.markers.append(marker)

        # Social zone circles around people
        for i, person_pos in enumerate(self.people_positions):
            marker = self.create_circle_marker(
                person_pos, self.social_zone_radius,
                [1.0, 1.0, 0.0, 0.2], marker_id
            )
            marker_id += 1
            markers.markers.append(marker)

        # Comfortable distance circles around people
        for i, person_pos in enumerate(self.people_positions):
            marker = self.create_circle_marker(
                person_pos, self.comfortable_distance,
                [0.0, 1.0, 0.0, 0.3], marker_id
            )
            marker_id += 1
            markers.markers.append(marker)

        return markers

    def create_circle_marker(self, center, radius, color, marker_id):
        """Create a circle marker for visualization"""
        from visualization_msgs.msg import Marker

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "social_navigation"
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = center.x
        marker.pose.position.y = center.y
        marker.pose.position.z = 0.5  # Height
        marker.pose.orientation.w = 1.0

        marker.scale.x = 2 * radius
        marker.scale.y = 2 * radius
        marker.scale.z = 0.1  # Thickness

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        return marker
```

### Stair and Step Navigation

```python
class HumanoidStairNavigation(Node):
    def __init__(self):
        super().__init__('humanoid_stair_navigation')

        # Subscribers
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/depth/camera_info', self.camera_info_callback, 10)

        # Publishers
        self.stair_plan_pub = self.create_publisher(Path, '/stair_navigation_plan', 10)

        # Parameters
        self.declare_parameter('step_height_threshold', 0.1)
        self.declare_parameter('step_depth_threshold', 0.2)
        self.declare_parameter('max_climbable_height', 0.15)

        self.step_height_threshold = self.get_parameter('step_height_threshold').value
        self.step_depth_threshold = self.get_parameter('step_depth_threshold').value
        self.max_climbable_height = self.get_parameter('max_climbable_height').value

        # Camera parameters
        self.camera_matrix = None
        self.latest_depth = None

    def depth_callback(self, msg):
        """Process depth image for stair detection"""
        # Convert depth image to numpy array
        if msg.encoding == '32FC1':
            depth_array = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        elif msg.encoding == '16UC1':
            depth_array = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
        else:
            self.get_logger().error(f'Unsupported depth encoding: {msg.encoding}')
            return

        self.latest_depth = depth_array

        # Detect stairs/steps
        stair_regions = self.detect_stairs(depth_array)

        if stair_regions:
            # Generate navigation plan for stairs
            stair_plan = self.generate_stair_navigation_plan(stair_regions)
            self.stair_plan_pub.publish(stair_plan)

    def detect_stairs(self, depth_array):
        """Detect stairs and steps in depth image"""
        # Calculate depth gradients to find step edges
        grad_x = np.gradient(depth_array, axis=1)
        grad_y = np.gradient(depth_array, axis=0)

        # Find regions with significant depth changes
        depth_change = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold to find potential step locations
        step_mask = depth_change > self.step_height_threshold

        # Find connected components (potential step regions)
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(step_mask)

        stair_regions = []
        for i in range(1, num_features + 1):
            region_mask = labeled_array == i
            region_coords = np.where(region_mask)

            if len(region_coords[0]) > 10:  # Minimum region size
                # Calculate average depth change in region
                avg_change = np.mean(depth_change[region_mask])

                if avg_change > self.step_height_threshold:
                    # Calculate region properties
                    y_center = int(np.mean(region_coords[0]))
                    x_center = int(np.mean(region_coords[1]))

                    # Project to 3D world coordinates
                    if self.camera_matrix is not None:
                        world_x = (x_center - self.camera_matrix[0, 2]) * depth_array[y_center, x_center] / self.camera_matrix[0, 0]
                        world_y = (y_center - self.camera_matrix[1, 2]) * depth_array[y_center, x_center] / self.camera_matrix[1, 1]
                        world_z = depth_array[y_center, x_center]

                        stair_regions.append({
                            'center': (world_x, world_y, world_z),
                            'depth_change': avg_change,
                            'pixel_region': region_coords
                        })

        return stair_regions

    def generate_stair_navigation_plan(self, stair_regions):
        """Generate navigation plan for traversing stairs"""
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        # This would generate a plan that accounts for step climbing
        # with proper foot placement and balance considerations
        for region in stair_regions:
            if region['depth_change'] <= self.max_climbable_height:
                # Add waypoints for safe navigation around step
                waypoint = PoseStamped()
                # Calculate safe position in front of step
                waypoint.pose.position.x = region['center'][0] - 0.3  # 30cm before step
                waypoint.pose.position.y = region['center'][1]
                waypoint.pose.position.z = region['center'][2]

                path.poses.append(waypoint)

        return path

    def camera_info_callback(self, msg):
        """Store camera intrinsic parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
```

## Integration with Isaac ROS

### Isaac ROS Perception Integration

```python
class IsaacROSNavIntegration(Node):
    def __init__(self):
        super().__init__('isaac_ros_nav_integration')

        # Isaac ROS perception subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/detectnet/detections', self.detection_callback, 10)
        self.segmentation_sub = self.create_subscription(
            Image, '/segmentation/image', self.segmentation_callback, 10)

        # Navigation publishers
        self.nav_goal_pub = self.create_publisher(PoseStamped, '/goal', 10)

        # Parameters
        self.declare_parameter('person_detection_class', 1)  # Person class ID
        self.declare_parameter('navigation_timeout', 30.0)

        self.person_class_id = self.get_parameter('person_detection_class').value
        self.navigation_timeout = self.get_parameter('navigation_timeout').value

        # State
        self.detected_objects = []
        self.navigation_start_time = None

        self.get_logger().info('Isaac ROS Navigation Integration initialized')

    def detection_callback(self, msg):
        """Process object detections from Isaac ROS"""
        self.detected_objects = []

        for detection in msg.detections:
            # Check if this is a person (or other relevant object)
            for result in detection.results:
                if result.hypothesis.class_id == self.person_class_id and result.hypothesis.score > 0.7:
                    # Store person detection with confidence
                    person_info = {
                        'bbox': detection.bbox,
                        'confidence': result.hypothesis.score,
                        'position': self.bbox_to_world_coordinates(detection.bbox)
                    }
                    self.detected_objects.append(person_info)

    def segmentation_callback(self, msg):
        """Process segmentation results"""
        # This could be used for more detailed scene understanding
        # For example, identifying walkable surfaces, obstacles, etc.
        pass

    def bbox_to_world_coordinates(self, bbox):
        """Convert bounding box to world coordinates"""
        # This would require camera calibration and depth information
        # to convert 2D image coordinates to 3D world coordinates
        # Simplified for example:
        return (bbox.center.x, bbox.center.y, 0.0)  # Placeholder

    def generate_navigation_goals_from_detections(self):
        """Generate navigation goals based on detections"""
        goals = []

        for obj in self.detected_objects:
            if obj['confidence'] > 0.8:  # High confidence detection
                # Create goal that navigates toward the detected object
                # but maintains safe distance
                goal = PoseStamped()
                goal.header.frame_id = "map"
                goal.header.stamp = self.get_clock().now().to_msg()

                # Calculate goal position (slightly offset from object)
                goal.pose.position.x = obj['position'][0] - 1.0  # 1m in front
                goal.pose.position.y = obj['position'][1]
                goal.pose.position.z = 0.0

                # Look toward the object
                direction = np.arctan2(
                    obj['position'][1] - goal.pose.position.y,
                    obj['position'][0] - goal.pose.position.x
                )
                goal.pose.orientation.z = np.sin(direction / 2)
                goal.pose.orientation.w = np.cos(direction / 2)

                goals.append(goal)

        return goals
```

## Performance Optimization and Tuning

### Navigation Parameter Tuning

```python
class Nav2ParameterTuner(Node):
    def __init__(self):
        super().__init__('nav2_parameter_tuner')

        # Service clients for dynamic parameter updates
        self.param_client = self.create_client(SetParameters, '/navigation_server/set_parameters')
        self.planner_client = self.create_client(SetParameters, '/planner_server/set_parameters')
        self.controller_client = self.create_client(SetParameters, '/controller_server/set_parameters')

        # Parameters to tune
        self.tunable_params = {
            'planner_frequency': [0.5, 10.0, 2.0],
            'controller_frequency': [10.0, 50.0, 20.0],
            'min_x_velocity_threshold': [0.001, 0.1, 0.01],
            'min_y_velocity_threshold': [0.1, 1.0, 0.5],
            'min_theta_velocity_threshold': [0.001, 0.1, 0.01],
            'robot_radius': [0.2, 0.5, 0.35],
            'inflation_radius': [0.3, 1.0, 0.55],
            'tolerance': [0.1, 1.0, 0.5],
            'xy_goal_tolerance': [0.1, 0.5, 0.25],
            'yaw_goal_tolerance': [0.1, 0.5, 0.25]
        }

        # Performance metrics
        self.performance_metrics = {
            'navigation_success_rate': 0.0,
            'average_time_to_goal': 0.0,
            'path_efficiency': 0.0,
            'collision_rate': 0.0
        }

        self.get_logger().info('Nav2 Parameter Tuner initialized')

    def tune_parameters(self, tuning_method='bayesian'):
        """Tune navigation parameters using specified method"""
        if tuning_method == 'bayesian':
            self.bayesian_optimization_tuning()
        elif tuning_method == 'grid_search':
            self.grid_search_tuning()
        elif tuning_method == 'genetic':
            self.genetic_algorithm_tuning()
        else:
            self.get_logger().error(f'Unknown tuning method: {tuning_method}')

    def bayesian_optimization_tuning(self):
        """Use Bayesian optimization for parameter tuning"""
        from skopt import gp_minimize
        from skopt.space import Real

        # Define search space
        dimensions = []
        param_names = []

        for param_name, (low, high, default) in self.tunable_params.items():
            dimensions.append(Real(low, high, name=param_name))
            param_names.append(param_name)

        # Objective function
        def objective(params):
            # Set parameters
            param_dict = dict(zip(param_names, params))
            self.set_navigation_parameters(param_dict)

            # Run navigation trials
            metrics = self.evaluate_navigation_performance()

            # Return negative success rate (to minimize)
            return -metrics['navigation_success_rate']

        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=50,
            random_state=42
        )

        # Apply best parameters
        best_params = dict(zip(param_names, result.x))
        self.set_navigation_parameters(best_params)

        self.get_logger().info(f'Best parameters found: {best_params}')

    def set_navigation_parameters(self, param_dict):
        """Set navigation parameters"""
        # Convert to ROS parameters and send to servers
        for param_name, param_value in param_dict.items():
            param = Parameter()
            param.name = param_name
            param.value = ParameterValue()
            param.value.double_value = param_value

            # Send to appropriate server based on parameter
            if param_name in ['planner_frequency', 'tolerance']:
                self.planner_client.call_async(param)
            elif param_name in ['controller_frequency', 'robot_radius']:
                self.controller_client.call_async(param)
            else:
                self.param_client.call_async(param)

    def evaluate_navigation_performance(self):
        """Evaluate navigation performance with current parameters"""
        # This would run navigation trials and collect metrics
        # For now, return placeholder metrics
        return {
            'navigation_success_rate': np.random.random(),
            'average_time_to_goal': np.random.uniform(10, 60),
            'path_efficiency': np.random.uniform(0.5, 1.0),
            'collision_rate': np.random.uniform(0, 0.1)
        }
```

## Safety and Recovery Behaviors

### Advanced Recovery Behaviors

```python
class HumanoidRecoveryBehaviors(Node):
    def __init__(self):
        super().__init__('humanoid_recovery_behaviors')

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        # Publishers
        self.recovery_cmd_pub = self.create_publisher(Twist, '/recovery_cmd_vel', 10)

        # Parameters
        self.declare_parameter('fall_threshold_angle', 0.5)
        self.declare_parameter('stuck_threshold_time', 10.0)
        self.declare_parameter('balance_recovery_time', 5.0)

        self.fall_threshold_angle = self.get_parameter('fall_threshold_angle').value
        self.stuck_threshold_time = self.get_parameter('stuck_threshold_time').value
        self.balance_recovery_time = self.get_parameter('balance_recovery_time').value

        # State
        self.current_odom = None
        self.current_imu = None
        self.last_position = None
        self.stuck_start_time = None
        self.recovery_mode = False

        self.get_logger().info('Humanoid Recovery Behaviors initialized')

    def odom_callback(self, msg):
        """Process odometry for stuck detection"""
        self.current_odom = msg

        # Check if robot is stuck
        current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        if self.last_position is not None:
            distance_moved = np.linalg.norm(current_pos - self.last_position)

            if distance_moved < 0.05:  # Less than 5cm in some time
                if self.stuck_start_time is None:
                    self.stuck_start_time = self.get_clock().now()
                elif (self.get_clock().now() - self.stuck_start_time).nanoseconds / 1e9 > self.stuck_threshold_time:
                    # Robot is stuck, trigger recovery
                    self.handle_stuck_recovery()
            else:
                # Robot moved, reset stuck timer
                self.stuck_start_time = None

        self.last_position = current_pos

    def imu_callback(self, msg):
        """Process IMU data for fall detection"""
        self.current_imu = msg

        # Convert quaternion to roll/pitch angles
        orientation = msg.orientation
        roll, pitch, _ = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )

        # Check if robot is falling
        if abs(roll) > self.fall_threshold_angle or abs(pitch) > self.fall_threshold_angle:
            self.handle_fall_recovery(roll, pitch)

    def handle_stuck_recovery(self):
        """Handle robot stuck situation"""
        self.get_logger().warn('Robot is stuck, initiating recovery')

        # Try backing up
        cmd_vel = Twist()
        cmd_vel.linear.x = -0.1  # Back up slowly
        cmd_vel.angular.z = 0.0

        self.recovery_cmd_pub.publish(cmd_vel)

        # Wait for a bit
        self.get_logger().info('Backing up for 2 seconds')
        self.wait_for_duration(2.0)

        # Try turning
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.3  # Turn slowly

        self.recovery_cmd_pub.publish(cmd_vel)

        self.get_logger().info('Turning for 2 seconds')
        self.wait_for_duration(2.0)

        # Stop
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.recovery_cmd_pub.publish(cmd_vel)

    def handle_fall_recovery(self, roll, pitch):
        """Handle robot fall situation"""
        self.get_logger().error(f'Robot falling! Roll: {roll}, Pitch: {pitch}')

        # Emergency stop
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.linear.y = 0.0
        cmd_vel.linear.z = 0.0
        cmd_vel.angular.x = 0.0
        cmd_vel.angular.y = 0.0
        cmd_vel.angular.z = 0.0

        self.recovery_cmd_pub.publish(cmd_vel)

        # This would trigger more complex recovery like:
        # - Fall protection behaviors
        # - Self-righting if possible
        # - Emergency shutdown if necessary

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles"""
        import math

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def wait_for_duration(self, duration_sec):
        """Wait for specified duration"""
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < duration_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
```

## Best Practices for Humanoid Navigation

### 1. Parameter Configuration
- Adjust robot radius for humanoid dimensions
- Set appropriate velocity limits for stability
- Configure costmap inflation for safety margins
- Tune planners for humanoid-specific constraints

### 2. Sensor Integration
- Use multiple sensors for redundancy
- Integrate IMU data for balance awareness
- Consider stereo vision for 3D obstacle detection
- Plan for sensor failures with fallback behaviors

### 3. Performance Optimization
- Optimize costmap resolution for humanoid size
- Use appropriate planning frequencies
- Implement efficient obstacle detection
- Monitor and tune performance metrics

### 4. Safety Considerations
- Implement comprehensive recovery behaviors
- Use IMU for fall detection
- Set conservative velocity limits
- Plan for emergency stops

## Troubleshooting Common Issues

### 1. Local Planner Oscillation
- **Problem**: Robot oscillates back and forth
- **Solution**: Increase minimum velocity thresholds, tune controller parameters

### 2. Global Planner Failure
- **Problem**: Cannot find path to goal
- **Solution**: Check costmap inflation, adjust tolerance parameters

### 3. Collision Issues
- **Problem**: Robot collides with obstacles
- **Solution**: Increase robot radius, tune costmap inflation, improve sensor coverage

### 4. Performance Issues
- **Problem**: Slow navigation or high CPU usage
- **Solution**: Optimize costmap resolution, adjust planning frequencies, tune parameters

## Summary

Nav2 provides a comprehensive navigation framework that can be adapted for humanoid robotics with several key considerations:

- **Humanoid Kinematics**: Account for bipedal locomotion constraints
- **Balance Requirements**: Integrate IMU and balance data into navigation
- **Anthropomorphic Motion**: Consider human-like movement patterns
- **Social Navigation**: Respect personal space and social norms
- **Safety Systems**: Implement comprehensive safety and recovery behaviors
- **Sensor Integration**: Leverage Isaac ROS perception capabilities

The combination of Nav2's flexible architecture with humanoid-specific modifications enables robots to navigate complex environments safely and effectively. Proper parameter tuning and integration with perception systems ensures robust navigation performance that accounts for the unique challenges of bipedal locomotion.

In the next section, we'll explore AI-enhanced robot control systems that build upon the navigation capabilities we've discussed.