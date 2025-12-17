---
sidebar_position: 4
title: "Planning and Navigation"
---

# Planning and Navigation

## Introduction to Planning and Navigation in Humanoid Robotics

Planning and navigation form the cognitive core of humanoid robotics, enabling robots to move purposefully through complex environments. Unlike wheeled robots, humanoid robots must navigate with bipedal locomotion, requiring sophisticated path planning that accounts for balance, step constraints, and dynamic stability. NVIDIA Isaac™ provides powerful tools for implementing advanced planning and navigation systems that leverage GPU acceleration for real-time performance.

## Navigation Stack Architecture

### 1. Overview of the Navigation System

The navigation system for humanoid robots typically consists of several interconnected components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Perception    │    │   World Model   │    │   Path Planner  │
│   Module        │───▶│   (Costmap)     │───▶│                 │
│                 │    │                 │    │   Global Planner│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Localization  │    │  Costmap        │    │   Local Planner │
│   & Mapping     │    │  Management     │    │   (Trajectory   │
│                 │    │                 │    │   Generation)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Motion Control Layer                         │
│              (Step Planning, Balance Control)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Isaac ROS Navigation Components

Isaac ROS provides specialized navigation packages optimized for GPU acceleration:

#### Isaac ROS Navigation2 Integration
```cpp
// Isaac ROS Navigation2 integration
#include "nav2_core/global_planner.hpp"
#include "nav2_core/local_planner.hpp"
#include "isaac_ros_nav2_babylon/global_planner.hpp"

class IsaacNavigationPlanner : public nav2_core::GlobalPlanner
{
public:
    IsaacNavigationPlanner() = default;

    void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
        std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
        std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override
    {
        node_ = parent.lock();
        name_ = name;
        tf_ = tf;
        costmap_ros_ = costmap_ros;
        costmap_ = costmap_ros_->getCostmap();

        // Initialize GPU-accelerated planning components
        InitializeGPUPlanners();
    }

    void cleanup() override
    {
        CleanupGPUPlanners();
    }

    void activate() override
    {
        RCLCPP_INFO(node_->get_logger(), "IsaacNavigationPlanner is active");
    }

    void deactivate() override
    {
        RCLCPP_INFO(node_->get_logger(), "IsaacNavigationPlanner is inactive");
    }

    nav_msgs::msg::Path createPlan(
        const geometry_msgs::msg::PoseStamped& start,
        const geometry_msgs::msg::PoseStamped& goal) override
    {
        nav_msgs::msg::Path path;

        // Use GPU-accelerated path planning
        if (UseGPUPlanning()) {
            path = CreateGPUPlan(start, goal);
        } else {
            path = CreateCPUPlan(start, goal);
        }

        return path;
    }

private:
    bool UseGPUPlanning() const
    {
        // Check if GPU is available and beneficial for current planning task
        return IsGPUAvailable() && costmap_->getSizeInCellsX() * costmap_->getSizeInCellsY() > 10000;
    }

    nav_msgs::msg::Path CreateGPUPlan(
        const geometry_msgs::msg::PoseStamped& start,
        const geometry_msgs::msg::PoseStamped& goal)
    {
        nav_msgs::msg::Path path;

        // Convert costmap to GPU-friendly format
        auto gpu_costmap = ConvertCostmapToGPU();

        // Run GPU-accelerated path planning (e.g., A* or Dijkstra)
        auto gpu_path = RunGPUPathPlanner(gpu_costmap, start, goal);

        // Convert result back to ROS format
        path = ConvertGPUPathToROS(gpu_path);

        return path;
    }

    std::shared_ptr<GPUPathPlanner> gpu_planner_;
    rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
    std::string name_;
    std::shared_ptr<tf2_ros::Buffer> tf_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    nav2_costmap_2d::Costmap2D* costmap_;
};
```

## Global Path Planning

### 1. GPU-Accelerated Path Planning Algorithms

#### A* Algorithm with GPU Acceleration
```cpp
// GPU-accelerated A* path planning
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class GPUAStarPlanner
{
public:
    GPUAStarPlanner(int width, int height)
        : width_(width), height_(height), size_(width * height)
    {
        // Allocate GPU memory
        cudaMalloc(&costmap_d_, size_ * sizeof(unsigned char));
        cudaMalloc(&g_score_d_, size_ * sizeof(float));
        cudaMalloc(&f_score_d_, size_ * sizeof(float));
        cudaMalloc(&came_from_d_, size_ * sizeof(int));
        cudaMalloc(&open_set_d_, size_ * sizeof(bool));
        cudaMalloc(&closed_set_d_, size_ * sizeof(bool));

        // Initialize device pointers
        InitializeDeviceMemory();
    }

    ~GPUAStarPlanner()
    {
        cudaFree(costmap_d_);
        cudaFree(g_score_d_);
        cudaFree(f_score_d_);
        cudaFree(came_from_d_);
        cudaFree(open_set_d_);
        cudaFree(closed_set_d_);
    }

    std::vector<geometry_msgs::msg::Point> PlanPath(
        const std::vector<std::vector<unsigned char>>& costmap,
        const geometry_msgs::msg::Point& start,
        const geometry_msgs::msg::Point& goal)
    {
        // Copy costmap to GPU
        CopyCostmapToGPU(costmap);

        // Initialize planning parameters
        int start_idx = PointToIndex(start);
        int goal_idx = PointToIndex(goal);

        // Launch GPU A* kernel
        LaunchAStarKernel(start_idx, goal_idx);

        // Retrieve path from GPU
        auto path_indices = RetrievePathFromGPU(start_idx, goal_idx);

        // Convert to world coordinates
        return ConvertIndicesToWorld(path_indices);
    }

private:
    void LaunchAStarKernel(int start_idx, int goal_idx)
    {
        // Configure kernel launch parameters
        int block_size = 256;
        int grid_size = (size_ + block_size - 1) / block_size;

        // Launch A* planning kernel
        a_star_kernel<<<grid_size, block_size>>>(
            costmap_d_, g_score_d_, f_score_d_, came_from_d_,
            open_set_d_, closed_set_d_, size_, width_, height_,
            start_idx, goal_idx
        );

        // Wait for kernel completion
        cudaDeviceSynchronize();
    }

    std::vector<int> RetrievePathFromGPU(int start_idx, int goal_idx)
    {
        std::vector<int> path_indices;
        std::vector<int> temp_path;

        // Copy came_from array to host
        std::vector<int> came_from_h(size_);
        cudaMemcpy(came_from_h.data(), came_from_d_, size_ * sizeof(int), cudaMemcpyDeviceToHost);

        // Reconstruct path by following came_from pointers
        int current = goal_idx;
        while (current != start_idx && current != -1) {
            temp_path.push_back(current);
            current = came_from_h[current];
        }
        temp_path.push_back(start_idx);

        // Reverse to get path from start to goal
        std::reverse(temp_path.begin(), temp_path.end());
        return temp_path;
    }

    geometry_msgs::msg::Point IndexToPoint(int idx)
    {
        geometry_msgs::msg::Point point;
        point.x = (idx % width_) * resolution_;
        point.y = (idx / width_) * resolution_;
        point.z = 0.0;
        return point;
    }

    int PointToIndex(const geometry_msgs::msg::Point& point)
    {
        int x = static_cast<int>(point.x / resolution_);
        int y = static_cast<int>(point.y / resolution_);
        return y * width_ + x;
    }

    std::vector<geometry_msgs::msg::Point> ConvertIndicesToWorld(const std::vector<int>& indices)
    {
        std::vector<geometry_msgs::msg::Point> world_path;
        for (int idx : indices) {
            world_path.push_back(IndexToPoint(idx));
        }
        return world_path;
    }

    void CopyCostmapToGPU(const std::vector<std::vector<unsigned char>>& costmap)
    {
        // Flatten 2D costmap to 1D array
        std::vector<unsigned char> flat_costmap;
        for (const auto& row : costmap) {
            flat_costmap.insert(flat_costmap.end(), row.begin(), row.end());
        }

        // Copy to GPU
        cudaMemcpy(costmap_d_, flat_costmap.data(), size_ * sizeof(unsigned char), cudaMemcpyHostToDevice);
    }

    void InitializeDeviceMemory()
    {
        // Initialize GPU memory with default values
        cudaMemset(g_score_d_, 0xFF, size_ * sizeof(float)); // Initialize with infinity
        cudaMemset(f_score_d_, 0xFF, size_ * sizeof(float)); // Initialize with infinity
        cudaMemset(came_from_d_, -1, size_ * sizeof(int));   // Initialize with -1 (no parent)
        cudaMemset(open_set_d_, 0, size_ * sizeof(bool));    // Initialize as false
        cudaMemset(closed_set_d_, 0, size_ * sizeof(bool));  // Initialize as false
    }

    // CUDA kernel for A* path planning (simplified)
    __global__ void a_star_kernel(
        unsigned char* costmap,
        float* g_score,
        float* f_score,
        int* came_from,
        bool* open_set,
        bool* closed_set,
        int size,
        int width,
        int height,
        int start_idx,
        int goal_idx)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size) return;

        // A* algorithm implementation on GPU
        // This is a simplified version - real implementation would be more complex
        if (idx == start_idx) {
            g_score[idx] = 0.0f;
            f_score[idx] = heuristic(idx, goal_idx);
            open_set[idx] = true;
        }
    }

    __device__ float heuristic(int idx1, int idx2)
    {
        // Manhattan distance heuristic
        int x1 = idx1 % width_;
        int y1 = idx1 / width_;
        int x2 = idx2 % width_;
        int y2 = idx2 / width_;
        return abs(x1 - x2) + abs(y1 - y2);
    }

    unsigned char* costmap_d_;
    float* g_score_d_;
    float* f_score_d_;
    int* came_from_d_;
    bool* open_set_d_;
    bool* closed_set_d_;

    int width_, height_, size_;
    double resolution_ = 0.05; // 5cm resolution
};
```

### 2. Humanoid-Specific Path Planning

#### Bipedal Path Planning with Step Constraints
```cpp
// Humanoid-specific path planning considering bipedal constraints
class BipedalPathPlanner
{
public:
    struct StepConstraint {
        double max_step_length = 0.3;     // 30cm max step
        double max_step_width = 0.2;      // 20cm max step width
        double max_step_height = 0.1;     // 10cm max step height
        double min_step_length = 0.1;     // 10cm min step
        double step_spacing = 0.15;       // 15cm between steps
    };

    struct StepSequence {
        std::vector<Step> steps;
        double total_energy;
        double stability_score;
    };

    struct Step {
        geometry_msgs::msg::Point position;
        double orientation;
        StepType type; // LEFT_FOOT, RIGHT_FOOT
        double support_time;
    };

    enum class StepType { LEFT_FOOT, RIGHT_FOOT };

    StepSequence PlanBipedalPath(
        const nav_msgs::msg::Path& global_path,
        const geometry_msgs::msg::Pose& start_pose)
    {
        StepSequence sequence;
        Step current_step;

        // Initialize with starting pose
        current_step.position = start_pose.position;
        current_step.orientation = tf2::getYaw(start_pose.orientation);
        current_step.type = StepType::LEFT_FOOT;
        current_step.support_time = 0.0;

        sequence.steps.push_back(current_step);

        // Generate step sequence following the global path
        for (size_t i = 0; i < global_path.poses.size(); i += step_sampling_rate_) {
            auto target_pose = global_path.poses[i].pose.position;

            // Generate next step based on constraints
            auto next_step = GenerateNextStep(current_step, target_pose);

            if (IsStepValid(next_step, sequence.steps)) {
                sequence.steps.push_back(next_step);
                current_step = next_step;
            } else {
                // Adjust path to accommodate step constraints
                auto adjusted_step = AdjustStepForConstraints(current_step, target_pose);
                sequence.steps.push_back(adjusted_step);
                current_step = adjusted_step;
            }
        }

        // Optimize step sequence for energy efficiency and stability
        sequence = OptimizeStepSequence(sequence);

        // Calculate sequence metrics
        sequence.total_energy = CalculateEnergy(sequence);
        sequence.stability_score = CalculateStability(sequence);

        return sequence;
    }

private:
    Step GenerateNextStep(const Step& current_step, const geometry_msgs::msg::Point& target)
    {
        Step next_step;

        // Calculate direction to target
        double dx = target.x - current_step.position.x;
        double dy = target.y - current_step.position.y;
        double distance = sqrt(dx * dx + dy * dy);

        if (distance > step_constraint_.max_step_length) {
            // Scale step to maximum allowed length
            next_step.position.x = current_step.position.x +
                                   (dx / distance) * step_constraint_.max_step_length;
            next_step.position.y = current_step.position.y +
                                   (dy / distance) * step_constraint_.max_step_length;
        } else {
            // Move toward target with appropriate step length
            double step_length = std::max(step_constraint_.min_step_length,
                                        std::min(distance, step_constraint_.max_step_length));
            next_step.position.x = current_step.position.x +
                                   (dx / distance) * step_length;
            next_step.position.y = current_step.position.y +
                                   (dy / distance) * step_length;
        }

        // Alternate foot placement
        next_step.type = (current_step.type == StepType::LEFT_FOOT) ?
                        StepType::RIGHT_FOOT : StepType::LEFT_FOOT;

        // Set orientation to face direction of movement
        next_step.orientation = atan2(dy, dx);

        // Set support time based on step length
        next_step.support_time = CalculateSupportTime(next_step.position, current_step.position);

        return next_step;
    }

    bool IsStepValid(const Step& step, const std::vector<Step>& existing_steps)
    {
        // Check if step is within constraints
        if (!IsWithinStepConstraints(step, existing_steps)) {
            return false;
        }

        // Check for collisions with environment
        if (IsStepInCollision(step)) {
            return false;
        }

        // Check for balance constraints
        if (!IsStepBalanced(step, existing_steps)) {
            return false;
        }

        return true;
    }

    bool IsWithinStepConstraints(const Step& step, const std::vector<Step>& existing_steps)
    {
        if (existing_steps.empty()) {
            return true; // First step is always valid
        }

        const Step& last_step = existing_steps.back();

        // Calculate step length
        double step_length = sqrt(pow(step.position.x - last_step.position.x, 2) +
                                 pow(step.position.y - last_step.position.y, 2));

        // Check step length constraints
        if (step_length > step_constraint_.max_step_length ||
            step_length < step_constraint_.min_step_length) {
            return false;
        }

        return true;
    }

    Step AdjustStepForConstraints(const Step& current_step, const geometry_msgs::msg::Point& target)
    {
        Step adjusted_step = current_step;

        // Find closest valid position to target that satisfies constraints
        double min_distance = std::numeric_limits<double>::max();
        geometry_msgs::msg::Point best_position = current_step.position;

        // Sample positions in a circle around current position
        for (double angle = 0; angle < 2 * M_PI; angle += M_PI / 8) {
            for (double radius = step_constraint_.min_step_length;
                 radius <= step_constraint_.max_step_length;
                 radius += 0.05) {

                geometry_msgs::msg::Point candidate;
                candidate.x = current_step.position.x + radius * cos(angle);
                candidate.y = current_step.position.y + radius * sin(angle);

                double distance_to_target = sqrt(pow(candidate.x - target.x, 2) +
                                               pow(candidate.y - target.y, 2));

                if (distance_to_target < min_distance) {
                    // Check if this position is valid
                    Step temp_step;
                    temp_step.position = candidate;
                    if (IsStepValid(temp_step, {current_step})) {
                        min_distance = distance_to_target;
                        best_position = candidate;
                    }
                }
            }
        }

        adjusted_step.position = best_position;
        adjusted_step.type = (current_step.type == StepType::LEFT_FOOT) ?
                            StepType::RIGHT_FOOT : StepType::LEFT_FOOT;
        adjusted_step.orientation = atan2(best_position.y - current_step.position.y,
                                         best_position.x - current_step.position.x);

        return adjusted_step;
    }

    StepSequence OptimizeStepSequence(const StepSequence& sequence)
    {
        StepSequence optimized = sequence;

        // Optimize for energy efficiency
        for (size_t i = 1; i < optimized.steps.size(); ++i) {
            // Smooth step positions to reduce energy consumption
            auto& current_step = optimized.steps[i];
            auto& prev_step = optimized.steps[i-1];

            // Apply smoothing based on previous and next steps
            if (i < optimized.steps.size() - 1) {
                auto& next_step = optimized.steps[i+1];

                // Smooth position
                current_step.position.x = 0.25 * prev_step.position.x +
                                         0.5 * current_step.position.x +
                                         0.25 * next_step.position.x;
                current_step.position.y = 0.25 * prev_step.position.y +
                                         0.5 * current_step.position.y +
                                         0.25 * next_step.position.y;
            }
        }

        return optimized;
    }

    double CalculateEnergy(const StepSequence& sequence)
    {
        double total_energy = 0.0;

        for (size_t i = 1; i < sequence.steps.size(); ++i) {
            const auto& prev_step = sequence.steps[i-1];
            const auto& curr_step = sequence.steps[i];

            // Calculate step energy (proportional to step length and height)
            double step_length = sqrt(pow(curr_step.position.x - prev_step.position.x, 2) +
                                     pow(curr_step.position.y - prev_step.position.y, 2));

            total_energy += step_length * energy_cost_per_meter_;
        }

        return total_energy;
    }

    double CalculateStability(const StepSequence& sequence)
    {
        double stability_score = 0.0;

        for (size_t i = 2; i < sequence.steps.size(); ++i) {
            // Calculate Zero Moment Point (ZMP) based stability
            const auto& prev_step = sequence.steps[i-2];
            const auto& curr_step = sequence.steps[i-1];
            const auto& next_step = sequence.steps[i];

            // Simplified stability calculation based on step pattern
            double support_polygon_area = CalculateSupportPolygonArea(prev_step, curr_step);
            double ZMP_deviation = CalculateZMPDeviation(prev_step, curr_step, next_step);

            stability_score += support_polygon_area / (1.0 + ZMP_deviation);
        }

        return stability_score / std::max(1.0, static_cast<double>(sequence.steps.size()));
    }

    StepConstraint step_constraint_;
    int step_sampling_rate_ = 5; // Sample every 5th point from global path
    double energy_cost_per_meter_ = 10.0; // Energy cost per meter of step
};
```

## Local Path Planning and Obstacle Avoidance

### 1. Dynamic Window Approach (DWA) with GPU Acceleration

```cpp
// GPU-accelerated local path planning using Dynamic Window Approach
#include "isaac_ros_nitros/types/nitros_format_agent.hpp"

class GPULocalPlanner
{
public:
    struct VelocitySample {
        double linear_vel;
        double angular_vel;
        double cost;
        bool valid;
    };

    struct Trajectory {
        std::vector<geometry_msgs::msg::Pose> poses;
        double time_to_execute;
        double clearance;
        double goal_distance;
        double heading_alignment;
    };

    geometry_msgs::msg::Twist ComputeVelocityCommands(
        const geometry_msgs::msg::PoseStamped& robot_pose,
        const geometry_msgs::msg::PoseStamped& goal_pose,
        const std::vector<geometry_msgs::msg::Point>& obstacles)
    {
        geometry_msgs::msg::Twist cmd_vel;

        // Sample velocity space
        auto velocity_samples = SampleVelocitySpace();

        // Evaluate trajectories for each velocity sample
        auto valid_trajectories = EvaluateTrajectories(
            velocity_samples, robot_pose, goal_pose, obstacles);

        // Select best trajectory
        auto best_trajectory = SelectBestTrajectory(valid_trajectories, goal_pose);

        // Convert to velocity command
        cmd_vel = ConvertTrajectoryToVelocity(best_trajectory);

        return cmd_vel;
    }

private:
    std::vector<VelocitySample> SampleVelocitySpace()
    {
        std::vector<VelocitySample> samples;

        // Sample linear velocities
        for (double v = min_linear_vel_; v <= max_linear_vel_; v += linear_vel_resolution_) {
            // Sample angular velocities
            for (double w = min_angular_vel_; w <= max_angular_vel_; w += angular_vel_resolution_) {
                VelocitySample sample;
                sample.linear_vel = v;
                sample.angular_vel = w;
                sample.valid = true;
                samples.push_back(sample);
            }
        }

        return samples;
    }

    std::vector<Trajectory> EvaluateTrajectories(
        const std::vector<VelocitySample>& samples,
        const geometry_msgs::msg::PoseStamped& robot_pose,
        const geometry_msgs::msg::PoseStamped& goal_pose,
        const std::vector<geometry_msgs::msg::Point>& obstacles)
    {
        std::vector<Trajectory> valid_trajectories;

        // Use GPU to evaluate trajectories in parallel
        auto gpu_trajectories = EvaluateTrajectoriesGPU(samples, robot_pose, goal_pose, obstacles);

        // Filter valid trajectories
        for (const auto& trajectory : gpu_trajectories) {
            if (IsTrajectoryValid(trajectory, obstacles)) {
                valid_trajectories.push_back(trajectory);
            }
        }

        return valid_trajectories;
    }

    std::vector<Trajectory> EvaluateTrajectoriesGPU(
        const std::vector<VelocitySample>& samples,
        const geometry_msgs::msg::PoseStamped& robot_pose,
        const geometry_msgs::msg::PoseStamped& goal_pose,
        const std::vector<geometry_msgs::msg::Point>& obstacles)
    {
        // Copy data to GPU
        CopyTrajectoryDataToGPU(samples, robot_pose, goal_pose, obstacles);

        // Launch GPU kernel to evaluate all trajectories in parallel
        int num_samples = samples.size();
        int block_size = 256;
        int grid_size = (num_samples + block_size - 1) / block_size;

        evaluate_trajectories_kernel<<<grid_size, block_size>>>(
            velocity_samples_d_, obstacles_d_, obstacles_count_,
            robot_pose_d_, goal_pose_d_, trajectories_d_, num_samples
        );

        // Copy results back to CPU
        return CopyTrajectoriesFromGPU(num_samples);
    }

    Trajectory PredictTrajectory(const VelocitySample& sample, double dt, int steps)
    {
        Trajectory trajectory;
        geometry_msgs::msg::Pose current_pose;

        // Initialize with robot pose
        current_pose = robot_pose_.pose;

        for (int i = 0; i < steps; ++i) {
            // Integrate motion model
            double dx = sample.linear_vel * cos(current_pose.orientation.z) * dt;
            double dy = sample.linear_vel * sin(current_pose.orientation.z) * dt;
            double dtheta = sample.angular_vel * dt;

            current_pose.position.x += dx;
            current_pose.position.y += dy;
            current_pose.orientation.z += dtheta;

            trajectory.poses.push_back(current_pose);
        }

        return trajectory;
    }

    bool IsTrajectoryValid(const Trajectory& trajectory,
                          const std::vector<geometry_msgs::msg::Point>& obstacles)
    {
        for (const auto& pose : trajectory.poses) {
            for (const auto& obstacle : obstacles) {
                double distance = sqrt(pow(pose.position.x - obstacle.x, 2) +
                                     pow(pose.position.y - obstacle.y, 2));

                if (distance < min_clearance_) {
                    return false; // Collision detected
                }
            }
        }

        return true;
    }

    Trajectory SelectBestTrajectory(const std::vector<Trajectory>& trajectories,
                                   const geometry_msgs::msg::PoseStamped& goal_pose)
    {
        if (trajectories.empty()) {
            // Return zero velocity if no valid trajectories found
            Trajectory empty_traj;
            return empty_traj;
        }

        // Score trajectories based on multiple criteria
        Trajectory best_trajectory = trajectories[0];
        double best_score = CalculateTrajectoryScore(best_trajectory, goal_pose);

        for (size_t i = 1; i < trajectories.size(); ++i) {
            double score = CalculateTrajectoryScore(trajectories[i], goal_pose);
            if (score > best_score) {
                best_score = score;
                best_trajectory = trajectories[i];
            }
        }

        return best_trajectory;
    }

    double CalculateTrajectoryScore(const Trajectory& trajectory,
                                   const geometry_msgs::msg::PoseStamped& goal_pose)
    {
        if (trajectory.poses.empty()) {
            return -1.0; // Invalid trajectory
        }

        // Calculate multiple scores and combine them
        double goal_distance_score = CalculateGoalDistanceScore(trajectory, goal_pose);
        double clearance_score = CalculateClearanceScore(trajectory);
        double velocity_score = CalculateVelocityScore(trajectory);
        double heading_score = CalculateHeadingScore(trajectory, goal_pose);

        // Weighted combination of scores
        return 0.4 * goal_distance_score +
               0.3 * clearance_score +
               0.2 * velocity_score +
               0.1 * heading_score;
    }

    geometry_msgs::msg::Twist ConvertTrajectoryToVelocity(const Trajectory& trajectory)
    {
        geometry_msgs::msg::Twist cmd_vel;

        if (!trajectory.poses.empty()) {
            // Calculate velocity based on first segment of trajectory
            const auto& pose1 = trajectory.poses[0];
            const auto& pose2 = trajectory.poses[1];

            double dt = 0.1; // Time step
            cmd_vel.linear.x = sqrt(pow(pose2.position.x - pose1.position.x, 2) +
                                   pow(pose2.position.y - pose1.position.y, 2)) / dt;
            cmd_vel.angular.z = (pose2.orientation.z - pose1.orientation.z) / dt;
        }

        return cmd_vel;
    }

    // GPU kernel for trajectory evaluation
    __global__ void evaluate_trajectories_kernel(
        VelocitySample* samples,
        geometry_msgs::msg::Point* obstacles,
        int obstacles_count,
        geometry_msgs::msg::Pose robot_pose,
        geometry_msgs::msg::Pose goal_pose,
        Trajectory* trajectories,
        int num_samples)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_samples) return;

        // Evaluate trajectory for this velocity sample
        VelocitySample sample = samples[idx];
        Trajectory trajectory;

        // Predict trajectory using motion model
        geometry_msgs::msg::Pose current_pose = robot_pose;
        for (int step = 0; step < trajectory_prediction_steps_; ++step) {
            // Integrate motion model
            double dx = sample.linear_vel * cos(current_pose.orientation.z) * trajectory_dt_;
            double dy = sample.linear_vel * sin(current_pose.orientation.z) * trajectory_dt_;
            double dtheta = sample.angular_vel * trajectory_dt_;

            current_pose.position.x += dx;
            current_pose.position.y += dy;
            current_pose.orientation.z += dtheta;

            trajectory.poses[step] = current_pose;

            // Check for collisions with obstacles
            for (int obs_idx = 0; obs_idx < obstacles_count; ++obs_idx) {
                double distance = sqrt(pow(current_pose.position.x - obstacles[obs_idx].x, 2) +
                                     pow(current_pose.position.y - obstacles[obs_idx].y, 2));
                if (distance < min_clearance_) {
                    trajectory.valid = false;
                    break;
                }
            }

            if (!trajectory.valid) break;
        }

        trajectories[idx] = trajectory;
    }

    geometry_msgs::msg::Pose robot_pose_;
    double min_linear_vel_ = 0.0;
    double max_linear_vel_ = 1.0;
    double min_angular_vel_ = -1.0;
    double max_angular_vel_ = 1.0;
    double linear_vel_resolution_ = 0.1;
    double angular_vel_resolution_ = 0.1;
    double min_clearance_ = 0.5;
    double trajectory_dt_ = 0.1;
    int trajectory_prediction_steps_ = 20;
};
```

### 2. Humanoid-Specific Local Planning

#### Balance-Aware Local Planning
```cpp
// Balance-aware local planning for humanoid robots
class BalanceAwareLocalPlanner
{
public:
    struct BalanceConstraint {
        double max_zmp_deviation = 0.05;  // 5cm max ZMP deviation
        double min_support_polygon_area = 0.01; // 100cm² min support area
        double max_com_height_variation = 0.1; // 10cm max CoM height change
        double balance_threshold = 0.8;   // 80% balance confidence required
    };

    struct BalanceAwareTrajectory {
        std::vector<geometry_msgs::msg::Pose> poses;
        std::vector<BalanceState> balance_states;
        double balance_score;
        bool is_stable;
    };

    struct BalanceState {
        geometry_msgs::msg::Point zmp;      // Zero Moment Point
        geometry_msgs::msg::Point com;      // Center of Mass
        double support_polygon_area;
        double balance_margin;
        bool is_balanced;
    };

    geometry_msgs::msg::Twist PlanBalancedTrajectory(
        const geometry_msgs::msg::PoseStamped& robot_pose,
        const geometry_msgs::msg::PoseStamped& goal_pose,
        const std::vector<geometry_msgs::msg::Point>& obstacles)
    {
        geometry_msgs::msg::Twist cmd_vel;

        // Generate candidate trajectories
        auto candidate_trajectories = GenerateCandidateTrajectories(robot_pose, goal_pose);

        // Evaluate balance for each trajectory
        auto balanced_trajectories = EvaluateBalanceConstraints(candidate_trajectories);

        // Select trajectory with best balance score
        auto best_trajectory = SelectBalancedTrajectory(balanced_trajectories, goal_pose);

        // Generate velocity command
        cmd_vel = GenerateVelocityCommand(best_trajectory, robot_pose);

        return cmd_vel;
    }

private:
    std::vector<BalanceAwareTrajectory> GenerateCandidateTrajectories(
        const geometry_msgs::msg::PoseStamped& robot_pose,
        const geometry_msgs::msg::PoseStamped& goal_pose)
    {
        std::vector<BalanceAwareTrajectory> candidates;

        // Generate trajectories with different walking patterns
        // 1. Straight line trajectory
        auto straight_traj = GenerateStraightTrajectory(robot_pose, goal_pose);
        candidates.push_back(straight_traj);

        // 2. Curved trajectory for obstacle avoidance
        auto curved_traj = GenerateCurvedTrajectory(robot_pose, goal_pose);
        candidates.push_back(curved_traj);

        // 3. Step-by-step trajectory for precise positioning
        auto step_traj = GenerateStepTrajectory(robot_pose, goal_pose);
        candidates.push_back(step_traj);

        return candidates;
    }

    BalanceAwareTrajectory GenerateStraightTrajectory(
        const geometry_msgs::msg::PoseStamped& start,
        const geometry_msgs::msg::PoseStamped& goal)
    {
        BalanceAwareTrajectory trajectory;

        // Calculate straight-line path
        double dx = goal.pose.position.x - start.pose.position.x;
        double dy = goal.pose.position.y - start.pose.position.y;
        double distance = sqrt(dx * dx + dy * dy);
        double steps = distance / step_size_;

        for (int i = 0; i <= static_cast<int>(steps); ++i) {
            geometry_msgs::msg::Pose pose;
            pose.position.x = start.pose.position.x + (dx / steps) * i;
            pose.position.y = start.pose.position.y + (dy / steps) * i;
            pose.position.z = nominal_com_height_; // Maintain nominal CoM height
            pose.orientation = goal.pose.orientation; // Face goal direction

            trajectory.poses.push_back(pose);

            // Calculate balance state for this pose
            BalanceState balance_state = CalculateBalanceState(pose, trajectory.poses);
            trajectory.balance_states.push_back(balance_state);
        }

        return trajectory;
    }

    BalanceAwareTrajectory GenerateCurvedTrajectory(
        const geometry_msgs::msg::PoseStamped& start,
        const geometry_msgs::msg::PoseStamped& goal)
    {
        BalanceAwareTrajectory trajectory;

        // Generate curved path (e.g., cubic spline) to avoid obstacles
        // This is a simplified implementation
        for (double t = 0; t <= 1.0; t += 0.1) {
            geometry_msgs::msg::Pose pose;

            // Cubic Bezier curve
            double x = (1-t)*(1-t)*(1-t)*start.pose.position.x +
                      3*(1-t)*(1-t)*t*start.pose.position.x +
                      3*(1-t)*t*t*goal.pose.position.x +
                      t*t*t*goal.pose.position.x;

            double y = (1-t)*(1-t)*(1-t)*start.pose.position.y +
                      3*(1-t)*(1-t)*t*start.pose.position.y +
                      3*(1-t)*t*t*goal.pose.position.y +
                      t*t*t*goal.pose.position.y;

            pose.position.x = x;
            pose.position.y = y;
            pose.position.z = nominal_com_height_;
            pose.orientation = goal.pose.orientation;

            trajectory.poses.push_back(pose);

            BalanceState balance_state = CalculateBalanceState(pose, trajectory.poses);
            trajectory.balance_states.push_back(balance_state);
        }

        return trajectory;
    }

    std::vector<BalanceAwareTrajectory> EvaluateBalanceConstraints(
        const std::vector<BalanceAwareTrajectory>& candidates)
    {
        std::vector<BalanceAwareTrajectory> valid_trajectories;

        for (auto trajectory : candidates) {
            bool is_trajectory_balanced = true;
            double total_balance_score = 0.0;

            for (size_t i = 0; i < trajectory.balance_states.size(); ++i) {
                const auto& balance_state = trajectory.balance_states[i];

                // Check balance constraints
                if (!balance_state.is_balanced) {
                    is_trajectory_balanced = false;
                    break;
                }

                if (balance_state.support_polygon_area < balance_constraint_.min_support_polygon_area) {
                    is_trajectory_balanced = false;
                    break;
                }

                total_balance_score += balance_state.balance_margin;
            }

            if (is_trajectory_balanced) {
                trajectory.balance_score = total_balance_score / trajectory.balance_states.size();
                trajectory.is_stable = true;
                valid_trajectories.push_back(trajectory);
            }
        }

        return valid_trajectories;
    }

    BalanceState CalculateBalanceState(const geometry_msgs::msg::Pose& pose,
                                      const std::vector<geometry_msgs::msg::Pose>& all_poses)
    {
        BalanceState state;

        // Calculate Zero Moment Point (ZMP)
        state.zmp = CalculateZMP(pose, all_poses);

        // Calculate Center of Mass (simplified)
        state.com = CalculateCoM(pose);

        // Calculate support polygon area (simplified - assume feet positions)
        state.support_polygon_area = CalculateSupportPolygonArea(pose);

        // Calculate balance margin
        state.balance_margin = CalculateBalanceMargin(state.zmp, state.support_polygon_area);

        // Determine if balanced
        state.is_balanced = state.balance_margin > balance_constraint_.balance_threshold;

        return state;
    }

    geometry_msgs::msg::Point CalculateZMP(const geometry_msgs::msg::Pose& pose,
                                          const std::vector<geometry_msgs::msg::Pose>& all_poses)
    {
        geometry_msgs::msg::Point zmp;

        // Simplified ZMP calculation
        // In reality, this would involve complex dynamics and foot placement
        zmp.x = pose.position.x;
        zmp.y = pose.position.y;

        // ZMP should be within support polygon for balance
        return zmp;
    }

    geometry_msgs::msg::Point CalculateCoM(const geometry_msgs::msg::Pose& pose)
    {
        geometry_msgs::msg::Point com;

        // Simplified CoM calculation - in reality, this depends on robot kinematics
        com.x = pose.position.x;
        com.y = pose.position.y;
        com.z = nominal_com_height_; // Nominal CoM height for humanoid

        return com;
    }

    double CalculateSupportPolygonArea(const geometry_msgs::msg::Pose& pose)
    {
        // Calculate area of support polygon based on foot positions
        // For bipedal walking, this is typically the area between feet
        // Simplified calculation assuming feet are 20cm apart
        double foot_separation = 0.2; // 20cm between feet
        double foot_size = 0.15; // 15cm foot length

        return foot_separation * foot_size; // Simplified area
    }

    double CalculateBalanceMargin(const geometry_msgs::msg::Point& zmp, double support_area)
    {
        // Calculate how much the ZMP is within the support polygon
        // Return value between 0 and 1 (1 = perfectly balanced)
        double distance_to_boundary = CalculateDistanceToSupportBoundary(zmp);

        // Normalize based on support polygon size
        double normalized_distance = distance_to_boundary / sqrt(support_area);

        // Convert to balance margin (higher is better)
        return std::min(1.0, normalized_distance * 10.0); // Scale factor
    }

    double CalculateDistanceToSupportBoundary(const geometry_msgs::msg::Point& zmp)
    {
        // Calculate minimum distance from ZMP to support polygon boundary
        // Simplified as distance to center of support polygon
        double center_x = 0.0; // Would be actual support polygon center
        double center_y = 0.0;

        return sqrt(pow(zmp.x - center_x, 2) + pow(zmp.y - center_y, 2));
    }

    BalanceAwareTrajectory SelectBalancedTrajectory(
        const std::vector<BalanceAwareTrajectory>& trajectories,
        const geometry_msgs::msg::PoseStamped& goal_pose)
    {
        if (trajectories.empty()) {
            // Return zero velocity if no balanced trajectories found
            BalanceAwareTrajectory empty_traj;
            return empty_traj;
        }

        // Select trajectory with best balance score and closest to goal
        BalanceAwareTrajectory best_trajectory = trajectories[0];
        double best_score = CalculateTrajectoryScore(best_trajectory, goal_pose);

        for (size_t i = 1; i < trajectories.size(); ++i) {
            double score = CalculateTrajectoryScore(trajectories[i], goal_pose);
            if (score > best_score) {
                best_score = score;
                best_trajectory = trajectories[i];
            }
        }

        return best_trajectory;
    }

    double CalculateTrajectoryScore(const BalanceAwareTrajectory& trajectory,
                                   const geometry_msgs::msg::PoseStamped& goal_pose)
    {
        if (trajectory.poses.empty()) {
            return -1.0;
        }

        // Calculate goal proximity score
        const auto& last_pose = trajectory.poses.back();
        double goal_distance = sqrt(pow(last_pose.position.x - goal_pose.pose.position.x, 2) +
                                   pow(last_pose.position.y - goal_pose.pose.position.y, 2));
        double goal_score = 1.0 / (1.0 + goal_distance);

        // Calculate balance score
        double balance_score = trajectory.balance_score;

        // Calculate smoothness score
        double smoothness_score = CalculateTrajectorySmoothness(trajectory);

        // Weighted combination
        return 0.4 * goal_score + 0.4 * balance_score + 0.2 * smoothness_score;
    }

    double CalculateTrajectorySmoothness(const BalanceAwareTrajectory& trajectory)
    {
        if (trajectory.poses.size() < 3) {
            return 1.0; // Single point is perfectly smooth
        }

        double total_curvature = 0.0;
        int segments = 0;

        for (size_t i = 1; i < trajectory.poses.size() - 1; ++i) {
            const auto& p1 = trajectory.poses[i-1];
            const auto& p2 = trajectory.poses[i];
            const auto& p3 = trajectory.poses[i+1];

            // Calculate curvature using three consecutive points
            double dx1 = p2.position.x - p1.position.x;
            double dy1 = p2.position.y - p1.position.y;
            double dx2 = p3.position.x - p2.position.x;
            double dy2 = p3.position.y - p2.position.y;

            // Approximate curvature
            double angle_change = atan2(dy2, dx2) - atan2(dy1, dx1);
            total_curvature += abs(angle_change);
            segments++;
        }

        if (segments > 0) {
            double average_curvature = total_curvature / segments;
            // Convert to smoothness (lower curvature = higher smoothness)
            return std::max(0.0, 1.0 - average_curvature);
        }

        return 1.0;
    }

    geometry_msgs::msg::Twist GenerateVelocityCommand(
        const BalanceAwareTrajectory& trajectory,
        const geometry_msgs::msg::PoseStamped& current_pose)
    {
        geometry_msgs::msg::Twist cmd_vel;

        if (trajectory.poses.size() < 2) {
            // Stop if trajectory is too short
            cmd_vel.linear.x = 0.0;
            cmd_vel.angular.z = 0.0;
            return cmd_vel;
        }

        // Calculate velocity to follow the trajectory
        const auto& next_pose = trajectory.poses[1]; // Next point in trajectory

        // Calculate linear velocity
        double dx = next_pose.position.x - current_pose.pose.position.x;
        double dy = next_pose.position.y - current_pose.pose.position.y;
        double distance = sqrt(dx * dx + dy * dy);

        cmd_vel.linear.x = std::min(max_linear_vel_, distance / trajectory_time_step_);

        // Calculate angular velocity to face the next direction
        double target_yaw = atan2(dy, dx);
        double current_yaw = tf2::getYaw(current_pose.pose.orientation);
        double yaw_error = target_yaw - current_yaw;

        // Normalize yaw error to [-π, π]
        while (yaw_error > M_PI) yaw_error -= 2 * M_PI;
        while (yaw_error < -M_PI) yaw_error += 2 * M_PI;

        cmd_vel.angular.z = std::max(-max_angular_vel_,
                                   std::min(max_angular_vel_, yaw_error / trajectory_time_step_));

        return cmd_vel;
    }

    BalanceConstraint balance_constraint_;
    double step_size_ = 0.1; // 10cm steps
    double nominal_com_height_ = 0.8; // 80cm nominal CoM height
    double trajectory_time_step_ = 0.1; // 100ms time step
    double max_linear_vel_ = 0.5; // 0.5 m/s max linear velocity
    double max_angular_vel_ = 0.5; // 0.5 rad/s max angular velocity
};
```

## Integration with Isaac Sim

### 1. Simulation-Based Planning Validation

```cpp
// Integration with Isaac Sim for planning validation
#include "isaac_ros_managed_nh/managed_node_handle.hpp"

class IsaacSimNavigationValidator
{
public:
    IsaacSimNavigationValidator(rclcpp::Node* node) : node_(node)
    {
        // Initialize Isaac Sim interface
        InitializeSimInterface();

        // Create validation topics
        validation_result_pub_ = node_->create_publisher<isaac_ros_nav_msgs::msg::NavigationValidation>(
            "/navigation/validation_result", 10);
    }

    NavigationValidationResult ValidateNavigationPlan(
        const nav_msgs::msg::Path& path,
        const std::string& environment_name)
    {
        NavigationValidationResult result;

        // Set up simulation environment
        SetupSimulationEnvironment(environment_name);

        // Execute navigation plan in simulation
        auto sim_result = ExecuteNavigationInSimulation(path);

        // Collect validation metrics
        result.success_rate = CalculateSuccessRate(sim_result);
        result.average_time = CalculateAverageTime(sim_result);
        result.collision_rate = CalculateCollisionRate(sim_result);
        result.energy_efficiency = CalculateEnergyEfficiency(sim_result);
        result.balance_stability = CalculateBalanceStability(sim_result);

        // Publish validation results
        PublishValidationResult(result);

        return result;
    }

private:
    struct NavigationValidationResult {
        double success_rate;
        double average_time;
        double collision_rate;
        double energy_efficiency;
        double balance_stability;
        std::vector<geometry_msgs::msg::Pose> executed_path;
    };

    struct SimulationResult {
        std::vector<geometry_msgs::msg::Pose> robot_poses;
        std::vector<double> time_stamps;
        std::vector<bool> collision_flags;
        std::vector<double> energy_consumption;
        std::vector<double> balance_metrics;
        bool navigation_successful;
    };

    void InitializeSimInterface()
    {
        // Initialize connection to Isaac Sim
        // This would typically involve setting up USD stage loading,
        // robot spawning, and sensor configuration
    }

    void SetupSimulationEnvironment(const std::string& environment_name)
    {
        // Load the specified environment in Isaac Sim
        // Configure obstacles, lighting, and other environmental factors
    }

    SimulationResult ExecuteNavigationInSimulation(const nav_msgs::msg::Path& path)
    {
        SimulationResult result;

        // Spawn humanoid robot in simulation
        SpawnHumanoidRobot();

        // Execute path following controller
        for (const auto& pose : path.poses) {
            // Send navigation goal to controller
            SendNavigationGoal(pose.pose);

            // Simulate robot movement
            auto robot_pose = SimulateRobotMovement(pose.pose);
            result.robot_poses.push_back(robot_pose);

            // Record metrics
            result.time_stamps.push_back(GetSimulationTime());
            result.collision_flags.push_back(CheckCollision());
            result.energy_consumption.push_back(CalculateEnergy());
            result.balance_metrics.push_back(CalculateBalanceMetric());

            // Check if navigation is still successful
            if (CheckCollision() || !IsBalanced()) {
                result.navigation_successful = false;
                break;
            }
        }

        result.navigation_successful = true;
        return result;
    }

    double CalculateSuccessRate(const SimulationResult& result)
    {
        // Success if robot reaches goal without collisions and maintains balance
        int successful_runs = 0;
        int total_runs = 1; // For single path validation

        if (result.navigation_successful) {
            successful_runs = 1;
        }

        return static_cast<double>(successful_runs) / total_runs;
    }

    double CalculateCollisionRate(const SimulationResult& result)
    {
        int collision_count = 0;
        for (bool collision : result.collision_flags) {
            if (collision) {
                collision_count++;
            }
        }

        return static_cast<double>(collision_count) / result.collision_flags.size();
    }

    double CalculateBalanceStability(const SimulationResult& result)
    {
        if (result.balance_metrics.empty()) {
            return 0.0;
        }

        double total_balance = 0.0;
        for (double balance : result.balance_metrics) {
            total_balance += balance;
        }

        return total_balance / result.balance_metrics.size();
    }

    void SpawnHumanoidRobot()
    {
        // Spawn humanoid robot model in Isaac Sim
        // Configure initial position, sensors, and controllers
    }

    geometry_msgs::msg::Pose SimulateRobotMovement(const geometry_msgs::msg::Pose& target_pose)
    {
        // Simulate robot movement toward target pose
        // This would involve physics simulation and controller execution
        return target_pose; // Simplified return
    }

    bool CheckCollision()
    {
        // Check for collisions in simulation
        return false; // Simplified
    }

    bool IsBalanced()
    {
        // Check if robot maintains balance
        return true; // Simplified
    }

    double CalculateEnergy()
    {
        // Calculate energy consumption in simulation
        return 0.0; // Simplified
    }

    double CalculateBalanceMetric()
    {
        // Calculate balance metric in simulation
        return 1.0; // Simplified (perfectly balanced)
    }

    double GetSimulationTime()
    {
        // Get current simulation time
        return 0.0; // Simplified
    }

    void SendNavigationGoal(const geometry_msgs::msg::Pose& pose)
    {
        // Send navigation goal to the robot's navigation stack in simulation
    }

    rclcpp::Node* node_;
    rclcpp::Publisher<isaac_ros_nav_msgs::msg::NavigationValidation>::SharedPtr validation_result_pub_;
};
```

## Performance Optimization and Real-time Considerations

### 1. Multi-Level Planning Hierarchy

```cpp
// Multi-level planning hierarchy for computational efficiency
class HierarchicalNavigationPlanner
{
public:
    struct PlanningLevel {
        std::string name;
        double resolution;
        double update_rate;
        std::function<nav_msgs::msg::Path(const geometry_msgs::msg::Pose&,
                                        const geometry_msgs::msg::Pose&)> planner_func;
    };

    HierarchicalNavigationPlanner()
    {
        InitializePlanningHierarchy();
    }

    nav_msgs::msg::Path PlanPath(const geometry_msgs::msg::PoseStamped& start,
                                const geometry_msgs::msg::PoseStamped& goal)
    {
        nav_msgs::msg::Path final_path;

        // Execute planning at different levels
        for (auto& level : planning_levels_) {
            if (ShouldExecuteLevel(level, start, goal)) {
                auto level_path = level.planner_func(start.pose, goal.pose);

                if (!level_path.poses.empty()) {
                    // Refine path for lower levels if needed
                    final_path = RefinePathForLevel(final_path, level_path, level);
                }
            }
        }

        return final_path;
    }

private:
    void InitializePlanningHierarchy()
    {
        // Global planning level (low resolution, infrequent updates)
        PlanningLevel global_level;
        global_level.name = "global";
        global_level.resolution = 1.0;  // 1m resolution
        global_level.update_rate = 0.1; // 0.1 Hz (every 10 seconds)
        global_level.planner_func = [this](const geometry_msgs::msg::Pose& start,
                                         const geometry_msgs::msg::Pose& goal) {
            return this->GlobalPlan(start, goal);
        };
        planning_levels_.push_back(global_level);

        // Mid-level planning (medium resolution, moderate updates)
        PlanningLevel mid_level;
        mid_level.name = "mid";
        mid_level.resolution = 0.2;   // 20cm resolution
        mid_level.update_rate = 1.0;  // 1 Hz
        mid_level.planner_func = [this](const geometry_msgs::msg::Pose& start,
                                      const geometry_msgs::msg::Pose& goal) {
            return this->MidPlan(start, goal);
        };
        planning_levels_.push_back(mid_level);

        // Local planning level (high resolution, frequent updates)
        PlanningLevel local_level;
        local_level.name = "local";
        local_level.resolution = 0.05; // 5cm resolution
        local_level.update_rate = 10.0; // 10 Hz
        local_level.planner_func = [this](const geometry_msgs::msg::Pose& start,
                                        const geometry_msgs::msg::Pose& goal) {
            return this->LocalPlan(start, goal);
        };
        planning_levels_.push_back(local_level);
    }

    bool ShouldExecuteLevel(const PlanningLevel& level,
                           const geometry_msgs::msg::PoseStamped& start,
                           const geometry_msgs::msg::PoseStamped& goal)
    {
        // Check if enough time has passed for this level
        auto current_time = std::chrono::steady_clock::now();
        auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - last_execution_time_[level.name]).count();

        double required_interval_ms = 1000.0 / level.update_rate;
        if (time_since_last < required_interval_ms) {
            return false;
        }

        // Check if goal has changed significantly
        double goal_distance = sqrt(pow(goal.pose.position.x - last_goals_[level.name].x, 2) +
                                   pow(goal.pose.position.y - last_goals_[level.name].y, 2));
        if (level.name == "global" && goal_distance < 0.5) { // 50cm threshold for global replanning
            return false;
        }

        // Update last execution time and goal
        last_execution_time_[level.name] = current_time;
        last_goals_[level.name] = goal.pose.position;

        return true;
    }

    nav_msgs::msg::Path GlobalPlan(const geometry_msgs::msg::Pose& start,
                                  const geometry_msgs::msg::Pose& goal)
    {
        // Use GPU-accelerated global planner for coarse path
        return gpu_global_planner_.Plan(start, goal);
    }

    nav_msgs::msg::Path MidPlan(const geometry_msgs::msg::Pose& start,
                               const geometry_msgs::msg::Pose& goal)
    {
        // Mid-level planner for path refinement
        return mid_level_planner_.Plan(start, goal);
    }

    nav_msgs::msg::Path LocalPlan(const geometry_msgs::msg::Pose& start,
                                 const geometry_msgs::msg::Pose& goal)
    {
        // Local planner for obstacle avoidance and fine adjustments
        return local_planner_.Plan(start, goal);
    }

    nav_msgs::msg::Path RefinePathForLevel(const nav_msgs::msg::Path& existing_path,
                                          const nav_msgs::msg::Path& level_path,
                                          const PlanningLevel& level)
    {
        if (level.name == "global") {
            // Global path becomes the base path
            return level_path;
        } else if (level.name == "mid") {
            // Mid-level refines global path
            return RefineGlobalWithMid(existing_path, level_path);
        } else if (level.name == "local") {
            // Local level provides fine adjustments to mid-level path
            return RefineMidWithLocal(existing_path, level_path);
        }

        return existing_path;
    }

    nav_msgs::msg::Path RefineGlobalWithMid(const nav_msgs::msg::Path& global_path,
                                           const nav_msgs::msg::Path& mid_path)
    {
        // Combine global and mid-level paths
        nav_msgs::msg::Path refined_path = global_path;

        // Replace sections of global path with mid-level refinements
        for (size_t i = 0; i < mid_path.poses.size(); ++i) {
            if (i < refined_path.poses.size()) {
                refined_path.poses[i] = mid_path.poses[i];
            } else {
                refined_path.poses.push_back(mid_path.poses[i]);
            }
        }

        return refined_path;
    }

    nav_msgs::msg::Path RefineMidWithLocal(const nav_msgs::msg::Path& mid_path,
                                          const nav_msgs::msg::Path& local_path)
    {
        // Apply local adjustments to mid-level path
        nav_msgs::msg::Path refined_path = mid_path;

        // For the immediate future path, use local planning results
        for (size_t i = 0; i < local_path.poses.size() && i < refined_path.poses.size(); ++i) {
            refined_path.poses[i] = local_path.poses[i];
        }

        return refined_path;
    }

    std::vector<PlanningLevel> planning_levels_;
    std::map<std::string, std::chrono::steady_clock::time_point> last_execution_time_;
    std::map<std::string, geometry_msgs::msg::Point> last_goals_;

    // Planner instances
    GPUAStarPlanner gpu_global_planner_;
    MidLevelPlanner mid_level_planner_;
    LocalPlanner local_planner_;
};
```

## Next Steps

In the next section, we'll explore AI model deployment and execution in Isaac ROS, learning how to leverage NVIDIA's AI frameworks for perception, planning, and control tasks in humanoid robotics applications.