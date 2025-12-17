---
sidebar_position: 3
title: "Perception Systems"
---

# Perception Systems

## Introduction to Perception in Humanoid Robotics

Perception systems are the eyes and ears of humanoid robots, enabling them to understand and interact with their environment. In humanoid robotics, perception systems must process multiple sensor modalities in real-time to provide accurate information about objects, people, obstacles, and navigable spaces. NVIDIA Isaac™ provides powerful tools for building sophisticated perception systems that leverage GPU acceleration for real-time performance.

## Sensor Modalities for Humanoid Robots

### 1. Vision Sensors

#### RGB Cameras
RGB cameras provide color information essential for object recognition, human detection, and scene understanding:

```cpp
// Isaac ROS RGB camera processing pipeline
#include "isaac_ros_image_pipeline/rectification_node.hpp"
#include "isaac_ros_detect_net/detect_net_node.hpp"

class RGBPerceptionPipeline
{
public:
    RGBPerceptionPipeline(rclcpp::Node* node)
        : node_(node),
          image_rectifier_("rectification_node"),
          object_detector_("detectnet", "ssd-mobilenet-v2", 0.5, true)
    {
        // Initialize CUDA streams for parallel processing
        cudaStreamCreate(&processing_stream_);
        cudaStreamCreate(&detection_stream_);
    }

    ~RGBPerceptionPipeline()
    {
        cudaStreamDestroy(processing_stream_);
        cudaStreamDestroy(detection_stream_);
    }

    PerceptionResult ProcessRGBImage(const sensor_msgs::msg::Image::SharedPtr& image)
    {
        PerceptionResult result;

        // Rectify image using GPU acceleration
        auto rectified_image = image_rectifier_.Process(image, processing_stream_);

        // Run object detection
        auto detections = object_detector_.Detect(rectified_image, detection_stream_);

        // Process detections
        result.objects = ProcessDetections(detections);
        result.humans = FilterHumanDetections(result.objects);
        result.navigation_objects = FilterNavigationRelevant(result.objects);

        // Calculate processing time
        cudaStreamSynchronize(processing_stream_);
        result.processing_time = GetElapsedTime();

        return result;
    }

private:
    struct PerceptionResult {
        std::vector<ObjectDetection> objects;
        std::vector<ObjectDetection> humans;
        std::vector<ObjectDetection> navigation_objects;
        double processing_time;
    };

    std::vector<ObjectDetection> ProcessDetections(
        const std::vector<vision_msgs::msg::Detection2D>& detections)
    {
        std::vector<ObjectDetection> processed_detections;

        for (const auto& detection : detections) {
            ObjectDetection obj;
            obj.class_name = GetClassName(detection.results[0].id);
            obj.confidence = detection.results[0].score;
            obj.bounding_box = detection.bbox;
            obj.center = CalculateCenter(detection.bbox);

            processed_detections.push_back(obj);
        }

        return processed_detections;
    }

    std::vector<ObjectDetection> FilterHumanDetections(
        const std::vector<ObjectDetection>& all_detections)
    {
        std::vector<ObjectDetection> humans;

        for (const auto& detection : all_detections) {
            if (IsHumanClass(detection.class_name) && detection.confidence > 0.7) {
                humans.push_back(detection);
            }
        }

        return humans;
    }

    rclcpp::Node* node_;
    isaac_ros::image_pipeline::RectificationNode image_rectifier_;
    isaac_ros::dnn_inference::DetectNetNode object_detector_;
    cudaStream_t processing_stream_;
    cudaStream_t detection_stream_;
};
```

#### RGB-D Cameras
RGB-D cameras provide both color and depth information:

```cpp
// RGB-D processing for 3D scene understanding
#include "image_geometry/pinhole_camera_model.hpp"
#include "tf2_ros/transform_listener.h"

class RGBDPerception
{
public:
    RGBDPerception(rclcpp::Node* node) : node_(node), tf_buffer_(node->get_clock())
    {
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(tf_buffer_);
        camera_model_.fromCameraInfo(camera_info_msg_);
    }

    SceneUnderstandingResult ProcessRGBD(const sensor_msgs::msg::Image::SharedPtr& color_image,
                                        const sensor_msgs::msg::Image::SharedPtr& depth_image)
    {
        SceneUnderstandingResult result;

        // Convert depth image to point cloud
        auto point_cloud = ConvertToPointCloud(color_image, depth_image);

        // Segment objects in 3D space
        result.segmented_objects = SegmentObjects(point_cloud);

        // Estimate surfaces and planes
        result.surfaces = EstimateSurfaces(point_cloud);

        // Calculate object poses in 3D
        result.object_poses = CalculateObjectPoses(result.segmented_objects);

        return result;
    }

private:
    struct SceneUnderstandingResult {
        std::vector<PointCloudSegment> segmented_objects;
        std::vector<Surface> surfaces;
        std::vector<ObjectPose> object_poses;
    };

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ConvertToPointCloud(
        const sensor_msgs::msg::Image::SharedPtr& color_image,
        const sensor_msgs::msg::Image::SharedPtr& depth_image)
    {
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        cloud->width = color_image->width;
        cloud->height = color_image->height;
        cloud->is_dense = false;
        cloud->points.resize(cloud->width * cloud->height);

        // Use camera model to convert 2D image coordinates to 3D points
        for (size_t v = 0; v < color_image->height; ++v) {
            for (size_t u = 0; u < color_image->width; ++u) {
                // Get depth value
                float depth = GetDepthValue(depth_image, u, v);

                if (depth > 0) {
                    // Convert to 3D point using camera model
                    cv::Point3d point3d = camera_model_.projectPixelTo3d(cv::Point2d(u, v), depth);

                    pcl::PointXYZRGB& point = cloud->points[v * cloud->width + u];
                    point.x = point3d.x;
                    point.y = point3d.y;
                    point.z = point3d.z;

                    // Get color value
                    size_t pixel_idx = v * color_image->step + u * 3;
                    if (pixel_idx + 2 < color_image->data.size()) {
                        point.r = color_image->data[pixel_idx];
                        point.g = color_image->data[pixel_idx + 1];
                        point.b = color_image->data[pixel_idx + 2];
                    }
                }
            }
        }

        return cloud;
    }

    std::vector<PointCloudSegment> SegmentObjects(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud)
    {
        // Use GPU-accelerated segmentation algorithms
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

        // Set segmentation parameters
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.02); // 2cm threshold

        std::vector<PointCloudSegment> segments;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered = cloud;

        // Extract planar surfaces first (floor, walls, tables)
        while (cloud_filtered->size() > 0.3 * cloud->size()) {
            seg.setInputCloud(cloud_filtered);
            seg.segment(*inliers, *coefficients);

            if (inliers->indices.size() == 0) {
                break; // No more planes
            }

            // Extract inliers as a surface
            PointCloudSegment surface;
            surface.type = PointCloudSegment::PLANE;
            surface.indices = inliers->indices;
            surface.coefficients = *coefficients;
            segments.push_back(surface);

            // Remove inliers from cloud
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_new(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::ExtractIndices<pcl::PointXYZRGB> extract;
            extract.setInputCloud(cloud_filtered);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*cloud_new);
            cloud_filtered = cloud_new;
        }

        // Segment remaining objects using clustering
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;

        ec.setClusterTolerance(0.02); // 2cm
        ec.setMinClusterSize(100);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_filtered);
        ec.extract(cluster_indices);

        for (const auto& cluster : cluster_indices) {
            PointCloudSegment object;
            object.type = PointCloudSegment::OBJECT;
            object.indices = cluster.indices;
            segments.push_back(object);
        }

        return segments;
    }

    struct PointCloudSegment {
        enum Type { PLANE, OBJECT };
        Type type;
        std::vector<int> indices;
        pcl::ModelCoefficients coefficients; // For planes
    };

    rclcpp::Node* node_;
    tf2_ros::Buffer tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    image_geometry::PinholeCameraModel camera_model_;
};
```

### 2. Depth Sensors

#### Stereo Cameras
Stereo cameras provide depth information through triangulation:

```cpp
// Isaac ROS stereo processing
#include "isaac_ros_stereo_image_proc/stereo_disparity_node.hpp"

class StereoPerception
{
public:
    StereoPerception(rclcpp::Node* node) : node_(node)
    {
        // Initialize stereo processing with GPU acceleration
        stereo_node_ = std::make_shared<isaac_ros::stereo_image_proc::DisparityNode>("stereo_node");
    }

    DepthResult ProcessStereo(const sensor_msgs::msg::Image::SharedPtr& left_image,
                             const sensor_msgs::msg::Image::SharedPtr& right_image)
    {
        DepthResult result;

        // Compute disparity map using GPU acceleration
        auto disparity_msg = stereo_node_->ComputeDisparity(left_image, right_image);

        // Convert to depth image
        result.depth_image = ConvertDisparityToDepth(*disparity_msg);

        // Extract depth features
        result.obstacles = DetectObstacles(result.depth_image);
        result.ground_plane = EstimateGroundPlane(result.depth_image);
        result.navigable_areas = ComputeNavigableAreas(result.depth_image);

        return result;
    }

private:
    struct DepthResult {
        sensor_msgs::msg::Image::SharedPtr depth_image;
        std::vector<Obstacle> obstacles;
        GroundPlane ground_plane;
        std::vector<NavigableArea> navigable_areas;
    };

    sensor_msgs::msg::Image::SharedPtr ConvertDisparityToDepth(
        const stereo_msgs::msg::DisparityImage& disparity_msg)
    {
        // Convert disparity to depth using camera parameters
        auto depth_image = std::make_shared<sensor_msgs::msg::Image>();
        depth_image->header = disparity_msg.header;
        depth_image->height = disparity_msg.image.height;
        depth_image->width = disparity_msg.image.width;
        depth_image->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        depth_image->is_bigendian = 0;
        depth_image->step = depth_image->width * sizeof(float);

        // Allocate memory for depth data
        depth_image->data.resize(depth_image->height * depth_image->step);

        // Convert disparity to depth
        float* depth_data = reinterpret_cast<float*>(depth_image->data.data());
        const uint8_t* disparity_data = disparity_msg.image.data.data();

        float focal_length = GetFocalLength(); // From camera info
        float baseline = GetBaseline();       // From stereo calibration

        for (size_t i = 0; i < depth_image->height * depth_image->width; ++i) {
            float disparity = reinterpret_cast<const float*>(disparity_data)[i];
            if (disparity > 0) {
                depth_data[i] = (baseline * focal_length) / disparity;
            } else {
                depth_data[i] = 0.0f; // Invalid depth
            }
        }

        return depth_image;
    }

    std::vector<Obstacle> DetectObstacles(const sensor_msgs::msg::Image::SharedPtr& depth_image)
    {
        std::vector<Obstacle> obstacles;

        // Process depth image to detect obstacles
        const float* depth_data = reinterpret_cast<const float*>(depth_image->data.data());
        float robot_height = 1.5f; // Example robot height
        float obstacle_threshold = 0.5f; // Minimum obstacle height

        // Simple obstacle detection: find depth discontinuities
        for (size_t y = 0; y < depth_image->height; y += 4) { // Downsample for efficiency
            for (size_t x = 0; x < depth_image->width; x += 4) {
                size_t idx = y * depth_image->width + x;
                float depth = depth_data[idx];

                if (depth > 0 && depth < 3.0f) { // Valid depth in range
                    // Check for height discontinuities that indicate obstacles
                    float neighbor_depth = GetMedianNeighborDepth(depth_data, x, y, depth_image->width, depth_image->height);

                    if (std::abs(depth - neighbor_depth) > obstacle_threshold) {
                        Obstacle obstacle;
                        obstacle.position.x = x;
                        obstacle.position.y = y;
                        obstacle.depth = depth;
                        obstacle.size = EstimateObstacleSize(depth_data, x, y);
                        obstacles.push_back(obstacle);
                    }
                }
            }
        }

        return obstacles;
    }

    float GetMedianNeighborDepth(const float* depth_data, size_t x, size_t y, size_t width, size_t height)
    {
        std::vector<float> neighbors;

        // Check 3x3 neighborhood
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue; // Skip center

                int nx = static_cast<int>(x) + dx;
                int ny = static_cast<int>(y) + dy;

                if (nx >= 0 && nx < static_cast<int>(width) &&
                    ny >= 0 && ny < static_cast<int>(height)) {
                    size_t idx = ny * width + nx;
                    if (depth_data[idx] > 0) {
                        neighbors.push_back(depth_data[idx]);
                    }
                }
            }
        }

        if (!neighbors.empty()) {
            std::sort(neighbors.begin(), neighbors.end());
            return neighbors[neighbors.size() / 2]; // Median
        }

        return 0.0f;
    }

    struct Obstacle {
        geometry_msgs::msg::Point position;
        float depth;
        float size;
    };

    struct GroundPlane {
        Eigen::Vector4f coefficients; // ax + by + cz + d = 0
        float confidence;
    };

    struct NavigableArea {
        std::vector<geometry_msgs::msg::Point> boundary;
        float clearance;
    };

    std::shared_ptr<isaac_ros::stereo_image_proc::DisparityNode> stereo_node_;
    rclcpp::Node* node_;
};
```

#### LiDAR Sensors
LiDAR provides accurate 3D measurements:

```cpp
// LiDAR processing for humanoid navigation
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "pcl_conversions/pcl_conversions.h"

class LidarPerception
{
public:
    LidarPerception(rclcpp::Node* node) : node_(node)
    {
        // Initialize GPU-accelerated LiDAR processing
        InitializeGPUProcessing();
    }

    LidarResult ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& pointcloud_msg)
    {
        LidarResult result;

        // Convert ROS message to PCL
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*pointcloud_msg, pcl_pc2);

        // Convert to organized point cloud for efficient processing
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromPCLPointCloud2(pcl_pc2, *cloud);

        // Segment ground plane
        result.ground_points = ExtractGroundPlane(cloud);
        result.obstacle_points = ExtractObstacles(cloud, result.ground_points);

        // Cluster obstacles
        result.obstacle_clusters = ClusterObstacles(result.obstacle_points);

        // Compute free space
        result.free_space_map = ComputeFreeSpaceMap(cloud);

        return result;
    }

private:
    struct LidarResult {
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground_points;
        pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_points;
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> obstacle_clusters;
        OccupancyGrid free_space_map;
    };

    pcl::PointCloud<pcl::PointXYZ>::Ptr ExtractGroundPlane(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
    {
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

        // Configure ground plane segmentation
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.05); // 5cm threshold
        seg.setAxis(Eigen::Vector3f(0, 0, 1)); // Expect plane perpendicular to Z-axis
        seg.setEpsAngle(15.0f * (M_PI / 180.0f)); // 15 degrees tolerance

        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        // Extract ground points
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*ground_cloud);

        return ground_cloud;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr ExtractObstacles(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& ground_cloud)
    {
        // Remove ground points from full cloud to get obstacles
        pcl::PointIndices::Ptr ground_indices(new pcl::PointIndices);

        // Find ground indices in original cloud
        std::vector<bool> is_ground(cloud->size(), false);
        for (const auto& point : ground_cloud->points) {
            // Find closest point in original cloud
            float min_dist = std::numeric_limits<float>::max();
            int closest_idx = -1;

            for (size_t i = 0; i < cloud->size(); ++i) {
                float dist = sqrt(pow(cloud->points[i].x - point.x, 2) +
                                pow(cloud->points[i].y - point.y, 2) +
                                pow(cloud->points[i].z - point.z, 2));
                if (dist < min_dist) {
                    min_dist = dist;
                    if (dist < 0.05) { // Ground point threshold
                        closest_idx = i;
                    }
                }
            }

            if (closest_idx >= 0) {
                is_ground[closest_idx] = true;
            }
        }

        // Extract non-ground points (obstacles)
        pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (size_t i = 0; i < cloud->size(); ++i) {
            if (!is_ground[i]) {
                obstacle_cloud->points.push_back(cloud->points[i]);
            }
        }

        return obstacle_cloud;
    }

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> ClusterObstacles(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& obstacle_cloud)
    {
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;

        // Use Euclidean clustering to group nearby points
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(obstacle_cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

        ec.setClusterTolerance(0.1); // 10cm
        ec.setMinClusterSize(50);    // Minimum 50 points per cluster
        ec.setMaxClusterSize(25000); // Maximum 25000 points per cluster
        ec.setSearchMethod(tree);
        ec.setInputCloud(obstacle_cloud);
        ec.extract(cluster_indices);

        // Create point clouds for each cluster
        for (const auto& cluster : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);

            for (const auto& idx : cluster.indices) {
                cluster_cloud->points.push_back(obstacle_cloud->points[idx]);
            }

            cluster_cloud->width = cluster_cloud->points.size();
            cluster_cloud->height = 1;
            cluster_cloud->is_dense = true;

            clusters.push_back(cluster_cloud);
        }

        return clusters;
    }

    struct OccupancyGrid {
        std::vector<std::vector<float>> grid; // 2D occupancy grid
        float resolution; // meters per cell
        geometry_msgs::msg::Point origin; // grid origin in world coordinates
    };

    OccupancyGrid ComputeFreeSpaceMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
    {
        OccupancyGrid grid;
        grid.resolution = 0.1f; // 10cm resolution
        grid.grid.resize(100, std::vector<float>(100, 0.5f)); // Initialize to unknown

        // Project 3D points to 2D occupancy grid
        float origin_x = -5.0f; // -5m to +5m in X
        float origin_y = -5.0f; // -5m to +5m in Y

        for (const auto& point : cloud->points) {
            // Convert world coordinates to grid coordinates
            int grid_x = static_cast<int>((point.x - origin_x) / grid.resolution);
            int grid_y = static_cast<int>((point.y - origin_y) / grid.resolution);

            // Check bounds
            if (grid_x >= 0 && grid_x < 100 && grid_y >= 0 && grid_y < 100) {
                // Mark as occupied if point is an obstacle (not on ground)
                if (point.z > 0.1f) { // Above ground threshold
                    grid.grid[grid_x][grid_y] = 1.0f; // Occupied
                } else {
                    grid.grid[grid_x][grid_y] = 0.0f; // Free space
                }
            }
        }

        grid.origin.x = origin_x;
        grid.origin.y = origin_y;

        return grid;
    }

    void InitializeGPUProcessing()
    {
        // Initialize GPU resources for LiDAR processing
        // This would include setting up CUDA kernels for point cloud operations
    }

    rclcpp::Node* node_;
};
```

### 3. Inertial and Proprioceptive Sensors

#### IMU Integration
IMU sensors provide orientation and motion information:

```cpp
// IMU processing for humanoid balance and motion
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"

class IMUPerception
{
public:
    IMUPerception(rclcpp::Node* node) : node_(node)
    {
        // Initialize orientation filter
        InitializeOrientationFilter();
    }

    IMUState ProcessIMU(const sensor_msgs::msg::Imu::SharedPtr& imu_msg)
    {
        IMUState state;

        // Update orientation using complementary filter
        state.orientation = UpdateOrientation(imu_msg);

        // Calculate angular velocity
        state.angular_velocity = imu_msg->angular_velocity;

        // Calculate linear acceleration (remove gravity)
        state.linear_acceleration = RemoveGravity(imu_msg->linear_acceleration, state.orientation);

        // Detect motion events
        state.motion_events = DetectMotionEvents(state);

        return state;
    }

private:
    struct IMUState {
        geometry_msgs::msg::Quaternion orientation;
        geometry_msgs::msg::Vector3 angular_velocity;
        geometry_msgs::msg::Vector3 linear_acceleration;
        std::vector<MotionEvent> motion_events;
    };

    struct MotionEvent {
        enum Type { GYROSCOPE_DRIFT, ACCELEROMETER_SPIKE, ORIENTATION_JUMP };
        Type type;
        rclcpp::Time timestamp;
        float magnitude;
    };

    geometry_msgs::msg::Quaternion UpdateOrientation(const sensor_msgs::msg::Imu::SharedPtr& imu_msg)
    {
        // Use a complementary filter to combine gyroscope and accelerometer data
        // This helps reduce drift from gyroscope integration while maintaining
        // responsiveness to rapid orientation changes

        rclcpp::Time current_time = imu_msg->header.stamp;
        double dt = (current_time - last_update_time_).seconds();

        if (dt > 0 && dt < 0.1) { // Valid time difference
            // Integrate gyroscope data for orientation change
            geometry_msgs::msg::Quaternion delta_orientation = IntegrateGyroscope(
                imu_msg->angular_velocity, dt);

            // Apply complementary filter
            geometry_msgs::msg::Quaternion accel_orientation = GetOrientationFromAccelerometer(
                imu_msg->linear_acceleration);

            // Complementary filter: blend gyroscope integration with accelerometer
            double alpha = 0.98; // Trust gyroscope more (reduce drift)
            current_orientation_ = Slerp(current_orientation_, delta_orientation, alpha);

            // Apply small correction from accelerometer
            geometry_msgs::msg::Quaternion corrected_orientation = Slerp(
                current_orientation_, accel_orientation, 1.0 - alpha);
            current_orientation_ = corrected_orientation;
        }

        last_update_time_ = current_time;
        return current_orientation_;
    }

    geometry_msgs::msg::Quaternion IntegrateGyroscope(
        const geometry_msgs::msg::Vector3& angular_velocity, double dt)
    {
        // Convert angular velocity to quaternion derivative
        double magnitude = sqrt(angular_velocity.x * angular_velocity.x +
                               angular_velocity.y * angular_velocity.y +
                               angular_velocity.z * angular_velocity.z);

        if (magnitude > 1e-6) { // Avoid division by zero
            double half_angle = magnitude * dt * 0.5;
            double sin_half = sin(half_angle);
            double cos_half = cos(half_angle);

            geometry_msgs::msg::Quaternion q;
            q.w = cos_half;
            q.x = (angular_velocity.x / magnitude) * sin_half;
            q.y = (angular_velocity.y / magnitude) * sin_half;
            q.z = (angular_velocity.z / magnitude) * sin_half;

            return q;
        }

        // No rotation
        geometry_msgs::msg::Quaternion q;
        q.w = 1.0;
        q.x = 0.0;
        q.y = 0.0;
        q.z = 0.0;
        return q;
    }

    geometry_msgs::msg::Quaternion GetOrientationFromAccelerometer(
        const geometry_msgs::msg::Vector3& linear_acceleration)
    {
        // Calculate orientation from accelerometer data
        // Assumes the robot is not accelerating (only gravity)

        // Normalize acceleration vector
        double norm = sqrt(linear_acceleration.x * linear_acceleration.x +
                          linear_acceleration.y * linear_acceleration.y +
                          linear_acceleration.z * linear_acceleration.z);

        if (norm > 1e-6) {
            double ax = linear_acceleration.x / norm;
            double ay = linear_acceleration.y / norm;
            double az = linear_acceleration.z / norm;

            // Calculate roll and pitch from accelerometer
            double roll = atan2(ay, az);
            double pitch = atan2(-ax, sqrt(ay * ay + az * az));

            // Convert to quaternion (assuming no yaw from accelerometer)
            double cy = 1.0;
            double sy = 0.0;
            double cp = cos(pitch * 0.5);
            double sp = sin(pitch * 0.5);
            double cr = cos(roll * 0.5);
            double sr = sin(roll * 0.5);

            geometry_msgs::msg::Quaternion q;
            q.w = cy * cp * cr + sy * sp * sr;
            q.x = cy * cp * sr - sy * sp * cr;
            q.y = sy * cp * sr + cy * sp * cr;
            q.z = sy * cp * cr - cy * sp * sr;

            return q;
        }

        // Default orientation
        geometry_msgs::msg::Quaternion q;
        q.w = 1.0;
        q.x = 0.0;
        q.y = 0.0;
        q.z = 0.0;
        return q;
    }

    geometry_msgs::msg::Vector3 RemoveGravity(
        const geometry_msgs::msg::Vector3& linear_acceleration,
        const geometry_msgs::msg::Quaternion& orientation)
    {
        // Transform gravity vector to body frame and subtract from acceleration
        Eigen::Quaterniond q(orientation.w, orientation.x, orientation.y, orientation.z);
        Eigen::Vector3d gravity_body = q.conjugate() * Eigen::Vector3d(0, 0, -9.81);

        geometry_msgs::msg::Vector3 corrected_acceleration;
        corrected_acceleration.x = linear_acceleration.x - gravity_body.x();
        corrected_acceleration.y = linear_acceleration.y - gravity_body.y();
        corrected_acceleration.z = linear_acceleration.z - gravity_body.z();

        return corrected_acceleration;
    }

    std::vector<MotionEvent> DetectMotionEvents(const IMUState& state)
    {
        std::vector<MotionEvent> events;

        // Detect gyroscope drift
        if (state.angular_velocity.x > gyroscope_drift_threshold_ ||
            state.angular_velocity.y > gyroscope_drift_threshold_ ||
            state.angular_velocity.z > gyroscope_drift_threshold_) {
            MotionEvent event;
            event.type = MotionEvent::GYROSCOPE_DRIFT;
            event.magnitude = std::max({abs(state.angular_velocity.x),
                                      abs(state.angular_velocity.y),
                                      abs(state.angular_velocity.z)});
            event.timestamp = node_->now();
            events.push_back(event);
        }

        // Detect accelerometer spikes
        double linear_accel_magnitude = sqrt(state.linear_acceleration.x * state.linear_acceleration.x +
                                           state.linear_acceleration.y * state.linear_acceleration.y +
                                           state.linear_acceleration.z * state.linear_acceleration.z);
        if (linear_accel_magnitude > accelerometer_spike_threshold_) {
            MotionEvent event;
            event.type = MotionEvent::ACCELEROMETER_SPIKE;
            event.magnitude = linear_accel_magnitude;
            event.timestamp = node_->now();
            events.push_back(event);
        }

        return events;
    }

    geometry_msgs::msg::Quaternion Slerp(const geometry_msgs::msg::Quaternion& q1,
                                        const geometry_msgs::msg::Quaternion& q2,
                                        double t)
    {
        // Spherical linear interpolation between quaternions
        Eigen::Quaterniond eq1(q1.w, q1.x, q1.y, q1.z);
        Eigen::Quaterniond eq2(q2.w, q2.x, q2.y, q2.z);

        Eigen::Quaterniond result = eq1.slerp(t, eq2);

        geometry_msgs::msg::Quaternion q;
        q.w = result.w();
        q.x = result.x();
        q.y = result.y();
        q.z = result.z();
        return q;
    }

    void InitializeOrientationFilter()
    {
        current_orientation_.w = 1.0;
        current_orientation_.x = 0.0;
        current_orientation_.y = 0.0;
        current_orientation_.z = 0.0;
        last_update_time_ = node_->now();
    }

    geometry_msgs::msg::Quaternion current_orientation_;
    rclcpp::Time last_update_time_;
    double gyroscope_drift_threshold_ = 0.1; // rad/s
    double accelerometer_spike_threshold_ = 15.0; // m/s²
    rclcpp::Node* node_;
};
```

## Multi-Sensor Fusion

### 1. Kalman Filtering for Sensor Fusion

```cpp
// Extended Kalman Filter for multi-sensor fusion
#include <Eigen/Dense>

class SensorFusionEKF
{
public:
    SensorFusionEKF()
    {
        // Initialize state vector: [x, y, z, vx, vy, vz, qx, qy, qz, qw]
        state_ = Eigen::VectorXd::Zero(10);
        state_(6) = 0; // Initialize quaternion (will be normalized)
        state_(7) = 0;
        state_(8) = 0;
        state_(9) = 1;

        // Initialize covariance matrix
        covariance_ = Eigen::MatrixXd::Identity(10, 10) * 1.0;

        // Process noise
        process_noise_ = Eigen::MatrixXd::Identity(10, 10);
        process_noise_.block<3,3>(0,0) *= 0.1;   // Position process noise
        process_noise_.block<3,3>(3,3) *= 0.5;   // Velocity process noise
        process_noise_.block<4,4>(6,6) *= 0.01;  // Orientation process noise
    }

    RobotState FuseSensors(const std::vector<SensorMeasurement>& measurements)
    {
        // Prediction step
        PredictState();

        // Update step for each sensor measurement
        for (const auto& measurement : measurements) {
            UpdateState(measurement);
        }

        return ExtractRobotState();
    }

private:
    struct RobotState {
        Eigen::Vector3d position;
        Eigen::Vector3d velocity;
        Eigen::Quaterniond orientation;
        Eigen::Vector3d angular_velocity;
    };

    struct SensorMeasurement {
        enum Type { CAMERA, LIDAR, IMU, ENCODER };
        Type type;
        Eigen::VectorXd measurement;
        Eigen::MatrixXd measurement_noise;
        rclcpp::Time timestamp;
    };

    void PredictState()
    {
        // State transition model: constant velocity with rotation
        double dt = 0.01; // 100Hz prediction

        // Extract state components
        Eigen::Vector3d pos = state_.segment<3>(0);
        Eigen::Vector3d vel = state_.segment<3>(3);
        Eigen::Vector4d quat = state_.segment<4>(6);
        Eigen::Quaterniond orientation(quat(3), quat(0), quat(1), quat(2));

        // Predict new state
        Eigen::VectorXd predicted_state = state_;

        // Position prediction: x_new = x + v*dt
        predicted_state.segment<3>(0) = pos + vel * dt;

        // Velocity prediction: assume constant (could include control input)
        // For now, assume velocity remains the same

        // Orientation prediction: integrate angular velocity
        // This is a simplified approach - in practice, use proper quaternion integration
        Eigen::Vector3d angular_vel(0, 0, 0); // Would come from IMU
        Eigen::Quaterniond delta_q(1, angular_vel.x() * dt / 2,
                                  angular_vel.y() * dt / 2,
                                  angular_vel.z() * dt / 2);
        delta_q.normalize();
        Eigen::Quaterniond new_orientation = orientation * delta_q;
        predicted_state.segment<4>(6) << new_orientation.x(), new_orientation.y(),
                                       new_orientation.z(), new_orientation.w();

        state_ = predicted_state;

        // Update covariance: P = F*P*F^T + Q
        Eigen::MatrixXd jacobian = ComputeStateJacobian(dt);
        covariance_ = jacobian * covariance_ * jacobian.transpose() + process_noise_;
    }

    Eigen::MatrixXd ComputeStateJacobian(double dt)
    {
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(10, 10);

        // Position-velocity relationship: dx/dt = v
        F.block<3,3>(0, 3) = Eigen::Matrix3d::Identity() * dt;

        // For a more accurate model, we would include the effects of rotation
        // on the state transition, but this is a simplified version

        return F;
    }

    void UpdateState(const SensorMeasurement& measurement)
    {
        // Compute measurement model Jacobian
        Eigen::MatrixXd H = ComputeMeasurementJacobian(measurement.type);

        // Compute Kalman gain
        Eigen::MatrixXd S = H * covariance_ * H.transpose() + measurement.measurement_noise;
        Eigen::MatrixXd K = covariance_ * H.transpose() * S.inverse();

        // Compute innovation
        Eigen::VectorXd predicted_measurement = PredictMeasurement(measurement.type);
        Eigen::VectorXd innovation = measurement.measurement - predicted_measurement;

        // Update state
        state_ = state_ + K * innovation;

        // Update covariance
        covariance_ = (Eigen::MatrixXd::Identity(10, 10) - K * H) * covariance_;
    }

    Eigen::VectorXd PredictMeasurement(SensorMeasurement::Type sensor_type)
    {
        Eigen::VectorXd predicted_measurement;

        switch (sensor_type) {
            case SensorMeasurement::CAMERA: {
                // Predict 2D image coordinates from 3D position
                // This would involve camera projection model
                predicted_measurement = Eigen::VectorXd::Zero(2); // [u, v]
                break;
            }
            case SensorMeasurement::LIDAR: {
                // Predict range measurements
                predicted_measurement = Eigen::VectorXd::Zero(1); // range
                break;
            }
            case SensorMeasurement::IMU: {
                // Predict IMU readings from state
                predicted_measurement = Eigen::VectorXd::Zero(6); // [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
                break;
            }
            case SensorMeasurement::ENCODER: {
                // Predict joint positions
                predicted_measurement = Eigen::VectorXd::Zero(28); // Example for 28 joints
                break;
            }
        }

        return predicted_measurement;
    }

    Eigen::MatrixXd ComputeMeasurementJacobian(SensorMeasurement::Type sensor_type)
    {
        // Return appropriate Jacobian based on sensor type
        switch (sensor_type) {
            case SensorMeasurement::CAMERA:
                return Eigen::MatrixXd::Zero(2, 10); // 2 measurements x 10 state
            case SensorMeasurement::LIDAR:
                return Eigen::MatrixXd::Zero(1, 10);
            case SensorMeasurement::IMU:
                return Eigen::MatrixXd::Zero(6, 10);
            case SensorMeasurement::ENCODER:
                return Eigen::MatrixXd::Zero(28, 10);
        }

        return Eigen::MatrixXd::Zero(1, 1);
    }

    RobotState ExtractRobotState()
    {
        RobotState robot_state;
        robot_state.position = state_.segment<3>(0);
        robot_state.velocity = state_.segment<3>(3);
        Eigen::Vector4d quat = state_.segment<4>(6);
        robot_state.orientation = Eigen::Quaterniond(quat(3), quat(0), quat(1), quat(2));
        robot_state.orientation.normalize();

        return robot_state;
    }

    Eigen::VectorXd state_;
    Eigen::MatrixXd covariance_;
    Eigen::MatrixXd process_noise_;
};
```

### 2. GPU-Accelerated Fusion

```cpp
// GPU-accelerated sensor fusion using Isaac ROS
#include "isaac_ros_fusion_interfaces/msg/pose_with_covariance_array_stamped.hpp"

class GPUSensorFusion
{
public:
    GPUSensorFusion()
    {
        // Initialize GPU memory for fusion operations
        InitializeGPUResources();
    }

    ~GPUSensorFusion()
    {
        CleanupGPUResources();
    }

    RobotPoseWithCovarianceArray FuseSensorsGPU(const std::vector<SensorDataGPU>& sensor_data)
    {
        RobotPoseWithCovarianceArray fused_result;

        // Copy sensor data to GPU
        CopySensorDataToGPU(sensor_data);

        // Execute fusion kernel
        LaunchFusionKernel();

        // Copy result back to CPU
        fused_result = CopyResultFromGPU();

        return fused_result;
    }

private:
    struct SensorDataGPU {
        float* data_ptr;
        size_t size;
        SensorType type;
        cudaStream_t stream;
    };

    struct RobotPoseWithCovarianceArray {
        std::vector<PoseWithCovariance> poses;
        rclcpp::Time timestamp;
    };

    struct PoseWithCovariance {
        geometry_msgs::msg::Pose pose;
        std::array<double, 36> covariance; // 6x6 covariance matrix
    };

    enum class SensorType {
        CAMERA, LIDAR, IMU, ENCODER, GPS
    };

    void InitializeGPUResources()
    {
        // Allocate GPU memory pools
        cudaMalloc(&sensor_data_buffer_, MAX_SENSOR_DATA_SIZE);
        cudaMalloc(&fusion_result_buffer_, MAX_FUSION_RESULT_SIZE);
        cudaMalloc(&covariance_buffer_, MAX_COVARIANCE_SIZE);

        // Create CUDA streams for parallel processing
        for (int i = 0; i < MAX_SENSOR_STREAMS; ++i) {
            cudaStreamCreate(&sensor_streams_[i]);
        }

        // Initialize fusion kernels
        InitializeFusionKernels();
    }

    void CopySensorDataToGPU(const std::vector<SensorDataGPU>& sensor_data)
    {
        for (size_t i = 0; i < sensor_data.size(); ++i) {
            cudaMemcpyAsync(
                sensor_data_buffer_ + i * SENSOR_DATA_STRIDE,
                sensor_data[i].data_ptr,
                sensor_data[i].size,
                cudaMemcpyHostToDevice,
                sensor_streams_[i % MAX_SENSOR_STREAMS]
            );
        }
    }

    void LaunchFusionKernel()
    {
        // Configure kernel launch parameters
        int block_size = 256;
        int grid_size = (MAX_SENSOR_DATA_SIZE / SENSOR_DATA_STRIDE + block_size - 1) / block_size;

        // Launch fusion kernel
        fusion_kernel<<<grid_size, block_size, 0, fusion_stream_>>>(
            sensor_data_buffer_,
            fusion_result_buffer_,
            covariance_buffer_,
            MAX_SENSOR_DATA_SIZE
        );

        // Synchronize fusion stream
        cudaStreamSynchronize(fusion_stream_);
    }

    RobotPoseWithCovarianceArray CopyResultFromGPU()
    {
        RobotPoseWithCovarianceArray result;

        // Copy fused result from GPU
        cudaMemcpy(
            &result_buffer_host_,
            fusion_result_buffer_,
            MAX_FUSION_RESULT_SIZE,
            cudaMemcpyDeviceToHost
        );

        // Parse result buffer into structured data
        result = ParseFusionResult(result_buffer_host_);

        return result;
    }

    void InitializeFusionKernels()
    {
        // Initialize CUDA streams
        cudaStreamCreate(&fusion_stream_);

        // Load or compile fusion kernels
        // This would typically involve loading PTX code or JIT compilation
    }

    void CleanupGPUResources()
    {
        cudaFree(sensor_data_buffer_);
        cudaFree(fusion_result_buffer_);
        cudaFree(covariance_buffer_);

        for (int i = 0; i < MAX_SENSOR_STREAMS; ++i) {
            cudaStreamDestroy(sensor_streams_[i]);
        }

        cudaStreamDestroy(fusion_stream_);
    }

    static constexpr size_t MAX_SENSOR_DATA_SIZE = 1024 * 1024; // 1MB
    static constexpr size_t MAX_FUSION_RESULT_SIZE = 64 * 1024; // 64KB
    static constexpr size_t MAX_COVARIANCE_SIZE = 36 * 8; // 6x6 double matrix
    static constexpr size_t SENSOR_DATA_STRIDE = 4096; // 4KB per sensor
    static constexpr int MAX_SENSOR_STREAMS = 8;

    float* sensor_data_buffer_;
    float* fusion_result_buffer_;
    float* covariance_buffer_;
    cudaStream_t sensor_streams_[MAX_SENSOR_STREAMS];
    cudaStream_t fusion_stream_;

    // Host buffer for results
    std::vector<float> result_buffer_host_;
};
```

## Real-time Performance Considerations

### 1. Pipeline Optimization

```cpp
// Optimized perception pipeline for real-time humanoid applications
class RealTimePerceptionPipeline
{
public:
    RealTimePerceptionPipeline(rclcpp::Node* node) : node_(node)
    {
        // Initialize processing pipeline with GPU streams
        InitializeProcessingPipeline();

        // Create processing threads
        CreateProcessingThreads();
    }

    void ProcessFrameAsync(const SensorFrame& frame)
    {
        // Asynchronously process different sensor modalities
        ProcessCameraAsync(frame.camera_data);
        ProcessLidarAsync(frame.lidar_data);
        ProcessIMUAsync(frame.imu_data);

        // Synchronize and fuse results
        FuseResultsAsync();
    }

private:
    struct SensorFrame {
        sensor_msgs::msg::Image::SharedPtr camera_data;
        sensor_msgs::msg::PointCloud2::SharedPtr lidar_data;
        sensor_msgs::msg::Imu::SharedPtr imu_data;
        rclcpp::Time timestamp;
    };

    struct ProcessingResult {
        std::vector<ObjectDetection> objects;
        pcl::PointCloud<pcl::PointXYZ>::Ptr obstacles;
        geometry_msgs::msg::Quaternion orientation;
        rclcpp::Time timestamp;
    };

    void InitializeProcessingPipeline()
    {
        // Create CUDA streams for parallel processing
        cudaStreamCreate(&camera_stream_);
        cudaStreamCreate(&lidar_stream_);
        cudaStreamCreate(&imu_stream_);
        cudaStreamCreate(&fusion_stream_);

        // Initialize processing nodes
        camera_processor_ = std::make_unique<IsaacCameraProcessor>(camera_stream_);
        lidar_processor_ = std::make_unique<IsaacLidarProcessor>(lidar_stream_);
        imu_processor_ = std::make_unique<IsaacIMUProcessor>(imu_stream_);
        fusion_processor_ = std::make_unique<IsaacFusionProcessor>(fusion_stream_);
    }

    void CreateProcessingThreads()
    {
        // Create dedicated threads for different processing tasks
        camera_thread_ = std::thread(&RealTimePerceptionPipeline::CameraProcessingLoop, this);
        lidar_thread_ = std::thread(&RealTimePerceptionPipeline::LidarProcessingLoop, this);
        fusion_thread_ = std::thread(&RealTimePerceptionPipeline::FusionProcessingLoop, this);
    }

    void CameraProcessingLoop()
    {
        while (rclcpp::ok()) {
            // Wait for camera data
            auto camera_data = GetNextCameraData();

            if (camera_data) {
                // Process with GPU acceleration
                auto camera_result = camera_processor_->ProcessAsync(camera_data);

                // Store result for fusion
                StoreCameraResult(camera_result);
            }
        }
    }

    void LidarProcessingLoop()
    {
        while (rclcpp::ok()) {
            // Wait for LiDAR data
            auto lidar_data = GetNextLidarData();

            if (lidar_data) {
                // Process with GPU acceleration
                auto lidar_result = lidar_processor_->ProcessAsync(lidar_data);

                // Store result for fusion
                StoreLidarResult(lidar_result);
            }
        }
    }

    void FusionProcessingLoop()
    {
        while (rclcpp::ok()) {
            // Check for synchronized data from all sensors
            auto sync_data = GetSynchronizedData();

            if (sync_data) {
                // Fuse sensor data
                auto fused_result = fusion_processor_->FuseAsync(
                    sync_data.camera_result,
                    sync_data.lidar_result,
                    sync_data.imu_result
                );

                // Publish fused perception result
                PublishFusedResult(fused_result);
            }

            // Sleep to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    std::unique_ptr<IsaacCameraProcessor> camera_processor_;
    std::unique_ptr<IsaacLidarProcessor> lidar_processor_;
    std::unique_ptr<IsaacIMUProcessor> imu_processor_;
    std::unique_ptr<IsaacFusionProcessor> fusion_processor_;

    cudaStream_t camera_stream_;
    cudaStream_t lidar_stream_;
    cudaStream_t imu_stream_;
    cudaStream_t fusion_stream_;

    std::thread camera_thread_;
    std::thread lidar_thread_;
    std::thread fusion_thread_;

    rclcpp::Node* node_;
};
```

## Quality Assurance and Validation

### 1. Perception Accuracy Testing

```cpp
// Perception validation and testing framework
class PerceptionValidator
{
public:
    struct ValidationMetrics {
        double object_detection_accuracy;
        double pose_estimation_error;
        double segmentation_iou;
        double processing_latency;
        double robustness_score;
    };

    ValidationMetrics ValidatePerception(const PerceptionResult& result,
                                       const GroundTruth& ground_truth)
    {
        ValidationMetrics metrics;

        // Object detection accuracy
        metrics.object_detection_accuracy = ComputeDetectionAccuracy(
            result.objects, ground_truth.objects);

        // Pose estimation error
        metrics.pose_estimation_error = ComputePoseError(
            result.poses, ground_truth.poses);

        // Segmentation IoU
        metrics.segmentation_iou = ComputeSegmentationIoU(
            result.segmentation, ground_truth.segmentation);

        // Processing latency
        metrics.processing_latency = ComputeProcessingLatency(result);

        // Robustness (performance under various conditions)
        metrics.robustness_score = ComputeRobustnessScore(result);

        return metrics;
    }

private:
    double ComputeDetectionAccuracy(const std::vector<ObjectDetection>& detections,
                                   const std::vector<ObjectDetection>& ground_truth)
    {
        // Compute mAP (mean Average Precision) or similar metric
        int true_positives = 0;
        int false_positives = 0;
        int false_negatives = 0;

        for (const auto& det : detections) {
            bool matched = false;
            for (const auto& gt : ground_truth) {
                if (ComputeIoU(det.bounding_box, gt.bounding_box) > 0.5) {
                    true_positives++;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                false_positives++;
            }
        }

        for (const auto& gt : ground_truth) {
            bool matched = false;
            for (const auto& det : detections) {
                if (ComputeIoU(det.bounding_box, gt.bounding_box) > 0.5) {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                false_negatives++;
            }
        }

        double precision = static_cast<double>(true_positives) /
                          (true_positives + false_positives);
        double recall = static_cast<double>(true_positives) /
                       (true_positives + false_negatives);

        return 2 * (precision * recall) / (precision + recall); // F1 score
    }

    double ComputeIoU(const BoundingBox& box1, const BoundingBox& box2)
    {
        // Calculate Intersection over Union
        double x1 = std::max(box1.x_offset, box2.x_offset);
        double y1 = std::max(box1.y_offset, box2.y_offset);
        double x2 = std::min(box1.x_offset + box1.width,
                            box2.x_offset + box2.width);
        double y2 = std::min(box1.y_offset + box1.height,
                            box2.y_offset + box2.height);

        if (x2 <= x1 || y2 <= y1) {
            return 0.0; // No intersection
        }

        double intersection = (x2 - x1) * (y2 - y1);
        double area1 = box1.width * box1.height;
        double area2 = box2.width * box2.height;
        double union_area = area1 + area2 - intersection;

        return intersection / union_area;
    }

    struct BoundingBox {
        double x_offset, y_offset, width, height;
    };

    struct ObjectDetection {
        std::string class_name;
        double confidence;
        BoundingBox bounding_box;
        geometry_msgs::msg::Point3D center;
    };

    struct GroundTruth {
        std::vector<ObjectDetection> objects;
        std::vector<ObjectPose> poses;
        cv::Mat segmentation;
    };

    struct ObjectPose {
        geometry_msgs::msg::Pose pose;
        std::string object_id;
    };

    struct PerceptionResult {
        std::vector<ObjectDetection> objects;
        std::vector<ObjectPose> poses;
        cv::Mat segmentation;
        rclcpp::Time processing_start_time;
        rclcpp::Time processing_end_time;
    };
};
```

## Next Steps

In the next section, we'll explore planning and navigation systems in humanoid robotics, learning how Isaac ROS enables sophisticated path planning, motion planning, and navigation capabilities that allow humanoid robots to move safely and efficiently in complex environments.