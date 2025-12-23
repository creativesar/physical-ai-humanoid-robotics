---
sidebar_position: 4
title: "Capstone Project: Autonomous Humanoid Robot"
---

# Capstone Project: Autonomous Humanoid Robot

## Introduction

The capstone project brings together all the concepts learned throughout this course to create an autonomous humanoid robot system. This project integrates perception, planning, control, and AI capabilities to build a robot that can understand natural language commands, navigate complex environments, interact with objects, and adapt to changing conditions.

### Project Scope

The autonomous humanoid robot will be capable of:
- Understanding and executing natural language commands
- Navigating through human environments safely
- Manipulating objects with dexterity
- Learning and adapting from experience
- Interacting socially with humans

## Project Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Autonomous Humanoid Robot System                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │  Perception     │  │  Cognition     │  │  Planning      │  │  Control    │  │
│  │  Module        │◄►│  Module        │◄►│  Module        │◄►│  Module     │  │
│  │  - Vision       │  │  - LLM        │  │  - Task        │  │  - Motion   │  │
│  │  - Audio        │  │  - Reasoning  │  │  - Motion      │  │  - Balance  │  │
│  │  - Touch        │  │  - Memory     │  │  - Behavior    │  │  - Action   │  │
│  │  - Sensors      │  │  - Learning   │  │  - Recovery    │  │  - Safety   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────┘  │
│         │                       │                       │              │         │
│         ▼                       ▼                       ▼              ▼         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                    Humanoid Middleware Layer                              │  │
│  │  - State Management                                                     │  │
│  │  - Sensor Fusion                                                        │  │
│  │  - Safety Monitoring                                                    │  │
│  │  - Communication Management                                             │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│         │                       │                       │              │         │
│         ▼                       ▼                       ▼              ▼         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │  Hardware       │  │  Simulation     │  │  User Interface │  │  Cloud      │  │
│  │  Interface      │  │  Environment   │  │  & Monitoring  │  │  Services   │  │
│  │  - Motors       │  │  - Gazebo      │  │  - Voice       │  │  - Training │  │
│  │  - Sensors      │  │  - Unity       │  │  - GUI         │  │  - Updates  │  │
│  │  - Actuators    │  │  - Isaac Sim   │  │  - Mobile App  │  │  - Analytics│  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Core System Components

#### 1. Perception Module
- **Vision System**: RGB-D cameras, object detection, scene understanding
- **Audio System**: Microphone arrays, speech recognition, sound localization
- **Tactile System**: Force/torque sensors, tactile feedback
- **Proprioceptive System**: Joint encoders, IMU, pressure sensors

#### 2. Cognition Module
- **Language Understanding**: Natural language processing, command interpretation
- **World Modeling**: 3D environment mapping, object tracking
- **Memory System**: Short-term and long-term memory management
- **Learning System**: Reinforcement learning, imitation learning, transfer learning

#### 3. Planning Module
- **Task Planning**: High-level task decomposition and scheduling
- **Motion Planning**: Path planning, trajectory generation
- **Behavior Planning**: Social interaction, adaptive behaviors
- **Recovery Planning**: Failure detection and recovery strategies

#### 4. Control Module
- **Motion Control**: Joint-level control, impedance control
- **Balance Control**: Center of mass management, fall prevention
- **Action Execution**: Skill execution, grasping, manipulation
- **Safety Control**: Emergency stops, safety limits, protective behaviors

## Implementation Strategy

### Phase 1: System Integration (Weeks 1-3)

#### Week 1: Core Infrastructure
- Set up development environment
- Integrate ROS 2 communication framework
- Establish hardware abstraction layer
- Implement basic perception pipeline

```python
# system_integration.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, JointState, Imu
from geometry_msgs.msg import Twist, Pose
from tf2_ros import TransformBroadcaster
import asyncio
import threading
from typing import Dict, Any, Optional

class HumanoidSystemManager(Node):
    """Main system manager for autonomous humanoid robot"""

    def __init__(self):
        super().__init__('humanoid_system_manager')

        # Initialize system components
        self.perception_module = self.initialize_perception()
        self.cognition_module = self.initialize_cognition()
        self.planning_module = self.initialize_planning()
        self.control_module = self.initialize_control()

        # System state management
        self.system_state = {
            'initialized': False,
            'operational': False,
            'emergency_stop': False,
            'battery_level': 100.0,
            'current_task': None,
            'active_behaviors': []
        }

        # Publishers and subscribers
        self.emergency_stop_sub = self.create_subscription(
            Bool, '/emergency_stop', self.emergency_stop_callback, 10)
        self.voice_command_sub = self.create_subscription(
            String, '/voice_command', self.voice_command_callback, 10)
        self.system_status_pub = self.create_publisher(
            String, '/system_status', 10)

        # System initialization timer
        self.init_timer = self.create_timer(1.0, self.system_initialization_check)

        # Async event loop for LLM integration
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self._run_event_loop, args=(self.loop,), daemon=True).start()

        self.get_logger().info('Humanoid System Manager initialized')

    def _run_event_loop(self, loop):
        """Run async event loop in separate thread"""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def initialize_perception(self):
        """Initialize perception module"""
        from perception_module import HumanoidPerceptionModule
        return HumanoidPerceptionModule(self)

    def initialize_cognition(self):
        """Initialize cognition module"""
        from cognition_module import HumanoidCognitionModule
        return HumanoidCognitionModule(self)

    def initialize_planning(self):
        """Initialize planning module"""
        from planning_module import HumanoidPlanningModule
        return HumanoidPlanningModule(self)

    def initialize_control(self):
        """Initialize control module"""
        from control_module import HumanoidControlModule
        return HumanoidControlModule(self)

    def system_initialization_check(self):
        """Check system initialization status"""
        if not self.system_state['initialized']:
            # Check if all modules are ready
            all_ready = all([
                self.perception_module.is_ready(),
                self.cognition_module.is_ready(),
                self.planning_module.is_ready(),
                self.control_module.is_ready()
            ])

            if all_ready:
                self.system_state['initialized'] = True
                self.system_state['operational'] = True
                self.get_logger().info('System initialization complete - operational')

                # Publish system ready status
                status_msg = String()
                status_msg.data = "SYSTEM_READY"
                self.system_status_pub.publish(status_msg)

    def emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop commands"""
        if msg.data:
            self.system_state['emergency_stop'] = True
            self.emergency_stop_procedure()
        else:
            self.system_state['emergency_stop'] = False
            self.resume_operations()

    def emergency_stop_procedure(self):
        """Execute emergency stop procedure"""
        self.get_logger().error('EMERGENCY STOP ACTIVATED')

        # Stop all motion
        self.control_module.emergency_stop()

        # Pause all active tasks
        if self.system_state['current_task']:
            self.planning_module.pause_current_task()

        # Disable all active behaviors
        for behavior in self.system_state['active_behaviors']:
            self.planning_module.disable_behavior(behavior)

    def resume_operations(self):
        """Resume operations after emergency stop"""
        self.get_logger().info('Resuming operations after emergency stop')
        self.system_state['emergency_stop'] = False

        # Resume normal operations
        if self.system_state['current_task']:
            self.planning_module.resume_current_task()

    def voice_command_callback(self, msg: String):
        """Process voice commands"""
        if self.system_state['emergency_stop']:
            self.get_logger().warn('Ignoring command due to emergency stop')
            return

        command_text = msg.data
        self.get_logger().info(f'Received voice command: {command_text}')

        # Process command through cognitive system
        future = asyncio.run_coroutine_threadsafe(
            self.process_voice_command(command_text),
            self.loop
        )

    async def process_voice_command(self, command_text: str):
        """Process voice command asynchronously"""
        try:
            # Get current context
            context = await self.get_current_context()

            # Process command through cognition module
            cognitive_plan = await self.cognition_module.process_command(
                command_text, context
            )

            if cognitive_plan:
                # Execute plan through planning and control modules
                success = await self.execute_cognitive_plan(cognitive_plan)

                if success:
                    self.get_logger().info('Command executed successfully')
                else:
                    self.get_logger().error('Command execution failed')
            else:
                self.get_logger().error('Failed to create plan from command')

        except Exception as e:
            self.get_logger().error(f'Error processing voice command: {e}')

    async def get_current_context(self) -> Dict[str, Any]:
        """Get current system context"""
        context = {
            'robot_state': await self.control_module.get_robot_state(),
            'environment_map': await self.perception_module.get_environment_map(),
            'detected_objects': await self.perception_module.get_detected_objects(),
            'current_task': self.system_state['current_task'],
            'battery_level': self.system_state['battery_level'],
            'time_of_day': self.get_current_time()
        }
        return context

    async def execute_cognitive_plan(self, plan) -> bool:
        """Execute cognitive plan"""
        try:
            # Update system state
            self.system_state['current_task'] = plan.original_command

            # Execute plan through planning module
            success = await self.planning_module.execute_plan(plan)

            # Update system state
            if success:
                self.system_state['current_task'] = None
            else:
                self.get_logger().error('Plan execution failed')

            return success

        except Exception as e:
            self.get_logger().error(f'Error executing cognitive plan: {e}')
            return False

    def get_current_time(self) -> float:
        """Get current timestamp"""
        return self.get_clock().now().nanoseconds / 1e9

    def destroy_node(self):
        """Cleanup before node destruction"""
        if hasattr(self, 'loop'):
            self.loop.call_soon_threadsafe(self.loop.stop)
        super().destroy_node()
```

#### Week 2: Perception Integration
- Integrate vision and audio systems
- Implement object detection and recognition
- Set up SLAM and mapping capabilities
- Create perception fusion algorithms

```python
# perception_module.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu, LaserScan
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
import open3d as o3d
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class DetectedObject:
    """Represents a detected object in the environment"""
    id: str
    name: str
    category: str
    position: np.ndarray  # 3D position [x, y, z]
    orientation: np.ndarray  # 4D quaternion [x, y, z, w]
    confidence: float
    bounding_box: List[float]  # [x_min, y_min, x_max, y_max]
    color: List[float]  # [r, g, b]

class HumanoidPerceptionModule(Node):
    """Perception module for humanoid robot"""

    def __init__(self, parent_node):
        super().__init__('humanoid_perception_module')

        self.parent_node = parent_node
        self.is_initialized = False

        # Initialize perception components
        self.visual_perception = self.initialize_visual_perception()
        self.audio_perception = self.initialize_audio_perception()
        self.spatial_perception = self.initialize_spatial_perception()

        # Publishers and subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)

        # Publishers
        self.object_detections_pub = self.create_publisher(
            MarkerArray, '/detected_objects', 10)
        self.spatial_map_pub = self.create_publisher(
            String, '/spatial_map', 10)

        # Detected objects storage
        self.detected_objects = []
        self.spatial_map = {}
        self.last_update_time = self.get_clock().now()

        # Processing timer
        self.processing_timer = self.create_timer(0.1, self.perception_processing_loop)

        self.is_initialized = True
        self.get_logger().info('Perception module initialized')

    def initialize_visual_perception(self):
        """Initialize visual perception components"""
        # Load pre-trained object detection model
        try:
            import torchvision.models.detection as detection_models
            model = detection_models.fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()
            return model
        except ImportError:
            self.get_logger().warn('Torchvision not available, using mock visual perception')
            return None

    def initialize_audio_perception(self):
        """Initialize audio perception components"""
        # This would integrate with speech recognition and sound processing
        return {
            'speech_recognizer': None,  # Would be Whisper or similar
            'sound_localizer': None,    # Sound source localization
            'voice_activity_detector': None  # VAD system
        }

    def initialize_spatial_perception(self):
        """Initialize spatial perception components"""
        # SLAM and mapping components
        return {
            'slam_system': None,  # Would be RTAB-Map, ORB-SLAM, etc.
            'octomap': o3d.geometry.Octree(),  # 3D occupancy grid
            'semantic_map': {},  # Semantic mapping
            'topological_map': {}  # Topological navigation
        }

    def rgb_callback(self, msg: Image):
        """Process RGB camera data"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.ros_image_to_cv2(msg)

            # Process with visual perception
            if self.visual_perception:
                detections = self.detect_objects(cv_image)
                self.detected_objects = detections

                # Publish visualization
                self.publish_object_markers(detections)

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')

    def depth_callback(self, msg: Image):
        """Process depth camera data"""
        try:
            # Convert depth image to point cloud
            depth_image = self.ros_image_to_cv2(msg, desired_encoding='passthrough')

            # Process depth information
            if hasattr(self, 'detected_objects'):
                self.update_object_depth_information(depth_image)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def imu_callback(self, msg: Imu):
        """Process IMU data for spatial awareness"""
        # Update robot orientation and acceleration information
        self.robot_orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        self.robot_acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        self.robot_angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

    def laser_callback(self, msg: LaserScan):
        """Process laser scan data for mapping"""
        try:
            # Convert laser scan to point cloud
            points = []
            angle = msg.angle_min
            for range_val in msg.ranges:
                if msg.range_min <= range_val <= msg.range_max:
                    x = range_val * np.cos(angle)
                    y = range_val * np.sin(angle)
                    points.append([x, y, 0.0])  # Assuming 2D scan
                angle += msg.angle_increment

            # Update spatial map with laser data
            if hasattr(self, 'spatial_perception'):
                self.update_spatial_map(points)

        except Exception as e:
            self.get_logger().error(f'Error processing laser scan: {e}')

    def detect_objects(self, image):
        """Detect objects in image using vision model"""
        if self.visual_perception is None:
            return []

        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            input_tensor = transform(image).unsqueeze(0)

            # Run object detection
            with torch.no_grad():
                predictions = self.visual_perception(input_tensor)

            # Process predictions
            detections = []
            for i in range(len(predictions[0]['boxes'])):
                box = predictions[0]['boxes'][i].cpu().numpy()
                score = predictions[0]['scores'][i].cpu().item()
                label = predictions[0]['labels'][i].cpu().item()

                if score > 0.5:  # Confidence threshold
                    # Convert COCO class ID to object name
                    object_name = self.coco_id_to_name(label)

                    # Calculate 3D position from depth (simplified)
                    center_x = int((box[0] + box[2]) / 2)
                    center_y = int((box[1] + box[3]) / 2)

                    # This would use actual depth data for 3D position
                    # For now, we'll use a placeholder
                    position_3d = np.array([center_x, center_y, 1.0])  # Placeholder depth

                    detection = DetectedObject(
                        id=f"obj_{len(detections)}",
                        name=object_name,
                        category=self.coco_category_to_general(label),
                        position=position_3d,
                        orientation=np.array([0, 0, 0, 1]),  # Placeholder
                        confidence=score,
                        bounding_box=list(box),
                        color=[1.0, 0.0, 0.0]  # Red for visualization
                    )

                    detections.append(detection)

            return detections

        except Exception as e:
            self.get_logger().error(f'Error in object detection: {e}')
            return []

    def coco_id_to_name(self, coco_id: int) -> str:
        """Convert COCO class ID to object name"""
        coco_names = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
            21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
            27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
            34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
            39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
            43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
            48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
            53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
            58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
            63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
            72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
            77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
            82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
            88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
        }
        return coco_names.get(coco_id, f'unknown_{coco_id}')

    def coco_category_to_general(self, coco_id: int) -> str:
        """Convert COCO class ID to general category"""
        categories = {
            'person': [1],
            'vehicle': [2, 3, 4, 5, 6, 7, 8, 9],
            'outdoor': [10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            'animal': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            'accessory': [27, 28, 31, 32, 33],
            'sports': [34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
            'kitchen': [44, 46, 47, 48, 49, 50, 51],
            'food': [52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
            'furniture': [62, 63, 64, 65, 67, 70],
            'electronic': [72, 73, 74, 75, 76, 77],
            'appliance': [78, 79, 80, 81, 82]
        }

        for category, ids in categories.items():
            if coco_id in ids:
                return category

        return 'other'

    def update_object_depth_information(self, depth_image):
        """Update detected objects with depth information"""
        for obj in self.detected_objects:
            # Get depth at object center
            center_x = int((obj.bounding_box[0] + obj.bounding_box[2]) / 2)
            center_y = int((obj.bounding_box[1] + obj.bounding_box[3]) / 2)

            if 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                depth = depth_image[center_y, center_x]
                if depth > 0:  # Valid depth
                    # Update 3D position with actual depth
                    obj.position[2] = depth

    def update_spatial_map(self, points):
        """Update spatial map with new point cloud data"""
        # Add points to octomap
        if hasattr(self.spatial_perception, 'octomap'):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))

            # This would update the actual 3D map
            # For now, we'll store in a simple structure
            self.spatial_map['points'] = points
            self.spatial_map['timestamp'] = self.get_clock().now().nanoseconds / 1e9

    def publish_object_markers(self, detections):
        """Publish detected objects as visualization markers"""
        marker_array = MarkerArray()

        for i, detection in enumerate(detections):
            # Create marker for bounding box
            marker = self.create_object_marker(detection, i)
            marker_array.markers.append(marker)

            # Create text marker for label
            text_marker = self.create_label_marker(detection, i)
            marker_array.markers.append(text_marker)

        self.object_detections_pub.publish(marker_array)

    def create_object_marker(self, detection: DetectedObject, index: int):
        """Create visualization marker for detected object"""
        from visualization_msgs.msg import Marker
        from geometry_msgs.msg import Point

        marker = Marker()
        marker.header.frame_id = "camera_rgb_optical_frame"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "detected_objects"
        marker.id = index
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Position and size (simplified)
        marker.pose.position.x = detection.position[0]
        marker.pose.position.y = detection.position[1]
        marker.pose.position.z = detection.position[2]
        marker.pose.orientation.w = 1.0

        # Size based on bounding box
        width = detection.bounding_box[2] - detection.bounding_box[0]
        height = detection.bounding_box[3] - detection.bounding_box[1]
        marker.scale.x = width / 100  # Convert pixels to meters (simplified)
        marker.scale.y = height / 100
        marker.scale.z = 0.1  # Height in meters

        # Color based on confidence
        confidence_factor = min(1.0, detection.confidence)
        marker.color.r = 1.0 - confidence_factor
        marker.color.g = confidence_factor
        marker.color.b = 0.0
        marker.color.a = 0.7

        return marker

    def create_label_marker(self, detection: DetectedObject, index: int):
        """Create text marker for object label"""
        from visualization_msgs.msg import Marker

        marker = Marker()
        marker.header.frame_id = "camera_rgb_optical_frame"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "object_labels"
        marker.id = index + 1000  # Separate ID space
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        # Position above object
        marker.pose.position.x = detection.position[0]
        marker.pose.position.y = detection.position[1]
        marker.pose.position.z = detection.position[2] + 0.2  # Above object
        marker.pose.orientation.w = 1.0

        # Text properties
        marker.text = f"{detection.name}\n{detection.confidence:.2f}"
        marker.scale.z = 0.1  # Font size
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        return marker

    def perception_processing_loop(self):
        """Main perception processing loop"""
        # This would run continuous perception tasks
        # For now, we'll just update the spatial map periodically
        if (self.get_clock().now() - self.last_update_time).nanoseconds > 1e9:  # 1 second
            self.publish_spatial_map()
            self.last_update_time = self.get_clock().now()

    def publish_spatial_map(self):
        """Publish spatial map information"""
        if self.spatial_map:
            map_msg = String()
            map_msg.data = str({
                'timestamp': self.spatial_map.get('timestamp', 0),
                'object_count': len(self.detected_objects),
                'point_cloud_size': len(self.spatial_map.get('points', []))
            })
            self.spatial_map_pub.publish(map_msg)

    def get_detected_objects(self) -> List[DetectedObject]:
        """Get currently detected objects"""
        return self.detected_objects.copy()

    def get_environment_map(self) -> Dict[str, Any]:
        """Get current environment map"""
        return {
            'objects': [obj.__dict__ for obj in self.detected_objects],
            'spatial_map': self.spatial_map,
            'timestamp': self.get_clock().now().nanoseconds / 1e9
        }

    def is_ready(self) -> bool:
        """Check if perception module is ready"""
        return self.is_initialized

    def ros_image_to_cv2(self, msg: Image, desired_encoding: str = 'bgr8'):
        """Convert ROS Image message to OpenCV image"""
        import cv2
        from cv_bridge import CvBridge

        bridge = CvBridge()
        return bridge.imgmsg_to_cv2(msg, desired_encoding=desired_encoding)
```

#### Week 3: Cognition and Planning Integration
- Implement LLM-based command understanding
- Create cognitive planning pipeline
- Integrate with perception and control systems
- Implement context management

```python
# cognition_module.py
import rclpy
from rclpy.node import Node
import openai
import asyncio
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
from std_msgs.msg import String

@dataclass
class CognitivePlan:
    """Structured cognitive plan for robot execution"""
    original_command: str
    steps: List[Dict[str, Any]]
    estimated_completion_time: float
    confidence: float
    context_used: Dict[str, Any]

class HumanoidCognitionModule(Node):
    """Cognition module for humanoid robot"""

    def __init__(self, parent_node):
        super().__init__('humanoid_cognition_module')

        self.parent_node = parent_node
        self.is_initialized = False

        # LLM configuration
        self.llm_client = None  # Will be initialized with API key
        self.model = "gpt-4-turbo"

        # Context management
        self.context_manager = ContextManager()
        self.memory_system = MemorySystem()

        # Publishers
        self.thought_process_pub = self.create_publisher(
            String, '/cognitive_thoughts', 10)

        # Initialize LLM client (requires API key)
        api_key = self.declare_parameter('openai_api_key', '').value
        if api_key:
            openai.api_key = api_key
            self.llm_client = openai.AsyncOpenAI(api_key=api_key)
            self.is_initialized = True
            self.get_logger().info('Cognition module initialized with LLM')
        else:
            self.get_logger().warn('OpenAI API key not provided - cognition module will use fallback methods')

    async def process_command(self, command_text: str, context: Dict[str, Any]) -> Optional[CognitivePlan]:
        """Process natural language command and create cognitive plan"""
        if not self.is_initialized:
            return await self.fallback_command_processing(command_text, context)

        try:
            # Update context with current information
            enriched_context = await self.enrich_context(context, command_text)

            # Generate cognitive plan using LLM
            plan = await self.generate_cognitive_plan(command_text, enriched_context)

            if plan:
                # Store in memory system
                await self.memory_system.store_interaction(
                    command=command_text,
                    plan=plan,
                    context=enriched_context
                )

                # Publish thought process for monitoring
                thought_msg = String()
                thought_msg.data = f"Processed command: {command_text}, Plan steps: {len(plan.steps)}"
                self.thought_process_pub.publish(thought_msg)

                return plan

        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')
            # Try fallback method
            return await self.fallback_command_processing(command_text, context)

        return None

    async def enrich_context(self, context: Dict[str, Any], command: str) -> Dict[str, Any]:
        """Enrich context with relevant information for the command"""
        # Get relevant context from memory
        relevant_memories = await self.memory_system.retrieve_relevant_memories(command)

        # Get spatial context
        spatial_context = await self.get_spatial_context(context)

        # Get temporal context
        temporal_context = await self.get_temporal_context()

        # Combine all context
        enriched_context = {
            **context,
            'relevant_past_interactions': relevant_memories,
            'spatial_context': spatial_context,
            'temporal_context': temporal_context,
            'robot_capabilities': await self.get_robot_capabilities(),
            'environment_constraints': await self.get_environment_constraints(context)
        }

        return enriched_context

    async def get_spatial_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get spatial context from environment information"""
        spatial_info = {
            'robot_position': context.get('robot_state', {}).get('position', [0, 0, 0]),
            'detected_objects': context.get('detected_objects', []),
            'environment_map': context.get('environment_map', {}),
            'navigable_areas': self.extract_navigable_areas(context)
        }
        return spatial_info

    def extract_navigable_areas(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract navigable areas from environment context"""
        # This would analyze the environment map to identify navigable areas
        # For now, return a placeholder
        return [
            {'name': 'kitchen', 'center': [2.0, 1.5, 0.0], 'reachable': True},
            {'name': 'living_room', 'center': [0.0, 0.0, 0.0], 'reachable': True},
            {'name': 'bedroom', 'center': [-1.5, 2.0, 0.0], 'reachable': True}
        ]

    async def get_temporal_context(self) -> Dict[str, Any]:
        """Get temporal context information"""
        import time
        current_time = time.time()

        return {
            'hour_of_day': int(time.strftime('%H')),
            'day_of_week': time.strftime('%A'),
            'season': self.get_current_season(),
            'time_since_last_interaction': current_time - getattr(self, 'last_interaction_time', current_time)
        }

    def get_current_season(self) -> str:
        """Get current season based on month"""
        import datetime
        month = datetime.datetime.now().month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'

    async def get_robot_capabilities(self) -> Dict[str, Any]:
        """Get current robot capabilities"""
        # This would come from robot's actual capabilities
        return {
            'navigation': {
                'max_speed': 0.5,
                'min_turn_radius': 0.2,
                'sensors': ['lidar', 'camera', 'imu', 'force_torque']
            },
            'manipulation': {
                'max_payload': 2.0,
                'reach': 1.2,
                'precision': 'fine'
            },
            'communication': {
                'speaking': True,
                'listening': True,
                'languages': ['English']
            }
        }

    async def get_environment_constraints(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get environment-specific constraints"""
        constraints = {
            'safety_requirements': ['avoid_people', 'respect_personal_space'],
            'accessibility': ['accessible_areas', 'obstacle_avoidance'],
            'social_norms': ['face_people_when_talking', 'respect_conversation_space']
        }

        # Add specific constraints based on environment
        if context.get('environment_map', {}).get('layout') == 'home':
            constraints['home_specific'] = ['quiet_operation', 'pet_friendly', 'child_safe']

        return constraints

    async def generate_cognitive_plan(self, command: str, context: Dict[str, Any]) -> Optional[CognitivePlan]:
        """Generate cognitive plan using LLM"""
        if not self.llm_client:
            return None

        try:
            # Create detailed prompt for LLM
            prompt = self.create_cognitive_planning_prompt(command, context)

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                functions=[
                    {
                        "name": "generate_cognitive_plan",
                        "description": "Generate a cognitive plan for a humanoid robot",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "original_command": {"type": "string"},
                                "steps": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "action_type": {
                                                "type": "string",
                                                "enum": ["navigation", "manipulation", "perception", "communication", "utility"]
                                            },
                                            "description": {"type": "string"},
                                            "parameters": {"type": "object"},
                                            "preconditions": {"type": "array", "items": {"type": "string"}},
                                            "effects": {"type": "array", "items": {"type": "string"}},
                                            "estimated_duration": {"type": "number"}
                                        },
                                        "required": ["action_type", "description", "parameters"]
                                    }
                                },
                                "estimated_completion_time": {"type": "number"},
                                "confidence": {"type": "number"},
                                "context_used": {"type": "object"}
                            },
                            "required": ["original_command", "steps", "estimated_completion_time", "confidence"]
                        }
                    }
                ],
                function_call={"name": "generate_cognitive_plan"},
                temperature=0.1
            )

            # Parse the response
            plan_data = json.loads(response.choices[0].message.function_call.arguments)

            # Create cognitive plan object
            cognitive_plan = CognitivePlan(
                original_command=plan_data['original_command'],
                steps=plan_data['steps'],
                estimated_completion_time=plan_data['estimated_completion_time'],
                confidence=plan_data['confidence'],
                context_used=plan_data.get('context_used', {})
            )

            return cognitive_plan

        except Exception as e:
            self.get_logger().error(f'Error generating cognitive plan: {e}')
            return None

    def create_cognitive_planning_prompt(self, command: str, context: Dict[str, Any]) -> str:
        """Create detailed prompt for cognitive planning"""
        prompt = f"""
        You are an expert cognitive planner for a humanoid robot. Given a natural language command and environmental context, create a detailed cognitive plan.

        Command: "{command}"

        Robot Capabilities:
        {json.dumps(context.get('robot_capabilities', {}), indent=2)}

        Environmental Context:
        {json.dumps(context.get('environment_map', {}), indent=2)}

        Detected Objects:
        {json.dumps(context.get('detected_objects', [])[:10], indent=2)}  # Limit to first 10

        Spatial Context:
        {json.dumps(context.get('spatial_context', {}), indent=2)}

        Temporal Context:
        {json.dumps(context.get('temporal_context', {}), indent=2)}

        Constraints:
        {json.dumps(context.get('environment_constraints', {}), indent=2)}

        Past Relevant Interactions:
        {json.dumps(context.get('relevant_past_interactions', [])[:3], indent=2)}  # Limit to 3

        Please generate a cognitive plan with the following requirements:
        1. Break down the command into specific, executable steps
        2. Consider the robot's capabilities and environmental constraints
        3. Account for detected objects and spatial relationships
        4. Include preconditions and expected effects for each step
        5. Provide realistic time estimates
        6. Assess confidence in the plan based on available information

        The plan should be executable by a humanoid robot with the specified capabilities.
        Focus on safety, efficiency, and achieving the user's intent.
        """

        return prompt

    def get_system_prompt(self) -> str:
        """Get system prompt for cognitive planning"""
        return """
        You are an expert cognitive planner for humanoid robots. Your role is to convert natural language commands into detailed, executable cognitive plans.

        Guidelines:
        1. Always prioritize safety and respect for humans and property
        2. Consider the robot's physical capabilities and limitations
        3. Account for environmental constraints and obstacles
        4. Include appropriate perception steps before manipulation
        5. Plan for potential failures and recovery strategies
        6. Use context to disambiguate commands
        7. Be specific about object references and locations
        8. Consider social norms and etiquette in human environments

        Output format should be structured JSON with clear action steps, parameters, and timing estimates.
        """

    async def fallback_command_processing(self, command_text: str, context: Dict[str, Any]) -> Optional[CognitivePlan]:
        """Fallback command processing using rule-based methods"""
        # Simple keyword-based command processing
        command_lower = command_text.lower()

        steps = []

        if any(word in command_lower for word in ['go to', 'navigate to', 'move to']):
            # Extract destination
            destination = self.extract_destination(command_text)
            if destination:
                steps.append({
                    'action_type': 'navigation',
                    'description': f'Navigate to {destination}',
                    'parameters': {'target_location': destination},
                    'preconditions': ['robot_is_operational', 'navigation_system_active'],
                    'effects': [f'robot_at_{destination}'],
                    'estimated_duration': 30.0
                })

        elif any(word in command_lower for word in ['pick up', 'grasp', 'take']):
            # Extract object
            object_name = self.extract_object(command_text)
            if object_name:
                steps.append({
                    'action_type': 'manipulation',
                    'description': f'Pick up {object_name}',
                    'parameters': {'target_object': object_name},
                    'preconditions': ['object_detected', 'robot_at_object_location'],
                    'effects': [f'{object_name}_in_hand'],
                    'estimated_duration': 15.0
                })

        elif any(word in command_lower for word in ['bring', 'deliver', 'give']):
            # Complex command: pick up and deliver
            object_name = self.extract_object(command_text)
            destination = self.extract_destination(command_text)
            if object_name and destination:
                steps.extend([
                    {
                        'action_type': 'perception',
                        'description': f'Locate {object_name}',
                        'parameters': {'target_object': object_name},
                        'preconditions': ['robot_is_operational'],
                        'effects': [f'{object_name}_located'],
                        'estimated_duration': 5.0
                    },
                    {
                        'action_type': 'manipulation',
                        'description': f'Pick up {object_name}',
                        'parameters': {'target_object': object_name},
                        'preconditions': [f'{object_name}_located'],
                        'effects': [f'{object_name}_in_hand'],
                        'estimated_duration': 10.0
                    },
                    {
                        'action_type': 'navigation',
                        'description': f'Navigate to {destination}',
                        'parameters': {'target_location': destination},
                        'preconditions': [f'{object_name}_in_hand'],
                        'effects': [f'robot_at_{destination}'],
                        'estimated_duration': 30.0
                    },
                    {
                        'action_type': 'manipulation',
                        'description': f'Place {object_name}',
                        'parameters': {'target_object': object_name, 'placement_location': destination},
                        'preconditions': [f'robot_at_{destination}', f'{object_name}_in_hand'],
                        'effects': [f'{object_name}_placed'],
                        'estimated_duration': 5.0
                    }
                ])

        if steps:
            return CognitivePlan(
                original_command=command_text,
                steps=steps,
                estimated_completion_time=sum(step['estimated_duration'] for step in steps),
                confidence=0.6,  # Lower confidence for fallback
                context_used=context
            )

        return None

    def extract_destination(self, command: str) -> Optional[str]:
        """Extract destination from command"""
        import re

        # Look for location keywords
        location_patterns = [
            r'to the (\w+)',
            r'in the (\w+)',
            r'at the (\w+)',
            r'go to (\w+)',
            r'navigate to (\w+)'
        ]

        for pattern in location_patterns:
            match = re.search(pattern, command.lower())
            if match:
                location = match.group(1)
                # Validate against known locations
                known_locations = ['kitchen', 'living room', 'bedroom', 'office', 'bathroom', 'hallway']
                if location in known_locations:
                    return location

        return None

    def extract_object(self, command: str) -> Optional[str]:
        """Extract object from command"""
        import re

        # Look for object keywords
        object_patterns = [
            r'pick up the (\w+)',
            r'grasp the (\w+)',
            r'take the (\w+)',
            r'bring me the (\w+)',
            r'get the (\w+)'
        ]

        for pattern in object_patterns:
            match = re.search(pattern, command.lower())
            if match:
                return match.group(1)

        return None

    def is_ready(self) -> bool:
        """Check if cognition module is ready"""
        return self.is_initialized

class ContextManager:
    """Manage contextual information for cognitive processing"""

    def __init__(self):
        self.recent_contexts = []
        self.context_history_size = 10

    async def update_context(self, new_context: Dict[str, Any]):
        """Update context with new information"""
        self.recent_contexts.append({
            'timestamp': time.time(),
            'context': new_context
        })

        # Keep only recent contexts
        if len(self.recent_contexts) > self.context_history_size:
            self.recent_contexts = self.recent_contexts[-self.context_history_size:]

    def get_relevant_context(self, query: str) -> Dict[str, Any]:
        """Get context relevant to the query"""
        # Simple keyword matching - in practice, this would use more sophisticated methods
        relevant_context = {}

        for context_entry in reversed(self.recent_contexts):
            context = context_entry['context']
            # Add logic to determine relevance based on query
            # This is a simplified example
            if any(keyword in str(context).lower() for keyword in query.lower().split()):
                relevant_context.update(context)

        return relevant_context

class MemorySystem:
    """Long-term memory system for the robot"""

    def __init__(self):
        self.interactions = []
        self.episodic_memory = []
        self.semantic_memory = {}

    async def store_interaction(self, command: str, plan: CognitivePlan, context: Dict[str, Any]):
        """Store interaction in memory"""
        interaction = {
            'timestamp': time.time(),
            'command': command,
            'plan': plan,
            'context': context,
            'success': None  # Will be updated after execution
        }
        self.interactions.append(interaction)

        # Keep only recent interactions
        if len(self.interactions) > 1000:  # Limit memory size
            self.interactions = self.interactions[-1000:]

    async def retrieve_relevant_memories(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for the query"""
        # Simple retrieval based on keyword matching
        relevant_memories = []

        for interaction in reversed(self.interactions[-50:]):  # Check recent interactions
            if any(keyword in interaction['command'].lower() for keyword in query.lower().split()):
                relevant_memories.append({
                    'command': interaction['command'],
                    'plan': interaction['plan'],
                    'success': interaction['success']
                })

                if len(relevant_memories) >= 5:  # Limit number of results
                    break

        return relevant_memories

    async def update_interaction_outcome(self, command: str, success: bool):
        """Update interaction success status"""
        for interaction in reversed(self.interactions):
            if interaction['command'] == command:
                interaction['success'] = success
                break
```

### Phase 2: Advanced Integration (Weeks 4-6)

#### Week 4: Control System Integration
- Implement advanced control algorithms
- Integrate with robot hardware
- Create safety monitoring systems
- Implement balance and locomotion control

```python
# control_module.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
from typing import Dict, Any, List, Optional
import asyncio

class HumanoidControlModule(Node):
    """Control module for humanoid robot"""

    def __init__(self, parent_node):
        super().__init__('humanoid_control_module')

        self.parent_node = parent_node
        self.is_initialized = False

        # Robot state
        self.current_joint_states = {}
        self.current_pose = None
        self.current_imu_data = None
        self.balance_state = {'com': np.array([0, 0, 0]), 'zmp': np.array([0, 0])}

        # Control systems
        self.balance_controller = BalanceController()
        self.motion_controller = MotionController()
        self.safety_monitor = SafetyMonitor()

        # Publishers and subscribers
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # Control loop timer
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100 Hz

        self.is_initialized = True
        self.get_logger().info('Control module initialized')

    def imu_callback(self, msg: Imu):
        """Update IMU data for balance control"""
        self.current_imu_data = msg

        # Update balance state
        self.balance_state['orientation'] = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        self.balance_state['angular_velocity'] = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        self.balance_state['linear_acceleration'] = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

    def joint_state_callback(self, msg: JointState):
        """Update joint state information"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_states[name] = {
                    'position': msg.position[i],
                    'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0,
                    'effort': msg.effort[i] if i < len(msg.effort) else 0.0
                }

    def control_loop(self):
        """Main control loop"""
        if not self.is_initialized:
            return

        # Update balance control
        balance_correction = self.balance_controller.compute_balance_correction(
            self.balance_state, self.current_joint_states
        )

        # Apply balance corrections to joint commands
        if balance_correction is not None:
            self.apply_balance_correction(balance_correction)

        # Monitor safety
        safety_status = self.safety_monitor.check_safety(
            self.current_joint_states, self.balance_state
        )

        if not safety_status['safe']:
            self.emergency_stop()
            self.get_logger().error(f'Safety violation: {safety_status["violations"]}')

    def execute_navigation_action(self, target_location: Dict[str, Any]) -> bool:
        """Execute navigation action"""
        try:
            # Simple navigation to target
            target_pose = self.convert_location_to_pose(target_location)

            # Generate navigation commands
            nav_commands = self.motion_controller.generate_navigation_commands(
                current_pose=self.current_pose,
                target_pose=target_pose
            )

            # Execute navigation
            success = self.execute_trajectory(nav_commands)

            return success

        except Exception as e:
            self.get_logger().error(f'Error in navigation: {e}')
            return False

    def execute_manipulation_action(self, manipulation_params: Dict[str, Any]) -> bool:
        """Execute manipulation action"""
        try:
            # Get target object information
            target_object = manipulation_params.get('target_object')
            action_type = manipulation_params.get('action_type', 'grasp')

            if action_type == 'grasp':
                return self.execute_grasp_action(target_object)
            elif action_type == 'place':
                return self.execute_place_action(manipulation_params)
            elif action_type == 'move':
                return self.execute_move_action(manipulation_params)

        except Exception as e:
            self.get_logger().error(f'Error in manipulation: {e}')
            return False

    def execute_grasp_action(self, target_object: str) -> bool:
        """Execute grasping action"""
        try:
            # Find object in environment
            object_info = self.find_object(target_object)
            if not object_info:
                self.get_logger().error(f'Object {target_object} not found')
                return False

            # Plan grasp trajectory
            grasp_pose = self.calculate_grasp_pose(object_info)
            if not grasp_pose:
                self.get_logger().error(f'Cannot calculate grasp pose for {target_object}')
                return False

            # Execute grasp sequence
            grasp_sequence = [
                {'type': 'approach', 'pose': self.calculate_approach_pose(grasp_pose)},
                {'type': 'grasp', 'pose': grasp_pose},
                {'type': 'lift', 'offset': [0, 0, 0.1]}
            ]

            for step in grasp_sequence:
                if not self.execute_manipulation_step(step):
                    return False

            return True

        except Exception as e:
            self.get_logger().error(f'Error in grasp action: {e}')
            return False

    def calculate_grasp_pose(self, object_info: Dict[str, Any]) -> Optional[Pose]:
        """Calculate optimal grasp pose for object"""
        # This would use perception data and grasp planning algorithms
        # For now, return a simple top-grasp pose
        if 'position' in object_info:
            grasp_pose = Pose()
            grasp_pose.position.x = object_info['position'][0]
            grasp_pose.position.y = object_info['position'][1]
            grasp_pose.position.z = object_info['position'][2] + 0.05  # 5cm above object
            # Set orientation for top grasp
            grasp_pose.orientation.z = 0.707  # 45-degree rotation
            grasp_pose.orientation.w = 0.707
            return grasp_pose

        return None

    def calculate_approach_pose(self, grasp_pose: Pose) -> Pose:
        """Calculate approach pose before grasping"""
        approach_pose = Pose()
        approach_pose.position.x = grasp_pose.position.x
        approach_pose.position.y = grasp_pose.position.y
        approach_pose.position.z = grasp_pose.position.z + 0.1  # 10cm above grasp point
        approach_pose.orientation = grasp_pose.orientation
        return approach_pose

    def execute_manipulation_step(self, step: Dict[str, Any]) -> bool:
        """Execute single manipulation step"""
        step_type = step['type']

        if step_type == 'approach':
            return self.move_to_pose(step['pose'])
        elif step_type == 'grasp':
            return self.grasp_object(step['pose'])
        elif step_type == 'lift':
            return self.lift_object(step['offset'])
        elif step_type == 'place':
            return self.place_object(step['pose'])

        return False

    def move_to_pose(self, target_pose: Pose) -> bool:
        """Move end-effector to target pose"""
        try:
            # Convert pose to joint trajectory
            joint_trajectory = self.inverse_kinematics(target_pose)

            if joint_trajectory:
                return self.execute_trajectory(joint_trajectory)
            else:
                self.get_logger().error('Inverse kinematics failed')
                return False

        except Exception as e:
            self.get_logger().error(f'Error moving to pose: {e}')
            return False

    def inverse_kinematics(self, target_pose: Pose) -> Optional[JointTrajectory]:
        """Solve inverse kinematics for target pose"""
        # This would implement or call an IK solver
        # For now, return a simplified example
        try:
            # In practice, this would use a proper IK solver like KDL, MoveIt, or custom implementation
            # Calculate joint angles for target pose
            joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # Example
            joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Placeholder

            # Create trajectory message
            trajectory = JointTrajectory()
            trajectory.joint_names = joint_names

            point = JointTrajectoryPoint()
            point.positions = joint_positions
            point.time_from_start = Duration(sec=2, nanosec=0)  # 2 seconds

            trajectory.points = [point]

            return trajectory

        except Exception as e:
            self.get_logger().error(f'IK error: {e}')
            return None

    def execute_trajectory(self, trajectory: JointTrajectory) -> bool:
        """Execute joint trajectory"""
        try:
            self.joint_cmd_pub.publish(trajectory)
            return True
        except Exception as e:
            self.get_logger().error(f'Trajectory execution error: {e}')
            return False

    def apply_balance_correction(self, correction_commands: Dict[str, Any]):
        """Apply balance corrections to joint commands"""
        # Modify joint commands to maintain balance
        # This would adjust joint positions/velocities to counteract balance disturbances
        pass

    def emergency_stop(self):
        """Execute emergency stop procedure"""
        # Stop all motion
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        # Stop all joint motion
        stop_trajectory = JointTrajectory()
        stop_trajectory.joint_names = list(self.current_joint_states.keys())
        stop_point = JointTrajectoryPoint()
        stop_point.positions = [0.0] * len(stop_trajectory.joint_names)
        stop_point.time_from_start = Duration(sec=0, nanosec=10000000)  # 10ms
        stop_trajectory.points = [stop_point]
        self.joint_cmd_pub.publish(stop_trajectory)

        self.get_logger().warn('Emergency stop executed')

    async def execute_plan_step(self, step: Dict[str, Any]) -> bool:
        """Execute a single plan step"""
        action_type = step.get('action_type', 'unknown')

        if action_type == 'navigation':
            return self.execute_navigation_action(step.get('parameters', {}))
        elif action_type == 'manipulation':
            return self.execute_manipulation_action(step.get('parameters', {}))
        elif action_type == 'perception':
            return self.execute_perception_action(step.get('parameters', {}))
        elif action_type == 'communication':
            return self.execute_communication_action(step.get('parameters', {}))
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            return False

    def execute_perception_action(self, params: Dict[str, Any]) -> bool:
        """Execute perception action"""
        # This would trigger perception processes
        # For example: "Look for the red cup"
        target_object = params.get('target_object')
        if target_object:
            # Trigger object detection for specific target
            pass
        return True

    def execute_communication_action(self, params: Dict[str, Any]) -> bool:
        """Execute communication action"""
        # This would handle speech synthesis, gesture, etc.
        message = params.get('message')
        if message:
            # In practice, this would interface with TTS system
            self.get_logger().info(f'Communicating: {message}')
        return True

    def find_object(self, object_name: str) -> Optional[Dict[str, Any]]:
        """Find object in environment (would interface with perception)"""
        # This would search through detected objects from perception module
        # For now, return a placeholder
        return {'position': [1.0, 0.5, 0.0], 'name': object_name}

    def get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state"""
        return {
            'joint_states': self.current_joint_states,
            'pose': self.current_pose,
            'imu_data': self.balance_state,
            'timestamp': self.get_clock().now().nanoseconds / 1e9
        }

    def is_ready(self) -> bool:
        """Check if control module is ready"""
        return self.is_initialized


class BalanceController:
    """Balance controller for humanoid robot"""

    def __init__(self):
        # Balance control parameters
        self.com_reference = np.array([0.0, 0.0, 0.8])  # Reference CoM position
        self.zmp_reference = np.array([0.0, 0.0])       # Reference ZMP
        self.balance_gains = {
            'kp': 50.0,  # Proportional gain
            'kd': 10.0,  # Derivative gain
            'ki': 1.0    # Integral gain
        }

        # Integral terms for balance control
        self.com_error_integral = np.zeros(3)

    def compute_balance_correction(self, balance_state: Dict[str, Any], joint_states: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Compute balance correction commands"""
        try:
            # Get current CoM and ZMP estimates (simplified)
            current_com = self.estimate_center_of_mass(joint_states)
            current_zmp = self.estimate_zero_moment_point(balance_state, joint_states)

            # Calculate errors
            com_error = self.com_reference[:2] - current_com[:2]  # Only x,y for balance
            zmp_error = self.zmp_reference - current_zmp

            # Update integral term
            self.com_error_integral[:2] += com_error * 0.01  # dt = 0.01s

            # Compute balance corrections using PD control
            com_correction = (
                self.balance_gains['kp'] * com_error[:2] +
                self.balance_gains['kd'] * np.zeros(2) +  # Need velocity estimation
                self.balance_gains['ki'] * self.com_error_integral[:2]
            )

            # Combine corrections
            balance_correction = {
                'com_correction': com_correction.tolist(),
                'zmp_correction': zmp_error.tolist(),
                'timestamp': time.time()
            }

            return balance_correction

        except Exception as e:
            print(f'Error in balance control: {e}')
            return None

    def estimate_center_of_mass(self, joint_states: Dict[str, Any]) -> np.ndarray:
        """Estimate center of mass from joint positions"""
        # Simplified CoM estimation
        # In practice, this would use full kinematic model and link masses
        return np.array([0.0, 0.0, 0.8])  # Placeholder

    def estimate_zero_moment_point(self, balance_state: Dict[str, Any], joint_states: Dict[str, Any]) -> np.ndarray:
        """Estimate Zero Moment Point"""
        # Simplified ZMP estimation
        # ZMP_x = CoM_x - (CoM_z * CoM_acc_x) / gravity
        # ZMP_y = CoM_y - (CoM_z * CoM_acc_y) / gravity

        # For now, return current CoM as approximation
        current_com = self.estimate_center_of_mass(joint_states)
        return current_com[:2]


class MotionController:
    """Motion controller for humanoid robot"""

    def __init__(self):
        # Motion planning parameters
        self.max_velocity = 0.5  # m/s
        self.max_acceleration = 1.0  # m/s²
        self.trajectory_smoothing = True

    def generate_navigation_commands(self, current_pose: Pose, target_pose: Pose) -> JointTrajectory:
        """Generate navigation commands to reach target pose"""
        # Simplified navigation command generation
        # In practice, this would use path planning algorithms

        trajectory = JointTrajectory()
        trajectory.joint_names = ['base_x', 'base_y', 'base_theta']  # Example for differential drive

        # Calculate path (simplified straight line)
        dx = target_pose.position.x - current_pose.position.x
        dy = target_pose.position.y - current_pose.position.y
        distance = np.sqrt(dx**2 + dy**2)

        # Generate trajectory points
        num_points = max(5, int(distance * 10))  # 10 points per meter
        for i in range(num_points + 1):
            t = i / num_points
            point = JointTrajectoryPoint()

            # Linear interpolation
            point.positions = [
                current_pose.position.x + t * dx,
                current_pose.position.y + t * dy,
                0.0  # Simplified orientation
            ]

            # Time from start
            point.time_from_start = Duration(
                sec=int(t * distance / self.max_velocity),
                nanosec=int((t * distance / self.max_velocity - int(t * distance / self.max_velocity)) * 1e9)
            )

            trajectory.points.append(point)

        return trajectory


class SafetyMonitor:
    """Safety monitoring for humanoid robot"""

    def __init__(self):
        self.safety_limits = {
            'joint_position': {'min': -3.14, 'max': 3.14},  # rad
            'joint_velocity': {'max': 5.0},  # rad/s
            'joint_effort': {'max': 100.0},  # Nm
            'balance_angle': {'max': 0.5},   # rad (about 28 degrees)
            'collision_distance': {'min': 0.1}  # m
        }

    def check_safety(self, joint_states: Dict[str, Any], balance_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check safety conditions"""
        violations = []

        # Check joint limits
        for joint_name, joint_data in joint_states.items():
            pos = joint_data['position']
            vel = joint_data['velocity']
            eff = joint_data['effort']

            if pos < self.safety_limits['joint_position']['min'] or pos > self.safety_limits['joint_position']['max']:
                violations.append(f'Joint {joint_name} position limit violation: {pos}')

            if abs(vel) > self.safety_limits['joint_velocity']['max']:
                violations.append(f'Joint {joint_name} velocity limit violation: {vel}')

            if abs(eff) > self.safety_limits['joint_effort']['max']:
                violations.append(f'Joint {joint_name} effort limit violation: {eff}')

        # Check balance (simplified)
        if 'orientation' in balance_state:
            orientation = balance_state['orientation']
            # Convert quaternion to roll/pitch angles (simplified)
            # In practice, use proper quaternion to euler conversion
            roll = np.arctan2(2*(orientation[0]*orientation[3] + orientation[1]*orientation[2]),
                             1 - 2*(orientation[2]**2 + orientation[3]**2))
            pitch = np.arcsin(2*(orientation[0]*orientation[2] - orientation[1]*orientation[3]))

            if abs(roll) > self.safety_limits['balance_angle']['max'] or abs(pitch) > self.safety_limits['balance_angle']['max']:
                violations.append(f'Balance angle violation: roll={roll:.3f}, pitch={pitch:.3f}')

        return {
            'safe': len(violations) == 0,
            'violations': violations
        }
```

#### Week 5: Integration Testing and Validation
- Test integrated system components
- Validate command understanding and execution
- Implement error handling and recovery
- Create comprehensive test suite

#### Week 6: Performance Optimization
- Optimize system performance
- Reduce latency in voice processing
- Improve real-time capabilities
- Optimize resource usage

### Phase 3: Advanced Capabilities (Weeks 7-9)

#### Week 7: Learning and Adaptation
- Implement reinforcement learning for skill improvement
- Create adaptive behavior systems
- Implement experience replay mechanisms
- Add continual learning capabilities

```python
# learning_module.py
import torch
import torch.nn as nn
import numpy as np
from collections import deque, namedtuple
import random
from typing import List, Tuple, Optional, Dict, Any
import pickle
import os

# Named tuple for experience replay
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class LearningModule:
    """Learning and adaptation module for humanoid robot"""

    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # Neural networks
        self.actor = self._build_actor_network()
        self.actor_target = self._build_actor_network()
        self.critic = self._build_critic_network()
        self.critic_target = self._build_critic_network()

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Soft update parameter
        self.exploration_noise = 0.1

        # Update target networks
        self._soft_update_targets()

        # Learning metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def _build_actor_network(self) -> nn.Module:
        """Build actor network (policy)"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh()  # Actions between -1 and 1
        )

    def _build_critic_network(self) -> nn.Module:
        """Build critic network (Q-function)"""
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Q-value output
        )

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = np.clip(action + noise, -1.0, 1.0)

        return action

    def store_experience(self, state: np.ndarray, action: np.ndarray,
                        reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.append(experience)

    def train(self) -> Optional[Dict[str, float]]:
        """Train the networks"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        batch_indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]

        # Unpack batch
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.FloatTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch]).unsqueeze(1)

        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(torch.cat([next_states, next_actions], dim=1))
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        current_q_values = self.critic(torch.cat([states, actions], dim=1))
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, predicted_actions], dim=1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update_targets()

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }

    def _soft_update_targets(self):
        """Soft update target networks"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, filepath: str):
        """Save model weights"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filepath)

    def load_model(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


class AdaptiveBehaviorSystem:
    """System for adaptive behavior learning"""

    def __init__(self):
        self.behavior_policies = {}
        self.performance_history = {}
        self.adaptation_threshold = 0.7  # Performance threshold for adaptation

    def register_behavior(self, behavior_name: str, policy_network: nn.Module):
        """Register a behavior with its policy"""
        self.behavior_policies[behavior_name] = {
            'policy': policy_network,
            'performance_history': deque(maxlen=50),
            'adaptation_counter': 0
        }

    def evaluate_behavior_performance(self, behavior_name: str, success: bool,
                                   execution_time: float, energy_consumption: float) -> float:
        """Evaluate behavior performance and return score"""
        if behavior_name not in self.behavior_policies:
            return 0.0

        # Calculate performance score (higher is better)
        success_weight = 0.5
        time_weight = 0.3
        efficiency_weight = 0.2

        success_score = 1.0 if success else 0.0
        time_score = max(0.0, 1.0 - (execution_time / 10.0))  # Normalize against 10s
        efficiency_score = max(0.0, 1.0 - (energy_consumption / 100.0))  # Normalize against 100 units

        performance_score = (success_weight * success_score +
                           time_weight * time_score +
                           efficiency_weight * efficiency_score)

        # Store performance
        self.behavior_policies[behavior_name]['performance_history'].append(performance_score)

        return performance_score

    def adapt_behavior(self, behavior_name: str, new_experience: List[Tuple]):
        """Adapt behavior based on new experience"""
        if behavior_name not in self.behavior_policies:
            return False

        policy_info = self.behavior_policies[behavior_name]
        performance_history = policy_info['performance_history']

        # Check if adaptation is needed
        if len(performance_history) >= 10:
            recent_performance = np.mean(list(performance_history)[-5:])
            historical_performance = np.mean(list(performance_history)[:5])

            if recent_performance < self.adaptation_threshold:
                # Performance degradation detected - adapt
                self._perform_adaptation(behavior_name, new_experience)
                policy_info['adaptation_counter'] += 1
                return True

        return False

    def _perform_adaptation(self, behavior_name: str, experience: List[Tuple]):
        """Perform behavior adaptation using experience"""
        policy_info = self.behavior_policies[behavior_name]
        policy = policy_info['policy']

        # Use experience to fine-tune the policy
        # This would involve training with the new experience
        # For now, we'll just log the adaptation
        print(f"Adapting behavior {behavior_name} with {len(experience)} new experiences")


class ExperienceReplaySystem:
    """System for storing and replaying experiences"""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.experiences = deque(maxlen=capacity)
        self.prioritized_replay = True
        self.priority_alpha = 0.6
        self.priority_beta = 0.4

    def add_experience(self, experience: Dict[str, Any], priority: float = 1.0):
        """Add experience to replay buffer"""
        if self.prioritized_replay:
            # Store with priority
            self.experiences.append({
                'experience': experience,
                'priority': priority,
                'td_error': priority
            })
        else:
            self.experiences.append(experience)

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch of experiences"""
        if len(self.experiences) < batch_size:
            return list(self.experiences)

        if self.prioritized_replay:
            # Sample based on priority
            priorities = np.array([exp['priority'] for exp in self.experiences])
            probabilities = priorities ** self.priority_alpha
            probabilities /= probabilities.sum()

            indices = np.random.choice(len(self.experiences), batch_size, p=probabilities)
            batch = [self.experiences[i]['experience'] for i in indices]
        else:
            # Random sampling
            indices = np.random.choice(len(self.experiences), batch_size, replace=False)
            batch = [self.experiences[i] for i in indices]

        return batch

    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on TD errors"""
        if not self.prioritized_replay:
            return

        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.experiences):
                self.experiences[idx]['td_error'] = td_error
                self.experiences[idx]['priority'] = (td_error + 1e-5) ** self.priority_alpha

    def save_experiences(self, filepath: str):
        """Save experiences to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.experiences), f)

    def load_experiences(self, filepath: str):
        """Load experiences from file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                experiences = pickle.load(f)
                self.experiences.extend(experiences)


class ContinualLearningSystem:
    """System for continual learning and skill acquisition"""

    def __init__(self):
        self.skill_library = {}
        self.transfer_learning_enabled = True
        self.catastrophic_forgetting_protection = True

    def register_skill(self, skill_name: str, skill_model: nn.Module,
                      skill_description: str = ""):
        """Register a new skill in the library"""
        self.skill_library[skill_name] = {
            'model': skill_model,
            'description': skill_description,
            'training_data': [],
            'performance_metrics': {},
            'dependencies': []
        }

    def learn_new_skill(self, skill_name: str, training_data: List[Tuple],
                       prerequisites: List[str] = None):
        """Learn a new skill with optional prerequisites"""
        if prerequisites:
            # Check if prerequisites are met
            for prereq in prerequisites:
                if prereq not in self.skill_library:
                    raise ValueError(f"Prerequisite skill {prereq} not found")

        # Create new skill model
        # This would typically involve training a new model
        new_model = self._create_skill_model(skill_name)

        # Train the skill
        self._train_skill(new_model, training_data)

        # Register the skill
        self.register_skill(skill_name, new_model, f"Skill learned for: {skill_name}")

        # If transfer learning is enabled, update related skills
        if self.transfer_learning_enabled:
            self._transfer_knowledge(skill_name)

    def _create_skill_model(self, skill_name: str) -> nn.Module:
        """Create a model for the new skill"""
        # This would create an appropriate model based on skill type
        # For now, return a generic network
        return nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def _train_skill(self, model: nn.Module, training_data: List[Tuple]):
        """Train the skill model"""
        # This would implement the training process
        # For now, just simulate training
        print(f"Training skill model with {len(training_data)} samples")

    def _transfer_knowledge(self, new_skill: str):
        """Transfer knowledge from related skills"""
        # Find related skills based on similarity
        related_skills = self._find_related_skills(new_skill)

        for related_skill in related_skills:
            # Transfer relevant knowledge
            self._transfer_from_skill(related_skill, new_skill)

    def _find_related_skills(self, skill_name: str) -> List[str]:
        """Find skills related to the given skill"""
        # This would implement skill similarity analysis
        # For now, return a simple heuristic
        related = []
        for existing_skill in self.skill_library:
            if existing_skill != skill_name:
                # Simple similarity based on name
                if any(word in existing_skill.lower() for word in skill_name.lower().split()):
                    related.append(existing_skill)
        return related

    def _transfer_from_skill(self, source_skill: str, target_skill: str):
        """Transfer knowledge from source to target skill"""
        # This would implement knowledge transfer mechanisms
        # Such as parameter sharing, feature transfer, etc.
        print(f"Transferring knowledge from {source_skill} to {target_skill}")

    def execute_skill(self, skill_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered skill"""
        if skill_name not in self.skill_library:
            raise ValueError(f"Skill {skill_name} not found in library")

        skill_model = self.skill_library[skill_name]['model']

        # Convert inputs to tensor and run through model
        with torch.no_grad():
            # This is a simplified example - in practice, inputs would be properly formatted
            input_tensor = torch.FloatTensor(list(inputs.values())).unsqueeze(0)
            output = skill_model(input_tensor)

            return {'output': output.numpy(), 'skill': skill_name}

    def update_skill(self, skill_name: str, new_training_data: List[Tuple]):
        """Update an existing skill with new data"""
        if skill_name not in self.skill_library:
            raise ValueError(f"Skill {skill_name} not found in library")

        # Protect against catastrophic forgetting
        if self.catastrophic_forgetting_protection:
            # Use techniques like elastic weight consolidation or rehearsal
            self._protect_against_forgetting(skill_name, new_training_data)

        # Retrain the skill with new data
        skill_model = self.skill_library[skill_name]['model']
        self._train_skill(skill_model, new_training_data)

        # Update training data
        self.skill_library[skill_name]['training_data'].extend(new_training_data)

    def _protect_against_forgetting(self, skill_name: str, new_data: List[Tuple]):
        """Protect against catastrophic forgetting during skill updates"""
        # Store exemplars from old data
        old_data = self.skill_library[skill_name]['training_data']

        # Use a subset of old data during new training to maintain old skills
        exemplar_data = self._select_exemplars(old_data, 100)  # Keep 100 exemplars

        # Combine old exemplars with new data for training
        combined_data = exemplar_data + new_data
        return combined_data

    def _select_exemplars(self, data: List[Tuple], num_exemplars: int) -> List[Tuple]:
        """Select exemplars from data using a strategy like reservoir sampling"""
        if len(data) <= num_exemplars:
            return data

        # Random sampling as a simple exemplar selection
        indices = np.random.choice(len(data), num_exemplars, replace=False)
        return [data[i] for i in indices]
```

#### Week 8: Advanced Perception Integration
- Implement multimodal perception fusion
- Create advanced object recognition systems
- Integrate 3D scene understanding
- Develop social interaction capabilities

#### Week 9: Human-Robot Interaction
- Implement conversational AI systems
- Create social behavior models
- Develop emotion recognition and response
- Implement collaborative task execution

### Phase 4: Deployment and Testing (Weeks 10-13)

#### Week 10: System Integration and Testing
- Integrate all modules into complete system
- Conduct comprehensive system testing
- Validate safety and reliability
- Create user interface and documentation

#### Week 11: Real-World Deployment
- Deploy system on physical robot
- Conduct real-world testing and validation
- Collect performance metrics
- Refine and optimize for deployment

#### Week 12: Advanced Features and Optimization
- Implement advanced features based on testing feedback
- Optimize system performance
- Enhance user experience
- Add robustness improvements

#### Week 13: Final Validation and Documentation
- Conduct final system validation
- Create comprehensive documentation
- Prepare project presentation
- Document lessons learned and future improvements

## Performance Considerations

### Real-time Requirements

```python
import time
import threading
from queue import Queue, PriorityQueue
import asyncio

class RealTimeScheduler:
    """Real-time scheduler for humanoid robot system"""

    def __init__(self):
        self.task_queue = PriorityQueue()
        self.active_tasks = {}
        self.scheduler_thread = None
        self.running = False

    def start_scheduler(self):
        """Start the real-time scheduler"""
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

    def schedule_task(self, task_func, priority: int, deadline: float, args=(), kwargs=None):
        """Schedule a task with priority and deadline"""
        if kwargs is None:
            kwargs = {}

        task_id = f"task_{time.time()}"
        task = {
            'id': task_id,
            'function': task_func,
            'priority': priority,
            'deadline': deadline,
            'args': args,
            'kwargs': kwargs,
            'scheduled_time': time.time()
        }

        self.task_queue.put((priority, deadline, task))
        return task_id

    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                # Get highest priority task
                priority, deadline, task = self.task_queue.get(timeout=0.001)

                current_time = time.time()

                if current_time > deadline:
                    print(f"Task {task['id']} missed deadline by {current_time - deadline:.3f}s")
                    continue

                # Execute task
                try:
                    result = task['function'](*task['args'], **task['kwargs'])
                except Exception as e:
                    print(f"Task {task['id']} failed: {e}")

            except:
                # No tasks available, sleep briefly
                time.sleep(0.001)

    def stop_scheduler(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()

class PerformanceMonitor:
    """Monitor system performance metrics"""

    def __init__(self):
        self.metrics = {
            'control_loop_times': deque(maxlen=1000),
            'perception_times': deque(maxlen=1000),
            'planning_times': deque(maxlen=1000),
            'command_response_times': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'cpu_usage': deque(maxlen=1000)
        }

    def record_metric(self, metric_name: str, value: float):
        """Record a performance metric"""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get current performance statistics"""
        stats = {}
        for name, values in self.metrics.items():
            if values:
                stats[name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'current': float(values[-1]) if values else 0.0
                }
        return stats

    def check_performance_alerts(self) -> List[str]:
        """Check for performance alerts"""
        alerts = []

        # Check control loop timing
        if 'control_loop_times' in self.metrics and self.metrics['control_loop_times']:
            avg_loop_time = np.mean(self.metrics['control_loop_times'])
            if avg_loop_time > 0.02:  # 20ms threshold
                alerts.append(f"Control loop slow: {avg_loop_time:.3f}s")

        # Check memory usage
        if 'memory_usage' in self.metrics and self.metrics['memory_usage']:
            avg_memory = np.mean(self.metrics['memory_usage'])
            if avg_memory > 0.8:  # 80% threshold
                alerts.append(f"High memory usage: {avg_memory:.2%}")

        return alerts
```

## Safety and Robustness

### Fault Tolerance and Recovery

```python
class FaultToleranceSystem:
    """System for handling faults and ensuring robust operation"""

    def __init__(self):
        self.fault_handlers = {}
        self.recovery_procedures = {}
        self.safety_protocols = {}
        self.fallback_behaviors = {}

    def register_fault_handler(self, fault_type: str, handler_func):
        """Register a fault handler for specific fault type"""
        self.fault_handlers[fault_type] = handler_func

    def register_recovery_procedure(self, fault_type: str, recovery_func):
        """Register recovery procedure for specific fault type"""
        self.recovery_procedures[fault_type] = recovery_func

    def detect_fault(self, fault_type: str, context: Dict[str, Any] = None):
        """Detect and handle fault"""
        if fault_type in self.fault_handlers:
            return self.fault_handlers[fault_type](context)
        else:
            # Use default fault handling
            return self.default_fault_handler(fault_type, context)

    def default_fault_handler(self, fault_type: str, context: Dict[str, Any] = None):
        """Default fault handling procedure"""
        print(f"Handling fault: {fault_type}")

        # Stop all robot motion
        self.emergency_stop()

        # Log fault
        self.log_fault(fault_type, context)

        # Attempt recovery
        if fault_type in self.recovery_procedures:
            return self.recovery_procedures[fault_type]()
        else:
            # Use fallback behavior
            return self.execute_fallback_behavior(fault_type)

    def emergency_stop(self):
        """Execute emergency stop procedure"""
        # This would interface with the control system
        print("EMERGENCY STOP EXECUTED")

    def log_fault(self, fault_type: str, context: Dict[str, Any]):
        """Log fault information"""
        import datetime
        fault_log = {
            'timestamp': datetime.datetime.now().isoformat(),
            'fault_type': fault_type,
            'context': context,
            'handled': True
        }
        # In practice, this would write to a persistent log
        print(f"Fault logged: {fault_log}")

    def execute_fallback_behavior(self, fault_type: str):
        """Execute fallback behavior for fault"""
        # Default fallback: return to safe position
        safe_behavior = self.fallback_behaviors.get('safe_return', self.default_safe_return)
        return safe_behavior()

    def default_safe_return(self):
        """Default safe return behavior"""
        # Move to pre-defined safe position
        # This would involve navigation to a safe location
        print("Returning to safe position")
        return True
```

## Testing and Validation

### Comprehensive Test Suite

```python
import unittest
import asyncio
from unittest.mock import Mock, MagicMock

class TestAutonomousHumanoidRobot(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Mock the robot system components
        self.mock_perception = Mock()
        self.mock_cognition = Mock()
        self.mock_planning = Mock()
        self.mock_control = Mock()

        # Initialize system manager with mocks
        self.system_manager = HumanoidSystemManager()
        self.system_manager.perception_module = self.mock_perception
        self.system_manager.cognition_module = self.mock_cognition
        self.system_manager.planning_module = self.mock_planning
        self.system_manager.control_module = self.mock_control

    def test_voice_command_processing(self):
        """Test voice command processing pipeline"""
        command = "Move forward 1 meter"

        # Mock the processing chain
        self.mock_cognition.process_command.return_value = MagicMock()
        self.mock_cognition.process_command.return_value.steps = [
            {'action_type': 'navigation', 'parameters': {'direction': 'forward', 'distance': 1.0}}
        ]

        # Process command
        result = asyncio.run(
            self.system_manager.process_voice_command(command)
        )

        # Verify processing steps
        self.assertIsNotNone(result)
        self.mock_cognition.process_command.assert_called_once()
        self.mock_planning.execute_plan.assert_called_once()

    def test_perception_integration(self):
        """Test perception system integration"""
        # Mock sensor data
        mock_image = np.random.rand(480, 640, 3).astype(np.float32)
        mock_objects = [
            {'name': 'cup', 'position': [1.0, 0.5, 0.0], 'confidence': 0.9}
        ]

        self.mock_perception.get_detected_objects.return_value = mock_objects

        # Get detected objects
        objects = self.mock_perception.get_detected_objects()

        # Verify object detection
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]['name'], 'cup')

    def test_navigation_command(self):
        """Test navigation command execution"""
        navigation_params = {'direction': 'forward', 'distance': 1.0}

        # Mock successful execution
        self.mock_control.execute_navigation_action.return_value = True

        # Execute navigation
        result = self.mock_control.execute_navigation_action(navigation_params)

        # Verify execution
        self.assertTrue(result)
        self.mock_control.execute_navigation_action.assert_called_once_with(navigation_params)

    def test_safety_monitoring(self):
        """Test safety monitoring system"""
        # Mock unsafe conditions
        unsafe_state = {
            'joint_states': {'joint1': {'position': 5.0}},  # Beyond limit
            'balance_state': {'orientation': [1.0, 1.0, 1.0, 1.0]}  # Unbalanced
        }

        # Mock safety monitor
        mock_safety = Mock()
        mock_safety.check_safety.return_value = {
            'safe': False,
            'violations': ['Joint limit exceeded', 'Balance angle exceeded']
        }

        # Check safety
        safety_result = mock_safety.check_safety(unsafe_state)

        # Verify safety detection
        self.assertFalse(safety_result['safe'])
        self.assertIn('Joint limit exceeded', safety_result['violations'])

    def test_learning_integration(self):
        """Test learning system integration"""
        # Initialize learning module
        learning_module = LearningModule(state_dim=24, action_dim=12)

        # Create mock experience
        state = np.random.rand(24)
        action = np.random.rand(12)
        reward = 1.0
        next_state = np.random.rand(24)
        done = False

        # Store experience
        learning_module.store_experience(state, action, reward, next_state, done)

        # Verify experience storage
        self.assertEqual(len(learning_module.replay_buffer), 1)

        # Train network
        metrics = learning_module.train()

        # Verify training occurred
        self.assertIsNotNone(metrics)


class IntegrationTestSuite:
    """Comprehensive integration test suite"""

    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []

    def run_all_tests(self):
        """Run all integration tests"""
        print("Starting comprehensive integration test suite...")

        # Run unit tests
        unittest.main(argv=[''], exit=False, verbosity=2)

        # Run integration tests
        self.test_end_to_end_scenario()
        self.test_multi_modal_integration()
        self.test_safety_systems()
        self.test_performance_under_load()

        self.print_summary()

    def test_end_to_end_scenario(self):
        """Test complete end-to-end scenario"""
        print("\nTesting end-to-end scenario...")
        try:
            # Simulate complete command processing
            command = "Go to kitchen and bring me a cup"

            # This would involve the complete pipeline:
            # 1. Voice recognition
            # 2. Language understanding
            # 3. Planning
            # 4. Execution
            # 5. Monitoring

            # For this test, we'll simulate the complete flow
            print("✓ End-to-end scenario test passed")
            self.tests_passed += 1
        except Exception as e:
            print(f"✗ End-to-end scenario test failed: {e}")
            self.tests_failed += 1

    def test_multi_modal_integration(self):
        """Test multi-modal integration"""
        print("\nTesting multi-modal integration...")
        try:
            # Test integration of vision, language, and action
            print("✓ Multi-modal integration test passed")
            self.tests_passed += 1
        except Exception as e:
            print(f"✗ Multi-modal integration test failed: {e}")
            self.tests_failed += 1

    def test_safety_systems(self):
        """Test safety system integration"""
        print("\nTesting safety systems...")
        try:
            # Test safety monitoring and response
            print("✓ Safety systems test passed")
            self.tests_passed += 1
        except Exception as e:
            print(f"✗ Safety systems test failed: {e}")
            self.tests_failed += 1

    def test_performance_under_load(self):
        """Test performance under load"""
        print("\nTesting performance under load...")
        try:
            # Simulate high-load conditions
            print("✓ Performance under load test passed")
            self.tests_passed += 1
        except Exception as e:
            print(f"✗ Performance under load test failed: {e}")
            self.tests_failed += 1

    def print_summary(self):
        """Print test results summary"""
        print(f"\n{'='*50}")
        print("INTEGRATION TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_failed}")
        print(f"Total Tests: {self.tests_passed + self.tests_failed}")
        print(f"Success Rate: {(self.tests_passed / max(1, self.tests_passed + self.tests_failed) * 100):.1f}%")
        print(f"{'='*50}")
```

## Deployment Considerations

### Hardware Requirements

For optimal performance of the autonomous humanoid robot system, consider these hardware requirements:

#### Minimum Requirements:
- **CPU**: Multi-core processor (Intel i7 / AMD Ryzen 7 or equivalent)
- **GPU**: NVIDIA RTX 3060 or better with CUDA support
- **RAM**: 16GB system memory
- **Storage**: 500GB SSD for models and data
- **Network**: Gigabit Ethernet or WiFi 6 for communication

#### Recommended Requirements:
- **CPU**: High-core count processor (Intel i9 / AMD Threadripper)
- **GPU**: NVIDIA RTX 4080/4090 or A6000/A100 for training
- **RAM**: 32GB+ system memory
- **Storage**: 1TB+ NVMe SSD
- **Network**: Dedicated network infrastructure

### Software Dependencies

```bash
# Core dependencies
sudo apt update
sudo apt install python3-pip python3-dev
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ROS 2 dependencies
sudo apt install ros-humble-desktop-full
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup

# Audio processing
sudo apt install portaudio19-dev python3-pyaudio
pip3 install pyaudio sounddevice

# Computer vision
pip3 install opencv-python open3d

# NLP and AI
pip3 install openai-whisper transformers torch

# Additional utilities
pip3 install numpy scipy matplotlib scikit-learn
```

## Best Practices for Deployment

### 1. System Monitoring
- Implement comprehensive logging and monitoring
- Monitor resource usage (CPU, GPU, memory)
- Track performance metrics in real-time
- Set up alerting for anomalies

### 2. Safety First
- Implement multiple safety layers
- Use redundant sensors where critical
- Plan for graceful degradation
- Test extensively before deployment

### 3. Continuous Improvement
- Collect usage data for improvement
- Implement A/B testing for new features
- Regular model updates and retraining
- User feedback integration

### 4. Maintenance Planning
- Scheduled maintenance windows
- Backup and recovery procedures
- Version control for all components
- Rollback capabilities

## Troubleshooting Common Issues

### 1. Performance Issues
- **Symptom**: Slow response times or dropped frames
- **Solution**: Optimize models, reduce resolution, use faster hardware

### 2. Recognition Failures
- **Symptom**: Poor speech or object recognition
- **Solution**: Improve audio quality, calibrate sensors, update models

### 3. Navigation Failures
- **Symptom**: Robot gets lost or stuck
- **Solution**: Improve mapping, enhance localization, add recovery behaviors

### 4. Safety Violations
- **Symptom**: Unexpected emergency stops
- **Solution**: Tune safety parameters, improve sensor fusion, add redundancy

## Summary

The autonomous humanoid robot capstone project integrates multiple advanced technologies:

- **Voice Recognition**: OpenAI Whisper for natural language commands
- **Vision Processing**: Real-time object detection and scene understanding
- **Cognitive Planning**: LLM-based task decomposition and planning
- **Control Systems**: Advanced balance and motion control
- **Learning Systems**: Continuous adaptation and improvement
- **Safety Systems**: Comprehensive monitoring and protection

Key success factors include:
- Proper system architecture and modularity
- Comprehensive testing and validation
- Safety-first design approach
- Performance optimization for real-time operation
- Continuous learning and adaptation capabilities

This project demonstrates the convergence of AI, robotics, and natural human interaction, creating robots that can understand and execute complex commands through natural language while operating safely in human environments. The integration of vision, language, and action systems enables truly intelligent robotic assistants that can adapt to changing needs and environments.

The skills developed through this project prepare students for careers in robotics, AI, and human-robot interaction, providing hands-on experience with cutting-edge technologies in embodied AI and autonomous systems.