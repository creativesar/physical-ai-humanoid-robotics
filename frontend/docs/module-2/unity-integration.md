---
sidebar_position: 4
title: "Unity Integration"
---

# Unity Integration

## Introduction to Unity for Robotics

Unity is a powerful real-time 3D development platform that has gained significant traction in robotics simulation, particularly for humanoid robots. Unlike traditional robotics simulators, Unity offers photorealistic rendering, advanced physics engines, and a rich ecosystem of tools that make it ideal for creating high-fidelity digital twins and immersive human-robot interaction experiences.

## Why Unity for Humanoid Robotics?

### 1. Photorealistic Rendering
- **High-quality graphics**: Realistic lighting, shadows, and materials
- **Advanced rendering pipeline**: HDRP (High Definition Render Pipeline) and URP (Universal Render Pipeline)
- **Real-time ray tracing**: For physically accurate lighting simulation
- **Post-processing effects**: Depth of field, motion blur, color grading

### 2. Physics Simulation
- **NVIDIA PhysX engine**: Industry-standard physics simulation
- **Advanced collision detection**: Complex mesh collisions and triggers
- **Soft body dynamics**: For simulating flexible materials
- **Fluid simulation**: For advanced environmental interactions

### 3. Cross-Platform Deployment
- **Multiple target platforms**: Windows, Linux, macOS, mobile, VR/AR
- **Web deployment**: WebGL for browser-based simulations
- **Real-time performance**: Optimized for interactive applications

## Unity Robotics Ecosystem

### 1. Unity Robotics Package

Unity provides the Unity Robotics Package for seamless integration with ROS 2:

```json
// Add to Packages/manifest.json
{
  "dependencies": {
    "com.unity.robotics.ros-tcp-connector": "https://github.com/Unity-Technologies/ROS-TCP-Connector.git?path=/com.unity.robotics.ros-tcp-connector#v0.7.0",
    "com.unity.robotics.urdf-importer": "https://github.com/Unity-Technologies/URDF-Importer.git?path=/com.unity.robotics.urdf-importer#v0.5.2"
  }
}
```

### 2. URDF Importer

The URDF Importer allows you to import ROS-compatible robot models:

```csharp
// Example of using URDF Importer
using Unity.Robotics.URDFImport;

public class HumanoidRobotLoader : MonoBehaviour
{
    [SerializeField]
    private string robotUrdfPath;

    [SerializeField]
    private GameObject robotPrefab;

    void Start()
    {
        // Load URDF model
        if (!string.IsNullOrEmpty(robotUrdfPath))
        {
            GameObject robot = URDFRobotExtensions.LoadRobotAtPath(robotUrdfPath);
            if (robot != null)
            {
                robot.transform.SetParent(transform);
                robot.transform.localPosition = Vector3.zero;
                robot.transform.localRotation = Quaternion.identity;
            }
        }
    }
}
```

### 3. ROS TCP Connector

The ROS TCP Connector enables communication between Unity and ROS 2:

```csharp
// Example ROS connection
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;

public class UnityRobotController : MonoBehaviour
{
    private ROSConnection ros;
    private string jointTopic = "/joint_states";
    private string cmdTopic = "/joint_commands";

    void Start()
    {
        // Get the ROS connection
        ros = ROSConnection.instance;

        // Subscribe to joint states
        ros.Subscribe<JointStateMsg>(jointTopic, OnJointStateReceived);
    }

    void OnJointStateReceived(JointStateMsg jointState)
    {
        // Process joint state messages
        for (int i = 0; i < jointState.name.Count; i++)
        {
            string jointName = jointState.name[i];
            float position = (float)jointState.position[i];

            // Update joint in Unity
            Transform jointTransform = FindJointByName(jointName);
            if (jointTransform != null)
            {
                // Apply position to Unity joint
                UpdateJointPosition(jointTransform, position);
            }
        }
    }

    void UpdateJointPosition(Transform joint, float position)
    {
        // Update the joint position in Unity
        // Implementation depends on your joint structure
        joint.localRotation = Quaternion.Euler(0, position * Mathf.Rad2Deg, 0);
    }

    Transform FindJointByName(string name)
    {
        // Find joint by name in the hierarchy
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            if (child.name == name)
                return child;
        }
        return null;
    }

    public void SendJointCommand(string[] jointNames, double[] positions)
    {
        // Send joint commands to ROS
        JointStateMsg cmd = new JointStateMsg();
        cmd.name = new System.Collections.Generic.List<string>(jointNames);
        cmd.position = new System.Collections.Generic.List<double>(positions);
        cmd.header = new HeaderMsg();
        cmd.header.stamp = new TimeStamp(ROSConnection.GetServerTime());

        ros.Publish(cmdTopic, cmd);
    }
}
```

## Setting Up Unity for Humanoid Robotics

### 1. Project Configuration

#### Unity Version Requirements
- Unity 2021.3 LTS or later recommended
- HDRP (High Definition Render Pipeline) for advanced rendering
- .NET Standard 2.1 for ROS integration

#### Installation Steps
1. Create new Unity project (3D Core template)
2. Install required packages via Package Manager
3. Configure ROS TCP connector settings
4. Set up physics and rendering settings

### 2. Physics Configuration

```csharp
// Physics settings for humanoid simulation
using UnityEngine;

public class PhysicsConfiguration : MonoBehaviour
{
    [Header("Physics Settings")]
    [SerializeField] private float gravity = -9.81f;
    [SerializeField] private int solverIterations = 8;
    [SerializeField] private int solverVelocityIterations = 2;

    [Header("Collision Settings")]
    [SerializeField] private float bounceThreshold = 2.0f;
    [SerializeField] private float sleepThreshold = 0.005f;

    void Start()
    {
        ConfigurePhysics();
    }

    void ConfigurePhysics()
    {
        // Set gravity for humanoid simulation
        Physics.gravity = new Vector3(0, gravity, 0);

        // Configure solver settings
        Physics.defaultSolverIterations = solverIterations;
        Physics.defaultSolverVelocityIterations = solverVelocityIterations;

        // Set collision detection parameters
        Physics.bounceThreshold = bounceThreshold;
        Physics.sleepThreshold = sleepThreshold;

        // Enable continuous collision detection for fast-moving parts
        Physics.defaultContactOffset = 0.01f;
        Physics.defaultSolverIterations = 8;
    }
}
```

## Creating Humanoid Robot Models in Unity

### 1. Manual Model Creation

```csharp
// Example of creating a humanoid robot hierarchy in Unity
using UnityEngine;

public class HumanoidRobotBuilder : MonoBehaviour
{
    [System.Serializable]
    public class JointDefinition
    {
        public string name;
        public JointType jointType;
        public Vector3 position;
        public Vector3 rotation;
        public float minAngle = -90f;
        public float maxAngle = 90f;
        public float maxForce = 100f;
    }

    [SerializeField] private JointDefinition[] jointDefinitions;
    [SerializeField] private Material robotMaterial;

    void Start()
    {
        BuildRobot();
    }

    void BuildRobot()
    {
        GameObject torso = CreateLink("torso", new Vector3(0.3f, 0.4f, 0.2f), Vector3.zero);
        torso.transform.SetParent(transform);

        // Create head
        GameObject head = CreateLink("head", new Vector3(0.2f, 0.2f, 0.2f), new Vector3(0, 0.4f, 0));
        CreateJoint(torso.transform, head.transform, "neck_joint", JointType.Revolute, Vector3.right, -30f, 30f);

        // Create arms
        CreateArm(torso.transform, "left", new Vector3(0.15f, 0.15f, 0.4f), new Vector3(0.2f, 0.2f, 0));
        CreateArm(torso.transform, "right", new Vector3(0.15f, 0.15f, 0.4f), new Vector3(-0.2f, 0.2f, 0));

        // Create legs
        CreateLeg(torso.transform, "left", new Vector3(0.15f, 0.15f, 0.5f), new Vector3(0.1f, -0.3f, 0));
        CreateLeg(torso.transform, "right", new Vector3(0.15f, 0.15f, 0.5f), new Vector3(-0.1f, -0.3f, 0));
    }

    GameObject CreateLink(string name, Vector3 size, Vector3 position)
    {
        GameObject link = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        link.name = name;
        link.transform.localScale = size;
        link.transform.position = position;

        // Add rigidbody for physics
        Rigidbody rb = link.AddComponent<Rigidbody>();
        rb.mass = CalculateMass(size);
        rb.collisionDetectionMode = CollisionDetectionMode.Continuous;

        // Add collider
        CapsuleCollider capsule = link.GetComponent<CapsuleCollider>();
        capsule.center = Vector3.up * size.y * 0.5f;
        capsule.height = size.y;
        capsule.radius = size.x * 0.5f;

        // Apply material
        link.GetComponent<Renderer>().material = robotMaterial;

        return link;
    }

    float CalculateMass(Vector3 size)
    {
        // Calculate mass based on volume (simplified)
        float volume = size.x * size.y * size.z;
        return volume * 1000f; // 1000 kg/m³ density
    }

    ConfigurableJoint CreateJoint(Transform parent, Transform child, string name, JointType jointType, Vector3 axis, float minAngle, float maxAngle)
    {
        child.SetParent(parent);

        ConfigurableJoint joint = child.gameObject.AddComponent<ConfigurableJoint>();
        joint.name = name;

        // Configure joint limits
        SoftJointLimit limit = new SoftJointLimit();
        limit.limit = maxAngle;
        joint.angularYLimit = limit;

        limit.limit = -minAngle;
        joint.angularZLimit = limit;

        // Lock other axes as needed
        joint.angularXMotion = ConfigurableJointMotion.Locked;
        joint.angularYMotion = ConfigurableJointMotion.Limited;
        joint.angularZMotion = ConfigurableJointMotion.Limited;

        // Set drive for actuation
        JointDrive drive = new JointDrive();
        drive.maximumForce = 1000f;
        drive.positionSpring = 1000f;
        drive.positionDamper = 100f;
        joint.slerpDrive = drive;

        return joint;
    }

    void CreateArm(Transform parent, string side, Vector3 upperArmSize, Vector3 position)
    {
        // Create upper arm
        GameObject upperArm = CreateLink($"{side}_upper_arm", upperArmSize, position);
        upperArm.transform.SetParent(parent);

        // Create lower arm
        Vector3 lowerArmPos = position + new Vector3(0, -upperArmSize.y, 0);
        GameObject lowerArm = CreateLink($"{side}_lower_arm", upperArmSize * 0.8f, lowerArmPos);

        // Create joint between upper and lower arm
        CreateJoint(upperArm.transform, lowerArm.transform,
                   $"{side}_elbow_joint", JointType.Revolute, Vector3.right, -120f, 0f);
    }

    void CreateLeg(Transform parent, string side, Vector3 thighSize, Vector3 position)
    {
        // Create thigh
        GameObject thigh = CreateLink($"{side}_thigh", thighSize, position);
        thigh.transform.SetParent(parent);

        // Create shin
        Vector3 shinPos = position + new Vector3(0, -thighSize.y, 0);
        GameObject shin = CreateLink($"{side}_shin", thighSize * 0.9f, shinPos);

        // Create foot
        Vector3 footPos = shinPos + new Vector3(0, -thighSize.y * 0.9f, 0.1f);
        GameObject foot = CreateLink($"{side}_foot", new Vector3(0.2f, 0.1f, 0.3f), footPos);

        // Create joints
        CreateJoint(thigh.transform, shin.transform,
                   $"{side}_knee_joint", JointType.Revolute, Vector3.right, -5f, 120f);
        CreateJoint(shin.transform, foot.transform,
                   $"{side}_ankle_joint", JointType.Revolute, Vector3.right, -30f, 30f);
    }
}
```

### 2. Animation and Control Systems

```csharp
// Animation controller for humanoid robot
using UnityEngine;
using System.Collections.Generic;

public class HumanoidAnimationController : MonoBehaviour
{
    [Header("Joint Control")]
    [SerializeField] private List<JointController> jointControllers;

    [Header("Animation Parameters")]
    [SerializeField] private float walkSpeed = 1.0f;
    [SerializeField] private float turnSpeed = 50.0f;

    private Dictionary<string, JointController> jointMap;
    private Animator animator;

    void Start()
    {
        InitializeJointControllers();
        animator = GetComponent<Animator>();
    }

    void InitializeJointControllers()
    {
        jointMap = new Dictionary<string, JointController>();
        foreach (JointController controller in jointControllers)
        {
            jointMap[controller.jointName] = controller;
        }
    }

    public void SetJointTarget(string jointName, float targetAngle)
    {
        if (jointMap.ContainsKey(jointName))
        {
            jointMap[jointName].SetTargetAngle(targetAngle);
        }
    }

    public void SetJointTargets(Dictionary<string, float> targets)
    {
        foreach (var target in targets)
        {
            SetJointTarget(target.Key, target.Value);
        }
    }

    public void Walk(float speed)
    {
        if (animator != null)
        {
            animator.SetFloat("Speed", speed * walkSpeed);
        }
    }

    public void Turn(float direction)
    {
        if (animator != null)
        {
            animator.SetFloat("Turn", direction * turnSpeed);
        }
    }

    void Update()
    {
        // Update all joint controllers
        foreach (var controller in jointControllers)
        {
            controller.UpdateJoint();
        }
    }
}

[System.Serializable]
public class JointController
{
    public string jointName;
    public Transform jointTransform;
    public float currentAngle;
    public float targetAngle;
    public float rotationSpeed = 100f;

    public void SetTargetAngle(float angle)
    {
        targetAngle = angle;
    }

    public void UpdateJoint()
    {
        // Smoothly rotate toward target angle
        currentAngle = Mathf.MoveTowards(currentAngle, targetAngle,
                                       rotationSpeed * Time.deltaTime * Mathf.Deg2Rad);

        // Apply rotation (assuming rotation around local Z axis)
        jointTransform.localRotation = Quaternion.Euler(0, 0, currentAngle * Mathf.Rad2Deg);
    }
}
```

## Advanced Unity Features for Humanoid Robotics

### 1. Visual Perception Simulation

```csharp
// Camera sensor simulation
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityCameraSensor : MonoBehaviour
{
    [Header("Camera Settings")]
    [SerializeField] private Camera cameraComponent;
    [SerializeField] private string rosTopic = "/camera/image_raw";
    [SerializeField] private int width = 640;
    [SerializeField] private int height = 480;

    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.instance;

        // Create render texture
        renderTexture = new RenderTexture(width, height, 24);
        cameraComponent.targetTexture = renderTexture;

        // Create texture for reading
        texture2D = new Texture2D(width, height, TextureFormat.RGB24, false);
    }

    void Update()
    {
        // Capture image from camera
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        texture2D.Apply();

        // Convert to ROS message and publish
        SendImageToROS();

        RenderTexture.active = null;
    }

    void SendImageToROS()
    {
        // Convert Unity texture to ROS Image message
        byte[] imageData = texture2D.EncodeToPNG();

        ImageMsg rosImage = new ImageMsg();
        rosImage.header = new std_msgs.HeaderMsg();
        rosImage.header.stamp = new TimeStamp(ROSConnection.GetServerTime());
        rosImage.header.frame_id = "camera_frame";

        rosImage.height = (uint)height;
        rosImage.width = (uint)width;
        rosImage.encoding = "rgb8";
        rosImage.is_bigendian = 0;
        rosImage.step = (uint)(width * 3); // 3 bytes per pixel (RGB)

        // Convert byte array to ROS message format
        rosImage.data = imageData;

        ros.Publish(rosTopic, rosImage);
    }
}
```

### 2. Physics-Based Simulation

```csharp
// Physics-based humanoid controller
using UnityEngine;

public class PhysicsBasedHumanoidController : MonoBehaviour
{
    [Header("Balance Control")]
    [SerializeField] private Transform centerOfMass;
    [SerializeField] private float balanceKp = 100f;
    [SerializeField] private float balanceKd = 10f;

    [Header("Walking Parameters")]
    [SerializeField] private float stepHeight = 0.1f;
    [SerializeField] private float stepLength = 0.3f;
    [SerializeField] private float walkSpeed = 0.5f;

    private Rigidbody[] rigidbodies;
    private Vector3 initialCOMPosition;
    private Vector3 previousCOMVelocity;

    void Start()
    {
        // Collect all rigidbodies in the robot
        rigidbodies = GetComponentsInChildren<Rigidbody>();

        // Set center of mass
        if (centerOfMass != null)
        {
            foreach (Rigidbody rb in rigidbodies)
            {
                rb.centerOfMass = centerOfMass.localPosition;
            }
        }

        initialCOMPosition = CalculateCenterOfMass();
    }

    void FixedUpdate()
    {
        // Balance control
        ApplyBalanceControl();

        // Walking gait (simplified)
        ApplyWalkingPattern();
    }

    Vector3 CalculateCenterOfMass()
    {
        Vector3 com = Vector3.zero;
        float totalMass = 0f;

        foreach (Rigidbody rb in rigidbodies)
        {
            com += rb.position * rb.mass;
            totalMass += rb.mass;
        }

        return com / totalMass;
    }

    void ApplyBalanceControl()
    {
        Vector3 currentCOM = CalculateCenterOfMass();
        Vector3 comVelocity = (currentCOM - previousCOMVelocity) / Time.fixedDeltaTime;

        // Calculate balance error
        Vector3 balanceError = currentCOM - initialCOMPosition;

        // Apply corrective forces based on balance error
        foreach (Rigidbody rb in rigidbodies)
        {
            Vector3 correctiveForce = -balanceKp * balanceError - balanceKd * comVelocity;

            // Apply force at center of mass
            rb.AddForceAtPosition(correctiveForce, rb.worldCenterOfMass, ForceMode.Force);
        }

        previousCOMVelocity = currentCOM;
    }

    void ApplyWalkingPattern()
    {
        // Simplified walking pattern
        // In a real implementation, this would use inverse kinematics
        // and more sophisticated gait planning

        // Move feet in walking pattern
        Transform leftFoot = FindChildByName("left_foot");
        Transform rightFoot = FindChildByName("right_foot");

        if (leftFoot != null && rightFoot != null)
        {
            // Apply walking motion to feet
            float time = Time.time;

            // Left foot trajectory
            Vector3 leftFootPos = leftFoot.position;
            leftFootPos.y += Mathf.Sin(time * walkSpeed) * stepHeight * 0.5f;
            leftFootPos.x += Mathf.Cos(time * walkSpeed) * stepLength * 0.5f;
            leftFoot.position = leftFootPos;

            // Right foot trajectory (opposite phase)
            Vector3 rightFootPos = rightFoot.position;
            rightFootPos.y += Mathf.Sin(time * walkSpeed + Mathf.PI) * stepHeight * 0.5f;
            rightFootPos.x += Mathf.Cos(time * walkSpeed) * stepLength * 0.5f;
            rightFoot.position = rightFootPos;
        }
    }

    Transform FindChildByName(string name)
    {
        Transform[] children = GetComponentsInChildren<Transform>();
        foreach (Transform child in children)
        {
            if (child.name.Contains(name))
                return child;
        }
        return null;
    }
}
```

## Integration with AI and Machine Learning

### 1. ML-Agents for Humanoid Training

```csharp
// ML-Agents Academy for humanoid robot training
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;

public class HumanoidRobotAcademy : Unity.MLAgents.Academy
{
    public override void InitializeAcademy()
    {
        // Initialize academy settings
        // Called once when the Academy is started
    }

    public override void AcademyStep()
    {
        // Called every step of the engine
        // Use for any academy-wide logic
    }

    public override void AcademyReset()
    {
        // Called when the Academy resets
        // Reset any academy-wide parameters
    }
}

public class HumanoidRobotAgent : Agent
{
    [Header("Agent Configuration")]
    [SerializeField] private Transform target;
    [SerializeField] private float moveSpeed = 5f;
    [SerializeField] private float rotationSpeed = 100f;

    [Header("Sensors")]
    [SerializeField] private float sensorRange = 10f;
    [SerializeField] private int sensorCount = 8;

    private Rigidbody[] rigidbodies;
    private Vector3 initialPosition;

    public override void Initialize()
    {
        rigidbodies = GetComponentsInChildren<Rigidbody>();
        initialPosition = transform.position;
    }

    public override void OnEpisodeBegin()
    {
        // Reset agent to initial state
        transform.position = initialPosition;
        transform.rotation = Quaternion.identity;

        // Reset all rigidbody velocities
        foreach (Rigidbody rb in rigidbodies)
        {
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Add observations for the neural network
        // Self position and rotation
        sensor.AddObservation(transform.position);
        sensor.AddObservation(transform.rotation.eulerAngles);

        // Velocity
        sensor.AddObservation(GetVelocity());

        // Joint angles and velocities
        CollectJointObservations(sensor);

        // Distance to target
        sensor.AddObservation(Vector3.Distance(transform.position, target.position));

        // Sensor rays for environment awareness
        CollectSensorRays(sensor);
    }

    void CollectJointObservations(VectorSensor sensor)
    {
        // Add joint angles and velocities to observations
        foreach (Rigidbody rb in rigidbodies)
        {
            // Position relative to root
            sensor.AddObservation(rb.transform.localPosition);

            // Velocity
            sensor.AddObservation(rb.velocity);

            // Angular velocity
            sensor.AddObservation(rb.angularVelocity);
        }
    }

    void CollectSensorRays(VectorSensor sensor)
    {
        // Add sensor rays for obstacle detection
        for (int i = 0; i < sensorCount; i++)
        {
            float angle = (float)i / sensorCount * Mathf.PI * 2f;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));

            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, sensorRange))
            {
                sensor.AddObservation(hit.distance / sensorRange); // Normalized distance
            }
            else
            {
                sensor.AddObservation(1f); // No obstacle detected
            }
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Process actions from the neural network
        float[] continuousActions = actions.ContinuousActions.ToArray();

        // Apply actions to joints
        ApplyActionsToJoints(continuousActions);

        // Calculate reward
        float reward = CalculateReward();
        SetReward(reward);

        // Check for terminal conditions
        if (IsTerminalCondition())
        {
            EndEpisode();
        }
    }

    void ApplyActionsToJoints(float[] actions)
    {
        // Apply continuous actions to robot joints
        // This is a simplified example - real implementation would use
        // inverse kinematics or joint-level control

        if (actions.Length >= 2)
        {
            // Move forward/backward
            transform.Translate(Vector3.forward * actions[0] * moveSpeed * Time.fixedDeltaTime);

            // Rotate left/right
            transform.Rotate(Vector3.up, actions[1] * rotationSpeed * Time.fixedDeltaTime);
        }
    }

    float CalculateReward()
    {
        float reward = 0f;

        // Reward for moving toward target
        float distanceToTarget = Vector3.Distance(transform.position, target.position);
        reward += (10f - distanceToTarget) * 0.1f; // Closer = higher reward

        // Penalty for falling
        if (transform.position.y < initialPosition.y - 1f)
        {
            reward -= 1f; // Fallen down
        }

        // Reward for staying upright
        if (Mathf.Abs(transform.rotation.eulerAngles.x) < 30f)
        {
            reward += 0.01f; // Small reward for staying upright
        }

        return reward;
    }

    bool IsTerminalCondition()
    {
        // Check if the agent should end the episode
        return transform.position.y < initialPosition.y - 2f; // Fallen too far
    }

    Vector3 GetVelocity()
    {
        // Calculate average velocity of the robot
        Vector3 avgVelocity = Vector3.zero;
        foreach (Rigidbody rb in rigidbodies)
        {
            avgVelocity += rb.velocity;
        }
        return avgVelocity / rigidbodies.Length;
    }

    public override float[] Heuristic(in ActionBuffers actionsOut)
    {
        // Manual control for testing (WASD + mouse)
        var continuousActionsOut = actionsOut.ContinuousActions;

        continuousActionsOut[0] = Input.GetAxis("Vertical"); // Forward/backward
        continuousActionsOut[1] = Input.GetAxis("Horizontal"); // Left/right

        return continuousActionsOut.ToArray();
    }
}
```

## Performance Optimization

### 1. LOD (Level of Detail) System

```csharp
// LOD system for humanoid robots
using UnityEngine;

public class HumanoidLODController : MonoBehaviour
{
    [System.Serializable]
    public class LODLevel
    {
        public int screenPercentage;
        public GameObject lodObject;
        public Material[] materials;
    }

    [SerializeField] private LODLevel[] lodLevels;
    [SerializeField] private Camera referenceCamera;

    private int currentLOD = 0;

    void Start()
    {
        if (referenceCamera == null)
            referenceCamera = Camera.main;
    }

    void Update()
    {
        UpdateLOD();
    }

    void UpdateLOD()
    {
        if (referenceCamera == null) return;

        // Calculate screen size of object
        Vector3 screenPoint = referenceCamera.WorldToViewportPoint(transform.position);
        if (screenPoint.z <= 0) return; // Object is behind camera

        float distance = Vector3.Distance(transform.position, referenceCamera.transform.position);
        float objectSize = CalculateObjectScreenSize(distance);

        // Determine appropriate LOD level
        int newLOD = 0;
        for (int i = 0; i < lodLevels.Length; i++)
        {
            if (objectSize * 100 > lodLevels[i].screenPercentage)
            {
                newLOD = i;
            }
        }

        // Activate appropriate LOD level
        if (newLOD != currentLOD)
        {
            SetLOD(newLOD);
            currentLOD = newLOD;
        }
    }

    float CalculateObjectScreenSize(float distance)
    {
        // Simplified calculation - in practice, use bounding box
        float objectHeight = 1.8f; // Approximate humanoid height
        float screenHeight = 2.0f * Mathf.Tan(referenceCamera.fieldOfView * Mathf.Deg2Rad / 2.0f) * distance;
        return objectHeight / screenHeight;
    }

    void SetLOD(int lodIndex)
    {
        for (int i = 0; i < lodLevels.Length; i++)
        {
            if (lodLevels[i].lodObject != null)
            {
                lodLevels[i].lodObject.SetActive(i == lodIndex);
            }
        }
    }
}
```

### 2. Occlusion Culling and Frustum Culling

```csharp
// Advanced culling system
using UnityEngine;

public class AdvancedCullingSystem : MonoBehaviour
{
    [Header("Culling Settings")]
    [SerializeField] private float cullDistance = 50f;
    [SerializeField] private LayerMask cullingMask = -1;
    [SerializeField] private bool useOcclusionCulling = true;

    private Camera mainCamera;
    private Plane[] frustumPlanes = new Plane[6];

    void Start()
    {
        mainCamera = Camera.main;
    }

    void Update()
    {
        UpdateCulling();
    }

    void UpdateCulling()
    {
        if (mainCamera == null) return;

        // Update frustum planes
        GeometryUtility.CalculateFrustumPlanes(mainCamera, frustumPlanes);

        // Cull distant objects
        CullDistantObjects();

        // Apply occlusion culling if enabled
        if (useOcclusionCulling)
        {
            ApplyOcclusionCulling();
        }
    }

    void CullDistantObjects()
    {
        // Disable rendering for objects beyond cull distance
        Renderer[] renderers = FindObjectsOfType<Renderer>();

        foreach (Renderer renderer in renderers)
        {
            float distance = Vector3.Distance(renderer.transform.position, mainCamera.transform.position);

            if (distance > cullDistance)
            {
                renderer.enabled = false;
            }
            else
            {
                // Check if in frustum
                if (GeometryUtility.TestPlanesAABB(frustumPlanes, renderer.bounds))
                {
                    renderer.enabled = true;
                }
                else
                {
                    renderer.enabled = false;
                }
            }
        }
    }

    void ApplyOcclusionCulling()
    {
        // Unity's built-in occlusion culling is enabled in the camera settings
        // This method can be extended with custom occlusion algorithms
    }
}
```

## Best Practices for Unity Robotics

### 1. Performance Considerations

- **Use Object Pooling**: For frequently created/destroyed objects
- **Optimize Colliders**: Use primitive colliders when possible
- **LOD Systems**: Implement level of detail for distant objects
- **Fixed Timestep**: Use consistent physics timestep for stable simulation

### 2. Architecture Patterns

- **Component-Based Design**: Use Unity's component system effectively
- **Event-Driven Communication**: Use UnityEvents for loose coupling
- **ScriptableObjects**: For configuration and data that doesn't change at runtime

### 3. Testing and Validation

- **Unit Testing**: Test individual components
- **Integration Testing**: Test ROS communication
- **Performance Testing**: Monitor frame rates and physics stability

## Next Steps

In the next section, we'll explore physics and dynamics in simulation, learning how to create realistic physical interactions for humanoid robots in both Gazebo and Unity environments.