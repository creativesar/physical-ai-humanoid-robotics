---
sidebar_position: 5
title: "Physics and Dynamics"
---

# Physics and Dynamics

## Introduction to Physics Simulation in Robotics

Physics simulation is fundamental to humanoid robotics, as it determines how robots interact with the environment and respond to forces. Accurate physics simulation is essential for developing robust control algorithms, testing robot behaviors safely, and achieving successful simulation-to-reality transfer. This module explores the principles and implementation of physics and dynamics in humanoid robotics simulation.

## Fundamentals of Rigid Body Dynamics

### Newton-Euler Equations

The motion of rigid bodies is governed by Newton's second law and Euler's rotation equations:

#### Linear Motion
```
F = m * a
```
Where:
- F is the net force applied to the body
- m is the mass of the body
- a is the linear acceleration

#### Angular Motion
```
τ = I * α
```
Where:
- τ is the net torque applied to the body
- I is the moment of inertia tensor
- α is the angular acceleration

### Rigid Body State

A rigid body in 3D space is characterized by:

#### Position State
- **Position**: (x, y, z) coordinates in world space
- **Orientation**: Represented by quaternions or rotation matrices
- **Linear velocity**: (vx, vy, vz)
- **Angular velocity**: (ωx, ωy, ωz)

#### Dynamic Properties
- **Mass**: Scalar value representing resistance to linear acceleration
- **Inertia tensor**: 3x3 matrix representing resistance to angular acceleration
- **Center of mass**: Point where mass can be considered concentrated

## Physics Engines in Robotics Simulation

### 1. Open Dynamics Engine (ODE)

ODE is one of the most commonly used physics engines in robotics simulation:

#### Key Features
- **Fast collision detection**: Optimized for robotic applications
- **Joint constraints**: Various joint types with limits and motors
- **Real-time performance**: Suitable for interactive simulation
- **Stable integration**: Implicit methods for stability

#### ODE Configuration for Humanoid Robots

```xml
<!-- Gazebo ODE configuration -->
<physics type="ode">
  <max_step_size>0.001</max_step_size>        <!-- 1ms time step -->
  <real_time_factor>1.0</real_time_factor>    <!-- Real-time simulation -->
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>

  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>                          <!-- Constraint Force Mixing -->
      <erp>0.2</erp>                          <!-- Error Reduction Parameter -->
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### 2. Bullet Physics

Bullet physics offers advanced features for complex robotic systems:

#### Key Features
- **Multi-body dynamics**: Efficient simulation of articulated systems
- **Soft body simulation**: For flexible components
- **Vehicle dynamics**: Specialized for wheeled robots
- **High-performance**: Optimized for parallel processing

### 3. NVIDIA PhysX (Unity)

Unity's physics engine provides high-fidelity simulation:

#### Key Features
- **GPU-accelerated physics**: For large-scale simulations
- **Advanced collision detection**: Continuous collision detection
- **Destruction and fracture**: For complex interactions
- **Fluid simulation**: For environmental modeling

## Collision Detection and Response

### Collision Detection Algorithms

#### 1. Broad Phase
- **Spatial partitioning**: Grid, octree, or bounding volume hierarchies
- **Sweep and prune**: Sort objects along axes to find potential collisions
- **Performance**: O(n) to O(n log n) complexity

#### 2. Narrow Phase
- **GJK Algorithm**: Gilbert-Johnson-Keerthi for convex shapes
- **SAT (Separating Axis Theorem)**: For collision detection between convex polyhedra
- **Performance**: O(1) to O(n) complexity depending on shape

### Contact Resolution

#### Impulse-Based Resolution
```cpp
// Simplified contact resolution algorithm
void ResolveContact(Contact& contact, float deltaTime) {
    // Calculate relative velocity at contact point
    Vector3 relativeVelocity = contact.body2->GetVelocityAtPoint(contact.point) -
                              contact.body1->GetVelocityAtPoint(contact.point);

    // Calculate impulse magnitude
    float normalVelocity = Vector3::Dot(relativeVelocity, contact.normal);
    if (normalVelocity > 0) return; // Objects separating

    // Calculate impulse
    float restitution = 0.2f; // Bounciness
    float impulseMagnitude = -(1 + restitution) * normalVelocity;
    impulseMagnitude /= (1/contact.body1->GetInvMass() + 1/contact.body2->GetInvMass());

    // Apply impulse
    Vector3 impulse = contact.normal * impulseMagnitude;
    contact.body1->ApplyImpulse(-impulse, contact.point);
    contact.body2->ApplyImpulse(impulse, contact.point);
}
```

### Friction Modeling

#### Coulomb Friction
```cpp
// Friction force calculation
Vector3 CalculateFrictionForce(const Contact& contact, float deltaTime) {
    // Calculate tangential velocity
    Vector3 relativeVelocity = contact.body2->GetVelocityAtPoint(contact.point) -
                              contact.body1->GetVelocityAtPoint(contact.point);
    Vector3 tangentialVelocity = relativeVelocity -
                                contact.normal * Vector3::Dot(relativeVelocity, contact.normal);

    // Calculate friction force
    float frictionCoeff = 0.5f; // Coefficient of friction
    float normalForce = contact.normalForce;
    float maxFrictionForce = frictionCoeff * normalForce;

    // Apply friction within limits
    float tangentialSpeed = tangentialVelocity.Length();
    if (tangentialSpeed > 0) {
        Vector3 frictionDir = -tangentialVelocity.Normalized();
        Vector3 frictionForce = frictionDir * std::min(maxFrictionForce, tangentialSpeed * 10.0f);
        return frictionForce;
    }

    return Vector3::Zero();
}
```

## Humanoid-Specific Physics Considerations

### 1. Balance and Stability

#### Center of Mass (CoM) Dynamics
```cpp
// Center of mass calculation for humanoid
Vector3 CalculateHumanoidCoM(const std::vector<BodyPart>& bodyParts) {
    Vector3 totalMomentum = Vector3::Zero();
    float totalMass = 0.0f;

    for (const auto& part : bodyParts) {
        totalMomentum += part.position * part.mass;
        totalMass += part.mass;
    }

    if (totalMass > 0) {
        return totalMomentum / totalMass;
    }
    return Vector3::Zero();
}

// Balance control using CoM
class BalanceController {
private:
    Vector3 desiredCoM;
    float kp_balance = 100.0f;
    float ki_balance = 10.0f;
    float kd_balance = 5.0f;

    Vector3 integral_error = Vector3::Zero();
    Vector3 previous_error = Vector3::Zero();

public:
    Vector3 CalculateBalanceCorrection(const Vector3& currentCoM, float deltaTime) {
        Vector3 error = desiredCoM - currentCoM;

        // PID control
        integral_error += error * deltaTime;
        Vector3 derivative_error = (error - previous_error) / deltaTime;

        Vector3 correction = kp_balance * error +
                           ki_balance * integral_error +
                           kd_balance * derivative_error;

        previous_error = error;
        return correction;
    }
};
```

### 2. Contact Modeling for Bipedal Locomotion

#### Foot-Ground Contact
```xml
<!-- Gazebo contact model for humanoid feet -->
<gazebo reference="left_foot_pad">
  <collision>
    <surface>
      <contact>
        <ode>
          <soft_cfm>0.001</soft_cfm>
          <soft_erp>0.8</soft_erp>
          <kp>1e+6</kp>  <!-- Contact stiffness -->
          <kd>1e+3</kd>  <!-- Contact damping -->
          <max_vel>100.0</max_vel>
          <min_depth>0.001</min_depth>
        </ode>
      </contact>
      <friction>
        <ode>
          <mu>0.8</mu>    <!-- Coefficient of friction -->
          <mu2>0.8</mu2>
          <fdir1>0 0 0</fdir1>
        </ode>
      </friction>
    </surface>
  </collision>
</gazebo>
```

### 3. Joint Dynamics and Actuator Modeling

#### Joint Compliance and Damping
```xml
<!-- Joint dynamics configuration -->
<joint name="knee_joint" type="revolute">
  <parent link="thigh"/>
  <child link="shin"/>
  <axis xyz="1 0 0"/>
  <limit lower="-0.1" upper="2.3" effort="100" velocity="3.0"/>
  <dynamics damping="1.0" friction="0.5" spring_reference="0" spring_stiffness="1000"/>
</joint>

<!-- In Gazebo plugin -->
<gazebo reference="knee_joint">
  <joint>
    <dynamics>
      <damping>1.0</damping>
      <friction>0.5</friction>
      <spring_reference>0.0</spring_reference>
      <spring_stiffness>1000.0</spring_stiffness>
    </dynamics>
  </joint>
</gazebo>
```

## Advanced Physics Concepts for Humanoid Robotics

### 1. Whole-Body Dynamics

#### Lagrangian Formulation
The dynamics of a humanoid robot can be described by the Lagrangian equation:

```
M(q)q̈ + C(q, q̇)q̇ + g(q) = τ + J^T * F
```

Where:
- M(q) is the mass matrix
- C(q, q̇) contains Coriolis and centrifugal terms
- g(q) is the gravity vector
- τ is the joint torques
- J is the Jacobian matrix
- F is the external forces

#### Implementation Example
```cpp
// Simplified whole-body dynamics computation
class WholeBodyDynamics {
private:
    MatrixXf massMatrix;
    VectorXf coriolisVector;
    VectorXf gravityVector;
    int numJoints;

public:
    // Compute mass matrix using Composite Rigid Body Algorithm
    MatrixXf ComputeMassMatrix(const VectorXf& q) {
        // Implementation of CRBA
        // This is a simplified placeholder
        MatrixXf M = MatrixXf::Identity(numJoints, numJoints);

        // In practice, this would involve recursive computation
        // through the kinematic tree

        return M;
    }

    // Compute Coriolis and centrifugal terms
    VectorXf ComputeCoriolisVector(const VectorXf& q, const VectorXf& qdot) {
        // Implementation of inverse dynamics
        VectorXf C = VectorXf::Zero(numJoints);

        // Compute using Christoffel symbols or RNEA
        return C;
    }

    // Compute gravity terms
    VectorXf ComputeGravityVector(const VectorXf& q) {
        VectorXf g = VectorXf::Zero(numJoints);

        // Compute gravity effects for each joint
        // based on link positions and masses

        return g;
    }

    // Forward dynamics: compute accelerations from torques
    VectorXf ForwardDynamics(const VectorXf& q, const VectorXf& qdot,
                           const VectorXf& tau, const VectorXf& externalForces) {
        MatrixXf M = ComputeMassMatrix(q);
        VectorXf C = ComputeCoriolisVector(q, qdot);
        VectorXf g = ComputeGravityVector(q);

        // M*qddot + C*qdot + g = tau + J^T*F
        // qddot = M^(-1) * (tau - C*qdot - g + J^T*F)

        VectorXf qddot = M.inverse() * (tau - C.cwiseProduct(qdot) - g + externalForces);
        return qddot;
    }
};
```

### 2. Contact Dynamics and Impacts

#### Rigid Impact Model
```cpp
// Contact dynamics for humanoid feet
class ContactDynamics {
public:
    struct ContactPoint {
        Vector3 position;
        Vector3 normal;
        float penetrationDepth;
        bool inContact;
    };

    // Detect and resolve contact
    void ProcessContact(const ContactPoint& contact, Body& body, float deltaTime) {
        if (contact.penetrationDepth > 0.0f) {
            // Apply contact forces to prevent penetration
            Vector3 contactForce = contact.normal * (contact.penetrationDepth * stiffness);

            // Add damping to prevent oscillation
            Vector3 velocityAtContact = body.GetVelocityAtPoint(contact.position);
            float normalVelocity = Vector3::Dot(velocityAtContact, contact.normal);
            if (normalVelocity < 0) { // Moving into contact
                contactForce += contact.normal * (normalVelocity * damping);
            }

            body.ApplyForceAtPoint(contactForce, contact.position);
        }
    }

private:
    float stiffness = 10000.0f;
    float damping = 100.0f;
};
```

### 3. Multi-Body Dynamics with Constraints

#### Constrained Dynamics
```cpp
// Handle closed-loop constraints in humanoid kinematic chains
class ConstraintHandler {
public:
    void SolveConstrainedDynamics(std::vector<Body>& bodies,
                                 std::vector<Constraint>& constraints) {
        // Build constraint Jacobian matrix
        MatrixXf J = BuildConstraintJacobian(constraints, bodies);

        // Constraint forces: λ = (J * M^(-1) * J^T)^(-1) * (J * M^(-1) * (g - C) - b)
        MatrixXf M_inv = BuildInverseMassMatrix(bodies);
        VectorXf g = ComputeGravityVector(bodies);
        VectorXf C = ComputeCoriolisVector(bodies);
        VectorXf b = ComputeConstraintBias(constraints);

        MatrixXf A = J * M_inv * J.transpose();
        VectorXf b_vec = J * M_inv * (g - C) - b;

        VectorXf lambda = A.inverse() * b_vec;

        // Apply constraint forces
        VectorXf constraintForces = J.transpose() * lambda;
        ApplyConstraintForces(bodies, constraintForces);
    }

private:
    MatrixXf BuildConstraintJacobian(const std::vector<Constraint>& constraints,
                                   const std::vector<Body>& bodies) {
        // Build Jacobian matrix for all constraints
        // Implementation depends on constraint types
        return MatrixXf::Zero(constraints.size(), bodies.size() * 6); // 6 DOF per body
    }
};
```

## Physics Simulation Parameters and Tuning

### 1. Time Step Selection

#### Stability vs. Accuracy Trade-offs
```xml
<!-- Physics configuration with different time step strategies -->
<physics type="ode">
  <!-- Fast simulation (less accurate) -->
  <max_step_size>0.01</max_step_size>        <!-- 10ms -->
  <real_time_update_rate>100</real_time_update_rate>

  <!-- Accurate simulation -->
  <max_step_size>0.001</max_step_size>       <!-- 1ms -->
  <real_time_update_rate>1000</real_time_update_rate>

  <!-- Very accurate simulation -->
  <max_step_size>0.0001</max_step_size>      <!-- 0.1ms -->
  <real_time_update_rate>10000</real_time_update_rate>
</physics>
```

### 2. Solver Parameters

#### Error Reduction and Constraint Force Mixing
```xml
<physics type="ode">
  <ode>
    <solver>
      <type>quick</type>
      <iters>20</iters>        <!-- More iterations = more accurate but slower -->
      <sor>1.0</sor>           <!-- Successive Over-Relaxation parameter -->
    </solver>
    <constraints>
      <cfm>1e-9</cfm>          <!-- Constraint Force Mixing -->
      <erp>0.2</erp>           <!-- Error Reduction Parameter -->
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### 3. Material Properties

#### Friction and Restitution Coefficients
```xml
<!-- Material properties for different surfaces -->
<gazebo reference="floor">
  <surface>
    <friction>
      <ode>
        <mu>0.8</mu>      <!-- Static friction coefficient -->
        <mu2>0.8</mu2>    <!-- Dynamic friction coefficient -->
      </ode>
    </friction>
    <bounce>
      <restitution_coefficient>0.1</restitution_coefficient>  <!-- Low bounciness -->
      <threshold>1.0</threshold>                              <!-- Bounce threshold -->
    </bounce>
  </surface>
</gazebo>

<!-- Humanoid foot material -->
<gazebo reference="foot_pad">
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>      <!-- High friction for good grip -->
        <mu2>0.9</mu2>
      </ode>
    </friction>
  </surface>
</gazebo>
```

## Physics Validation and Verification

### 1. Conservation Laws

#### Energy Conservation Testing
```cpp
// Verify physics simulation accuracy
class PhysicsValidator {
public:
    bool CheckEnergyConservation(const std::vector<Body>& bodies) {
        float kineticEnergy = CalculateKineticEnergy(bodies);
        float potentialEnergy = CalculatePotentialEnergy(bodies);
        float totalEnergy = kineticEnergy + potentialEnergy;

        // Check if energy is conserved (within numerical tolerance)
        static float previousEnergy = totalEnergy;
        float energyDrift = std::abs(totalEnergy - previousEnergy);

        if (energyDrift > energyTolerance) {
            // Energy is not conserved - simulation may be unstable
            return false;
        }

        previousEnergy = totalEnergy;
        return true;
    }

    bool CheckMomentumConservation(const std::vector<Body>& bodies) {
        Vector3 linearMomentum = CalculateLinearMomentum(bodies);
        Vector3 angularMomentum = CalculateAngularMomentum(bodies);

        // In absence of external forces, momentum should be conserved
        // Implementation depends on external force tracking

        return true; // Simplified check
    }

private:
    float energyTolerance = 0.01f;

    float CalculateKineticEnergy(const std::vector<Body>& bodies) {
        float energy = 0.0f;
        for (const auto& body : bodies) {
            energy += 0.5f * body.mass * body.linearVelocity.LengthSquared();
            energy += 0.5f * body.angularVelocity.Dot(body.inertiaTensor * body.angularVelocity);
        }
        return energy;
    }

    float CalculatePotentialEnergy(const std::vector<Body>& bodies) {
        float energy = 0.0f;
        Vector3 gravity(0, 0, -9.81f);
        for (const auto& body : bodies) {
            energy += body.mass * (-gravity.z) * body.position.z; // Assuming gravity in -Z direction
        }
        return energy;
    }
};
```

### 2. Analytical Validation

#### Comparing with Analytical Solutions
```cpp
// Validate simulation against analytical solutions
class AnalyticalValidator {
public:
    // Simple pendulum validation
    bool ValidatePendulum(float length, float mass, float initialAngle) {
        // Analytical period: T = 2π * sqrt(L/g)
        float analyticalPeriod = 2.0f * M_PI * sqrt(length / 9.81f);

        // Simulate pendulum and measure period
        float simulatedPeriod = MeasurePendulumPeriod(length, mass, initialAngle);

        // Compare with tolerance
        float tolerance = 0.05f; // 5% tolerance
        return std::abs(simulatedPeriod - analyticalPeriod) / analyticalPeriod < tolerance;
    }

    // Double pendulum validation (more complex)
    bool ValidateDoublePendulum() {
        // Compare with known chaotic behavior characteristics
        // Check energy conservation
        // Verify Lyapunov exponent behavior

        return true; // Implementation would be complex
    }

private:
    float MeasurePendulumPeriod(float length, float mass, float initialAngle) {
        // Implementation would run simulation and measure oscillation period
        return 2.0f * M_PI * sqrt(length / 9.81f); // Placeholder
    }
};
```

## Performance Optimization

### 1. Parallel Physics Simulation

#### Multi-threaded Physics Updates
```cpp
// Parallel physics update for humanoid simulation
#include <thread>
#include <vector>

class ParallelPhysicsEngine {
public:
    void UpdatePhysics(std::vector<Body>& bodies, float deltaTime) {
        int numThreads = std::thread::hardware_concurrency();
        int bodiesPerThread = bodies.size() / numThreads;

        std::vector<std::thread> threads;

        for (int i = 0; i < numThreads; ++i) {
            int start = i * bodiesPerThread;
            int end = (i == numThreads - 1) ? bodies.size() : (i + 1) * bodiesPerThread;

            threads.emplace_back([this, &bodies, start, end, deltaTime]() {
                UpdateBodiesRange(bodies, start, end, deltaTime);
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        // Handle inter-body interactions in single thread
        HandleInteractions(bodies, deltaTime);
    }

private:
    void UpdateBodiesRange(std::vector<Body>& bodies, int start, int end, float deltaTime) {
        for (int i = start; i < end; ++i) {
            bodies[i].Update(deltaTime);
        }
    }

    void HandleInteractions(std::vector<Body>& bodies, float deltaTime) {
        // Handle collisions and constraints between bodies
        // This needs to be thread-safe
    }
};
```

### 2. Adaptive Time Stepping

#### Variable Time Step Based on Dynamics
```cpp
// Adaptive time stepping for humanoid simulation
class AdaptiveTimestep {
public:
    float CalculateTimestep(const std::vector<Body>& bodies) {
        float minTimestep = maxTimestep;

        for (const auto& body : bodies) {
            // Calculate timestep based on highest frequency mode
            float bodyTimestep = CalculateBodyTimestep(body);
            minTimestep = std::min(minTimestep, bodyTimestep);
        }

        // Ensure timestep doesn't exceed maximum
        return std::max(minTimestep, minTimestepLimit);
    }

private:
    float CalculateBodyTimestep(const Body& body) {
        // Based on natural frequency of the system
        // For harmonic oscillator: dt < 2π / ω_max
        float maxFrequency = EstimateMaxFrequency(body);
        return 2.0f * M_PI / (maxFrequency * safetyFactor);
    }

    float EstimateMaxFrequency(const Body& body) {
        // Estimate based on stiffness and mass
        // This is a simplified estimation
        return sqrt(body.stiffness / body.mass);
    }

    float maxTimestep = 0.01f;
    float minTimestepLimit = 0.0001f;
    float safetyFactor = 10.0f;
};
```

## Real-time Physics Considerations

### 1. Fixed vs. Variable Timestep

#### Fixed Timestep for Determinism
```cpp
// Fixed timestep physics update
class FixedTimestepPhysics {
private:
    float fixedTimestep = 0.001f; // 1ms
    float accumulator = 0.0f;

public:
    void Update(float deltaTime, std::vector<Body>& bodies) {
        accumulator += deltaTime;

        while (accumulator >= fixedTimestep) {
            // Update physics with fixed timestep
            UpdatePhysics(bodies, fixedTimestep);
            accumulator -= fixedTimestep;
        }

        // Interpolate rendering based on accumulator
        float alpha = accumulator / fixedTimestep;
        InterpolateRendering(bodies, alpha);
    }

    void UpdatePhysics(std::vector<Body>& bodies, float dt) {
        // Apply forces
        ApplyForces(bodies);

        // Integrate equations of motion
        IntegrateMotion(bodies, dt);

        // Handle collisions
        DetectAndResolveCollisions(bodies);
    }
};
```

### 2. Physics Pipeline Optimization

#### Multi-stage Physics Update
```cpp
// Optimized physics pipeline for humanoid robots
class OptimizedPhysicsPipeline {
public:
    void UpdatePipeline(std::vector<Body>& bodies, float deltaTime) {
        // Stage 1: Broad phase collision detection (parallel)
        auto potentialCollisions = BroadPhaseDetection(bodies);

        // Stage 2: Narrow phase collision detection (parallel)
        auto actualCollisions = NarrowPhaseDetection(potentialCollisions);

        // Stage 3: Constraint solving (parallel where possible)
        SolveConstraints(actualCollisions, bodies);

        // Stage 4: Integration (parallel)
        IntegrateBodies(bodies, deltaTime);
    }

private:
    std::vector<Contact> BroadPhaseDetection(const std::vector<Body>& bodies) {
        // Use spatial partitioning for O(n) complexity
        // Return potential collision pairs
        return std::vector<Contact>();
    }

    std::vector<Contact> NarrowPhaseDetection(const std::vector<Contact>& candidates) {
        // Detailed collision detection for candidate pairs
        return std::vector<Contact>();
    }

    void SolveConstraints(const std::vector<Contact>& contacts, std::vector<Body>& bodies) {
        // Solve contact constraints using iterative methods
        for (int iteration = 0; iteration < maxIterations; ++iteration) {
            for (const auto& contact : contacts) {
                ResolveContact(contact, bodies);
            }
        }
    }

    int maxIterations = 10;
};
```

## Best Practices for Physics Simulation

### 1. Model Validation
- Validate models against analytical solutions
- Test with simple cases before complex ones
- Monitor conservation laws
- Check for numerical stability

### 2. Parameter Tuning
- Start with conservative parameters
- Gradually increase complexity
- Monitor simulation stability
- Use adaptive methods where appropriate

### 3. Performance Considerations
- Balance accuracy with performance
- Use appropriate collision geometries
- Implement efficient broad-phase algorithms
- Consider multi-threading for large systems

## Next Steps

In the next section, we'll explore simulation-to-reality transfer, learning how to bridge the gap between simulation and real-world humanoid robot deployment, including techniques for domain randomization, system identification, and controller adaptation.