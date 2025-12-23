---
sidebar_position: 4
title: "Cognitive Planning: LLMs for Action Translation"
---

# Cognitive Planning: LLMs for Action Translation

## Introduction to Cognitive Planning in Robotics

Cognitive planning represents the bridge between high-level natural language commands and low-level robot actions. It involves using sophisticated reasoning systems, particularly Large Language Models (LLMs), to decompose complex, natural language instructions into executable action sequences that robots can understand and execute.

Unlike traditional robotics approaches that rely on pre-programmed behavior trees or finite state machines, cognitive planning with LLMs enables robots to:
- Understand complex, multi-step commands in natural language
- Reason about the environment and available actions
- Adapt to novel situations without pre-programming
- Learn from experience and improve over time

### The Cognitive Planning Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     Cognitive Planning                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │  Natural        │    │  LLM-based     │    │  Action         │  │
│  │  Language       │───▶│  Reasoning     │───▶│  Sequence       │  │
│  │  Command        │    │  & Planning    │    │  Execution      │  │
│  │                 │    │                 │    │                 │  │
│  │  "Go to the    │    │  - Decompose    │    │  - Primitive    │  │
│  │  kitchen and    │    │    command      │    │    actions      │  │
│  │  bring me the   │    │  - Generate     │    │  - Task         │  │
│  │  red cup"      │    │    plan         │    │    coordination │  │
│  └─────────────────┘    │  - Validate     │    │  - Execution    │  │
│                         │    constraints  │    │    monitoring   │  │
│  ┌─────────────────┐    │  - Reason about │    │                 │  │
│  │  Environmental  │    │    context      │    │  ┌─────────────┐  │
│  │  Context       │───▶│  - Adapt to     │    │  │  Robot      │  │
│  │                 │    │    situation    │    │  │  Actions    │  │
│  │  - Robot state  │    │                 │    │  │             │  │
│  │  - Environment  │    │  ┌─────────────┐ │    │  │  - Move     │  │
│  │  - Objects      │    │  │  Plan       │ │    │  │  - Grasp    │  │
│  │  - Constraints  │    │  │  Validator │ │    │  │  - Speak    │  │
│  └─────────────────┘    │  └─────────────┘ │    │  │  - Navigate │  │
│                         └─────────────────┘    │  └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Understanding LLMs for Robot Planning

### Why LLMs for Cognitive Planning?

Large Language Models excel at cognitive planning for robotics because they possess:

1. **World Knowledge**: Trained on vast amounts of text, LLMs have implicit knowledge about objects, actions, and their relationships
2. **Reasoning Capabilities**: Can decompose complex tasks and reason about causality
3. **Natural Language Understanding**: Can interpret human commands in natural language
4. **Generalization**: Can handle novel combinations of known concepts
5. **Context Awareness**: Can maintain context across conversations and tasks

### Types of Planning with LLMs

#### 1. Task Planning
Breaking down high-level goals into sequences of subtasks:
```
Goal: "Clean the living room"
├── Find cleaning supplies
├── Pick up scattered items
├── Vacuum the floor
└── Organize furniture
```

#### 2. Motion Planning
Generating specific movement sequences:
```
Action: "Move the book to the shelf"
├── Navigate to book location
├── Align gripper with book
├── Grasp the book
├── Lift the book
├── Navigate to shelf
├── Align book with shelf position
└── Place the book on shelf
```

#### 3. Contingency Planning
Preparing for potential failures:
```
Plan: "Pour water from bottle to glass"
├── Approach bottle
├── Grasp bottle
├── Move to glass location
├── Pour water into glass
└── Handle contingencies:
    ├── If bottle is empty → Find water source
    ├── If glass is full → Stop pouring
    └── If spill occurs → Clean up
```

## Architecture for LLM-Based Cognitive Planning

### Core Components

```python
import openai
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

class ActionType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    COMMUNICATION = "communication"
    UTILITY = "utility"

@dataclass
class RobotAction:
    """Represents a single robot action"""
    action_type: ActionType
    parameters: Dict[str, Any]
    priority: int = 1
    estimated_duration: float = 0.0  # in seconds
    preconditions: List[str] = None
    effects: List[str] = None

@dataclass
class PlanStep:
    """A step in the plan"""
    action: RobotAction
    description: str
    dependencies: List[int] = None  # indices of steps this depends on

@dataclass
class CognitivePlan:
    """A complete cognitive plan"""
    steps: List[PlanStep]
    original_command: str
    estimated_completion_time: float
    confidence: float

class LLMCognitivePlanner:
    """Main cognitive planning system using LLMs"""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        openai.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)

        # Planning schema for structured output
        self.planning_schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action_type": {
                                "type": "string",
                                "enum": [action_type.value for action_type in ActionType]
                            },
                            "parameters": {"type": "object"},
                            "description": {"type": "string"},
                            "estimated_duration": {"type": "number"},
                            "priority": {"type": "integer"},
                            "preconditions": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "effects": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["action_type", "parameters", "description"]
                    }
                },
                "estimated_completion_time": {"type": "number"},
                "confidence": {"type": "number"}
            },
            "required": ["steps", "estimated_completion_time", "confidence"]
        }

    async def create_plan(self,
                         command: str,
                         robot_context: Dict[str, Any],
                         environment_context: Dict[str, Any]) -> Optional[CognitivePlan]:
        """
        Create a cognitive plan for the given command

        Args:
            command: Natural language command
            robot_context: Information about robot capabilities and state
            environment_context: Information about environment and objects

        Returns:
            CognitivePlan or None if planning fails
        """
        try:
            # Create structured prompt
            prompt = self._create_planning_prompt(command, robot_context, environment_context)

            # Call LLM with function
            response = await self._call_llm_with_schema(prompt)

            if response is None:
                return None

            # Convert LLM response to CognitivePlan
            plan = self._convert_to_cognitive_plan(response, command)

            self.logger.info(f"Created plan for command: {command}")
            return plan

        except Exception as e:
            self.logger.error(f"Error creating plan: {e}")
            return None

    def _create_planning_prompt(self, command: str, robot_context: Dict, environment_context: Dict) -> str:
        """Create structured prompt for planning"""
        prompt = f"""
        You are an expert cognitive planner for a robot. Given a natural language command,
        create a detailed plan of actions for the robot to execute.

        Robot Capabilities and Current State:
        {json.dumps(robot_context, indent=2)}

        Environment Information:
        {json.dumps(environment_context, indent=2)}

        Command: "{command}"

        Please provide a detailed plan with the following structure:
        - Break down the command into specific, executable actions
        - Specify action types from: navigation, manipulation, perception, communication, utility
        - Include necessary parameters for each action
        - Estimate duration for each action
        - Identify preconditions and effects
        - Assign priorities (1-5, where 5 is highest priority)

        Consider:
        - Robot's physical constraints and capabilities
        - Environmental constraints and obstacles
        - Safety requirements
        - Efficiency of the plan
        - Potential failure modes and contingency plans
        """

        return prompt

    async def _call_llm_with_schema(self, prompt: str) -> Optional[Dict]:
        """Call LLM with structured schema"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert cognitive planner for robots. Always respond with structured JSON following the provided schema."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                functions=[{
                    "name": "generate_robot_plan",
                    "description": "Generate a detailed robot action plan",
                    "parameters": self.planning_schema
                }],
                function_call={"name": "generate_robot_plan"},
                temperature=0.1  # Low temperature for consistency
            )

            # Extract function arguments
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            return function_args

        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            return None

    def _convert_to_cognitive_plan(self, llm_response: Dict, original_command: str) -> CognitivePlan:
        """Convert LLM response to CognitivePlan object"""
        steps = []

        for i, step_data in enumerate(llm_response['steps']):
            # Convert action type string to enum
            action_type = ActionType(step_data['action_type'])

            # Create RobotAction
            robot_action = RobotAction(
                action_type=action_type,
                parameters=step_data.get('parameters', {}),
                priority=step_data.get('priority', 1),
                estimated_duration=step_data.get('estimated_duration', 0.0),
                preconditions=step_data.get('preconditions', []),
                effects=step_data.get('effects', [])
            )

            # Create PlanStep
            plan_step = PlanStep(
                action=robot_action,
                description=step_data['description'],
                dependencies=step_data.get('dependencies', [])
            )

            steps.append(plan_step)

        return CognitivePlan(
            steps=steps,
            original_command=original_command,
            estimated_completion_time=llm_response['estimated_completion_time'],
            confidence=llm_response['confidence']
        )
```

### Context Management System

```python
class ContextManager:
    """Manages contextual information for cognitive planning"""

    def __init__(self):
        self.robot_state = {}
        self.environment_state = {}
        self.object_database = {}
        self.task_history = []
        self.current_plan = None

    def update_robot_state(self, state: Dict[str, Any]):
        """Update robot state information"""
        self.robot_state.update(state)

    def update_environment_state(self, state: Dict[str, Any]):
        """Update environment state information"""
        self.environment_state.update(state)

    def add_object(self, obj_id: str, properties: Dict[str, Any]):
        """Add object to database"""
        self.object_database[obj_id] = {
            'id': obj_id,
            'properties': properties,
            'last_seen': self.get_current_time(),
            'location': properties.get('location', 'unknown')
        }

    def get_relevant_context(self, command: str) -> Dict[str, Any]:
        """Get context relevant to the current command"""
        # Analyze command to determine relevant context
        relevant_objects = self._find_relevant_objects(command)
        relevant_locations = self._find_relevant_locations(command)

        context = {
            'robot_capabilities': self._get_robot_capabilities(),
            'robot_current_state': self.robot_state,
            'environment_map': self.environment_state.get('map', {}),
            'relevant_objects': relevant_objects,
            'relevant_locations': relevant_locations,
            'recent_interactions': self.task_history[-5:],  # Last 5 tasks
            'current_time': self.get_current_time(),
            'safety_constraints': self._get_safety_constraints()
        }

        return context

    def _find_relevant_objects(self, command: str) -> List[Dict[str, Any]]:
        """Find objects relevant to the command"""
        relevant = []
        command_lower = command.lower()

        for obj_id, obj_data in self.object_database.items():
            # Check if object name appears in command
            obj_name = obj_data['properties'].get('name', '').lower()
            obj_type = obj_data['properties'].get('type', '').lower()

            if obj_name in command_lower or obj_type in command_lower:
                relevant.append(obj_data)

        return relevant

    def _find_relevant_locations(self, command: str) -> List[str]:
        """Find locations relevant to the command"""
        # Simple keyword matching - in practice, this would use more sophisticated NLP
        location_keywords = ['kitchen', 'living room', 'bedroom', 'office', 'bathroom', 'hallway']
        relevant = []

        command_lower = command.lower()
        for location in location_keywords:
            if location in command_lower:
                relevant.append(location)

        return relevant

    def _get_robot_capabilities(self) -> Dict[str, Any]:
        """Get robot's current capabilities"""
        return {
            'navigation': {
                'max_speed': 0.5,
                'min_turn_radius': 0.2,
                'sensors': ['lidar', 'camera', 'imu']
            },
            'manipulation': {
                'max_payload': 2.0,
                'reach': 1.2,
                'gripper_type': 'parallel_jaw'
            },
            'communication': {
                'speaking': True,
                'listening': True,
                'languages': ['English']
            }
        }

    def _get_safety_constraints(self) -> Dict[str, Any]:
        """Get current safety constraints"""
        return {
            'max_speed': 0.3,
            'min_distance_to_people': 0.5,
            'restricted_areas': [],
            'emergency_stop': True
        }

    def get_current_time(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()
```

## Advanced Planning Techniques

### Hierarchical Task Network (HTN) Planning

```python
class HTNPlanner:
    """Hierarchical Task Network planner using LLMs"""

    def __init__(self, cognitive_planner: LLMCognitivePlanner):
        self.cognitive_planner = cognitive_planner
        self.task_methods = self._initialize_task_methods()

    def _initialize_task_methods(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize methods for decomposing high-level tasks"""
        return {
            'fetch_object': [
                {
                    'name': 'basic_fetch',
                    'subtasks': [
                        {'action': 'navigate_to', 'parameters': {'target': 'object_location'}},
                        {'action': 'perceive_object', 'parameters': {'target': 'object'}},
                        {'action': 'grasp_object', 'parameters': {'object': 'object'}},
                        {'action': 'navigate_to', 'parameters': {'target': 'delivery_location'}}
                    ]
                }
            ],
            'clean_area': [
                {
                    'name': 'tidy_clean',
                    'subtasks': [
                        {'action': 'scan_area', 'parameters': {'area': 'target_area'}},
                        {'action': 'identify_items', 'parameters': {'area': 'target_area'}},
                        {'action': 'organize_items', 'parameters': {'items': 'scattered_items'}},
                        {'action': 'vacuum_floor', 'parameters': {'area': 'target_area'}}
                    ]
                }
            ],
            'prepare_food': [
                {
                    'name': 'simple_prepare',
                    'subtasks': [
                        {'action': 'navigate_to', 'parameters': {'target': 'kitchen'}},
                        {'action': 'identify_ingredients', 'parameters': {}},
                        {'action': 'assemble_ingredients', 'parameters': {}},
                        {'action': 'cook_food', 'parameters': {'recipe': 'simple_recipe'}}
                    ]
                }
            ]
        }

    async def create_hierarchical_plan(self,
                                     high_level_task: str,
                                     context: Dict[str, Any]) -> Optional[CognitivePlan]:
        """Create a hierarchical plan using HTN methods"""
        try:
            # Decompose high-level task into subtasks
            subtasks = await self._decompose_task(high_level_task, context)

            # Convert subtasks to detailed actions
            detailed_plan = await self._create_detailed_plan(subtasks, context)

            return detailed_plan

        except Exception as e:
            print(f"Error in HTN planning: {e}")
            return None

    async def _decompose_task(self, task: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose a high-level task into subtasks"""
        # Use LLM to decompose the task
        decomposition_prompt = f"""
        Decompose the following high-level task into subtasks that can be executed by a robot:

        Task: "{task}"

        Robot Capabilities: {json.dumps(context.get('robot_capabilities', {}), indent=2)}
        Environment: {json.dumps(context.get('environment_map', {}), indent=2)}
        Available Objects: {json.dumps(context.get('relevant_objects', []), indent=2)}

        Please provide a decomposition of the task into 3-7 subtasks that can be executed sequentially.
        Each subtask should be specific enough to be understood by a robot.

        Respond with a JSON array of subtasks.
        """

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at decomposing complex tasks into subtasks for robots. Respond with a JSON array of subtasks."
                    },
                    {
                        "role": "user",
                        "content": decomposition_prompt
                    }
                ],
                temperature=0.1
            )

            subtasks = json.loads(response.choices[0].message.content)
            return subtasks

        except Exception as e:
            print(f"Error decomposing task: {e}")
            # Fallback: return simple decomposition
            return [{"action": "unknown", "description": f"Handle task: {task}"}]

    async def _create_detailed_plan(self, subtasks: List[Dict], context: Dict[str, Any]) -> CognitivePlan:
        """Create detailed plan from subtasks"""
        all_steps = []

        for subtask in subtasks:
            # Create detailed plan for each subtask
            command = subtask.get('description', str(subtask))

            # Use cognitive planner to create detailed action plan
            detailed_plan = await self.cognitive_planner.create_plan(
                command=command,
                robot_context=context.get('robot_capabilities', {}),
                environment_context=context
            )

            if detailed_plan:
                all_steps.extend(detailed_plan.steps)

        # Combine all steps into a single plan
        combined_plan = CognitivePlan(
            steps=all_steps,
            original_command=subtasks[0].get('description', 'Combined plan'),
            estimated_completion_time=sum(step.action.estimated_duration for step in all_steps),
            confidence=min(step.action.parameters.get('confidence', 1.0) for step in all_steps) if all_steps else 0.0
        )

        return combined_plan
```

### Reactive Planning and Adaptation

```python
class ReactivePlanner:
    """Handles plan adaptation when execution fails or conditions change"""

    def __init__(self, cognitive_planner: LLMCognitivePlanner):
        self.cognitive_planner = cognitive_planner
        self.execution_history = []

    async def handle_execution_failure(self,
                                     plan: CognitivePlan,
                                     failed_step_index: int,
                                     failure_reason: str,
                                     current_context: Dict[str, Any]) -> Optional[CognitivePlan]:
        """Handle plan failure and create adapted plan"""

        # Get the failed step
        failed_step = plan.steps[failed_step_index]

        print(f"Plan failure at step {failed_step_index}: {failure_reason}")
        print(f"Failed step: {failed_step.description}")

        # Analyze failure and create recovery plan
        recovery_plan = await self._create_recovery_plan(
            failed_step,
            failure_reason,
            current_context
        )

        if recovery_plan:
            # Integrate recovery plan with remaining plan
            adapted_plan = await self._integrate_recovery_plan(
                plan,
                failed_step_index,
                recovery_plan
            )
            return adapted_plan

        return None

    async def _create_recovery_plan(self,
                                  failed_step: PlanStep,
                                  failure_reason: str,
                                  context: Dict[str, Any]) -> Optional[CognitivePlan]:
        """Create recovery plan for failed step"""

        recovery_prompt = f"""
        A robot's plan failed at a specific step. Create a recovery plan to handle the failure.

        Failed Step: "{failed_step.description}"
        Action Type: {failed_step.action.action_type.value}
        Parameters: {json.dumps(failed_step.action.parameters)}
        Failure Reason: "{failure_reason}"

        Current Context:
        - Robot State: {json.dumps(context.get('robot_current_state', {}))}
        - Environment: {json.dumps(context.get('environment_map', {}))}
        - Available Objects: {json.dumps(context.get('relevant_objects', []))}

        Please create a recovery plan that addresses the failure and allows the robot to continue with its original goal.
        Consider alternative approaches and available resources.

        Respond with a JSON plan for the recovery action.
        """

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating recovery plans for robots. Respond with a JSON plan for recovery."
                    },
                    {
                        "role": "user",
                        "content": recovery_prompt
                    }
                ],
                temperature=0.2
            )

            recovery_data = json.loads(response.choices[0].message.content)

            # Convert to CognitivePlan
            recovery_steps = []
            for step_data in recovery_data.get('steps', []):
                action_type = ActionType(step_data['action_type'])
                robot_action = RobotAction(
                    action_type=action_type,
                    parameters=step_data.get('parameters', {}),
                    priority=step_data.get('priority', 1),
                    estimated_duration=step_data.get('estimated_duration', 0.0)
                )

                plan_step = PlanStep(
                    action=robot_action,
                    description=step_data['description']
                )

                recovery_steps.append(plan_step)

            return CognitivePlan(
                steps=recovery_steps,
                original_command=f"Recovery for: {failed_step.description}",
                estimated_completion_time=recovery_data.get('estimated_completion_time', 0.0),
                confidence=recovery_data.get('confidence', 0.5)
            )

        except Exception as e:
            print(f"Error creating recovery plan: {e}")
            return None

    async def _integrate_recovery_plan(self,
                                     original_plan: CognitivePlan,
                                     failed_step_index: int,
                                     recovery_plan: CognitivePlan) -> CognitivePlan:
        """Integrate recovery plan with original plan"""

        # Create new plan by combining:
        # 1. Successful steps before failure
        # 2. Recovery steps
        # 3. Remaining steps after failure

        new_steps = []

        # Add successful steps before failure
        new_steps.extend(original_plan.steps[:failed_step_index])

        # Add recovery steps
        new_steps.extend(recovery_plan.steps)

        # Add remaining steps after failure (potentially modified)
        remaining_steps = original_plan.steps[failed_step_index + 1:]

        # Potentially adjust remaining steps based on recovery
        adjusted_remaining = await self._adjust_remaining_steps(
            remaining_steps,
            recovery_plan
        )

        new_steps.extend(adjusted_remaining)

        return CognitivePlan(
            steps=new_steps,
            original_command=original_plan.original_command,
            estimated_completion_time=sum(step.action.estimated_duration for step in new_steps),
            confidence=min(step.action.parameters.get('confidence', 1.0) for step in new_steps) if new_steps else 0.0
        )

    async def _adjust_remaining_steps(self,
                                    remaining_steps: List[PlanStep],
                                    recovery_plan: CognitivePlan) -> List[PlanStep]:
        """Adjust remaining steps based on recovery actions"""
        # In a real implementation, this would analyze how the recovery
        # affects the remaining steps and make necessary adjustments
        # For now, return the steps unchanged
        return remaining_steps

    async def handle_context_change(self,
                                  current_plan: CognitivePlan,
                                  new_context: Dict[str, Any]) -> Optional[CognitivePlan]:
        """Handle changes in environment/context during plan execution"""

        # Check if context change affects current plan
        if await self._context_change_affects_plan(current_plan, new_context):
            # Recreate plan with new context
            new_plan = await self.cognitive_planner.create_plan(
                command=current_plan.original_command,
                robot_context=new_context.get('robot_capabilities', {}),
                environment_context=new_context
            )
            return new_plan

        return current_plan

    async def _context_change_affects_plan(self,
                                         plan: CognitivePlan,
                                         new_context: Dict[str, Any]) -> bool:
        """Check if context change affects the current plan"""
        # Analyze if new context invalidates any plan assumptions
        # This is a simplified check - in practice, this would be more sophisticated
        return False  # Placeholder
```

## Integration with Robot Execution

### Plan Execution Monitor

```python
import asyncio
from typing import Callable, Awaitable
import time

class PlanExecutionMonitor:
    """Monitors plan execution and handles interruptions"""

    def __init__(self,
                 robot_interface,  # Robot-specific interface
                 reactive_planner: ReactivePlanner,
                 context_manager: ContextManager):
        self.robot_interface = robot_interface
        self.reactive_planner = reactive_planner
        self.context_manager = context_manager
        self.current_plan = None
        self.current_step_index = 0
        self.execution_active = False

        # Callbacks for external events
        self.on_plan_start: Optional[Callable] = None
        self.on_step_complete: Optional[Callable] = None
        self.on_plan_complete: Optional[Callable] = None
        self.on_failure: Optional[Callable] = None

    async def execute_plan(self, plan: CognitivePlan) -> bool:
        """Execute a cognitive plan step by step"""

        self.current_plan = plan
        self.current_step_index = 0
        self.execution_active = True

        if self.on_plan_start:
            await self.on_plan_start(plan)

        success = True

        while self.current_step_index < len(plan.steps) and self.execution_active:
            current_step = plan.steps[self.current_step_index]

            try:
                # Execute the current step
                step_success = await self._execute_step(current_step)

                if step_success:
                    # Update context with step completion
                    await self._update_context_after_step(current_step)

                    if self.on_step_complete:
                        await self.on_step_complete(current_step, self.current_step_index)

                    self.current_step_index += 1

                    # Small delay between steps
                    await asyncio.sleep(0.1)

                else:
                    # Step failed - handle failure
                    success = await self._handle_step_failure(current_step, self.current_step_index)
                    if not success:
                        break

            except Exception as e:
                print(f"Error executing step {self.current_step_index}: {e}")
                success = False
                break

        self.execution_active = False

        if success and self.on_plan_complete:
            await self.on_plan_complete(plan, True)
        elif not success and self.on_failure:
            await self.on_failure(plan, self.current_step_index)

        return success

    async def _execute_step(self, step: PlanStep) -> bool:
        """Execute a single plan step"""
        print(f"Executing step: {step.description}")

        try:
            # Convert plan step to robot command
            robot_command = self._convert_to_robot_command(step.action)

            # Execute command via robot interface
            result = await self.robot_interface.execute_command(robot_command)

            return result.get('success', False)

        except Exception as e:
            print(f"Error executing step: {e}")
            return False

    def _convert_to_robot_command(self, action: RobotAction) -> Dict[str, Any]:
        """Convert RobotAction to robot-specific command"""
        # This would be robot-specific implementation
        # For example, converting to ROS messages, API calls, etc.

        command = {
            'action_type': action.action_type.value,
            'parameters': action.parameters,
            'priority': action.priority
        }

        return command

    async def _update_context_after_step(self, step: PlanStep):
        """Update context after step completion"""
        # Update robot state based on action effects
        if step.action.effects:
            new_state = {}
            for effect in step.action.effects:
                # Parse effect and update state
                # This is a simplified example
                if 'moved_to' in effect:
                    location = effect.split('moved_to ')[-1]
                    new_state['current_location'] = location

            if new_state:
                self.context_manager.update_robot_state(new_state)

    async def _handle_step_failure(self, failed_step: PlanStep, step_index: int) -> bool:
        """Handle step failure and attempt recovery"""
        print(f"Step failed: {failed_step.description}")

        # Get current context
        current_context = self.context_manager.get_relevant_context(
            self.current_plan.original_command
        )

        # Attempt recovery
        recovery_plan = await self.reactive_planner.handle_execution_failure(
            plan=self.current_plan,
            failed_step_index=step_index,
            failure_reason="Step execution failed",  # In practice, get real failure reason
            current_context=current_context
        )

        if recovery_plan:
            print("Recovery plan created, executing...")
            # Execute recovery plan
            recovery_success = await self.execute_plan(recovery_plan)

            if recovery_success:
                # Resume original plan from the failed step
                remaining_plan = CognitivePlan(
                    steps=self.current_plan.steps[step_index + 1:],
                    original_command=self.current_plan.original_command,
                    estimated_completion_time=sum(s.action.estimated_duration for s in self.current_plan.steps[step_index + 1:]),
                    confidence=self.current_plan.confidence
                )

                return await self.execute_plan(remaining_plan)

        return False

    def interrupt_execution(self):
        """Interrupt current plan execution"""
        self.execution_active = False

    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        return {
            'active': self.execution_active,
            'current_plan': self.current_plan.original_command if self.current_plan else None,
            'current_step': self.current_step_index,
            'total_steps': len(self.current_plan.steps) if self.current_plan else 0,
            'progress': self.current_step_index / len(self.current_plan.steps) if self.current_plan and len(self.current_plan.steps) > 0 else 0
        }
```

## Multi-Modal Cognitive Planning

### Integration with Vision and Perception

```python
class MultiModalCognitivePlanner(LLMCognitivePlanner):
    """Cognitive planner that integrates visual and perceptual information"""

    def __init__(self, api_key: str, model: str = "gpt-4-vision-preview"):
        super().__init__(api_key, model)
        self.perception_interface = None  # Would be connected to robot perception
        self.vision_schema = {
            "type": "object",
            "properties": {
                "objects_detected": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "category": {"type": "string"},
                            "location": {"type": "array", "items": {"type": "number"}},
                            "confidence": {"type": "number"}
                        }
                    }
                },
                "scene_description": {"type": "string"},
                "relevant_features": {"type": "array", "items": {"type": "string"}}
            }
        }

    async def create_plan_with_vision(self,
                                    command: str,
                                    image_data: Optional[bytes] = None) -> Optional[CognitivePlan]:
        """Create plan with visual context"""

        # Get visual information
        vision_context = await self._analyze_vision_data(image_data)

        # Get environmental context
        env_context = self.context_manager.get_relevant_context(command)

        # Combine contexts
        combined_context = {
            **env_context,
            'visual_context': vision_context
        }

        # Create plan with combined context
        return await self.create_plan(command, {}, combined_context)

    async def _analyze_vision_data(self, image_data: Optional[bytes]) -> Dict[str, Any]:
        """Analyze image data using vision-capable LLM"""
        if not image_data:
            return {'objects_detected': [], 'scene_description': 'No visual data provided'}

        try:
            # Convert image to base64 for API
            import base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            response = await openai.ChatCompletion.acreate(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image and describe the objects, their locations, and relevant features for a robot to understand the scene. Respond with structured JSON."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                functions=[{
                    "name": "describe_scene",
                    "description": "Describe objects and scene features",
                    "parameters": self.vision_schema
                }],
                function_call={"name": "describe_scene"},
                max_tokens=500
            )

            vision_data = json.loads(response.choices[0].message.function_call.arguments)
            return vision_data

        except Exception as e:
            print(f"Error analyzing vision data: {e}")
            return {'objects_detected': [], 'scene_description': 'Error analyzing image'}

    def update_object_knowledge(self, vision_data: Dict[str, Any]):
        """Update object database with new visual information"""
        for obj in vision_data.get('objects_detected', []):
            obj_id = obj.get('name', f"object_{len(self.context_manager.object_database)}")
            properties = {
                'name': obj.get('name'),
                'category': obj.get('category'),
                'location': obj.get('location'),
                'confidence': obj.get('confidence')
            }
            self.context_manager.add_object(obj_id, properties)
```

## Safety and Validation

### Plan Validation System

```python
class PlanValidator:
    """Validates plans for safety and feasibility"""

    def __init__(self):
        self.safety_rules = self._load_safety_rules()
        self.feasibility_rules = self._load_feasibility_rules()

    def _load_safety_rules(self) -> List[Dict[str, Any]]:
        """Load safety validation rules"""
        return [
            {
                'name': 'collision_avoidance',
                'condition': lambda action: action.action_type == ActionType.NAVIGATION,
                'validation': self._validate_navigation_safety
            },
            {
                'name': 'manipulation_safety',
                'condition': lambda action: action.action_type == ActionType.MANIPULATION,
                'validation': self._validate_manipulation_safety
            },
            {
                'name': 'speed_limits',
                'condition': lambda action: action.action_type in [ActionType.NAVIGATION, ActionType.MANIPULATION],
                'validation': self._validate_speed_limits
            }
        ]

    def _load_feasibility_rules(self) -> List[Dict[str, Any]]:
        """Load feasibility validation rules"""
        return [
            {
                'name': 'capability_check',
                'condition': lambda action: True,
                'validation': self._validate_robot_capabilities
            },
            {
                'name': 'precondition_check',
                'condition': lambda action: len(action.preconditions) > 0,
                'validation': self._validate_preconditions
            }
        ]

    def validate_plan(self, plan: CognitivePlan, robot_capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Validate entire plan"""
        results = {
            'overall_valid': True,
            'safety_issues': [],
            'feasibility_issues': [],
            'warnings': []
        }

        for i, step in enumerate(plan.steps):
            step_results = self.validate_step(step, robot_capabilities)

            if not step_results['safety_valid']:
                results['safety_issues'].extend([
                    f"Step {i}: {issue}" for issue in step_results['safety_issues']
                ])
                results['overall_valid'] = False

            if not step_results['feasibility_valid']:
                results['feasibility_issues'].extend([
                    f"Step {i}: {issue}" for issue in step_results['feasibility_issues']
                ])
                results['overall_valid'] = False

            if step_results['warnings']:
                results['warnings'].extend([
                    f"Step {i}: {warning}" for warning in step_results['warnings']
                ])

        return results

    def validate_step(self, step: PlanStep, robot_capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual plan step"""
        results = {
            'safety_valid': True,
            'feasibility_valid': True,
            'safety_issues': [],
            'feasibility_issues': [],
            'warnings': []
        }

        # Check safety rules
        for rule in self.safety_rules:
            if rule['condition'](step.action):
                is_safe, issues = rule['validation'](step.action, robot_capabilities)
                if not is_safe:
                    results['safety_valid'] = False
                    results['safety_issues'].extend(issues)

        # Check feasibility rules
        for rule in self.feasibility_rules:
            if rule['condition'](step.action):
                is_feasible, issues = rule['validation'](step.action, robot_capabilities)
                if not is_feasible:
                    results['feasibility_valid'] = False
                    results['feasibility_issues'].extend(issues)

        return results

    def _validate_navigation_safety(self, action: RobotAction, capabilities: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate navigation action safety"""
        issues = []

        # Check speed limits
        max_speed = capabilities.get('navigation', {}).get('max_speed', 0.5)
        requested_speed = action.parameters.get('speed', 0.3)

        if requested_speed > max_speed:
            issues.append(f"Requested speed {requested_speed}m/s exceeds maximum {max_speed}m/s")

        # Check for safety constraints in parameters
        if 'avoid_people' not in action.parameters:
            action.parameters['avoid_people'] = True

        return len(issues) == 0, issues

    def _validate_manipulation_safety(self, action: RobotAction, capabilities: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate manipulation action safety"""
        issues = []

        # Check payload limits
        max_payload = capabilities.get('manipulation', {}).get('max_payload', 2.0)
        requested_payload = action.parameters.get('payload', 1.0)

        if requested_payload > max_payload:
            issues.append(f"Requested payload {requested_payload}kg exceeds maximum {max_payload}kg")

        return len(issues) == 0, issues

    def _validate_speed_limits(self, action: RobotAction, capabilities: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate speed limits for actions"""
        issues = []

        # Generic speed validation
        speed_param = action.parameters.get('speed')
        if speed_param:
            max_speed = capabilities.get('navigation', {}).get('max_speed', 0.5)
            if speed_param > max_speed:
                issues.append(f"Speed parameter {speed_param} exceeds safe limits")

        return len(issues) == 0, issues

    def _validate_robot_capabilities(self, action: RobotAction, capabilities: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate that robot can perform the action"""
        issues = []

        action_type = action.action_type
        params = action.parameters

        if action_type == ActionType.NAVIGATION:
            nav_caps = capabilities.get('navigation', {})
            if not nav_caps.get('enabled', True):
                issues.append("Navigation capability is disabled")

        elif action_type == ActionType.MANIPULATION:
            manip_caps = capabilities.get('manipulation', {})
            if not manip_caps.get('enabled', True):
                issues.append("Manipulation capability is disabled")

        return len(issues) == 0, issues

    def _validate_preconditions(self, action: RobotAction, capabilities: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate action preconditions"""
        issues = []

        for precondition in action.preconditions:
            # This would check if preconditions are satisfied
            # For now, we'll just warn about unverified preconditions
            issues.append(f"Precondition not verified: {precondition}")

        # In a real implementation, this would check current state
        # against preconditions and return appropriate validation
        return True, []  # Placeholder
```

## Performance Optimization

### Efficient Planning Strategies

```python
import time
from functools import wraps

def timing_decorator(func):
    """Decorator to time planning operations"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f}s")
        return result
    return wrapper

class OptimizedCognitivePlanner(LLMCognitivePlanner):
    """Optimized version with caching and efficiency improvements"""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        super().__init__(api_key, model)
        self.plan_cache = {}  # Cache for similar plans
        self.template_cache = {}  # Cache for plan templates
        self.response_cache = {}  # Cache for LLM responses

    @timing_decorator
    async def create_plan(self, command: str, robot_context: Dict, environment_context: Dict) -> Optional[CognitivePlan]:
        """Create plan with caching and optimization"""

        # Create cache key
        cache_key = self._create_cache_key(command, robot_context, environment_context)

        # Check cache first
        cached_plan = self._get_cached_plan(cache_key)
        if cached_plan:
            print("Using cached plan")
            return cached_plan

        # Proceed with normal planning
        plan = await super().create_plan(command, robot_context, environment_context)

        # Cache the result if successful
        if plan:
            self._cache_plan(cache_key, plan)

        return plan

    def _create_cache_key(self, command: str, robot_context: Dict, environment_context: Dict) -> str:
        """Create cache key for the command and context"""
        import hashlib
        import json

        # Simplify context for caching (exclude volatile information)
        simplified_context = {
            'robot_caps': {k: v for k, v in robot_context.items() if k != 'timestamp'},
            'env_map': environment_context.get('map', {}).get('layout', 'unknown')
        }

        cache_string = f"{command}_{json.dumps(simplified_context, sort_keys=True)}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _get_cached_plan(self, cache_key: str) -> Optional[CognitivePlan]:
        """Retrieve plan from cache"""
        if cache_key in self.plan_cache:
            cached_item = self.plan_cache[cache_key]
            # Check if cache is still valid (not too old)
            if time.time() - cached_item['timestamp'] < 3600:  # 1 hour cache
                return cached_item['plan']
            else:
                # Remove expired cache
                del self.plan_cache[cache_key]
        return None

    def _cache_plan(self, cache_key: str, plan: CognitivePlan):
        """Cache a plan"""
        self.plan_cache[cache_key] = {
            'plan': plan,
            'timestamp': time.time()
        }

    async def batch_create_plans(self, commands: List[str], context: Dict) -> List[Optional[CognitivePlan]]:
        """Create multiple plans efficiently"""
        import asyncio

        # Create tasks for concurrent execution
        tasks = [
            self.create_plan(cmd, context, context)
            for cmd in commands
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        plans = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Error creating plan: {result}")
                plans.append(None)
            else:
                plans.append(result)

        return plans
```

## Integration with Real Systems

### ROS 2 Integration Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from action_msgs.msg import GoalStatus
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class CognitivePlannerROS2Node(Node):
    """ROS 2 node for cognitive planning"""

    def __init__(self):
        super().__init__('cognitive_planner_node')

        # Initialize cognitive planning components
        api_key = self.declare_parameter('openai_api_key', '').value
        if not api_key:
            self.get_logger().error("OpenAI API key not provided!")
            return

        self.cognitive_planner = LLMCognitivePlanner(api_key)
        self.context_manager = ContextManager()
        self.reactive_planner = ReactivePlanner(self.cognitive_planner)
        self.execution_monitor = PlanExecutionMonitor(
            robot_interface=self,  # This would be a real robot interface
            reactive_planner=self.reactive_planner,
            context_manager=self.context_manager
        )

        # Publishers and subscribers
        self.voice_command_sub = self.create_subscription(
            String, '/voice_command', self.voice_command_callback, 10)
        self.plan_status_pub = self.create_publisher(
            String, '/plan_status', 10)

        # Action servers for planning and execution
        self.plan_server = ActionServer(
            self,
            PlanAction,  # Custom action type
            'generate_plan',
            self.execute_plan_callback,
            callback_group=MutuallyExclusiveCallbackGroup())

        self.get_logger().info('Cognitive Planner node initialized')

    def voice_command_callback(self, msg: String):
        """Handle voice commands"""
        command = msg.data
        self.get_logger().info(f'Received voice command: {command}')

        # Create and execute plan
        self.create_and_execute_plan(command)

    async def create_and_execute_plan(self, command: str):
        """Create and execute a plan for the given command"""
        try:
            # Get current context
            context = self.context_manager.get_relevant_context(command)

            # Create plan
            plan = await self.cognitive_planner.create_plan(
                command=command,
                robot_context=context.get('robot_capabilities', {}),
                environment_context=context
            )

            if plan:
                self.get_logger().info(f'Plan created with {len(plan.steps)} steps')

                # Execute plan
                success = await self.execution_monitor.execute_plan(plan)

                if success:
                    self.get_logger().info('Plan executed successfully')
                else:
                    self.get_logger().error('Plan execution failed')
            else:
                self.get_logger().error('Failed to create plan')

        except Exception as e:
            self.get_logger().error(f'Error in plan creation/execution: {e}')

    async def execute_plan_callback(self, goal_handle):
        """Execute plan action callback"""
        self.get_logger().info('Executing plan goal')

        command = goal_handle.request.command

        # Create and execute plan
        await self.create_and_execute_plan(command)

        # Set result
        goal_handle.succeed()
        result = PlanActionResult()  # Custom result type
        result.success = True
        return result

    def execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a robot command (placeholder for real robot interface)"""
        # This would interface with the actual robot
        # For now, return a mock response
        return {'success': True, 'message': 'Command executed'}

def main(args=None):
    rclpy.init(args=args)

    cognitive_planner_node = CognitivePlannerROS2Node()

    executor = MultiThreadedExecutor()
    rclpy.spin(cognitive_planner_node, executor=executor)

    cognitive_planner_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Cognitive Planning

### 1. Planning Hierarchy
- Use hierarchical planning for complex tasks
- Break down high-level goals into manageable subtasks
- Maintain clear separation between strategic and tactical planning

### 2. Context Management
- Keep context up-to-date during plan execution
- Handle context changes gracefully
- Maintain relevant history for decision making

### 3. Error Handling
- Implement robust error detection and recovery
- Use multiple validation layers
- Provide graceful degradation when possible

### 4. Performance Optimization
- Cache frequently used plans
- Use appropriate LLM models for different tasks
- Implement efficient prompting strategies

### 5. Safety and Validation
- Always validate plans before execution
- Implement safety constraints and limits
- Monitor execution and intervene when necessary

## Troubleshooting Common Issues

### 1. Planning Failures
- **Problem**: LLM fails to generate valid plans
- **Solution**: Improve prompts, use structured schemas, implement fallback strategies

### 2. Execution Failures
- **Problem**: Planned actions fail during execution
- **Solution**: Implement robust reactive planning, improve perception integration

### 3. Performance Issues
- **Problem**: Slow planning or execution
- **Solution**: Optimize caching, use smaller models for simple tasks, implement parallel processing

### 4. Context Confusion
- **Problem**: Robot loses track of state or context
- **Solution**: Implement proper state management, frequent context updates, error recovery

## Future Directions

### 1. Learning from Experience
- Plan adaptation based on execution outcomes
- Learning from human demonstrations
- Improving through trial and error

### 2. Multi-Agent Coordination
- Coordinating multiple robots
- Distributed planning and execution
- Communication and synchronization

### 3. Real-World Integration
- Better integration with real environments
- Improved perception and localization
- Enhanced safety systems

## Summary

Cognitive planning with LLMs represents a significant advancement in robotics, enabling robots to understand and execute complex, natural language commands. Key concepts include:

- **Hierarchical Planning**: Breaking down complex tasks into manageable subtasks
- **Context Awareness**: Understanding and adapting to environmental conditions
- **Reactive Planning**: Handling failures and changing conditions during execution
- **Multi-Modal Integration**: Combining language, vision, and other sensory information
- **Safety and Validation**: Ensuring safe and reliable plan execution

The integration of LLMs with robotics enables more intuitive human-robot interaction and allows robots to operate in complex, dynamic environments without extensive pre-programming. As these systems mature, they will enable more capable and adaptable robotic assistants.

In the next section, we'll explore capstone projects that integrate all the concepts we've learned into comprehensive humanoid robotics applications.