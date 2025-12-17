---
sidebar_position: 4
title: "Language-Guided Action Planning"
---

# Language-Guided Action Planning

## Introduction to Language-Guided Action Planning

Language-guided action planning is a critical component of Vision-Language-Action (VLA) systems, enabling humanoid robots to interpret natural language commands and translate them into executable actions in complex environments. This capability allows robots to understand high-level instructions, reason about their meaning in the context of the current situation, and generate appropriate behavioral responses. The integration of language understanding with action planning enables more intuitive human-robot interaction and flexible task execution.

## Architecture of Language-Guided Planning Systems

### 1. Hierarchical Planning Architecture

The language-guided planning system operates at multiple levels of abstraction:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Natural Language Command                     │
│                    "Pick up the red cup from the table"         │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Language Understanding                       │
│  • Parse command structure                                      │
│  • Identify objects (red cup)                                   │
│  • Identify locations (table)                                   │
│  • Identify actions (pick up)                                   │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Task Decomposition                           │
│  • Break down into sub-tasks:                                   │
│    1. Navigate to table                                         │
│    2. Locate red cup                                            │
│    3. Plan grasp motion                                         │
│    4. Execute grasp                                             │
│    5. Lift cup                                                  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Action Generation                            │
│  • Generate low-level motor commands                            │
│  • Plan joint trajectories                                      │
│  • Execute coordinated movements                                │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Execution Monitoring                         │
│  • Track execution progress                                     │
│  • Handle exceptions and failures                               │
│  • Provide feedback to user                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Language Understanding Module

The language understanding module parses natural language commands and extracts structured information:

```python
# Advanced language understanding for robotic commands
import torch
import torch.nn as nn
import spacy
from transformers import AutoTokenizer, AutoModel
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class ParsedCommand:
    """Structured representation of a parsed command"""
    intent: str
    objects: List[Dict[str, str]]
    locations: List[Dict[str, str]]
    actions: List[str]
    attributes: Dict[str, str]
    spatial_relations: List[Tuple[str, str, str]]  # (entity1, relation, entity2)

class LanguageUnderstandingModule(nn.Module):
    def __init__(self, model_name='bert-base-uncased', nlp_model='en_core_web_sm'):
        super().__init__()

        # Load transformer model for semantic understanding
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.language_model = AutoModel.from_pretrained(model_name)

        # Load spaCy for syntactic parsing
        self.nlp = spacy.load(nlp_model)

        # Task-specific heads
        self.intent_classifier = nn.Linear(self.language_model.config.hidden_size, 50)  # 50 task types
        self.entity_recognizer = nn.Linear(self.language_model.config.hidden_size, 20)  # 20 entity types

        # Spatial relation extractor
        self.spatial_relation_extractor = nn.Sequential(
            nn.Linear(self.language_model.config.hidden_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 spatial relations (on, under, next_to, etc.)
        )

        # Action sequence predictor
        self.action_sequence_predictor = nn.Sequential(
            nn.Linear(self.language_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100)  # 100 possible actions, max sequence of 5
        )

    def forward(self, command_text: str) -> ParsedCommand:
        """
        Parse natural language command into structured representation
        """
        # Tokenize and encode command
        encoded = self.tokenizer(
            command_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )

        # Get language embeddings
        outputs = self.language_model(**encoded)
        sequence_output = outputs.last_hidden_state  # [1, seq_len, hidden_size]
        pooled_output = outputs.pooler_output  # [1, hidden_size]

        # Intent classification
        intent_logits = self.intent_classifier(pooled_output)
        intent_probs = torch.softmax(intent_logits, dim=-1)
        intent_id = torch.argmax(intent_probs, dim=-1).item()
        intent = self.get_intent_label(intent_id)

        # Entity recognition
        entity_logits = self.entity_recognizer(sequence_output)  # [1, seq_len, num_entities]
        entity_probs = torch.softmax(entity_logits, dim=-1)
        entities = self.extract_entities(command_text, entity_probs[0])

        # Spatial relation extraction
        spatial_relations = self.extract_spatial_relations(command_text, sequence_output[0])

        # Action sequence prediction
        action_logits = self.action_sequence_predictor(pooled_output)
        actions = self.decode_action_sequence(action_logits)

        # Parse using spaCy for syntactic structure
        doc = self.nlp(command_text)

        # Extract objects and locations from entities
        objects = [entity for entity in entities if entity['type'] == 'object']
        locations = [entity for entity in entities if entity['type'] == 'location']

        # Extract attributes (colors, sizes, etc.)
        attributes = self.extract_attributes(doc)

        return ParsedCommand(
            intent=intent,
            objects=objects,
            locations=locations,
            actions=actions,
            attributes=attributes,
            spatial_relations=spatial_relations
        )

    def extract_entities(self, command_text: str, entity_probs: torch.Tensor) -> List[Dict[str, str]]:
        """Extract named entities from command text"""
        doc = self.nlp(command_text)
        entities = []

        entity_types = ['object', 'location', 'person', 'direction', 'color', 'size']

        for i, token in enumerate(doc):
            if i < len(entity_probs):
                entity_type_id = torch.argmax(entity_probs[i]).item()

                # Map entity type ID to label
                if entity_type_id < len(entity_types):
                    entity_type = entity_types[entity_type_id]
                    confidence = entity_probs[i][entity_type_id].item()

                    if confidence > 0.5:  # Threshold
                        entities.append({
                            'text': token.text,
                            'type': entity_type,
                            'confidence': confidence,
                            'lemma': token.lemma_
                        })

        return entities

    def extract_spatial_relations(self, command_text: str, token_embeddings: torch.Tensor) -> List[Tuple[str, str, str]]:
        """Extract spatial relations between entities in the command"""
        doc = self.nlp(command_text)
        relations = []

        spatial_relations = ['on', 'under', 'next_to', 'in_front_of', 'behind', 'above', 'below']

        # Find spatial prepositions and their arguments
        for token in doc:
            if token.dep_ == 'prep' and token.lemma_ in spatial_relations:
                # Find the object of the preposition
                prep_obj = None
                for child in token.children:
                    if child.dep_ == 'pobj':
                        prep_obj = child.text
                        break

                # Find the subject that the preposition modifies
                subject = None
                for ancestor in token.ancestors:
                    if ancestor.pos_ == 'NOUN':
                        subject = ancestor.text
                        break

                if subject and prep_obj:
                    relations.append((subject, token.lemma_, prep_obj))

        return relations

    def decode_action_sequence(self, action_logits: torch.Tensor) -> List[str]:
        """Decode action sequence from logits"""
        action_ids = torch.argmax(action_logits.reshape(5, 20), dim=-1)  # 5 steps, 20 actions per step
        actions = []

        action_labels = [
            'move_to', 'grasp', 'release', 'rotate', 'lift', 'lower',
            'approach', 'retreat', 'turn', 'walk', 'stop', 'start',
            'open_gripper', 'close_gripper', 'point', 'wave', 'nod',
            'shake', 'look_at', 'track'
        ]

        for action_id in action_ids:
            if action_id < len(action_labels):
                actions.append(action_labels[action_id.item()])

        return actions

    def extract_attributes(self, doc) -> Dict[str, str]:
        """Extract attributes like colors, sizes from command"""
        attributes = {}

        for token in doc:
            if token.pos_ == 'ADJ':  # Adjectives often represent attributes
                # Look for nearby nouns to associate with
                for child in token.children:
                    if child.pos_ == 'NOUN':
                        if child.text not in attributes:
                            attributes[child.text] = token.text
                        break

        return attributes

    def get_intent_label(self, intent_id: int) -> str:
        """Map intent ID to label"""
        intent_labels = [
            'pick_up', 'put_down', 'move_to', 'navigate', 'grasp',
            'release', 'open', 'close', 'push', 'pull', 'lift',
            'lower', 'rotate', 'point', 'wave', 'speak', 'listen',
            'follow', 'stop', 'start', 'turn', 'walk', 'run',
            'jump', 'sit', 'stand', 'lie', 'crawl', 'climb',
            'descend', 'enter', 'exit', 'search', 'find', 'locate',
            'identify', 'recognize', 'inspect', 'examine', 'measure',
            'count', 'compare', 'match', 'sort', 'arrange', 'organize',
            'assemble', 'disassemble', 'repair', 'clean', 'cook',
            'serve', 'eat', 'drink', 'give', 'receive', 'carry'
        ]

        if intent_id < len(intent_labels):
            return intent_labels[intent_id]
        else:
            return 'unknown'
```

### 3. Task Decomposition Module

The task decomposition module breaks down high-level commands into executable sub-tasks:

```python
# Task decomposition for complex commands
class TaskDecomposer:
    def __init__(self):
        self.action_templates = self.load_action_templates()
        self.primitive_actions = self.define_primitive_actions()

    def decompose_task(self, parsed_command: ParsedCommand) -> List[Dict[str, any]]:
        """
        Decompose high-level command into sequence of primitive actions
        """
        if parsed_command.intent in ['pick_up', 'grasp']:
            return self.decompose_pick_up_task(parsed_command)
        elif parsed_command.intent in ['put_down', 'place']:
            return self.decompose_place_task(parsed_command)
        elif parsed_command.intent == 'navigate':
            return self.decompose_navigation_task(parsed_command)
        elif parsed_command.intent == 'move_to':
            return self.decompose_move_to_task(parsed_command)
        else:
            return self.decompose_generic_task(parsed_command)

    def decompose_pick_up_task(self, command: ParsedCommand) -> List[Dict[str, any]]:
        """Decompose pick-up task into sub-actions"""
        sub_tasks = []

        # 1. Navigate to object location
        if command.locations:
            sub_tasks.append({
                'action': 'navigate',
                'target': command.locations[0]['text'],
                'priority': 1,
                'preconditions': [],
                'effects': ['robot_at_location']
            })

        # 2. Identify the specific object to pick up
        if command.objects:
            sub_tasks.append({
                'action': 'identify_object',
                'target': command.objects[0]['text'],
                'attributes': command.attributes,
                'priority': 2,
                'preconditions': ['robot_at_location'],
                'effects': ['object_located']
            })

        # 3. Approach the object
        sub_tasks.append({
            'action': 'approach_object',
            'target': command.objects[0]['text'] if command.objects else None,
            'priority': 3,
            'preconditions': ['object_located'],
            'effects': ['robot_near_object']
        })

        # 4. Plan grasp motion
        sub_tasks.append({
            'action': 'plan_grasp',
            'target': command.objects[0]['text'] if command.objects else None,
            'priority': 4,
            'preconditions': ['robot_near_object'],
            'effects': ['grasp_plan_computed']
        })

        # 5. Execute grasp
        sub_tasks.append({
            'action': 'execute_grasp',
            'target': command.objects[0]['text'] if command.objects else None,
            'priority': 5,
            'preconditions': ['grasp_plan_computed'],
            'effects': ['object_grasped']
        })

        # 6. Lift object
        sub_tasks.append({
            'action': 'lift_object',
            'target': command.objects[0]['text'] if command.objects else None,
            'priority': 6,
            'preconditions': ['object_grasped'],
            'effects': ['object_lifted']
        })

        return sub_tasks

    def decompose_place_task(self, command: ParsedCommand) -> List[Dict[str, any]]:
        """Decompose place task into sub-actions"""
        sub_tasks = []

        # 1. Navigate to destination
        if command.locations:
            sub_tasks.append({
                'action': 'navigate',
                'target': command.locations[0]['text'],
                'priority': 1,
                'preconditions': [],
                'effects': ['robot_at_destination']
            })

        # 2. Position object above destination
        sub_tasks.append({
            'action': 'position_above',
            'target': command.locations[0]['text'] if command.locations else None,
            'priority': 2,
            'preconditions': ['robot_at_destination'],
            'effects': ['object_positioned']
        })

        # 3. Release object
        sub_tasks.append({
            'action': 'release_object',
            'priority': 3,
            'preconditions': ['object_positioned'],
            'effects': ['object_released']
        })

        # 4. Retract gripper
        sub_tasks.append({
            'action': 'retract_gripper',
            'priority': 4,
            'preconditions': ['object_released'],
            'effects': ['gripper_retracted']
        })

        return sub_tasks

    def decompose_navigation_task(self, command: ParsedCommand) -> List[Dict[str, any]]:
        """Decompose navigation task into sub-actions"""
        sub_tasks = []

        # 1. Parse destination
        destination = command.locations[0]['text'] if command.locations else command.objects[0]['text'] if command.objects else None

        # 2. Plan path to destination
        sub_tasks.append({
            'action': 'plan_path',
            'destination': destination,
            'priority': 1,
            'preconditions': [],
            'effects': ['path_planned']
        })

        # 3. Execute navigation
        sub_tasks.append({
            'action': 'execute_navigation',
            'destination': destination,
            'priority': 2,
            'preconditions': ['path_planned'],
            'effects': ['robot_arrived']
        })

        return sub_tasks

    def decompose_move_to_task(self, command: ParsedCommand) -> List[Dict[str, any]]:
        """Decompose move-to task into sub-actions"""
        return self.decompose_navigation_task(command)

    def decompose_generic_task(self, command: ParsedCommand) -> List[Dict[str, any]]:
        """Decompose generic task using template matching"""
        # Match command to action templates
        template_match = self.match_to_template(command)

        if template_match:
            return self.instantiate_template(template_match, command)
        else:
            # Fallback: use a general task decomposition
            return self.general_decomposition(command)

    def match_to_template(self, command: ParsedCommand):
        """Match command to predefined action templates"""
        # This would implement template matching logic
        # For now, return a simple match based on intent
        return self.action_templates.get(command.intent)

    def instantiate_template(self, template, command: ParsedCommand):
        """Instantiate action template with command-specific parameters"""
        # Replace placeholders in template with actual values from command
        instantiated_tasks = []

        for task_template in template:
            instantiated_task = task_template.copy()

            # Replace placeholders
            if '{object}' in str(instantiated_task):
                if command.objects:
                    instantiated_task = self.replace_placeholder(
                        instantiated_task, '{object}', command.objects[0]['text']
                    )

            if '{location}' in str(instantiated_task):
                if command.locations:
                    instantiated_task = self.replace_placeholder(
                        instantiated_task, '{location}', command.locations[0]['text']
                    )

            instantiated_tasks.append(instantiated_task)

        return instantiated_tasks

    def replace_placeholder(self, task, placeholder, value):
        """Replace placeholder in task dictionary with actual value"""
        import json
        task_str = json.dumps(task)
        task_str = task_str.replace(placeholder, value)
        return json.loads(task_str)

    def general_decomposition(self, command: ParsedCommand) -> List[Dict[str, any]]:
        """General task decomposition for unknown intents"""
        # A fallback decomposition strategy
        sub_tasks = []

        # 1. Understand the command context
        sub_tasks.append({
            'action': 'analyze_command',
            'command': str(command),
            'priority': 1,
            'preconditions': [],
            'effects': ['command_analyzed']
        })

        # 2. Plan high-level approach
        sub_tasks.append({
            'action': 'plan_approach',
            'intent': command.intent,
            'priority': 2,
            'preconditions': ['command_analyzed'],
            'effects': ['approach_planned']
        })

        # 3. Execute the planned approach
        sub_tasks.append({
            'action': 'execute_plan',
            'priority': 3,
            'preconditions': ['approach_planned'],
            'effects': ['task_attempted']
        })

        return sub_tasks

    def load_action_templates(self):
        """Load predefined action templates"""
        return {
            'pick_up': [
                {'action': 'navigate_to', 'target': '{location}', 'priority': 1},
                {'action': 'locate_object', 'target': '{object}', 'priority': 2},
                {'action': 'approach_object', 'target': '{object}', 'priority': 3},
                {'action': 'grasp_object', 'target': '{object}', 'priority': 4}
            ],
            'place': [
                {'action': 'navigate_to', 'target': '{location}', 'priority': 1},
                {'action': 'position_object', 'target': '{object}', 'location': '{location}', 'priority': 2},
                {'action': 'release_object', 'target': '{object}', 'priority': 3}
            ],
            'move_to': [
                {'action': 'plan_path_to', 'target': '{location}', 'priority': 1},
                {'action': 'navigate_to', 'target': '{location}', 'priority': 2}
            ]
        }

    def define_primitive_actions(self):
        """Define basic primitive actions that the robot can execute"""
        return {
            'navigate_to': {
                'parameters': ['target_location'],
                'preconditions': ['robot_is_mobile'],
                'effects': ['robot_at_target']
            },
            'locate_object': {
                'parameters': ['object_name'],
                'preconditions': ['camera_functional'],
                'effects': ['object_position_known']
            },
            'approach_object': {
                'parameters': ['object_name'],
                'preconditions': ['object_position_known'],
                'effects': ['robot_near_object']
            },
            'grasp_object': {
                'parameters': ['object_name'],
                'preconditions': ['robot_near_object', 'gripper_free'],
                'effects': ['object_grasped']
            },
            'release_object': {
                'parameters': [],
                'preconditions': ['object_grasped'],
                'effects': ['object_released', 'gripper_free']
            },
            'plan_path_to': {
                'parameters': ['target_location'],
                'preconditions': ['map_known'],
                'effects': ['path_computed']
            }
        }
```

## Action Planning Algorithms

### 1. Symbolic Planning

Symbolic planning uses logical representations to plan sequences of actions:

```python
# Symbolic planning for robotic tasks
class SymbolicPlanner:
    def __init__(self):
        self.operators = self.define_operators()
        self.state_variables = self.initialize_state_variables()

    def define_operators(self):
        """Define STRIPS-style operators for robotic actions"""
        return {
            'navigate': {
                'preconditions': [
                    'robot_at(?from)',
                    'connected(?from, ?to)',
                    'robot_mobile'
                ],
                'effects': [
                    '-robot_at(?from)',
                    '+robot_at(?to)'
                ]
            },
            'grasp': {
                'preconditions': [
                    'robot_at(?loc)',
                    'object_at(?obj, ?loc)',
                    'robot_has_free_gripper',
                    'grasp_possible(?obj)'
                ],
                'effects': [
                    '-robot_has_free_gripper',
                    '+robot_holding(?obj)',
                    '-object_at(?obj, ?loc)'
                ]
            },
            'place': {
                'preconditions': [
                    'robot_at(?loc)',
                    'robot_holding(?obj)',
                    'surface_available(?loc)'
                ],
                'effects': [
                    '+robot_has_free_gripper',
                    '-robot_holding(?obj)',
                    '+object_at(?obj, ?loc)'
                ]
            },
            'approach_object': {
                'preconditions': [
                    'robot_at(?robot_loc)',
                    'object_at(?obj, ?obj_loc)',
                    'accessible(?obj_loc, ?robot_loc)'
                ],
                'effects': [
                    '+robot_near(?obj)'
                ]
            }
        }

    def plan(self, initial_state, goal_state):
        """
        Plan a sequence of actions to achieve the goal
        """
        # Use forward state-space search
        return self.forward_search(initial_state, goal_state)

    def forward_search(self, initial_state, goal_state):
        """Forward state-space search for planning"""
        from collections import deque

        # Initialize search
        queue = deque([(initial_state, [])])  # (state, action_sequence)
        visited = set()
        visited.add(self.state_to_tuple(initial_state))

        while queue:
            current_state, action_sequence = queue.popleft()

            # Check if goal is satisfied
            if self.goal_satisfied(current_state, goal_state):
                return action_sequence

            # Apply applicable operators
            applicable_ops = self.get_applicable_operators(current_state)

            for op_name, bindings in applicable_ops:
                # Apply operator to get successor state
                successor_state = self.apply_operator(current_state, op_name, bindings)

                # Check if state has been visited
                state_tuple = self.state_to_tuple(successor_state)
                if state_tuple not in visited:
                    visited.add(state_tuple)
                    new_action_sequence = action_sequence + [(op_name, bindings)]
                    queue.append((successor_state, new_action_sequence))

        # No plan found
        return None

    def get_applicable_operators(self, state):
        """Get all applicable operators in the current state"""
        applicable = []

        for op_name, op_def in self.operators.items():
            # Get all possible bindings for the operator
            bindings_list = self.get_operator_bindings(op_def, state)

            for bindings in bindings_list:
                # Check if preconditions are satisfied
                if self.check_preconditions(op_def, state, bindings):
                    applicable.append((op_name, bindings))

        return applicable

    def get_operator_bindings(self, operator, state):
        """Get all possible variable bindings for an operator in the current state"""
        # This would implement unification and binding logic
        # For now, return a simple binding based on state predicates
        bindings = []

        # Example: if operator has variables ?obj and ?loc
        # Extract all objects and locations from state
        objects = self.extract_objects_from_state(state)
        locations = self.extract_locations_from_state(state)

        # Generate all combinations of bindings
        for obj in objects:
            for loc in locations:
                bindings.append({'?obj': obj, '?loc': loc})

        return bindings

    def check_preconditions(self, operator, state, bindings):
        """Check if operator preconditions are satisfied in state with bindings"""
        for precondition in operator['preconditions']:
            # Apply bindings to precondition
            bound_condition = self.apply_bindings(precondition, bindings)

            # Check if condition is positive or negative
            is_positive = not bound_condition.startswith('-')
            if not is_positive:
                bound_condition = bound_condition[1:]  # Remove negation

            # Check if predicate is in state
            predicate_in_state = self.predicate_in_state(bound_condition, state)

            if is_positive and not predicate_in_state:
                return False
            if not is_positive and predicate_in_state:
                return False

        return True

    def apply_operator(self, state, op_name, bindings):
        """Apply operator to state and return resulting state"""
        operator = self.operators[op_name]
        new_state = state.copy()

        for effect in operator['effects']:
            # Apply bindings to effect
            bound_effect = self.apply_bindings(effect, bindings)

            # Check if effect is positive or negative
            is_positive = not bound_effect.startswith('-')
            if not is_positive:
                bound_effect = bound_effect[1:]  # Remove negation

            if is_positive:
                # Add predicate to state
                new_state.add(bound_effect)
            else:
                # Remove predicate from state
                if bound_effect in new_state:
                    new_state.remove(bound_effect)

        return new_state

    def goal_satisfied(self, state, goal):
        """Check if goal state is satisfied in current state"""
        for goal_literal in goal:
            if goal_literal.startswith('-'):
                # Negative goal literal
                predicate = goal_literal[1:]
                if predicate in state:
                    return False
            else:
                # Positive goal literal
                if goal_literal not in state:
                    return False

        return True

    def apply_bindings(self, predicate, bindings):
        """Apply variable bindings to a predicate"""
        result = predicate
        for var, value in bindings.items():
            result = result.replace(var, value)
        return result

    def predicate_in_state(self, predicate, state):
        """Check if predicate is in state"""
        return predicate in state

    def extract_objects_from_state(self, state):
        """Extract all objects from state predicates"""
        objects = set()
        for pred in state:
            # Simple parsing: extract terms from predicate
            # Format: predicate(arg1, arg2, ...)
            if '(' in pred and ')' in pred:
                args_part = pred[pred.index('(')+1:pred.index(')')]
                args = [arg.strip() for arg in args_part.split(',')]
                objects.update(args)
        return list(objects)

    def extract_locations_from_state(self, state):
        """Extract all locations from state predicates"""
        locations = set()
        for pred in state:
            if 'at' in pred or 'location' in pred:
                # Extract location terms
                if '(' in pred and ')' in pred:
                    args_part = pred[pred.index('(')+1:pred.index(')')]
                    args = [arg.strip() for arg in args_part.split(',')]
                    # Assume second argument is usually the location
                    if len(args) > 1:
                        locations.add(args[1])
        return list(locations)

    def state_to_tuple(self, state):
        """Convert state to hashable tuple for visited set"""
        return tuple(sorted(list(state)))
```

### 2. Hierarchical Task Network (HTN) Planning

HTN planning decomposes high-level tasks into lower-level methods:

```python
# Hierarchical Task Network planning
class HTNPlanner:
    def __init__(self):
        self.methods = self.define_methods()
        self.operators = self.define_operators()

    def define_methods(self):
        """Define HTN methods for task decomposition"""
        return {
            'complex_pick_up': [
                {
                    'task': 'navigate_to_object',
                    'decomposition': [
                        ('navigate', {'target': '?object_location'}),
                        ('locate_object', {'target': '?object'})
                    ]
                },
                {
                    'task': 'grasp_object',
                    'decomposition': [
                        ('approach_object', {'target': '?object'}),
                        ('compute_grasp_pose', {'target': '?object'}),
                        ('execute_grasp', {'target': '?object'})
                    ]
                }
            ],
            'move_object': [
                {
                    'task': 'pickup_object',
                    'decomposition': [('complex_pick_up', {'object': '?obj'})]
                },
                {
                    'task': 'navigate_to_destination',
                    'decomposition': [('navigate', {'target': '?dest'})]
                },
                {
                    'task': 'place_object',
                    'decomposition': [
                        ('position_object', {'object': '?obj', 'location': '?dest'}),
                        ('release_object', {'object': '?obj'})
                    ]
                }
            ]
        }

    def define_operators(self):
        """Define primitive operators"""
        return {
            'navigate': {
                'preconditions': ['robot_operational', 'path_exists(?target)'],
                'effects': ['robot_at(?target)']
            },
            'locate_object': {
                'preconditions': ['camera_operational', 'object_visible(?target)'],
                'effects': ['object_position_known(?target)']
            },
            'approach_object': {
                'preconditions': ['robot_at(?current)', 'object_at(?target, ?obj_location)'],
                'effects': ['robot_adjacent_to(?target)']
            },
            'execute_grasp': {
                'preconditions': ['robot_adjacent_to(?target)', 'gripper_free'],
                'effects': ['object_grasped(?target)', 'gripper_occupied']
            },
            'release_object': {
                'preconditions': ['object_grasped(?target)', 'gripper_occupied'],
                'effects': ['object_released(?target)', 'gripper_free']
            }
        }

    def plan(self, task_network, state):
        """
        Plan using Hierarchical Task Network
        """
        return self.hierarchical_search(task_network, state, [])

    def hierarchical_search(self, task_network, state, partial_plan):
        """Recursive search through task network"""
        if not task_network:
            # All tasks completed
            return partial_plan

        # Get first task to accomplish
        current_task = task_network[0]
        remaining_tasks = task_network[1:]

        # Check if it's a primitive operator
        if current_task[0] in self.operators:
            # Apply primitive operator
            if self.operator_applicable(current_task, state):
                new_state = self.apply_operator(current_task, state)
                return self.hierarchical_search(
                    remaining_tasks, new_state, partial_plan + [current_task]
                )
        else:
            # It's a compound task, try to decompose using methods
            methods = self.get_applicable_methods(current_task, state)

            for method in methods:
                # Decompose task using method
                decomposition = method['decomposition']

                # Create new task network with decomposition
                new_task_network = decomposition + remaining_tasks

                # Recursively plan for new task network
                plan = self.hierarchical_search(new_task_network, state, partial_plan)

                if plan is not None:
                    return plan

        # No applicable methods or operators found
        return None

    def get_applicable_methods(self, task, state):
        """Get all applicable methods for a task in current state"""
        task_name = task[0]
        bindings = task[1] if len(task) > 1 else {}

        applicable_methods = []

        if task_name in self.methods:
            for method in self.methods[task_name]:
                # Check if method preconditions are satisfied
                if self.method_applicable(method, state, bindings):
                    applicable_methods.append(method)

        return applicable_methods

    def method_applicable(self, method, state, bindings):
        """Check if method is applicable in current state"""
        # For now, assume methods are always applicable
        # In practice, you'd check preconditions
        return True

    def operator_applicable(self, operator, state):
        """Check if operator is applicable in current state"""
        op_name, params = operator[0], operator[1] if len(operator) > 1 else {}

        if op_name in self.operators:
            preconditions = self.operators[op_name]['preconditions']

            # Check each precondition
            for prec in preconditions:
                # Apply bindings
                bound_prec = self.apply_bindings(prec, params)

                # Check if precondition is satisfied in state
                if not self.check_predicate_in_state(bound_prec, state):
                    return False

            return True

        return False

    def apply_operator(self, operator, state):
        """Apply operator to state and return new state"""
        op_name, params = operator[0], operator[1] if len(operator) > 1 else {}

        if op_name in self.operators:
            effects = self.operators[op_name]['effects']
            new_state = state.copy()

            for effect in effects:
                bound_effect = self.apply_bindings(effect, params)

                # Apply effect (positive or negative)
                if effect.startswith('-'):
                    actual_effect = bound_effect[1:]
                    if actual_effect in new_state:
                        new_state.remove(actual_effect)
                else:
                    new_state.add(bound_effect)

            return new_state

        return state

    def apply_bindings(self, predicate, bindings):
        """Apply variable bindings to a predicate"""
        result = predicate
        for var, value in bindings.items():
            result = result.replace(var, value)
        return result

    def check_predicate_in_state(self, predicate, state):
        """Check if predicate is in state"""
        return predicate in state
```

## Language-to-Action Mapping

### 1. Semantic Parsing for Action Generation

```python
# Semantic parsing for converting language to actions
class SemanticParser:
    def __init__(self):
        self.semantic_forms = self.define_semantic_forms()
        self.action_mapping = self.create_action_mapping()

    def define_semantic_forms(self):
        """Define semantic forms for common command patterns"""
        return {
            # "Pick up the red ball"
            'pick_up_pattern': {
                'pattern': r'pick\s+up\s+(the\s+)?(\w+)\s+(\w+)',
                'semantic_template': {
                    'action': 'grasp',
                    'object': '{adjective}_{noun}',
                    'attributes': ['{adjective}']
                }
            },
            # "Put the book on the table"
            'put_on_pattern': {
                'pattern': r'put\s+(the\s+)?(\w+)\s+on\s+(the\s+)?(\w+)',
                'semantic_template': {
                    'action': 'place',
                    'object': '{object_noun}',
                    'destination': '{location_noun}',
                    'relation': 'on'
                }
            },
            # "Move to the kitchen"
            'navigate_pattern': {
                'pattern': r'move\s+to\s+(the\s+)?(\w+)',
                'semantic_template': {
                    'action': 'navigate',
                    'destination': '{location}'
                }
            },
            # "Go to the living room"
            'go_to_pattern': {
                'pattern': r'go\s+to\s+(the\s+)?(\w+)',
                'semantic_template': {
                    'action': 'navigate',
                    'destination': '{location}'
                }
            },
            # "Bring the cup to me"
            'bring_to_pattern': {
                'pattern': r'bring\s+(the\s+)?(\w+)\s+to\s+(\w+)',
                'semantic_template': {
                    'action': 'transport',
                    'object': '{object_noun}',
                    'recipient': '{recipient}',
                    'subtasks': ['grasp_{object_noun}', 'navigate_to_{recipient}', 'release_{object_noun}']
                }
            }
        }

    def create_action_mapping(self):
        """Create mapping from semantic forms to robot actions"""
        return {
            'grasp': {
                'primitive': 'execute_grasp',
                'required_args': ['object'],
                'planning_template': [
                    'navigate_to_object_location',
                    'align_gripper_with_object',
                    'execute_grasp_manipulation'
                ]
            },
            'place': {
                'primitive': 'release_object',
                'required_args': ['object', 'destination'],
                'planning_template': [
                    'navigate_to_destination',
                    'position_object_above_destination',
                    'execute_release'
                ]
            },
            'navigate': {
                'primitive': 'move_to_waypoint',
                'required_args': ['destination'],
                'planning_template': [
                    'get_path_to_destination',
                    'execute_navigation_trajectory'
                ]
            },
            'transport': {
                'primitive': 'object_transport',
                'required_args': ['object', 'destination'],
                'planning_template': [
                    'grasp_object',
                    'navigate_with_object',
                    'place_object'
                ]
            }
        }

    def parse_command(self, command_text: str) -> Dict[str, any]:
        """Parse command and generate semantic representation"""
        import re

        # Try each semantic pattern
        for pattern_name, pattern_info in self.semantic_forms.items():
            match = re.search(pattern_info['pattern'], command_text, re.IGNORECASE)

            if match:
                # Extract matched groups
                groups = match.groups()

                # Apply semantic template
                semantic_template = pattern_info['semantic_template']
                semantic_form = self.instantiate_template(semantic_template, groups)

                # Map to action
                action_spec = self.map_to_action(semantic_form)

                return {
                    'command': command_text,
                    'semantic_form': semantic_form,
                    'action_specification': action_spec,
                    'pattern_matched': pattern_name
                }

        # If no pattern matches, use general NLP approach
        return self.fallback_parse(command_text)

    def instantiate_template(self, template, groups):
        """Instantiate semantic template with matched groups"""
        import json
        import re

        template_str = json.dumps(template)

        # Replace numbered placeholders {1}, {2}, etc.
        for i, group in enumerate(groups):
            if group:  # Skip None groups
                template_str = template_str.replace(f'{{{i+1}}}', group.strip())

        # Replace named placeholders like {adjective}, {noun}, etc.
        # This is a simplified approach - in practice you'd have more sophisticated matching
        if len(groups) >= 2:
            template_str = template_str.replace('{adjective}', groups[0] or '')
            template_str = template_str.replace('{noun}', groups[1] or '')
            if len(groups) >= 3:
                template_str = template_str.replace('{object_noun}', groups[1] or '')
                template_str = template_str.replace('{location_noun}', groups[2] or '')
            if len(groups) >= 4:
                template_str = template_str.replace('{location}', groups[3] or '')

        return json.loads(template_str)

    def map_to_action(self, semantic_form: Dict[str, any]) -> Dict[str, any]:
        """Map semantic form to executable action"""
        action_type = semantic_form.get('action', 'unknown')

        if action_type in self.action_mapping:
            action_def = self.action_mapping[action_type]

            # Validate required arguments
            required_args = action_def.get('required_args', [])
            provided_args = {k: v for k, v in semantic_form.items()
                           if k in required_args}

            missing_args = set(required_args) - set(provided_args.keys())

            if missing_args:
                return {
                    'action_type': action_type,
                    'status': 'incomplete',
                    'missing_arguments': list(missing_args),
                    'provided_arguments': provided_args
                }

            # Generate planning sequence
            planning_template = action_def.get('planning_template', [])
            planning_sequence = self.instantiate_planning_template(
                planning_template, provided_args
            )

            return {
                'action_type': action_type,
                'primitive_action': action_def.get('primitive'),
                'arguments': provided_args,
                'planning_sequence': planning_sequence,
                'status': 'ready'
            }

        return {
            'action_type': action_type,
            'status': 'unsupported',
            'error': f'Action type {action_type} not supported'
        }

    def instantiate_planning_template(self, template, args):
        """Instantiate planning template with arguments"""
        instantiated = []

        for step_template in template:
            step = step_template.format(**args)
            instantiated.append(step)

        return instantiated

    def fallback_parse(self, command_text: str) -> Dict[str, any]:
        """Fallback parsing using general NLP techniques"""
        # Use the language understanding module as fallback
        lu_module = LanguageUnderstandingModule()
        parsed_command = lu_module(command_text)

        # Convert to action specification
        action_spec = self.convert_parsed_command_to_action(parsed_command)

        return {
            'command': command_text,
            'semantic_form': {
                'intent': parsed_command.intent,
                'objects': parsed_command.objects,
                'locations': parsed_command.locations,
                'attributes': parsed_command.attributes
            },
            'action_specification': action_spec,
            'pattern_matched': 'fallback_nlp'
        }

    def convert_parsed_command_to_action(self, parsed_command: ParsedCommand):
        """Convert parsed command to action specification"""
        # Map intent to action type
        intent_to_action = {
            'pick_up': 'grasp',
            'grasp': 'grasp',
            'put_down': 'place',
            'place': 'place',
            'move_to': 'navigate',
            'navigate': 'navigate',
            'transport': 'transport',
            'carry': 'transport'
        }

        action_type = intent_to_action.get(parsed_command.intent, 'unknown')

        # Collect arguments
        args = {}
        if parsed_command.objects:
            args['object'] = parsed_command.objects[0]['text']
        if parsed_command.locations:
            args['destination'] = parsed_command.locations[0]['text']

        return {
            'action_type': action_type,
            'arguments': args,
            'status': 'parsed' if action_type != 'unknown' else 'unknown_intent'
        }
```

## Execution and Monitoring

### 1. Action Execution Framework

```python
# Action execution and monitoring framework
import asyncio
import threading
from typing import Callable, Any
from dataclasses import dataclass

@dataclass
class ActionResult:
    """Result of action execution"""
    success: bool
    message: str
    execution_time: float
    details: dict = None

class ActionExecutor:
    def __init__(self, robot_interface):
        self.robot_interface = robot_interface
        self.current_action = None
        self.action_queue = asyncio.Queue()
        self.executor_thread = threading.Thread(target=self.execution_loop, daemon=True)
        self.running = True
        self.executor_thread.start()

    def execution_loop(self):
        """Main execution loop for actions"""
        asyncio.run(self.async_execution_loop())

    async def async_execution_loop(self):
        """Async execution loop"""
        while self.running:
            try:
                # Get next action from queue
                action_spec = await asyncio.wait_for(self.action_queue.get(), timeout=0.1)

                # Execute action
                result = await self.execute_single_action(action_spec)

                # Handle result
                if result.success:
                    print(f"Action completed successfully: {action_spec['action_type']}")
                else:
                    print(f"Action failed: {result.message}")
                    # Handle failure (retry, skip, etc.)
                    await self.handle_action_failure(action_spec, result)

            except asyncio.TimeoutError:
                continue  # No action available, continue loop
            except Exception as e:
                print(f"Execution error: {e}")

    async def execute_single_action(self, action_spec: Dict[str, any]) -> ActionResult:
        """Execute a single action"""
        import time

        start_time = time.time()
        self.current_action = action_spec

        try:
            action_type = action_spec['action_type']

            if action_type == 'grasp':
                result = await self.execute_grasp_action(action_spec)
            elif action_type == 'place':
                result = await self.execute_place_action(action_spec)
            elif action_type == 'navigate':
                result = await self.execute_navigate_action(action_spec)
            elif action_type == 'transport':
                result = await self.execute_transport_action(action_spec)
            else:
                return ActionResult(
                    success=False,
                    message=f"Unknown action type: {action_type}",
                    execution_time=time.time() - start_time
                )

            return result

        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Action execution error: {str(e)}",
                execution_time=time.time() - start_time,
                details={'exception': str(e)}
            )

    async def execute_grasp_action(self, action_spec: Dict[str, any]) -> ActionResult:
        """Execute grasp action"""
        import time

        object_name = action_spec['arguments'].get('object')
        if not object_name:
            return ActionResult(
                success=False,
                message="No object specified for grasp action",
                execution_time=0
            )

        # 1. Navigate to object location
        nav_result = await self.robot_interface.navigate_to_object(object_name)
        if not nav_result.success:
            return ActionResult(
                success=False,
                message=f"Failed to navigate to {object_name}: {nav_result.message}",
                execution_time=0,
                details={'navigation_result': nav_result}
            )

        # 2. Align gripper with object
        align_result = await self.robot_interface.align_gripper(object_name)
        if not align_result.success:
            return ActionResult(
                success=False,
                message=f"Failed to align gripper with {object_name}: {align_result.message}",
                execution_time=0,
                details={'alignment_result': align_result}
            )

        # 3. Execute grasp
        grasp_result = await self.robot_interface.execute_grasp(object_name)
        if not grasp_result.success:
            return ActionResult(
                success=False,
                message=f"Failed to grasp {object_name}: {grasp_result.message}",
                execution_time=0,
                details={'grasp_result': grasp_result}
            )

        return ActionResult(
            success=True,
            message=f"Successfully grasped {object_name}",
            execution_time=time.time() - time.time(),  # Placeholder
            details={
                'object': object_name,
                'navigation': nav_result,
                'alignment': align_result,
                'grasp': grasp_result
            }
        )

    async def execute_place_action(self, action_spec: Dict[str, any]) -> ActionResult:
        """Execute place action"""
        import time

        object_name = action_spec['arguments'].get('object')
        destination = action_spec['arguments'].get('destination')

        if not object_name or not destination:
            return ActionResult(
                success=False,
                message="Object and destination required for place action",
                execution_time=0
            )

        # 1. Navigate to destination
        nav_result = await self.robot_interface.navigate_to_location(destination)
        if not nav_result.success:
            return ActionResult(
                success=False,
                message=f"Failed to navigate to {destination}: {nav_result.message}",
                execution_time=0
            )

        # 2. Position object above destination
        position_result = await self.robot_interface.position_above(destination)
        if not position_result.success:
            return ActionResult(
                success=False,
                message=f"Failed to position above {destination}: {position_result.message}",
                execution_time=0
            )

        # 3. Release object
        release_result = await self.robot_interface.release_object(object_name)
        if not release_result.success:
            return ActionResult(
                success=False,
                message=f"Failed to release {object_name}: {release_result.message}",
                execution_time=0
            )

        return ActionResult(
            success=True,
            message=f"Successfully placed {object_name} at {destination}",
            execution_time=time.time() - time.time(),  # Placeholder
            details={
                'object': object_name,
                'destination': destination,
                'navigation': nav_result,
                'positioning': position_result,
                'release': release_result
            }
        )

    async def execute_navigate_action(self, action_spec: Dict[str, any]) -> ActionResult:
        """Execute navigation action"""
        import time

        destination = action_spec['arguments'].get('destination')
        if not destination:
            return ActionResult(
                success=False,
                message="No destination specified for navigate action",
                execution_time=0
            )

        # Navigate to destination
        result = await self.robot_interface.navigate_to_location(destination)

        return ActionResult(
            success=result.success,
            message=result.message,
            execution_time=time.time() - time.time(),  # Placeholder
            details={'destination': destination, 'navigation_result': result}
        )

    async def execute_transport_action(self, action_spec: Dict[str, any]) -> ActionResult:
        """Execute transport action (grasp + navigate + place)"""
        import time

        object_name = action_spec['arguments'].get('object')
        destination = action_spec['arguments'].get('destination')

        if not object_name or not destination:
            return ActionResult(
                success=False,
                message="Object and destination required for transport action",
                execution_time=0
            )

        # 1. Grasp the object
        grasp_spec = {
            'action_type': 'grasp',
            'arguments': {'object': object_name}
        }
        grasp_result = await self.execute_single_action(grasp_spec)
        if not grasp_result.success:
            return ActionResult(
                success=False,
                message=f"Transport failed at grasp stage: {grasp_result.message}",
                execution_time=0,
                details={'grasp_result': grasp_result}
            )

        # 2. Navigate with object
        nav_spec = {
            'action_type': 'navigate',
            'arguments': {'destination': destination}
        }
        nav_result = await self.execute_single_action(nav_spec)
        if not nav_result.success:
            # Try to safely release the object before returning failure
            release_spec = {
                'action_type': 'place',
                'arguments': {'object': object_name, 'destination': 'current_location'}
            }
            await self.execute_single_action(release_spec)

            return ActionResult(
                success=False,
                message=f"Transport failed at navigation stage: {nav_result.message}",
                execution_time=0,
                details={'grasp_result': grasp_result, 'navigation_result': nav_result}
            )

        # 3. Place the object
        place_spec = {
            'action_type': 'place',
            'arguments': {'object': object_name, 'destination': destination}
        }
        place_result = await self.execute_single_action(place_spec)
        if not place_result.success:
            return ActionResult(
                success=False,
                message=f"Transport failed at place stage: {place_result.message}",
                execution_time=0,
                details={
                    'grasp_result': grasp_result,
                    'navigation_result': nav_result,
                    'place_result': place_result
                }
            )

        return ActionResult(
            success=True,
            message=f"Successfully transported {object_name} to {destination}",
            execution_time=time.time() - time.time(),  # Placeholder
            details={
                'object': object_name,
                'destination': destination,
                'grasp_result': grasp_result,
                'navigation_result': nav_result,
                'place_result': place_result
            }
        )

    async def handle_action_failure(self, action_spec: Dict[str, any], result: ActionResult):
        """Handle action execution failure"""
        print(f"Handling action failure: {result.message}")

        # Possible failure handling strategies:
        # 1. Retry the action
        # 2. Skip to next action
        # 3. Request human intervention
        # 4. Modify the plan

        # For now, implement a simple retry mechanism
        max_retries = 3
        retry_count = action_spec.get('retry_count', 0)

        if retry_count < max_retries:
            # Add action back to queue with incremented retry count
            action_spec['retry_count'] = retry_count + 1
            await self.action_queue.put(action_spec)
            print(f"Retrying action ({retry_count + 1}/{max_retries})")
        else:
            print("Max retries exceeded, skipping action")

    def submit_action(self, action_spec: Dict[str, any]):
        """Submit action for execution"""
        asyncio.run_coroutine_threadsafe(
            self.action_queue.put(action_spec),
            asyncio.get_event_loop()
        )

    def shutdown(self):
        """Shutdown the executor"""
        self.running = False
        self.executor_thread.join()
```

## Integration with VLA Systems

### 1. Complete Language-Guided Planning Pipeline

```python
# Complete language-guided action planning pipeline
class LanguageGuidedPlanner:
    def __init__(self, robot_interface):
        self.language_understanding = LanguageUnderstandingModule()
        self.task_decomposer = TaskDecomposer()
        self.symbolic_planner = SymbolicPlanner()
        self.hierarchical_planner = HTNPlanner()
        self.semantic_parser = SemanticParser()
        self.action_executor = ActionExecutor(robot_interface)

    def process_command(self, command_text: str) -> Dict[str, any]:
        """
        Process natural language command and execute corresponding actions
        """
        # Step 1: Parse the command
        print(f"Parsing command: {command_text}")
        parsed_command = self.language_understanding(command_text)

        # Step 2: Generate semantic representation
        print("Generating semantic representation...")
        semantic_result = self.semantic_parser.parse_command(command_text)

        # Step 3: Decompose task into sub-tasks
        print("Decomposing task...")
        sub_tasks = self.task_decomposer.decompose_task(parsed_command)

        # Step 4: Generate action plan
        print("Generating action plan...")
        action_plan = self.generate_action_plan(semantic_result, sub_tasks)

        # Step 5: Execute the plan
        print("Executing action plan...")
        execution_results = self.execute_action_plan(action_plan)

        return {
            'original_command': command_text,
            'parsed_command': parsed_command,
            'semantic_result': semantic_result,
            'sub_tasks': sub_tasks,
            'action_plan': action_plan,
            'execution_results': execution_results
        }

    def generate_action_plan(self, semantic_result, sub_tasks):
        """Generate detailed action plan from semantic result and sub-tasks"""
        action_plan = []

        # If we have a direct semantic mapping, use it
        if semantic_result['action_specification']['status'] == 'ready':
            action_spec = semantic_result['action_specification']
            action_plan.append(action_spec)

        # Otherwise, use task decomposition
        else:
            for sub_task in sub_tasks:
                action_spec = {
                    'action_type': sub_task['action'],
                    'arguments': self.extract_arguments(sub_task),
                    'priority': sub_task.get('priority', 1),
                    'preconditions': sub_task.get('preconditions', []),
                    'effects': sub_task.get('effects', [])
                }
                action_plan.append(action_spec)

        return action_plan

    def extract_arguments(self, sub_task):
        """Extract arguments from sub-task definition"""
        # This would involve mapping sub-task parameters to actual values
        # For now, return empty dict
        return {}

    def execute_action_plan(self, action_plan):
        """Execute the action plan"""
        results = []

        for action_spec in action_plan:
            print(f"Executing action: {action_spec['action_type']}")

            # Submit action for execution
            self.action_executor.submit_action(action_spec)

            # Wait for result (in a real system, this would be asynchronous)
            # For now, just add the action spec to results
            results.append({
                'action_spec': action_spec,
                'status': 'submitted'
            })

        return results

    def validate_plan(self, command_text: str, current_state: set) -> bool:
        """Validate if the plan is feasible given current state"""
        # Parse command to get goal state
        parsed_command = self.language_understanding(command_text)

        # Define goal based on command intent
        goal_state = self.command_to_goal_state(parsed_command, current_state)

        # Use symbolic planner to check if goal is achievable
        plan = self.symbolic_planner.plan(current_state, goal_state)

        return plan is not None

    def command_to_goal_state(self, parsed_command: ParsedCommand, current_state: set):
        """Convert command to goal state representation"""
        goal_literals = set()

        if parsed_command.intent == 'pick_up':
            # Goal: robot should be holding the object
            if parsed_command.objects:
                obj = parsed_command.objects[0]['text']
                goal_literals.add(f'robot_holding({obj})')

        elif parsed_command.intent == 'place':
            # Goal: object should be at destination
            if parsed_command.objects and parsed_command.locations:
                obj = parsed_command.objects[0]['text']
                loc = parsed_command.locations[0]['text']
                goal_literals.add(f'object_at({obj}, {loc})')

        elif parsed_command.intent == 'navigate':
            # Goal: robot should be at location
            if parsed_command.locations:
                loc = parsed_command.locations[0]['text']
                goal_literals.add(f'robot_at({loc})')

        return goal_literals
```

## Next Steps

In the next section, we'll explore VLA models and architectures in detail, examining how vision, language, and action components are integrated into unified neural architectures. We'll cover state-of-the-art models like OpenVLA, RT-1, and other foundational VLA models, learning how to implement and train these systems for humanoid robotics applications.