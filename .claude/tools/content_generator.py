#!/usr/bin/env python3
"""
Claude Code Subagent: Content Generator for Physical AI & Humanoid Robotics Textbook
This tool generates textbook content following the established structure and style.
"""

import os
import json
import argparse
from pathlib import Path


def create_module_content(module_title, topics, target_dir="frontend/docs"):
    """
    Creates a new module with specified topics in the textbook structure.

    Args:
        module_title (str): Title of the module (e.g., "Module 5: Advanced Control Systems")
        topics (list): List of topics to include in the module
        target_dir (str): Directory where the content will be created
    """
    # Create module directory
    module_dir = Path(target_dir) / module_title.lower().replace(" ", "-").replace(":", "")
    module_dir.mkdir(exist_ok=True)

    # Create index file for the module
    index_file = module_dir / "index.md"
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(f"""---
sidebar_position: 99
title: "{module_title}"
---

# {module_title}

## Overview

This module covers the fundamental concepts of {module_title.split(':')[-1].strip()} in the context of Physical AI and Humanoid Robotics.

## Learning Objectives

After completing this module, you will be able to:
- Understand the core principles of {module_title.split(':')[-1].strip()}
- Apply these concepts to humanoid robotics scenarios
- Implement practical examples using ROS 2, NVIDIA Isaac, and other relevant frameworks

## Table of Contents

""")
        for i, topic in enumerate(topics, 1):
            topic_slug = topic.lower().replace(" ", "-").replace(":", "").replace("(", "").replace(")", "")
            f.write(f"- [{topic}](./{topic_slug})\n")
            # Create individual topic files
            topic_file = module_dir / f"{topic_slug}.md"
            with open(topic_file, 'w', encoding='utf-8') as tf:
                tf.write(f"""---
sidebar_position: {i}
title: "{topic}"
---

# {topic}

## Introduction

TODO: Add introduction for {topic}

## Key Concepts

TODO: Add key concepts for {topic}

## Practical Implementation

TODO: Add practical implementation details for {topic} in the context of humanoid robotics

## Examples

TODO: Add code examples or practical examples for {topic}

## Exercises

TODO: Add exercises related to {topic}

## Further Reading

TODO: Add references and further reading for {topic}

## Summary

TODO: Add summary of {topic}
""")

    print(f"Module '{module_title}' created successfully with {len(topics)} topics.")
    print(f"Location: {module_dir}")
    return str(module_dir)


def create_assessment(module_number, module_title, target_dir="frontend/docs/assessments"):
    """
    Creates an assessment for a specific module.

    Args:
        module_number (int): Module number
        module_title (str): Title of the module
        target_dir (str): Directory where the assessment will be created
    """
    assessment_dir = Path(target_dir)
    assessment_dir.mkdir(exist_ok=True)

    assessment_file = assessment_dir / f"module-{module_number}-assessment.md"
    with open(assessment_file, 'w', encoding='utf-8') as f:
        f.write(f"""---
sidebar_position: {module_number}
title: "Module {module_number} Assessment - {module_title}"
---

# Module {module_number} Assessment: {module_title}

## Multiple Choice Questions

1. Question about key concept from Module {module_number}:
   - a) Option A
   - b) Option B
   - c) Option C
   - d) Option D

2. Another question about Module {module_number} content:
   - a) Option A
   - b) Option B
   - c) Option C
   - d) Option D

## Short Answer Questions

3. Explain the main principles covered in Module {module_number}.

4. How would you apply the concepts from Module {module_number} to a humanoid robotics scenario?

## Practical Exercise

5. Implement a basic example that demonstrates the concepts learned in Module {module_number} using ROS 2 or NVIDIA Isaac.

## Answer Key

1. Correct answer
2. Correct answer
3. Detailed explanation
4. Detailed explanation
5. Sample implementation
""")

    print(f"Assessment for '{module_title}' created successfully.")
    print(f"Location: {assessment_file}")
    return str(assessment_file)


def create_practice_questions(chapter_content, num_questions=5):
    """
    Generates practice questions based on chapter content.

    Args:
        chapter_content (str): Content of the chapter
        num_questions (int): Number of questions to generate
    """
    # This would typically use an LLM to generate questions
    # For now, we'll create a template
    questions = []
    for i in range(1, num_questions + 1):
        questions.append({
            "question": f"Practice question {i} based on the chapter content",
            "type": "short_answer",  # or "multiple_choice", "true_false"
            "difficulty": "medium",
            "answer": f"Sample answer for question {i}"
        })

    return questions


def main():
    parser = argparse.ArgumentParser(description="Content Generator for Physical AI & Humanoid Robotics Textbook")
    parser.add_argument("action", choices=["create_module", "create_assessment", "generate_questions"],
                       help="Action to perform")
    parser.add_argument("--title", type=str, help="Title for the module or content")
    parser.add_argument("--topics", type=str, nargs="*", help="Topics to include in the module")
    parser.add_argument("--module-number", type=int, help="Module number for assessments")
    parser.add_argument("--content", type=str, help="Chapter content for question generation")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions to generate")
    parser.add_argument("--output-dir", type=str, default="frontend/docs", help="Output directory")

    args = parser.parse_args()

    if args.action == "create_module":
        if not args.title or not args.topics:
            print("Error: --title and --topics are required for create_module")
            return
        create_module_content(args.title, args.topics, args.output_dir)

    elif args.action == "create_assessment":
        if not args.title or not args.module_number:
            print("Error: --title and --module-number are required for create_assessment")
            return
        create_assessment(args.module_number, args.title)

    elif args.action == "generate_questions":
        if not args.content:
            print("Error: --content is required for generate_questions")
            return
        questions = create_practice_questions(args.content, args.num_questions)
        print(json.dumps(questions, indent=2))


if __name__ == "__main__":
    main()