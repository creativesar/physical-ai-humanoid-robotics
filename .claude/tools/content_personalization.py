#!/usr/bin/env python3
"""
Claude Code Subagent: Content Personalization System
This tool personalizes textbook content based on user background and expertise level.
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml


def load_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Loads user profile information from a database or file.
    In a real implementation, this would connect to a database.

    Args:
        user_id (str): ID of the user

    Returns:
        Dict[str, Any]: User profile information
    """
    # In a real implementation, this would fetch from a database
    # For now, we'll return a default profile or load from a file
    profile_file = Path(f"frontend/src/data/user_profiles/{user_id}.json")

    default_profile = {
        "user_id": user_id,
        "background": "beginner",  # beginner, intermediate, advanced
        "interests": ["robotics", "ai"],
        "programming_experience": "none",  # none, basic, intermediate, advanced
        "robotics_experience": "none",  # none, basic, intermediate, advanced
        "learning_goals": ["understand_basics", "build_simple_robot"],
        "preferred_content_style": "detailed"  # concise, detailed, hands_on
    }

    if profile_file.exists():
        with open(profile_file, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        # Merge with defaults to ensure all fields exist
        for key, value in default_profile.items():
            if key not in profile:
                profile[key] = value
        return profile
    else:
        # Return default profile if no specific profile exists
        return default_profile


def personalize_content(content: str, user_profile: Dict[str, Any]) -> str:
    """
    Personalizes content based on user profile.

    Args:
        content (str): Original content
        user_profile (Dict[str, Any]): User profile information

    Returns:
        str: Personalized content
    """
    background = user_profile.get("background", "beginner")
    programming_exp = user_profile.get("programming_experience", "none")
    robotics_exp = user_profile.get("robotics_experience", "none")
    content_style = user_profile.get("preferred_content_style", "detailed")

    # Make a copy of the content to modify
    personalized_content = content

    # Adjust complexity based on background
    if background == "beginner":
        # Add more explanations and simpler language
        personalized_content = add_beginner_explanations(personalized_content)
    elif background == "advanced":
        # Add more technical depth
        personalized_content = add_advanced_details(personalized_content)

    # Adjust code examples based on programming experience
    if programming_exp in ["none", "basic"]:
        # Simplify or remove complex code examples
        personalized_content = simplify_code_examples(personalized_content)
    elif programming_exp == "advanced":
        # Add more complex examples
        personalized_content = add_complex_code_examples(personalized_content)

    # Adjust robotics examples based on robotics experience
    if robotics_exp in ["none", "basic"]:
        # Add more basic examples
        personalized_content = add_basic_robotics_examples(personalized_content)
    elif robotics_exp == "advanced":
        # Add more complex robotics applications
        personalized_content = add_advanced_robotics_examples(personalized_content)

    # Adjust content style
    if content_style == "concise":
        personalized_content = make_content_concise(personalized_content)
    elif content_style == "hands_on":
        personalized_content = add_hands_on_elements(personalized_content)

    return personalized_content


def add_beginner_explanations(content: str) -> str:
    """Add explanations for beginners."""
    # Add more introductory text
    beginner_additions = [
        ("\n## Introduction\n", "\n## Introduction\nFor beginners, it's important to understand the foundational concepts before diving deeper. We'll start with the basics.\n"),
        ("For example,", "For example (beginners start here):"),
        ("First,", "First (as a beginner, focus on understanding this):"),
    ]

    for old, new in beginner_additions:
        content = content.replace(old, new)

    # Add beginner tips
    content = re.sub(r'([.!?]\s+)([A-Z])', r'\1> **Beginner Tip**: \2', content, count=2)  # Add beginner tips at the beginning

    return content


def add_advanced_details(content: str) -> str:
    """Add more technical depth for advanced users."""
    # Add more technical details
    content += "\n\n> **Advanced Note**: For users with advanced knowledge, consider exploring the implementation details and optimization strategies for this concept."

    # Replace basic explanations with more advanced ones
    content = content.replace("This works because...", "This works because... (Advanced: This implementation uses...")

    return content


def simplify_code_examples(content: str) -> str:
    """Simplify code examples for users with less programming experience."""
    # Remove complex code blocks or add more explanation
    # This is a simplified implementation - in reality, you'd want more sophisticated code analysis
    content = re.sub(r'```(python|javascript|cpp|c)\n(.*?)\n```',
                     r'> **Code Explanation**: This code does the following:\n> \2\n\n```python\n\2\n```',
                     content, flags=re.DOTALL)

    return content


def add_complex_code_examples(content: str) -> str:
    """Add more complex code examples for advanced users."""
    # Add more complex code examples
    complex_examples = [
        ("```python", "```python\n# Advanced implementation with optimizations:\n"),
        ("## Implementation", "## Implementation\nFor advanced users, here's an optimized implementation:\n"),
    ]

    for old, new in complex_examples:
        content = content.replace(old, new)

    return content


def add_basic_robotics_examples(content: str) -> str:
    """Add basic robotics examples."""
    # Add simple robotics examples
    basic_example = "\n> **Robotics Example**: A simple example is a robot that moves forward when it detects an obstacle. This is a basic behavior that demonstrates the concept.\n"
    content += basic_example

    return content


def add_advanced_robotics_examples(content: str) -> str:
    """Add advanced robotics examples."""
    # Add complex robotics examples
    advanced_example = "\n> **Advanced Robotics Application**: In real-world applications, this concept is used in humanoid robots for dynamic balance control, where multiple sensors and actuators must coordinate in real-time to maintain stability.\n"
    content += advanced_example

    return content


def make_content_concise(content: str) -> str:
    """Make content more concise."""
    # Remove verbose explanations
    content = re.sub(r'\n> \*\*Beginner Tip\*\*:.*?\n', '\n', content)
    content = re.sub(r'\nFor beginners, it\'s important to understand.*?\n', '\n', content)

    return content


def add_hands_on_elements(content: str) -> str:
    """Add hands-on elements to content."""
    # Add practical exercises
    hands_on_elements = [
        ("## Summary", "## Hands-On Exercise\nTry implementing this concept with your own robot or simulation.\n\n## Summary"),
        ("## Further Reading", "## Hands-On Challenge\nBuild a simple prototype based on this concept.\n\n## Further Reading"),
    ]

    for old, new in hands_on_elements:
        content = content.replace(old, new)

    return content


def create_personalized_version(chapter_path: str, user_id: str, output_path: str = None) -> str:
    """
    Creates a personalized version of a chapter based on user profile.

    Args:
        chapter_path (str): Path to the original chapter file
        user_id (str): ID of the user
        output_path (str): Path to save the personalized version (optional)

    Returns:
        str: Path to the personalized chapter
    """
    # Load the original content
    with open(chapter_path, 'r', encoding='utf-8') as f:
        original_content = f.read()

    # Extract frontmatter if it exists
    frontmatter_match = re.match(r'(^---\n.*?\n---\n)', original_content, re.DOTALL)
    frontmatter = frontmatter_match.group(1) if frontmatter_match else ""
    main_content = original_content[len(frontmatter):] if frontmatter_match else original_content

    # Load user profile
    user_profile = load_user_profile(user_id)

    # Personalize the main content
    personalized_main_content = personalize_content(main_content, user_profile)

    # Combine frontmatter and personalized content
    personalized_content = frontmatter + personalized_main_content

    # Determine output path
    if not output_path:
        chapter_dir = Path(chapter_path).parent
        user_dir = chapter_dir / "personalized" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        output_path = user_dir / Path(chapter_path).name

    # Write personalized content to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(personalized_content)

    print(f"Personalized version created for user {user_id}")
    print(f"Original: {chapter_path}")
    print(f"Personalized: {output_path}")

    return str(output_path)


def create_personalized_module(module_dir: str, user_id: str, output_dir: str = None) -> str:
    """
    Creates a personalized version of an entire module.

    Args:
        module_dir (str): Path to the module directory
        user_id (str): ID of the user
        output_dir (str): Directory to save personalized module (optional)

    Returns:
        str: Path to the personalized module directory
    """
    module_path = Path(module_dir)
    if not module_path.exists():
        raise FileNotFoundError(f"Module directory {module_dir} does not exist")

    # Determine output directory
    if not output_dir:
        output_dir = Path("frontend/docs/personalized") / user_id / module_path.name
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each chapter in the module
    chapter_files = list(module_path.glob("*.md"))
    chapter_files = [f for f in chapter_files if f.name != "index.md"]  # Skip index for now

    for chapter_file in chapter_files:
        output_path = output_dir / chapter_file.name
        create_personalized_version(str(chapter_file), user_id, str(output_path))

    # Also personalize the index file if it exists
    index_file = module_path / "index.md"
    if index_file.exists():
        output_index = output_dir / "index.md"
        create_personalized_version(str(index_file), user_id, str(output_index))

    print(f"Personalized module created for user {user_id}")
    print(f"Personalized module: {output_dir}")

    return str(output_dir)


def generate_personalization_report(user_id: str, module_dir: str) -> Dict[str, Any]:
    """
    Generates a report on how content was personalized for a user.

    Args:
        user_id (str): ID of the user
        module_dir (str): Path to the module directory

    Returns:
        Dict[str, Any]: Personalization report
    """
    user_profile = load_user_profile(user_id)
    module_path = Path(module_dir)

    report = {
        "user_id": user_id,
        "module": module_path.name,
        "personalization_applied": {
            "background_level": user_profile.get("background"),
            "programming_experience": user_profile.get("programming_experience"),
            "robotics_experience": user_profile.get("robotics_experience"),
            "content_style": user_profile.get("preferred_content_style")
        },
        "modifications": [],
        "generated_at": "2025-12-17"
    }

    # Add specific modifications based on profile
    if user_profile.get("background") == "beginner":
        report["modifications"].append("Added beginner explanations and simplified concepts")
    elif user_profile.get("background") == "advanced":
        report["modifications"].append("Added advanced technical details and complex examples")

    if user_profile.get("programming_experience") in ["none", "basic"]:
        report["modifications"].append("Simplified code examples with more explanations")
    elif user_profile.get("programming_experience") == "advanced":
        report["modifications"].append("Added complex code examples and optimization strategies")

    if user_profile.get("preferred_content_style") == "concise":
        report["modifications"].append("Made content more concise by removing verbose explanations")

    return report


def main():
    parser = argparse.ArgumentParser(description="Content Personalization System for Physical AI & Humanoid Robotics Textbook")
    parser.add_argument("action", choices=["personalize_chapter", "personalize_module", "generate_report"],
                       help="Action to perform")
    parser.add_argument("--chapter-path", type=str, help="Path to the chapter file")
    parser.add_argument("--module-dir", type=str, help="Path to the module directory")
    parser.add_argument("--user-id", type=str, required=True, help="ID of the user")
    parser.add_argument("--output-path", type=str, help="Path to save personalized content")
    parser.add_argument("--output-dir", type=str, help="Directory to save personalized module")

    args = parser.parse_args()

    if args.action == "personalize_chapter":
        if not args.chapter_path:
            print("Error: --chapter-path is required for personalize_chapter")
            return
        create_personalized_version(args.chapter_path, args.user_id, args.output_path)

    elif args.action == "personalize_module":
        if not args.module_dir:
            print("Error: --module-dir is required for personalize_module")
            return
        create_personalized_module(args.module_dir, args.user_id, args.output_dir)

    elif args.action == "generate_report":
        if not args.module_dir:
            print("Error: --module-dir is required for generate_report")
            return
        report = generate_personalization_report(args.user_id, args.module_dir)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()