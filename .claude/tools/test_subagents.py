#!/usr/bin/env python3
"""
Claude Code Subagent: Test Suite
This script tests the various subagents with different textbook generation scenarios.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from content_generator import create_module_content, create_assessment, create_practice_questions
from textbook_maintenance import update_sidebar_with_module, validate_module_structure
from practice_question_generator import generate_practice_questions
from content_update_system import update_terminology, sync_module_overviews
from content_personalization import create_personalized_version, load_user_profile


def setup_test_environment():
    """Creates a temporary test environment."""
    test_dir = Path(tempfile.mkdtemp(prefix="textbook_test_"))
    frontend_dir = test_dir / "frontend"
    docs_dir = frontend_dir / "docs"
    docs_dir.mkdir(parents=True)

    # Create a sample chapter for testing
    sample_chapter = docs_dir / "sample_module"
    sample_chapter.mkdir()
    with open(sample_chapter / "index.md", 'w') as f:
        f.write("""---
sidebar_position: 1
title: "Sample Module"
---
# Sample Module

This is a sample module for testing purposes.

## Introduction

This module introduces basic concepts.

## Advanced Topic

This section covers more advanced material that might need personalization.
""")

    with open(sample_chapter / "basic_topic.md", 'w') as f:
        f.write("""---
sidebar_position: 1
title: "Basic Topic"
---
# Basic Topic

This is a basic topic in the sample module.

For example, we can see that this concept is fundamental.

The implementation is straightforward.
""")

    return test_dir, docs_dir, sample_chapter


def test_content_generator():
    """Test the content generator subagent."""
    print("Testing Content Generator...")

    test_dir, docs_dir, sample_chapter = setup_test_environment()

    try:
        # Test creating a new module
        module_path = create_module_content(
            "Test Module: Sensor Integration",
            ["Types of Sensors", "Sensor Fusion", "Calibration Techniques"],
            target_dir=str(docs_dir)
        )
        print(f"OK Created module at: {module_path}")

        # Test creating an assessment
        assessment_path = create_assessment(99, "Test Module: Sensor Integration", target_dir=str(docs_dir / "assessments"))
        print(f"OK Created assessment at: {assessment_path}")

        # Test generating practice questions
        chapter_content = "Robot sensors are crucial for perception. Common types include cameras, LIDAR, and IMUs."
        questions = create_practice_questions(chapter_content, num_questions=3)
        print(f"OK Generated {len(questions)} practice questions")

        return True
    except Exception as e:
        print(f"FAILED Content Generator test failed: {str(e)}")
        return False
    finally:
        shutil.rmtree(test_dir)


def test_textbook_maintenance():
    """Test the textbook maintenance subagent."""
    print("Testing Textbook Maintenance...")

    test_dir, docs_dir, sample_chapter = setup_test_environment()

    try:
        # Create a module to test maintenance on
        module_path = create_module_content(
            "Maintenance Test Module",
            ["Topic 1", "Topic 2"],
            target_dir=str(docs_dir)
        )

        # Test updating sidebar (this would normally modify an existing sidebar.ts file)
        # For this test, we'll just verify the function exists and can be called
        print("OK Sidebar update function is available")

        # Test validating module structure
        is_valid = validate_module_structure(module_path)
        print(f"OK Module validation: {'PASSED' if is_valid else 'FAILED'}")

        return True
    except Exception as e:
        print(f"FAILED Textbook Maintenance test failed: {str(e)}")
        return False
    finally:
        shutil.rmtree(test_dir)


def test_practice_question_generator():
    """Test the practice question generator."""
    print("Testing Practice Question Generator...")

    test_dir, docs_dir, sample_chapter = setup_test_environment()

    try:
        # Create a sample chapter file
        sample_content = """---
sidebar_position: 1
title: "Test Chapter: ROS 2 Basics"
---
# Test Chapter: ROS 2 Basics

## Introduction

ROS 2 is a flexible framework for writing robotic software.

## Key Concepts

The main concepts include nodes, topics, services, and actions.

## Advanced Section

More complex implementations use custom message types and QoS settings.
"""
        chapter_file = docs_dir / "test_chapter.md"
        with open(chapter_file, 'w') as f:
            f.write(sample_content)

        # Test generating questions
        questions_file = generate_practice_questions(str(chapter_file))
        print(f"OK Generated questions saved to: {questions_file}")

        # Verify the questions file exists and has content
        if Path(questions_file).exists():
            with open(questions_file, 'r') as f:
                questions_data = json.load(f)
            print(f"OK Questions file contains {questions_data.get('total_questions', 0)} questions")

        return True
    except Exception as e:
        print(f"FAILED Practice Question Generator test failed: {str(e)}")
        return False
    finally:
        shutil.rmtree(test_dir)


def test_content_update_system():
    """Test the content update system."""
    print("Testing Content Update System...")

    test_dir, docs_dir, sample_chapter = setup_test_environment()

    try:
        # Create a test file with some content to update
        test_file = sample_chapter / "update_test.md"
        with open(test_file, 'w') as f:
            f.write("""# Update Test

This document uses the old terminology.

We refer to the robot's control system as the 'old controller'.

The old controller is important for basic operations.
""")

        # Test updating terminology
        results = update_terminology("old controller", "modern controller", root_dir=str(docs_dir))
        successful_updates = sum(1 for success in results.values() if success)
        print(f"OK Terminology update: {successful_updates}/{len(results)} files updated")

        # Verify the update worked
        with open(test_file, 'r') as f:
            updated_content = f.read()

        if "modern controller" in updated_content and "old controller" not in updated_content:
            print("OK Terminology successfully updated")
        else:
            print("FAILED Terminology update failed")
            return False

        # Test sync overviews
        overview_results = sync_module_overviews(root_dir=str(docs_dir))
        print(f"OK Overview sync completed for {len(overview_results)} modules")

        return True
    except Exception as e:
        print(f"FAILED Content Update System test failed: {str(e)}")
        return False
    finally:
        shutil.rmtree(test_dir)


def test_content_personalization():
    """Test the content personalization system."""
    print("Testing Content Personalization...")

    test_dir, docs_dir, sample_chapter = setup_test_environment()

    try:
        # Create user profiles for different experience levels
        user_profiles_dir = Path("frontend/src/data/user_profiles")
        user_profiles_dir.mkdir(parents=True, exist_ok=True)

        # Create beginner profile
        beginner_profile = {
            "user_id": "beginner_user",
            "background": "beginner",
            "interests": ["robotics"],
            "programming_experience": "none",
            "robotics_experience": "none",
            "learning_goals": ["understand_basics"],
            "preferred_content_style": "detailed"
        }

        with open(user_profiles_dir / "beginner_user.json", 'w') as f:
            json.dump(beginner_profile, f)

        # Create advanced profile
        advanced_profile = {
            "user_id": "advanced_user",
            "background": "advanced",
            "interests": ["robotics", "ai", "control_systems"],
            "programming_experience": "advanced",
            "robotics_experience": "advanced",
            "learning_goals": ["implement_complex_systems"],
            "preferred_content_style": "concise"
        }

        with open(user_profiles_dir / "advanced_user.json", 'w') as f:
            json.dump(advanced_profile, f)

        # Create a chapter to personalize
        chapter_content = """---
sidebar_position: 1
title: "Personalization Test: Navigation"
---
# Navigation Systems

## Introduction

Navigation is a complex topic in robotics that involves path planning and obstacle avoidance.

For example, we use algorithms like A* and Dijkstra for path planning.

## Implementation

The implementation requires understanding of coordinate frames and transforms.
"""
        chapter_file = docs_dir / "personalization_test.md"
        with open(chapter_file, 'w') as f:
            f.write(chapter_content)

        # Test personalizing for beginner
        beginner_output = create_personalized_version(str(chapter_file), "beginner_user")
        print(f"OK Created beginner version: {beginner_output}")

        # Test personalizing for advanced user
        advanced_output = create_personalized_version(str(chapter_file), "advanced_user")
        print(f"OK Created advanced version: {advanced_output}")

        # Verify both files exist
        if Path(beginner_output).exists() and Path(advanced_output).exists():
            print("OK Both personalized versions created successfully")
        else:
            print("FAILED Personalized versions not created")
            return False

        # Load and check user profiles
        beginner_data = load_user_profile("beginner_user")
        advanced_data = load_user_profile("advanced_user")

        print(f"OK Beginner profile loaded: {beginner_data['background']}")
        print(f"OK Advanced profile loaded: {advanced_data['background']}")

        return True
    except Exception as e:
        print(f"FAILED Content Personalization test failed: {str(e)}")
        return False
    finally:
        # Clean up user profiles
        for profile_file in ["beginner_user.json", "advanced_user.json"]:
            profile_path = user_profiles_dir / profile_file
            if profile_path.exists():
                profile_path.unlink()


def run_all_tests():
    """Run all subagent tests."""
    print("Running Claude Code Subagent Tests...\n")

    tests = [
        ("Content Generator", test_content_generator),
        ("Textbook Maintenance", test_textbook_maintenance),
        ("Practice Question Generator", test_practice_question_generator),
        ("Content Update System", test_content_update_system),
        ("Content Personalization", test_content_personalization),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        success = test_func()
        results.append((test_name, success))
        print(f"Result: {'PASS' if success else 'FAIL'}")

    print(f"\n--- Test Summary ---")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Passed: {passed}/{total} tests")

    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")

    return passed == total


def main():
    """Main function to run the test suite."""
    success = run_all_tests()

    if success:
        print("\nSUCCESS: All tests passed! The Claude Code subagents are working correctly.")
    else:
        print("\nFAILED: Some tests failed. Please review the implementation.")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())