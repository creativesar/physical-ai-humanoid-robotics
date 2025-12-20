#!/usr/bin/env python3
"""
Claude Code Skill: Practice Question Generator
This tool generates practice questions based on textbook chapter content.
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any


def extract_chapter_content(file_path: str) -> str:
    """
    Extracts the main content from a chapter file, excluding frontmatter.

    Args:
        file_path (str): Path to the chapter file

    Returns:
        str: Extracted chapter content
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove frontmatter if present
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            content = parts[2]

    return content


def generate_multiple_choice_questions(content: str, num_questions: int = 5) -> List[Dict[str, Any]]:
    """
    Generates multiple choice questions based on chapter content.
    In a real implementation, this would use an LLM to analyze the content.

    Args:
        content (str): Chapter content to base questions on
        num_questions (int): Number of questions to generate

    Returns:
        List[Dict[str, Any]]: List of generated questions
    """
    # This is a simplified implementation - in practice, you'd use an LLM
    # to analyze the content and generate relevant questions
    questions = []

    # Extract key terms and concepts from the content
    # This is a basic approach - in practice, you'd use NLP techniques
    words = re.findall(r'\b[A-Z][a-z]+\w*\b', content)
    unique_words = list(set(words))

    for i in range(min(num_questions, len(unique_words))):
        word = unique_words[i]
        question = {
            "id": f"mcq_{i+1}",
            "type": "multiple_choice",
            "question": f"What is the significance of {word} in the context of Physical AI and Humanoid Robotics?",
            "options": [
                f"Option A: Related to {word} implementation",
                f"Option B: Related to {word} theory",
                f"Option C: Related to {word} applications",
                f"Option D: Related to {word} challenges"
            ],
            "correct_answer": "A",
            "explanation": f"Explanation about {word} based on chapter content"
        }
        questions.append(question)

    return questions


def generate_short_answer_questions(content: str, num_questions: int = 3) -> List[Dict[str, Any]]:
    """
    Generates short answer questions based on chapter content.

    Args:
        content (str): Chapter content to base questions on
        num_questions (int): Number of questions to generate

    Returns:
        List[Dict[str, Any]]: List of generated questions
    """
    questions = []

    # Extract sentences that could form the basis of questions
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter short sentences

    for i in range(min(num_questions, len(sentences))):
        sentence = sentences[i][:100]  # Take first 100 chars as context
        question = {
            "id": f"saq_{i+1}",
            "type": "short_answer",
            "question": f"Explain the concept mentioned in this excerpt: '{sentence}...'",
            "answer_guidelines": "Provide a detailed explanation of the concept and its application in humanoid robotics",
            "difficulty": "medium"
        }
        questions.append(question)

    return questions


def generate_practice_questions(chapter_path: str, output_path: str = None) -> str:
    """
    Generates practice questions for a given chapter.

    Args:
        chapter_path (str): Path to the chapter file
        output_path (str): Path to save the generated questions (optional)

    Returns:
        str: Path to the generated questions file
    """
    # Extract content from the chapter
    content = extract_chapter_content(chapter_path)
    chapter_name = Path(chapter_path).stem

    # Generate different types of questions
    mcq_questions = generate_multiple_choice_questions(content, 5)
    saq_questions = generate_short_answer_questions(content, 3)

    # Combine all questions
    all_questions = {
        "chapter": chapter_name,
        "generated_at": "2025-12-17",
        "multiple_choice": mcq_questions,
        "short_answer": saq_questions,
        "total_questions": len(mcq_questions) + len(saq_questions)
    }

    # Determine output path
    if not output_path:
        chapter_dir = Path(chapter_path).parent
        output_path = chapter_dir / f"{chapter_name}_questions.json"

    # Save questions to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, indent=2)

    print(f"Generated {all_questions['total_questions']} practice questions for '{chapter_name}'")
    print(f"Questions saved to: {output_path}")

    return str(output_path)


def generate_assessment_from_module(module_dir: str, output_path: str = None) -> str:
    """
    Generates a comprehensive assessment from all chapters in a module.

    Args:
        module_dir (str): Path to the module directory
        output_path (str): Path to save the generated assessment (optional)

    Returns:
        str: Path to the generated assessment file
    """
    module_path = Path(module_dir)
    if not module_path.exists():
        raise FileNotFoundError(f"Module directory {module_dir} does not exist")

    # Find all markdown files in the module (excluding index.md for now)
    chapter_files = list(module_path.glob("*.md"))
    chapter_files = [f for f in chapter_files if f.name != "index.md"]

    if not chapter_files:
        print(f"No chapter files found in {module_dir}")
        return ""

    # Generate questions for each chapter
    all_assessment_data = {
        "module": module_path.name,
        "generated_at": "2025-12-17",
        "chapters": {}
    }

    for chapter_file in chapter_files:
        print(f"Processing chapter: {chapter_file.name}")
        questions_file = generate_practice_questions(str(chapter_file))

        # Load the generated questions
        with open(questions_file, 'r', encoding='utf-8') as f:
            chapter_questions = json.load(f)

        all_assessment_data["chapters"][chapter_file.stem] = chapter_questions

    # Determine output path
    if not output_path:
        module_name = module_path.name
        output_path = module_path / f"{module_name}_comprehensive_assessment.json"

    # Save comprehensive assessment
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_assessment_data, f, indent=2)

    print(f"Comprehensive assessment generated for module '{module_path.name}'")
    print(f"Assessment saved to: {output_path}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Practice Question Generator for Physical AI & Humanoid Robotics Textbook")
    parser.add_argument("action", choices=["generate_questions", "generate_assessment"],
                       help="Action to perform")
    parser.add_argument("--chapter-path", type=str, help="Path to the chapter file")
    parser.add_argument("--module-dir", type=str, help="Path to the module directory")
    parser.add_argument("--output-path", type=str, help="Path to save generated questions")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions to generate")

    args = parser.parse_args()

    if args.action == "generate_questions":
        if not args.chapter_path:
            print("Error: --chapter-path is required for generate_questions")
            return
        generate_practice_questions(args.chapter_path, args.output_path)

    elif args.action == "generate_assessment":
        if not args.module_dir:
            print("Error: --module-dir is required for generate_assessment")
            return
        generate_assessment_from_module(args.module_dir, args.output_path)


if __name__ == "__main__":
    main()