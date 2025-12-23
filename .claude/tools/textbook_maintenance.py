#!/usr/bin/env python3
"""
Claude Code Subagent: Textbook Maintenance Tools
This tool provides utilities for maintaining the Physical AI & Humanoid Robotics textbook.
"""

import os
import json
import argparse
from pathlib import Path
import re


def update_sidebar_with_module(module_name, sidebar_file="frontend/sidebars.ts"):
    """
    Updates the sidebar configuration to include a new module.

    Args:
        module_name (str): Name of the module to add
        sidebar_file (str): Path to the sidebar configuration file
    """
    module_slug = module_name.lower().replace(" ", "-").replace(":", "").replace("(", "").replace(")", "")

    # Read the current sidebar file
    with open(sidebar_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the 'docs' section and add the new module
    # This is a simplified approach - in practice, you'd want more sophisticated AST parsing
    if f'"{module_slug}"' not in content:
        # Find the docs section and add the new module
        docs_match = re.search(r'(\s*docs:\s*\{[^}]*items:\s*\[)([^\]]*)(\][^}]*\})', content)
        if docs_match:
            before_items = docs_match.group(1)
            items_content = docs_match.group(2)
            after_items = docs_match.group(3)

            # Add the new module to the items list
            if items_content.strip() and not items_content.strip().endswith(','):
                items_content += ','
            items_content += f'\n        "{module_slug}"'

            new_content = content.replace(docs_match.group(0), before_items + items_content + after_items)

            with open(sidebar_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"Module '{module_name}' added to sidebar successfully.")
        else:
            print(f"Could not find docs section in {sidebar_file}")
    else:
        print(f"Module '{module_name}' already exists in sidebar.")


def update_table_of_contents(module_name, toc_file="frontend/docs/intro.md"):
    """
    Updates the table of contents to include a new module.

    Args:
        module_name (str): Name of the module to add
        toc_file (str): Path to the table of contents file
    """
    module_slug = module_name.lower().replace(" ", "-").replace(":", "").replace("(", "").replace(")", "")

    with open(toc_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if the module is already in the TOC
    if f"/docs/{module_slug}" not in content:
        # Add the new module to the table of contents
        # Find the end of the list of modules and add the new one
        new_module_entry = f"\n- [{module_name}](/docs/{module_slug})"
        content = content.replace("\n- [Hardware Requirements](/docs/hardware-requirements)",
                                 f"\n- [Hardware Requirements](/docs/hardware-requirements){new_module_entry}")

        with open(toc_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Module '{module_name}' added to table of contents successfully.")
    else:
        print(f"Module '{module_name}' already exists in table of contents.")


def validate_module_structure(module_dir):
    """
    Validates that a module directory has the required structure.

    Args:
        module_dir (str): Path to the module directory
    """
    module_path = Path(module_dir)
    if not module_path.exists():
        print(f"Error: Module directory {module_dir} does not exist")
        return False

    # Check for required files
    required_files = ["index.md"]
    missing_files = []

    for file in required_files:
        if not (module_path / file).exists():
            missing_files.append(file)

    # Check for at least one topic file
    topic_files = list(module_path.glob("*.md"))
    topic_files = [f for f in topic_files if f.name != "index.md"]

    if missing_files:
        print(f"Missing required files: {missing_files}")

    if not topic_files:
        print("Warning: No topic files found in module directory")

    is_valid = len(missing_files) == 0 and len(topic_files) > 0
    print(f"Module structure validation: {'PASSED' if is_valid else 'FAILED'}")
    return is_valid


def update_assessment_links(module_number, module_title):
    """
    Updates assessment links in the module to point to the correct assessment.

    Args:
        module_number (int): Module number
        module_title (str): Title of the module
    """
    # This would update the module's index.md to include a link to the assessment
    module_dir = Path(f"frontend/docs/module-{module_number}")
    if module_dir.exists():
        index_file = module_dir / "index.md"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Add assessment link if not already present
            assessment_link = f"[Module {module_number} Assessment](/docs/assessments/module-{module_number}-assessment)"
            if assessment_link not in content:
                content += f"\n\n## Assessment\n\nComplete the [Module {module_number} Assessment]({assessment_link}) to test your understanding of {module_title}.\n"

                with open(index_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                print(f"Assessment link added to Module {module_number}.")
            else:
                print(f"Assessment link already exists in Module {module_number}.")


def main():
    parser = argparse.ArgumentParser(description="Textbook Maintenance Tools for Physical AI & Humanoid Robotics Textbook")
    parser.add_argument("action", choices=["update_sidebar", "update_toc", "validate_module", "update_assessment_links"],
                       help="Action to perform")
    parser.add_argument("--module-name", type=str, help="Name of the module")
    parser.add_argument("--module-number", type=int, help="Module number")
    parser.add_argument("--module-dir", type=str, help="Path to module directory")
    parser.add_argument("--sidebar-file", type=str, default="frontend/sidebars.ts", help="Path to sidebar file")
    parser.add_argument("--toc-file", type=str, default="frontend/docs/intro.md", help="Path to table of contents file")

    args = parser.parse_args()

    if args.action == "update_sidebar":
        if not args.module_name:
            print("Error: --module-name is required for update_sidebar")
            return
        update_sidebar_with_module(args.module_name, args.sidebar_file)

    elif args.action == "update_toc":
        if not args.module_name:
            print("Error: --module-name is required for update_toc")
            return
        update_table_of_contents(args.module_name, args.toc_file)

    elif args.action == "validate_module":
        if not args.module_dir:
            print("Error: --module-dir is required for validate_module")
            return
        validate_module_structure(args.module_dir)

    elif args.action == "update_assessment_links":
        if not args.module_number or not args.module_name:
            print("Error: --module-number and --module-name are required for update_assessment_links")
            return
        update_assessment_links(args.module_number, args.module_name)


if __name__ == "__main__":
    main()