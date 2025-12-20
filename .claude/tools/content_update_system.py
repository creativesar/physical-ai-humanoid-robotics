#!/usr/bin/env python3
"""
Claude Code Subagent: Automated Content Update System
This tool automatically updates textbook content based on new information, corrections, or improvements.
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime


def update_content_with_changes(file_path: str, updates: List[Dict[str, Any]], backup: bool = True) -> bool:
    """
    Updates content in a file based on a list of changes.

    Args:
        file_path (str): Path to the file to update
        updates (List[Dict]): List of updates to apply
        backup (bool): Whether to create a backup before updating

    Returns:
        bool: True if update was successful, False otherwise
    """
    # Create backup if requested
    if backup:
        backup_path = f"{file_path}.backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(file_path, 'r', encoding='utf-8') as original:
            with open(backup_path, 'w', encoding='utf-8') as backup_file:
                backup_file.write(original.read())
        print(f"Backup created: {backup_path}")

    # Read the current content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Apply each update
    for update in updates:
        if update['type'] == 'replace':
            # Replace specific text
            old_text = update['old_text']
            new_text = update['new_text']
            if old_text in content:
                content = content.replace(old_text, new_text)
                print(f"Replaced text in {file_path}")
            else:
                print(f"Warning: Old text not found in {file_path} for update: {update.get('description', 'N/A')}")
        elif update['type'] == 'append':
            # Append content to the end of the file
            content += "\n" + update['new_content']
            print(f"Appended content to {file_path}")
        elif update['type'] == 'prepend':
            # Prepend content to the beginning of the file
            content = update['new_content'] + "\n" + content
            print(f"Prepended content to {file_path}")
        elif update['type'] == 'update_section':
            # Update a specific section based on a pattern
            section_pattern = update['section_pattern']
            new_section = update['new_section']
            if re.search(section_pattern, content):
                content = re.sub(section_pattern, new_section, content)
                print(f"Updated section in {file_path}")
            else:
                print(f"Warning: Section pattern not found in {file_path} for update: {update.get('description', 'N/A')}")

    # Write the updated content back to the file
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Content updated successfully: {file_path}")
        return True
    else:
        print(f"No changes made to {file_path}")
        return False


def find_content_files(root_dir: str = "frontend/docs", pattern: str = "*.md") -> List[str]:
    """
    Finds all content files in the textbook directory.

    Args:
        root_dir (str): Root directory to search
        pattern (str): File pattern to match

    Returns:
        List[str]: List of file paths
    """
    content_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(pattern.replace("*.", "")):
                content_files.append(os.path.join(root, file))
    return content_files


def update_all_modules_with_common_info(updates: List[Dict[str, Any]], root_dir: str = "frontend/docs") -> Dict[str, bool]:
    """
    Updates all module files with common information.

    Args:
        updates (List[Dict]): List of updates to apply
        root_dir (str): Root directory containing modules

    Returns:
        Dict[str, bool]: Dictionary mapping file paths to update success status
    """
    results = {}
    content_files = find_content_files(root_dir)

    for file_path in content_files:
        # Skip index files for this operation unless specifically needed
        if "index.md" in file_path:
            continue

        try:
            success = update_content_with_changes(file_path, updates, backup=True)
            results[file_path] = success
        except Exception as e:
            print(f"Error updating {file_path}: {str(e)}")
            results[file_path] = False

    return results


def update_references_and_links(old_url: str, new_url: str, root_dir: str = "frontend/docs") -> Dict[str, bool]:
    """
    Updates all references and links from an old URL to a new URL.

    Args:
        old_url (str): Old URL to replace
        new_url (str): New URL to use
        root_dir (str): Root directory to search

    Returns:
        Dict[str, bool]: Dictionary mapping file paths to update success status
    """
    update_spec = [{
        'type': 'replace',
        'old_text': old_url,
        'new_text': new_url,
        'description': f'Update URL from {old_url} to {new_url}'
    }]

    return update_all_modules_with_common_info(update_spec, root_dir)


def update_terminology(old_term: str, new_term: str, root_dir: str = "frontend/docs", case_sensitive: bool = False) -> Dict[str, bool]:
    """
    Updates all instances of an old term with a new term.

    Args:
        old_term (str): Old term to replace
        new_term (str): New term to use
        root_dir (str): Root directory to search
        case_sensitive (bool): Whether the search should be case sensitive

    Returns:
        Dict[str, bool]: Dictionary mapping file paths to update success status
    """
    content_files = find_content_files(root_dir)
    results = {}

    for file_path in content_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Perform replacement
            if case_sensitive:
                content = content.replace(old_term, new_term)
            else:
                # Case insensitive replacement using regex
                pattern = re.compile(re.escape(old_term), re.IGNORECASE)
                content = pattern.sub(new_term, content)

            # Write back if changes were made
            if content != original_content:
                # Create backup
                backup_path = f"{file_path}.backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with open(backup_path, 'w', encoding='utf-8') as backup_file:
                    backup_file.write(original_content)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                print(f"Updated terminology in {file_path}: {old_term} -> {new_term}")
                results[file_path] = True
            else:
                results[file_path] = False
        except Exception as e:
            print(f"Error updating terminology in {file_path}: {str(e)}")
            results[file_path] = False

    return results


def sync_module_overviews(root_dir: str = "frontend/docs") -> Dict[str, bool]:
    """
    Synchronizes module overview content with the latest information.

    Args:
        root_dir (str): Root directory containing modules

    Returns:
        Dict[str, bool]: Dictionary mapping file paths to update success status
    """
    # Find all module directories
    module_dirs = [d for d in Path(root_dir).iterdir() if d.is_dir() and d.name.startswith('module-')]

    results = {}
    for module_dir in module_dirs:
        index_file = module_dir / "index.md"
        if index_file.exists():
            # Read the current content
            with open(index_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract frontmatter
            frontmatter_match = re.match(r'(^---\n.*?\n---\n)', content, re.DOTALL)
            frontmatter = frontmatter_match.group(1) if frontmatter_match else ""
            main_content = content[len(frontmatter):] if frontmatter_match else content

            # Update the overview section
            if "## Overview" in main_content:
                # For now, we'll just add a standard update notice
                # In a real implementation, this would update with new content
                update_notice = f"\n\n> **Content Updated**: This module was last updated on {datetime.date.today().isoformat()}\n"

                # Find the position after the overview section
                overview_match = re.search(r'(## Overview\n.*?\n)(##|$)', main_content, re.DOTALL)
                if overview_match:
                    before_overview = main_content[:overview_match.end(1)]
                    after_overview = main_content[overview_match.end(1):]
                    new_content = frontmatter + before_overview + update_notice + after_overview
                else:
                    new_content = frontmatter + main_content + update_notice
            else:
                new_content = content

            # Write the updated content back
            if new_content != content:
                # Create backup
                backup_path = f"{index_file}.backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with open(backup_path, 'w', encoding='utf-8') as backup_file:
                    backup_file.write(content)

                with open(index_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                print(f"Updated overview in {index_file}")
                results[str(index_file)] = True
            else:
                results[str(index_file)] = False

    return results


def main():
    parser = argparse.ArgumentParser(description="Automated Content Update System for Physical AI & Humanoid Robotics Textbook")
    parser.add_argument("action", choices=["update_content", "update_references", "update_terminology", "sync_overviews"],
                       help="Action to perform")
    parser.add_argument("--file-path", type=str, help="Path to specific file to update")
    parser.add_argument("--root-dir", type=str, default="frontend/docs", help="Root directory to search")
    parser.add_argument("--old-text", type=str, help="Old text to replace (for update_content)")
    parser.add_argument("--new-text", type=str, help="New text to use (for update_content)")
    parser.add_argument("--old-url", type=str, help="Old URL to replace (for update_references)")
    parser.add_argument("--new-url", type=str, help="New URL to use (for update_references)")
    parser.add_argument("--old-term", type=str, help="Old term to replace (for update_terminology)")
    parser.add_argument("--new-term", type=str, help="New term to use (for update_terminology)")
    parser.add_argument("--backup", action="store_true", default=True, help="Create backup before updating")
    parser.add_argument("--case-sensitive", action="store_true", default=False, help="Case sensitive replacement")

    args = parser.parse_args()

    if args.action == "update_content":
        if not args.file_path or not args.old_text or not args.new_text:
            print("Error: --file-path, --old-text, and --new-text are required for update_content")
            return

        updates = [{
            'type': 'replace',
            'old_text': args.old_text,
            'new_text': args.new_text,
            'description': f'Update from "{args.old_text}" to "{args.new_text}"'
        }]

        success = update_content_with_changes(args.file_path, updates, args.backup)
        print(f"Update {'successful' if success else 'failed'}")

    elif args.action == "update_references":
        if not args.old_url or not args.new_url:
            print("Error: --old-url and --new-url are required for update_references")
            return

        results = update_references_and_links(args.old_url, args.new_url, args.root_dir)
        successful_updates = sum(1 for success in results.values() if success)
        print(f"Reference updates completed: {successful_updates}/{len(results)} files updated")

    elif args.action == "update_terminology":
        if not args.old_term or not args.new_term:
            print("Error: --old-term and --new-term are required for update_terminology")
            return

        results = update_terminology(args.old_term, args.new_term, args.root_dir, args.case_sensitive)
        successful_updates = sum(1 for success in results.values() if success)
        print(f"Terminology updates completed: {successful_updates}/{len(results)} files updated")

    elif args.action == "sync_overviews":
        results = sync_module_overviews(args.root_dir)
        successful_updates = sum(1 for success in results.values() if success)
        print(f"Overview synchronization completed: {successful_updates}/{len(results)} modules updated")


if __name__ == "__main__":
    main()