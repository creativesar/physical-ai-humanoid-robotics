# Claude Code Subagents and Skills Documentation

This document explains how to use the Claude Code subagents and skills created for the Physical AI & Humanoid Robotics Textbook project.

## Available Subagents and Skills

### 1. Content Generator (`content_generator.py`)

The content generator creates new textbook modules and assessments.

#### Creating a New Module
```bash
python .claude/tools/content_generator.py create_module \
  --title "Module 5: Advanced Control Systems" \
  --topics "Introduction to Control Systems" "PID Controllers" "Advanced Control Algorithms" \
  --output-dir "frontend/docs"
```

#### Creating an Assessment
```bash
python .claude/tools/content_generator.py create_assessment \
  --module-number 5 \
  --title "Module 5: Advanced Control Systems"
```

#### Generating Practice Questions
```bash
python .claude/tools/content_generator.py generate_questions \
  --content "Your chapter content here..." \
  --num-questions 10
```

### 2. Textbook Maintenance (`textbook_maintenance.py`)

The maintenance tools help keep the textbook organized and up-to-date.

#### Updating Sidebar with New Module
```bash
python .claude/tools/textbook_maintenance.py update_sidebar \
  --module-name "Module 5: Advanced Control Systems"
```

#### Updating Table of Contents
```bash
python .claude/tools/textbook_maintenance.py update_toc \
  --module-name "Module 5: Advanced Control Systems"
```

#### Validating Module Structure
```bash
python .claude/tools/textbook_maintenance.py validate_module \
  --module-dir "frontend/docs/module-5"
```

#### Updating Assessment Links
```bash
python .claude/tools/textbook_maintenance.py update_assessment_links \
  --module-number 5 \
  --module-name "Module 5: Advanced Control Systems"
```

### 3. Practice Question Generator (`practice_question_generator.py`)

Generates practice questions based on chapter content.

#### Generating Questions for a Chapter
```bash
python .claude/tools/practice_question_generator.py generate_questions \
  --chapter-path "frontend/docs/module-1/introduction-to-ros2.md" \
  --output-path "frontend/docs/module-1/introduction-to-ros2_questions.json"
```

#### Generating Module Assessment
```bash
python .claude/tools/practice_question_generator.py generate_assessment \
  --module-dir "frontend/docs/module-1" \
  --output-path "frontend/docs/module-1/module-1_comprehensive_assessment.json"
```

### 4. Content Update System (`content_update_system.py`)

Automatically updates content across the textbook.

#### Updating Specific Content
```bash
python .claude/tools/content_update_system.py update_content \
  --file-path "frontend/docs/module-1/introduction-to-ros2.md" \
  --old-text "old information" \
  --new-text "new information"
```

#### Updating All References to a URL
```bash
python .claude/tools/content_update_system.py update_references \
  --old-url "http://old-url.com" \
  --new-url "http://new-url.com" \
  --root-dir "frontend/docs"
```

#### Updating Terminology Throughout Textbook
```bash
python .claude/tools/content_update_system.py update_terminology \
  --old-term "old term" \
  --new-term "new term" \
  --root-dir "frontend/docs" \
  --case-sensitive
```

#### Synchronizing Module Overviews
```bash
python .claude/tools/content_update_system.py sync_overviews \
  --root-dir "frontend/docs"
```

### 5. Content Personalization (`content_personalization.py`)

Personalizes content based on user background and expertise.

#### Creating Personalized Chapter
```bash
python .claude/tools/content_personalization.py personalize_chapter \
  --chapter-path "frontend/docs/module-1/introduction-to-ros2.md" \
  --user-id "user123"
```

#### Creating Personalized Module
```bash
python .claude/tools/content_personalization.py personalize_module \
  --module-dir "frontend/docs/module-1" \
  --user-id "user123" \
  --output-dir "frontend/docs/personalized/user123/module-1"
```

#### Generating Personalization Report
```bash
python .claude/tools/content_personalization.py generate_report \
  --module-dir "frontend/docs/module-1" \
  --user-id "user123"
```

## User Profile Structure

The personalization system uses user profiles stored in `frontend/src/data/user_profiles/{user_id}.json`:

```json
{
  "user_id": "user123",
  "background": "beginner",  // beginner, intermediate, advanced
  "interests": ["robotics", "ai"],
  "programming_experience": "basic",  // none, basic, intermediate, advanced
  "robotics_experience": "none",  // none, basic, intermediate, advanced
  "learning_goals": ["understand_basics", "build_simple_robot"],
  "preferred_content_style": "detailed"  // concise, detailed, hands_on
}
```

## Integration with Claude Code

To use these tools with Claude Code, you can reference them in your prompts:

```
Use the content generator tool to create a new module about "Computer Vision for Robotics".
```

Or call them directly:

```
Run: python .claude/tools/content_generator.py create_module --title "Module X: Topic" --topics "Topic 1" "Topic 2" "Topic 3"
```

## Best Practices

1. Always create backups before running update operations
2. Test personalization with sample user profiles before deploying
3. Validate module structure after creating new modules
4. Use the assessment generators to maintain quality standards
5. Update terminology consistently across the textbook when concepts evolve