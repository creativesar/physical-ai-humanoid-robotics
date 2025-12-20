---
id: 0023
title: "Fix Broken Documentation Links on Homepage"
stage: green
date: "2025-12-20"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "001-physical-ai-textbook"
branch: "001-physical-ai-textbook"
user: "Claude Code"
command: "sp.implement"
labels:
  - "documentation"
  - "ui"
  - "fix"
links:
  - spec: null
  - ticket: null
  - adr: null
  - pr: null
files:
  - frontend/src/components/PremiumHero.tsx
  - frontend/src/components/LuxuryAboutUs.tsx
  - frontend/docusaurus.config.ts
tests: []
outcome: "Successfully updated broken documentation links on the homepage to ensure seamless navigation from landing page to curriculum"
evaluation: "All links now point to the correct destination (/docs/module-1/) instead of the deprecated /docs/intro path"
---

# Fix Broken Documentation Links on Homepage

## Summary
Fixed broken and deprecated documentation links on the homepage to ensure a seamless transition from the landing page to the curriculum.

## Changes Made

### PremiumHero.tsx
- Updated "Start Learning" button link from `/docs/intro` to `/docs/module-1/`
- Updated "Begin Module 1" button link from `/docs/module-1/introduction-to-ros2` to `/docs/module-1/`

### LuxuryAboutUs.tsx
- Updated "Read Textbook" button link from `/docs/intro` to `/docs/module-1/`
- Updated "Our Vision" button link from `/docs/intro#why-physical-ai--humanoid-robotics` to `/docs/module-1/index#overview`

### docusaurus.config.ts
- Updated footer link "Introduction" from `/docs/intro` to `/docs/module-1/`

## Verification
All links have been updated to point to the correct module-1 index page, ensuring users can navigate properly from the homepage to the curriculum content.

## Impact
- Improved user experience by fixing broken navigation paths
- Ensured consistency in documentation linking
- Maintained proper information architecture for the textbook