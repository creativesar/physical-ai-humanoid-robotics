# Personalization System Guide

This guide explains how to use the personalization system in the Physical AI & Humanoid Robotics Textbook.

## Overview

The personalization system allows users to:
- Choose difficulty level (Beginner, Intermediate, Advanced)
- Switch between English and Urdu languages
- Auto-detect difficulty based on their background
- Persist preferences across sessions

## Components

### 1. PersonalizationContext
Manages user preferences and profile data.

```tsx
import { usePersonalization } from '../contexts/PersonalizationContext';

function MyComponent() {
  const {
    userProfile,
    difficultyLevel,
    language,
    setDifficultyLevel,
    setLanguage,
    updateUserProfile,
    isPersonalized,
  } = usePersonalization();

  return (
    <div>
      <p>Current Level: {difficultyLevel}</p>
      <p>Current Language: {language}</p>
    </div>
  );
}
```

### 2. ChapterPersonalizationBar
Sticky bar at the top of each chapter for controlling preferences.

```tsx
import ChapterPersonalizationBar from '../components/ChapterPersonalizationBar';

function ChapterPage() {
  return (
    <>
      <ChapterPersonalizationBar />
      <div>
        {/* Your chapter content */}
      </div>
    </>
  );
}
```

### 3. PersonalizedContent
Component to render different content based on user preferences.

```tsx
import PersonalizedContent from '../components/PersonalizedContent';

// Example 1: Difficulty-based content
<PersonalizedContent
  beginner={
    <p>ROS 2 is a robot operating system. It helps robots communicate.</p>
  }
  intermediate={
    <p>ROS 2 uses a publish-subscribe pattern for node communication.</p>
  }
  advanced={
    <p>ROS 2 leverages DDS for real-time, distributed communication with QoS policies.</p>
  }
/>

// Example 2: Language-based content
<PersonalizedContent
  english={<h1>Introduction to ROS 2</h1>}
  urdu={<h1>ROS 2 کا تعارف</h1>}
/>

// Example 3: Both difficulty and language
<PersonalizedContent
  beginner={
    language === 'urdu' ?
      <p>ROS 2 ایک روبوٹ آپریٹنگ سسٹم ہے۔</p> :
      <p>ROS 2 is a robot operating system.</p>
  }
  intermediate={
    language === 'urdu' ?
      <p>ROS 2 نوڈ مواصلات کے لیے publish-subscribe pattern استعمال کرتا ہے۔</p> :
      <p>ROS 2 uses a publish-subscribe pattern for node communication.</p>
  }
  advanced={
    language === 'urdu' ?
      <p>ROS 2 QoS پالیسیوں کے ساتھ real-time، distributed مواصلات کے لیے DDS استعمال کرتا ہے۔</p> :
      <p>ROS 2 leverages DDS for real-time, distributed communication with QoS policies.</p>
  }
/>

// Example 4: Dynamic translation (translates English to Urdu in real-time)
<PersonalizedContent
  dynamicTranslation={true}
  fallbackToDynamic={true}
>
  <p>This content will be translated to Urdu dynamically using the backend API</p>
</PersonalizedContent>

// Example 5: Dynamic translation for difficulty levels
<PersonalizedContent
  beginner="ROS 2 is a robot operating system."
  intermediate="ROS 2 uses a publish-subscribe pattern for node communication."
  advanced="ROS 2 leverages DDS for real-time, distributed communication."
  dynamicTranslation={true}
/>
```

### 4. Utility Hooks

```tsx
import { useDifficultyContent, useLanguageContent } from '../components/PersonalizedContent';

// Get content based on difficulty
const title = useDifficultyContent({
  beginner: 'Getting Started with ROS 2',
  intermediate: 'ROS 2 Architecture',
  advanced: 'Advanced ROS 2 Patterns',
});

// Get content based on language
const greeting = useLanguageContent({
  english: 'Welcome',
  urdu: 'خوش آمدید',
});
```

## How to Add Personalization to a Chapter

### Step 1: Add the Personalization Bar

At the top of your MDX file or component:

```tsx
import ChapterPersonalizationBar from '@site/src/components/ChapterPersonalizationBar';

<ChapterPersonalizationBar />
```

### Step 2: Add Personalized Content

Replace static content with personalized versions:

**Before:**
```mdx
## Introduction to ROS 2

ROS 2 uses a publish-subscribe pattern for communication.
```

**After:**
```mdx
import PersonalizedContent from '@site/src/components/PersonalizedContent';

## <PersonalizedContent
     english="Introduction to ROS 2"
     urdu="ROS 2 کا تعارف"
   />

<PersonalizedContent
  beginner="ROS 2 helps robots talk to each other using messages."
  intermediate="ROS 2 uses a publish-subscribe pattern for communication."
  advanced="ROS 2 implements DDS middleware for distributed real-time communication."
/>
```

## Data Storage

### LocalStorage
User preferences are automatically saved to browser's LocalStorage:
- Key: `physical-ai-personalization`
- Data includes: userProfile, difficultyLevel, language, timestamp

### Session Persistence
- Preferences persist across page refreshes
- Data is loaded on app initialization
- Updates are saved automatically

## Auto-Detection Logic

When a user signs up, the system automatically detects their difficulty level:

```
Software Background + Hardware Background
├─ Both present (>10 chars each) → Advanced
├─ One present → Intermediate
└─ None present → Beginner
```

## Urdu Support

### RTL Text Direction
For Urdu content, add `dir="rtl"` to containers:

```tsx
<div dir={language === 'urdu' ? 'rtl' : 'ltr'}>
  {content}
</div>
```

### Font Recommendations
For better Urdu rendering, add to your CSS:

```css
[dir="rtl"] {
  font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif;
}
```

## Example: Full Chapter Implementation

```tsx
import React from 'react';
import Layout from '@theme/Layout';
import ChapterPersonalizationBar from '@site/src/components/ChapterPersonalizationBar';
import PersonalizedContent from '@site/src/components/PersonalizedContent';
import { usePersonalization } from '@site/src/contexts/PersonalizationContext';

export default function ROS2Chapter() {
  const { language } = usePersonalization();

  return (
    <Layout>
      <ChapterPersonalizationBar />

      <div
        style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}
        dir={language === 'urdu' ? 'rtl' : 'ltr'}
      >
        <h1>
          <PersonalizedContent
            english="Chapter 1: Introduction to ROS 2"
            urdu="باب 1: ROS 2 کا تعارف"
          />
        </h1>

        <PersonalizedContent
          beginner={
            <div>
              <p>ROS 2 is like a language that robots use to talk to each other.</p>
              <p>Imagine robots sending messages to each other!</p>
            </div>
          }
          intermediate={
            <div>
              <p>ROS 2 (Robot Operating System 2) is a middleware framework for building robot applications.</p>
              <p>It uses a publish-subscribe communication pattern between nodes.</p>
            </div>
          }
          advanced={
            <div>
              <p>ROS 2 is a distributed middleware built on top of DDS (Data Distribution Service).</p>
              <p>It provides QoS policies for real-time communication, supports multiple programming languages, and offers robust tools for debugging and visualization.</p>
            </div>
          }
        />

        {/* Add more content sections */}
      </div>
    </Layout>
  );
}
```

## Best Practices

1. **Always provide fallbacks**: If one difficulty level is missing, the system falls back to available levels
2. **Keep content aligned**: Ensure all three difficulty levels cover the same concepts
3. **Test both languages**: Verify that Urdu translations are accurate and properly formatted
4. **Use semantic HTML**: Maintain proper heading hierarchy regardless of personalization
5. **Performance**: Personalization is client-side only, no API calls needed

## API Reference

### PersonalizationContext

```typescript
interface PersonalizationContextType {
  userProfile: UserProfile | null;
  difficultyLevel: 'beginner' | 'intermediate' | 'advanced';
  language: 'english' | 'urdu';
  setDifficultyLevel: (level: DifficultyLevel) => void;
  setLanguage: (lang: Language) => void;
  updateUserProfile: (profile: Partial<UserProfile>) => void;
  isPersonalized: boolean;
}
```

### UserProfile

```typescript
interface UserProfile {
  name?: string;
  email?: string;
  softwareBackground?: string;
  hardwareBackground?: string;
  difficultyLevel?: DifficultyLevel;
  preferredLanguage?: Language;
}
```

## Troubleshooting

### Preferences not persisting
- Check browser's LocalStorage is enabled
- Clear cache and try again
- Check browser console for errors

### Urdu text not displaying correctly
- Ensure proper UTF-8 encoding
- Add Urdu font support
- Check RTL direction is set

### Personalization not working in MDX
- Ensure component is properly imported
- Check PersonalizationProvider is wrapped in Root.tsx
- Verify component is client-side rendered
