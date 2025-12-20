import React from 'react';
import { usePersonalization, DifficultyLevel } from '../contexts/PersonalizationContext';
import DynamicTranslation from './DynamicTranslation';

interface PersonalizedContentProps {
  beginner?: React.ReactNode;
  intermediate?: React.ReactNode;
  advanced?: React.ReactNode;
  english?: React.ReactNode;
  urdu?: React.ReactNode;
  children?: React.ReactNode;
  dynamicTranslation?: boolean; // Whether to use dynamic translation API
  fallbackToDynamic?: boolean; // Whether to fall back to dynamic translation if static content not provided
}

/**
 * Component to render content based on user's difficulty level and language preference
 *
 * Usage:
 *
 * 1. Difficulty-based content:
 * <PersonalizedContent
 *   beginner={<p>Basic explanation...</p>}
 *   intermediate={<p>Moderate explanation...</p>}
 *   advanced={<p>Advanced explanation...</p>}
 * />
 *
 * 2. Language-based content:
 * <PersonalizedContent
 *   english={<p>English content...</p>}
 *   urdu={<p>اردو مواد...</p>}
 * />
 *
 * 3. Both (difficulty takes precedence):
 * <PersonalizedContent
 *   beginner={<p>Basic English...</p>}
 *   intermediate={<p>Moderate English...</p>}
 *   advanced={<p>Advanced English...</p>}
 *   urdu={<p>اردو مواد...</p>}
 * />
 */
export default function PersonalizedContent({
  beginner,
  intermediate,
  advanced,
  english,
  urdu,
  children,
  dynamicTranslation = false,
  fallbackToDynamic = false,
}: PersonalizedContentProps) {
  const { difficultyLevel, language } = usePersonalization();

  // If difficulty-based content is provided
  if (beginner || intermediate || advanced) {
    const difficultyContent: Record<DifficultyLevel, React.ReactNode> = {
      beginner: beginner || intermediate || advanced || children,
      intermediate: intermediate || beginner || advanced || children,
      advanced: advanced || intermediate || beginner || children,
    };

    const content = difficultyContent[difficultyLevel];

    // If using Urdu and we want dynamic translation, wrap in DynamicTranslation
    if (language === 'urdu' && dynamicTranslation && typeof content === 'string') {
      return <DynamicTranslation>{content}</DynamicTranslation>;
    }

    return <>{content}</>;
  }

  // If language-based content is provided
  if (english || urdu) {
    const content = language === 'urdu' ? (urdu || english || children) : (english || urdu || children);

    // If using dynamic translation and no Urdu content provided but we want to translate
    if (language === 'urdu' && dynamicTranslation && !urdu && typeof english === 'string') {
      return <DynamicTranslation>{english}</DynamicTranslation>;
    }

    return <>{content}</>;
  }

  // If no specific content provided but we want dynamic translation
  if (dynamicTranslation || (fallbackToDynamic && language === 'urdu')) {
    if (typeof children === 'string') {
      return <DynamicTranslation>{children}</DynamicTranslation>;
    }
    return <>{children}</>;
  }

  // Default: return children
  return <>{children}</>;
}

/**
 * Hook to get content based on difficulty level
 */
export function useDifficultyContent<T>(content: Record<DifficultyLevel, T>): T {
  const { difficultyLevel } = usePersonalization();
  return content[difficultyLevel];
}

/**
 * Hook to get content based on language
 */
export function useLanguageContent<T>(content: { english: T; urdu: T }): T {
  const { language } = usePersonalization();
  return language === 'urdu' ? content.urdu : content.english;
}
