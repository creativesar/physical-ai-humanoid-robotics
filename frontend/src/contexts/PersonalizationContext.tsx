import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export type DifficultyLevel = 'beginner' | 'intermediate' | 'advanced';
export type Language = 'english' | 'urdu';

interface UserProfile {
  name?: string;
  email?: string;
  softwareBackground?: string;
  hardwareBackground?: string;
  difficultyLevel?: DifficultyLevel;
  preferredLanguage?: Language;
}

interface PersonalizationContextType {
  userProfile: UserProfile | null;
  difficultyLevel: DifficultyLevel;
  language: Language;
  setDifficultyLevel: (level: DifficultyLevel) => void;
  setLanguage: (lang: Language) => void;
  updateUserProfile: (profile: Partial<UserProfile>) => void;
  isPersonalized: boolean;
}

const PersonalizationContext = createContext<PersonalizationContextType | undefined>(undefined);

const STORAGE_KEY = 'physical-ai-personalization';

export function PersonalizationProvider({ children }: { children: ReactNode }) {
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [difficultyLevel, setDifficultyLevelState] = useState<DifficultyLevel>('intermediate');
  const [language, setLanguageState] = useState<Language>('english');

  // Load from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        try {
          const data = JSON.parse(stored);
          setUserProfile(data.userProfile || null);
          setDifficultyLevelState(data.difficultyLevel || 'intermediate');
          setLanguageState(data.language || 'english');
        } catch (error) {
          console.error('Failed to load personalization data:', error);
        }
      }
    }
  }, []);

  // Auto-detect difficulty level based on user background
  useEffect(() => {
    if (userProfile && !userProfile.difficultyLevel) {
      const hasSoftwareExp = userProfile.softwareBackground && userProfile.softwareBackground.length > 10;
      const hasHardwareExp = userProfile.hardwareBackground && userProfile.hardwareBackground.length > 10;

      let detectedLevel: DifficultyLevel = 'beginner';
      if (hasSoftwareExp && hasHardwareExp) {
        detectedLevel = 'advanced';
      } else if (hasSoftwareExp || hasHardwareExp) {
        detectedLevel = 'intermediate';
      }

      setDifficultyLevelState(detectedLevel);
    }
  }, [userProfile]);

  // Save to localStorage whenever data changes
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const data = {
        userProfile,
        difficultyLevel,
        language,
        timestamp: Date.now(),
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    }
  }, [userProfile, difficultyLevel, language]);

  const setDifficultyLevel = (level: DifficultyLevel) => {
    setDifficultyLevelState(level);
    if (userProfile) {
      setUserProfile({ ...userProfile, difficultyLevel: level });
    }
  };

  const setLanguage = (lang: Language) => {
    setLanguageState(lang);
    if (userProfile) {
      setUserProfile({ ...userProfile, preferredLanguage: lang });
    }
  };

  const updateUserProfile = (profile: Partial<UserProfile>) => {
    setUserProfile(prev => ({
      ...prev,
      ...profile,
    }));
  };

  const isPersonalized = !!(userProfile || difficultyLevel !== 'intermediate' || language !== 'english');

  return (
    <PersonalizationContext.Provider
      value={{
        userProfile,
        difficultyLevel,
        language,
        setDifficultyLevel,
        setLanguage,
        updateUserProfile,
        isPersonalized,
      }}
    >
      {children}
    </PersonalizationContext.Provider>
  );
}

export function usePersonalization() {
  const context = useContext(PersonalizationContext);
  if (context === undefined) {
    throw new Error('usePersonalization must be used within a PersonalizationProvider');
  }
  return context;
}
