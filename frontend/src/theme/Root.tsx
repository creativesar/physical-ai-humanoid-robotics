import React from 'react';
import { PersonalizationProvider } from '../contexts/PersonalizationContext';
import { AuthProvider } from '../hooks/useAuth';
import LuxuryChatbotWidget from '../components/LuxuryChatbotWidget';
import ContentTranslator from '../components/ContentTranslator';

export default function Root({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <PersonalizationProvider>
        <>
          {children}
          <LuxuryChatbotWidget />
          <ContentTranslator />
        </>
      </PersonalizationProvider>
    </AuthProvider>
  );
}