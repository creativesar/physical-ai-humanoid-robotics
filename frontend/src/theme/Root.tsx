import React from 'react';
import { PersonalizationProvider } from '../contexts/PersonalizationContext';
import { AuthProvider } from '../hooks/useAuth';
import LuxuryChatbotWidget from '../components/LuxuryChatbotWidget';

export default function Root({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <PersonalizationProvider>
        <>
          {children}
          <LuxuryChatbotWidget />
        </>
      </PersonalizationProvider>
    </AuthProvider>
  );
}