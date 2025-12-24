import React, { lazy, Suspense } from 'react';
import { PersonalizationProvider } from '../contexts/PersonalizationContext';
import { AuthProvider } from '../hooks/useAuth';

// OPTIMIZED: Lazy load chatbot to improve initial page load
const LuxuryChatbotWidget = lazy(() => import('../components/LuxuryChatbotWidget'));

export default function Root({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <PersonalizationProvider>
        <>
          {children}
          <Suspense fallback={null}>
            <LuxuryChatbotWidget />
          </Suspense>
        </>
      </PersonalizationProvider>
    </AuthProvider>
  );
}