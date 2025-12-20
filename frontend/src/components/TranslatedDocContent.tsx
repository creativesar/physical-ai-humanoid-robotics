import React, { useState, useEffect } from 'react';
import { usePersonalization } from '../contexts/PersonalizationContext';

interface TranslatedDocContentProps {
  children: React.ReactNode;
}

/**
 * Wrapper component that handles document content translation
 * When language is set to Urdu, it applies proper RTL layout and Urdu fonts
 * Note: Actual translation would require backend API integration
 */
export default function TranslatedDocContent({ children }: TranslatedDocContentProps) {
  const { language } = usePersonalization();
  const [isUrdu, setIsUrdu] = useState(false);

  useEffect(() => {
    setIsUrdu(language === 'urdu');
  }, [language]);

  // For now, we just apply proper styling and directionality
  // In the future, you can integrate actual translation API here
  return (
    <div
      className={isUrdu ? 'urdu-content' : ''}
      dir={isUrdu ? 'rtl' : 'ltr'}
      lang={isUrdu ? 'ur' : 'en'}
      style={{
        direction: isUrdu ? 'rtl' : 'ltr',
        textAlign: isUrdu ? 'right' : 'left',
      }}
    >
      {children}
    </div>
  );
}
