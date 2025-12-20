import React, { useState, useEffect } from 'react';
import { useTranslation } from '../hooks/useTranslation';

interface DynamicTranslationProps {
  children: string;
  sourceLang?: string;
  targetLang?: string;
  fallbackToOriginal?: boolean;
}

const DynamicTranslation: React.FC<DynamicTranslationProps> = ({
  children,
  sourceLang = 'en',
  targetLang = 'ur',
  fallbackToOriginal = true,
}) => {
  const [translatedText, setTranslatedText] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const { translateToUrdu, isLoading: isHookLoading, error: hookError } = useTranslation();

  useEffect(() => {
    const translateContent = async () => {
      if (!children || typeof children !== 'string') {
        setTranslatedText(children as string);
        setIsLoading(false);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        if (targetLang === 'ur') {
          // Use the Urdu-specific endpoint
          const result = await translateToUrdu(children);
          setTranslatedText(result.translated_text);
        } else {
          // For other languages, we could implement a more general translation
          setTranslatedText(children); // Fallback to original for now
        }
      } catch (err) {
        console.error('Translation error:', err);
        setError(err instanceof Error ? err.message : 'Translation failed');
        if (fallbackToOriginal) {
          setTranslatedText(children);
        }
      } finally {
        setIsLoading(false);
      }
    };

    if (children && typeof children === 'string') {
      translateContent();
    } else {
      setTranslatedText(children as string);
      setIsLoading(false);
    }
  }, [children, sourceLang, targetLang, translateToUrdu, fallbackToOriginal]);

  if (isLoading || isHookLoading) {
    return (
      <span style={{ opacity: 0.6, fontStyle: 'italic' }}>
        [Translating...]
      </span>
    );
  }

  if (error || hookError) {
    return (
      <span title={error || hookError || 'Translation error'}>
        {children}
      </span>
    );
  }

  return <>{translatedText || children}</>;
};

export default DynamicTranslation;