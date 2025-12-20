import React, { useEffect, useState, useRef } from 'react';
import { usePersonalization } from '../contexts/PersonalizationContext';

// Translation cache to avoid re-translating same content
const translationCache: Map<string, string> = new Map();

/**
 * ContentTranslator component that uses Mistral AI backend
 * to translate page content when language changes
 */
export default function ContentTranslator() {
  const { language } = usePersonalization();
  const [isTranslating, setIsTranslating] = useState(false);
  const [translationError, setTranslationError] = useState<string | null>(null);
  const previousLanguage = useRef<string>('english');
  const originalContent = useRef<Map<Element, string>>(new Map());

  // Backend URL - can be configured via environment variable
  const BACKEND_URL = typeof window !== 'undefined' && (window as any).__BACKEND_URL__
    ? (window as any).__BACKEND_URL__
    : 'http://localhost:8000';

  useEffect(() => {
    const translateContent = async () => {
      // Only translate when switching to Urdu
      if (language === 'urdu' && previousLanguage.current === 'english') {
        setIsTranslating(true);
        setTranslationError(null);

        try {
          // Get all text nodes that need translation
          const article = document.querySelector('article.markdown');
          if (!article) {
            console.warn('Article element not found');
            setIsTranslating(false);
            return;
          }

          // Store original content before translation
          if (originalContent.current.size === 0) {
            const textElements = article.querySelectorAll('h1, h2, h3, h4, h5, h6, p, li, td, th, blockquote');
            textElements.forEach((element) => {
              originalContent.current.set(element, element.textContent || '');
            });
          }

          // Translate all text elements
          const textElements = article.querySelectorAll('h1, h2, h3, h4, h5, h6, p, li, td, th, blockquote');

          for (const element of Array.from(textElements)) {
            const originalText = element.textContent?.trim();
            if (!originalText || originalText.length === 0) continue;

            // Skip code blocks
            if (element.closest('pre') || element.closest('code')) continue;

            // Check cache first
            const cacheKey = `en_ur_${originalText}`;
            let translatedText = translationCache.get(cacheKey);

            if (!translatedText) {
              // Call backend API for translation
              try {
                const response = await fetch(`${BACKEND_URL}/api/translate/urdu`, {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json',
                  },
                  body: JSON.stringify({
                    content: originalText,
                    source_language: 'en',
                    target_language: 'ur',
                  }),
                });

                if (!response.ok) {
                  throw new Error(`Translation API error: ${response.status}`);
                }

                const data = await response.json();
                translatedText = data.translated_content;

                // Cache the translation
                translationCache.set(cacheKey, translatedText);
              } catch (error) {
                console.error('Translation error:', error);
                // Continue with next element if one fails
                continue;
              }
            }

            // Update the element with translated text
            if (translatedText && element.textContent) {
              element.textContent = translatedText;
              // Apply RTL styling for Urdu content
              if (language === 'urdu') {
                element.dir = 'rtl';
                element.style.textAlign = 'right';
              } else {
                element.dir = 'ltr';
                element.style.textAlign = 'left';
              }
            }
          }
        } catch (error) {
          console.error('Translation error:', error);
          setTranslationError('Translation service unavailable. Please start the backend server.');
        } finally {
          setIsTranslating(false);
        }
      } else if (language === 'english' && previousLanguage.current === 'urdu') {
        // Restore original content when switching back to English
        originalContent.current.forEach((originalText, element) => {
          if (element.textContent) {
            element.textContent = originalText;
          }
        });
      }

      previousLanguage.current = language;
    };

    // Small delay to ensure DOM is ready
    const timeoutId = setTimeout(translateContent, 300);
    return () => clearTimeout(timeoutId);
  }, [language, BACKEND_URL]);

  return (
    <>
      {/* Translation loading indicator */}
      {isTranslating && (
        <div
          style={{
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 9999,
            padding: '2rem 3rem',
            background: 'rgba(0, 0, 0, 0.95)',
            borderRadius: '16px',
            border: '1px solid rgba(0, 102, 255, 0.3)',
            color: '#3385FF',
            fontSize: '1.1rem',
            fontWeight: 600,
            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.8)',
            backdropFilter: 'blur(10px)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div
              style={{
                width: '24px',
                height: '24px',
                border: '3px solid rgba(0, 102, 255, 0.2)',
                borderTop: '3px solid #3385FF',
                borderRadius: '50%',
                animation: 'spin 0.8s linear infinite',
              }}
            />
            <div>
              <div>Translating to Urdu...</div>
              <div style={{ fontSize: '0.85rem', color: '#9CA3AF', marginTop: '0.5rem' }}>
                Using Mistral AI
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Translation error message */}
      {translationError && (
        <div
          style={{
            position: 'fixed',
            bottom: '2rem',
            right: '2rem',
            zIndex: 9999,
            padding: '1rem 1.5rem',
            background: 'rgba(239, 68, 68, 0.95)',
            borderRadius: '12px',
            border: '1px solid rgba(239, 68, 68, 0.5)',
            color: 'white',
            fontSize: '0.9rem',
            fontWeight: 500,
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
            maxWidth: '400px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
            <span style={{ fontSize: '1.2rem' }}>⚠️</span>
            <div>
              <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>Translation Error</div>
              <div style={{ fontSize: '0.85rem', opacity: 0.9 }}>{translationError}</div>
              <button
                onClick={() => setTranslationError(null)}
                style={{
                  marginTop: '0.5rem',
                  padding: '0.25rem 0.75rem',
                  background: 'rgba(255, 255, 255, 0.2)',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  borderRadius: '6px',
                  color: 'white',
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                }}
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </>
  );
}
