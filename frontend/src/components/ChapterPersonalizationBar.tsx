import React, { useState } from 'react';
import { usePersonalization, DifficultyLevel, Language } from '../contexts/PersonalizationContext';

export default function ChapterPersonalizationBar() {
  const { difficultyLevel, language, setDifficultyLevel, setLanguage, isPersonalized } = usePersonalization();
  const [showOptions, setShowOptions] = useState(false);

  const difficultyOptions: { value: DifficultyLevel; label: string; labelUrdu: string; description: string; descriptionUrdu: string }[] = [
    {
      value: 'beginner',
      label: 'Beginner',
      labelUrdu: 'شروعاتی',
      description: 'Detailed explanations with basic concepts',
      descriptionUrdu: 'بنیادی تصورات کے ساتھ تفصیلی وضاحت'
    },
    {
      value: 'intermediate',
      label: 'Intermediate',
      labelUrdu: 'درمیانی',
      description: 'Balanced pace with practical examples',
      descriptionUrdu: 'عملی مثالوں کے ساتھ متوازن رفتار'
    },
    {
      value: 'advanced',
      label: 'Advanced',
      labelUrdu: 'جدید',
      description: 'Advanced topics and concepts',
      descriptionUrdu: 'جدید موضوعات اور تصورات'
    },
  ];

  const languageOptions: { value: Language; label: string; flag: string }[] = [
    { value: 'english', label: 'English', flag: '🇬🇧' },
    { value: 'urdu', label: 'اردو', flag: '🇵🇰' },
  ];

  return (
    <div style={{
      position: 'sticky',
      top: '60px',
      zIndex: 100,
      background: 'rgba(20, 20, 20, 0.95)',
      backdropFilter: 'blur(20px)',
      WebkitBackdropFilter: 'blur(20px)',
      borderBottom: '1px solid rgba(176, 224, 230, 0.2)',
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
    }}>
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '0.8rem 1.5rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '1rem',
        flexWrap: 'wrap',
      }}>
        {/* Left side - Info */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.8rem',
        }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: isPersonalized
              ? 'linear-gradient(135deg, #10b981, #34d399)'
              : 'rgba(176, 224, 230, 0.3)',
            boxShadow: isPersonalized ? '0 0 10px rgba(16, 185, 129, 0.5)' : 'none',
          }} />
          <span style={{
            color: 'rgba(224, 240, 240, 0.8)',
            fontSize: '0.9rem',
            fontWeight: 500,
          }}>
            {language === 'urdu' ? 'ذاتی نوعیت' : 'Personalization'}
          </span>
        </div>

        {/* Right side - Controls */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.8rem',
          flexWrap: 'wrap',
        }}>
          {/* Difficulty Selector */}
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => setShowOptions(!showOptions)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.6rem 1rem',
                borderRadius: '8px',
                background: 'rgba(176, 224, 230, 0.1)',
                border: '1px solid rgba(176, 224, 230, 0.2)',
                color: '#b0e0e6',
                fontSize: '0.85rem',
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'all 0.3s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(176, 224, 230, 0.15)';
                e.currentTarget.style.borderColor = 'rgba(176, 224, 230, 0.3)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(176, 224, 230, 0.1)';
                e.currentTarget.style.borderColor = 'rgba(176, 224, 230, 0.2)';
              }}
            >
              <span>📚</span>
              <span style={{ textTransform: 'capitalize' }}>
                {language === 'urdu'
                  ? difficultyOptions.find(o => o.value === difficultyLevel)?.labelUrdu
                  : difficultyLevel}
              </span>
              <span style={{ fontSize: '0.7rem', opacity: 0.7 }}>▼</span>
            </button>

            {showOptions && (
              <div style={{
                position: 'absolute',
                top: 'calc(100% + 8px)',
                right: 0,
                minWidth: '280px',
                padding: '0.8rem',
                borderRadius: '12px',
                background: 'rgba(20, 20, 20, 0.98)',
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
                border: '1px solid rgba(176, 224, 230, 0.2)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
                zIndex: 1000,
              }}>
                <div style={{
                  fontSize: '0.75rem',
                  color: 'rgba(224, 240, 240, 0.6)',
                  marginBottom: '0.8rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                }}>
                  {language === 'urdu' ? 'مشکل کی سطح منتخب کریں' : 'Select Difficulty Level'}
                </div>
                {difficultyOptions.map((option) => (
                  <button
                    key={option.value}
                    onClick={() => {
                      setDifficultyLevel(option.value);
                      setShowOptions(false);
                    }}
                    style={{
                      width: '100%',
                      padding: '0.8rem',
                      marginBottom: '0.5rem',
                      borderRadius: '8px',
                      background: difficultyLevel === option.value
                        ? 'rgba(176, 224, 230, 0.15)'
                        : 'transparent',
                      border: '1px solid ' + (difficultyLevel === option.value
                        ? 'rgba(176, 224, 230, 0.3)'
                        : 'rgba(176, 224, 230, 0.1)'),
                      color: '#e0f0f0',
                      fontSize: '0.9rem',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      textAlign: 'left',
                    }}
                    onMouseEnter={(e) => {
                      if (difficultyLevel !== option.value) {
                        e.currentTarget.style.background = 'rgba(176, 224, 230, 0.08)';
                        e.currentTarget.style.borderColor = 'rgba(176, 224, 230, 0.2)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (difficultyLevel !== option.value) {
                        e.currentTarget.style.background = 'transparent';
                        e.currentTarget.style.borderColor = 'rgba(176, 224, 230, 0.1)';
                      }
                    }}
                  >
                    <div style={{ fontWeight: 600, marginBottom: '0.3rem' }}>
                      {language === 'urdu' ? option.labelUrdu : option.label}
                    </div>
                    <div style={{
                      fontSize: '0.8rem',
                      color: 'rgba(224, 240, 240, 0.6)',
                      direction: language === 'urdu' ? 'rtl' : 'ltr',
                    }}>
                      {language === 'urdu' ? option.descriptionUrdu : option.description}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Language Selector */}
          <div style={{
            display: 'flex',
            gap: '0.4rem',
            padding: '0.3rem',
            borderRadius: '8px',
            background: 'rgba(176, 224, 230, 0.05)',
            border: '1px solid rgba(176, 224, 230, 0.15)',
          }}>
            {languageOptions.map((option) => (
              <button
                key={option.value}
                onClick={() => setLanguage(option.value)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  padding: '0.5rem 0.8rem',
                  borderRadius: '6px',
                  background: language === option.value
                    ? 'rgba(176, 224, 230, 0.2)'
                    : 'transparent',
                  border: 'none',
                  color: language === option.value ? '#b0e0e6' : 'rgba(224, 240, 240, 0.6)',
                  fontSize: '0.85rem',
                  fontWeight: language === option.value ? 600 : 500,
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
                onMouseEnter={(e) => {
                  if (language !== option.value) {
                    e.currentTarget.style.background = 'rgba(176, 224, 230, 0.08)';
                    e.currentTarget.style.color = '#e0f0f0';
                  }
                }}
                onMouseLeave={(e) => {
                  if (language !== option.value) {
                    e.currentTarget.style.background = 'transparent';
                    e.currentTarget.style.color = 'rgba(224, 240, 240, 0.6)';
                  }
                }}
              >
                <span>{option.flag}</span>
                <span>{option.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Click outside to close dropdown */}
      {showOptions && (
        <div
          onClick={() => setShowOptions(false)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            zIndex: 999,
          }}
        />
      )}
    </div>
  );
}
