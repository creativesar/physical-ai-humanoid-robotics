import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { useColorMode } from '@docusaurus/theme-common';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import RAGChatbotSDK from '../../sdk/rag-chatbot-sdk';
import styles from './InlineHighlightChat.module.css';

interface Position {
  top: number;
  left: number;
}

const InlineHighlightChat: React.FC = () => {
  const { colorMode } = useColorMode();
  const { siteConfig } = useDocusaurusContext();
  const [selectedText, setSelectedText] = useState('');
  const [position, setPosition] = useState<Position | null>(null);
  const [showInput, setShowInput] = useState(false);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const ragSDK = useRef<RAGChatbotSDK | null>(null);

  // Initialize SDK
  useEffect(() => {
    const backendUrl = (siteConfig.customFields?.backendUrl as string) || 'http://localhost:8000';
    ragSDK.current = new RAGChatbotSDK(backendUrl);
  }, [siteConfig.customFields?.backendUrl]);

  // Handle text selection
  const handleSelection = useCallback(() => {
    const selection = window.getSelection();
    const text = selection?.toString().trim();

    if (text && text.length > 3) {
      const range = selection?.getRangeAt(0);
      if (range) {
        const rect = range.getBoundingClientRect();
        setSelectedText(text);
        setPosition({
          top: rect.bottom + window.scrollY + 10,
          left: rect.left + window.scrollX + (rect.width / 2),
        });
        setShowInput(false);
        setAnswer('');
        setError('');
        setQuestion('');
      }
    }
  }, []);

  // Listen for mouseup events to detect selection
  useEffect(() => {
    const handleMouseUp = (e: MouseEvent) => {
      // Ignore clicks inside our popup
      if (popupRef.current?.contains(e.target as Node)) {
        return;
      }

      // Small delay to ensure selection is complete
      setTimeout(handleSelection, 10);
    };

    const handleClickOutside = (e: MouseEvent) => {
      if (popupRef.current && !popupRef.current.contains(e.target as Node)) {
        // Only close if clicking outside and not selecting new text
        const selection = window.getSelection();
        if (!selection?.toString().trim()) {
          setPosition(null);
          setSelectedText('');
          setShowInput(false);
          setAnswer('');
        }
      }
    };

    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('mousedown', handleClickOutside);

    return () => {
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [handleSelection]);

  // Focus input when showing
  useEffect(() => {
    if (showInput && inputRef.current) {
      inputRef.current.focus();
    }
  }, [showInput]);

  // Handle asking question
  const handleAskQuestion = async () => {
    if (!question.trim() || isLoading || !ragSDK.current) return;

    setIsLoading(true);
    setError('');

    try {
      const response = await ragSDK.current.chat({
        question: question,
        session_id: null,
        selected_text: selectedText,
      });

      setAnswer(response.answer);
    } catch (err) {
      console.error('Error getting answer:', err);
      setError('Could not get answer. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAskQuestion();
    }
    if (e.key === 'Escape') {
      setPosition(null);
      setSelectedText('');
    }
  };

  // Close popup
  const handleClose = () => {
    setPosition(null);
    setSelectedText('');
    setShowInput(false);
    setAnswer('');
    setQuestion('');
    setError('');
  };

  if (!position || !selectedText) return null;

  const popupContent = (
    <div
      ref={popupRef}
      className={styles.popup}
      style={{
        top: position.top,
        left: position.left,
        transform: 'translateX(-50%)',
      }}
      data-theme={colorMode}
    >
      {/* Arrow pointing up */}
      <div className={styles.arrow} data-theme={colorMode} />

      {/* Close button */}
      <button className={styles.closeBtn} onClick={handleClose}>
        ×
      </button>

      {!showInput && !answer ? (
        /* Initial state - Ask button */
        <button
          className={styles.askBtn}
          onClick={() => setShowInput(true)}
          data-theme={colorMode}
        >
          <span className={styles.askIcon}>💬</span>
          Ask about this
        </button>
      ) : (
        /* Question input and answer display */
        <div className={styles.chatContainer}>
          {/* Selected text preview */}
          <div className={styles.selectedPreview} data-theme={colorMode}>
            <span className={styles.quoteIcon}>"</span>
            {selectedText.length > 100
              ? selectedText.substring(0, 100) + '...'
              : selectedText}
          </div>

          {/* Question input */}
          {!answer && (
            <div className={styles.inputContainer}>
              <input
                ref={inputRef}
                type="text"
                className={styles.input}
                placeholder="Ask your question..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={handleKeyPress}
                disabled={isLoading}
                data-theme={colorMode}
              />
              <button
                className={styles.sendBtn}
                onClick={handleAskQuestion}
                disabled={!question.trim() || isLoading}
                data-theme={colorMode}
              >
                {isLoading ? (
                  <span className={styles.spinner} />
                ) : (
                  '→'
                )}
              </button>
            </div>
          )}

          {/* Loading state */}
          {isLoading && (
            <div className={styles.loading} data-theme={colorMode}>
              <div className={styles.loadingDots}>
                <span></span>
                <span></span>
                <span></span>
              </div>
              Thinking...
            </div>
          )}

          {/* Error message */}
          {error && (
            <div className={styles.error}>
              {error}
            </div>
          )}

          {/* Answer display */}
          {answer && (
            <div className={styles.answer} data-theme={colorMode}>
              <div className={styles.answerHeader}>
                <span className={styles.botIcon}>🤖</span>
                Answer
              </div>
              <div className={styles.answerText}>
                {answer}
              </div>
              <button
                className={styles.newQuestionBtn}
                onClick={() => {
                  setAnswer('');
                  setQuestion('');
                  setShowInput(true);
                }}
                data-theme={colorMode}
              >
                Ask more
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );

  return createPortal(popupContent, document.body);
};

export default InlineHighlightChat;
