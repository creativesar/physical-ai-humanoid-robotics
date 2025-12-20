import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '@site/src/hooks/useAuth';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './LuxuryChatbotWidget.module.css';

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  sources?: Array<{
    title: string;
    url: string;
    score: number;
  }>;
}

interface ChatResponse {
  answer: string;
  sources: Array<{
    title: string;
    url: string;
    score: number;
    chapter_id: string;
  }>;
  query: string;
  retrieved_chunks: number;
}

const SUGGESTED_QUESTIONS = [
  "What is ROS 2 and how does it work?",
  "Explain Gazebo simulation environment",
  "How does NVIDIA Isaac platform work?",
  "What are Vision-Language-Action systems?",
  "How to set up a humanoid robot?",
  "Explain digital twin concepts"
];

const LuxuryChatbotWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('chatbot_conversations');
      if (saved) {
        try {
          return JSON.parse(saved).map((conv: any) => ({
            ...conv,
            createdAt: new Date(conv.createdAt),
            updatedAt: new Date(conv.updatedAt),
            messages: conv.messages.map((msg: any) => ({
              ...msg,
              timestamp: new Date(msg.timestamp)
            }))
          }));
        } catch (e) {
          console.error('Error loading conversations from localStorage:', e);
        }
      }
    }
    // Default conversation
    return [{
      id: '1',
      title: 'Welcome Conversation',
      messages: [
        {
          id: '1',
          text: 'Welcome to Physical AI Humanoid Robotics Assistant! 👋\n\nI\'m here to help you master Physical AI and Humanoid Robotics. Ask me anything about ROS 2, Gazebo, NVIDIA Isaac™, or Vision-Language-Action systems.',
          sender: 'bot',
          timestamp: new Date(),
        },
      ],
      createdAt: new Date(),
      updatedAt: new Date(),
    }];
  });
  const [currentConversationId, setCurrentConversationId] = useState<string>('1');
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState<string | null>(null);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const { user, isAuthenticated } = useAuth();
  const { siteConfig } = useDocusaurusContext();

  const API_URL = (siteConfig.customFields?.backendUrl as string) || 'http://localhost:8000';

  // Get current conversation
  const currentConversation = conversations.find(conv => conv.id === currentConversationId) || conversations[0];

  // Save conversations to localStorage whenever they change
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('chatbot_conversations', JSON.stringify(conversations));
    }
  }, [conversations]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentConversation?.messages]);

  useEffect(() => {
    if (inputRef.current) {
      adjustTextareaHeight(inputRef.current);
    }
  }, [inputValue]);

  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      if (selection && selection.toString().trim().length > 0) {
        setSelectedText(selection.toString().trim());
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  const handleSendMessage = async (messageText?: string) => {
    const text = messageText || inputValue;
    if (!text.trim() || isLoading) return;

    // Translate user message to Urdu if Urdu mode is enabled
    let processedText = text;
    if (isUrduMode) {
      processedText = await translateToUrdu(text);
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      text: processedText,
      sender: 'user',
      timestamp: new Date(),
    };

    // Update the conversation with the new message
    setConversations(prev => prev.map(conv => {
      if (conv.id === currentConversationId) {
        const updatedMessages = [...conv.messages, userMessage];
        return {
          ...conv,
          messages: updatedMessages,
          updatedAt: new Date(),
          // Update title if this is the first user message
          title: conv.title === 'Welcome Conversation' && updatedMessages.length === 2
            ? processedText.substring(0, 30) + (processedText.length > 30 ? '...' : '')
            : conv.title
        };
      }
      return conv;
    }));

    setInputValue('');
    setIsLoading(true);
    setShowSuggestions(false);

    try {
      const response = await fetch(`${API_URL}/api/chat/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: text, // Send original query to backend
          user_id: user?.id || null,
          selected_text: selectedText,
        }),
      });

      if (!response.ok) {
        if (response.status === 503) {
          throw new Error('Service temporarily unavailable. This may be due to API rate limits. Please try again later.');
        } else {
          throw new Error('Failed to get response from backend');
        }
      }

      const data: ChatResponse = await response.json();

      let botResponse = data.answer;
      // Translate bot response to Urdu if Urdu mode is enabled
      if (isUrduMode) {
        botResponse = await translateToUrdu(data.answer);
      }

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: botResponse,
        sender: 'bot',
        timestamp: new Date(),
        sources: data.sources,
      };

      // Update the conversation with the bot's response
      setConversations(prev => prev.map(conv => {
        if (conv.id === currentConversationId) {
          return {
            ...conv,
            messages: [...conv.messages, botMessage],
            updatedAt: new Date()
          };
        }
        return conv;
      }));

      setSelectedText(null);
    } catch (error) {
      console.error('Error sending message:', error);
      let errorMessageText = '⚠️ I\'m having trouble connecting right now. Please ensure the backend server is running.';
      if (isUrduMode) {
        errorMessageText = await translateToUrdu(errorMessageText);
      }

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: errorMessageText,
        sender: 'bot',
        timestamp: new Date(),
      };

      // Update the conversation with the error message
      setConversations(prev => prev.map(conv => {
        if (conv.id === currentConversationId) {
          return {
            ...conv,
            messages: [...conv.messages, errorMessage],
            updatedAt: new Date()
          };
        }
        return conv;
      }));
    } finally {
      setIsLoading(false);
    }
  };

  const adjustTextareaHeight = (textarea: HTMLTextAreaElement) => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 80) + 'px';
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const textarea = e.target;
    setInputValue(textarea.value);

    // Adjust height after state update
    setTimeout(() => {
      adjustTextareaHeight(textarea);
    }, 0);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInputValue(suggestion);
    inputRef.current?.focus();
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };


  const clearChat = async () => {
    let welcomeMessage = 'Physical AI Humanoid Robotics chat cleared! How can I assist you today?';
    if (isUrduMode) {
      welcomeMessage = await translateToUrdu(welcomeMessage);
    }

    setConversations(prev => prev.map(conv => {
      if (conv.id === currentConversationId) {
        return {
          ...conv,
          messages: [
            {
              id: '1',
              text: welcomeMessage,
              sender: 'bot',
              timestamp: new Date(),
            },
          ],
          title: 'New Conversation',
          updatedAt: new Date(),
        };
      }
      return conv;
    }));
    setShowSuggestions(true);
  };

  const getRandomSuggestions = () => {
    return SUGGESTED_QUESTIONS.sort(() => 0.5 - Math.random()).slice(0, 3);
  };



  const [isUrduMode, setIsUrduMode] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<any>(null);

  const quickActions = [
    { id: 'examples', label: 'Examples', icon: '💡', prompt: 'Can you provide practical examples related to our discussion?' },
  ];

  const handleQuickAction = (prompt: string) => {
    setInputValue(prompt);
    setTimeout(() => {
      inputRef.current?.focus();
    }, 100);
  };

  const toggleUrduMode = () => {
    setIsUrduMode(!isUrduMode);
  };

  const translateToUrdu = async (text: string): Promise<string> => {
    try {
      // Call the backend translation API
      const response = await fetch(`${API_URL}/api/translate/urdu`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: text,
          source_language: 'en',
          target_language: 'ur'
        }),
      });

      if (!response.ok) {
        throw new Error(`Translation API error: ${response.status}`);
      }

      const data = await response.json();
      return data.translated_content;
    } catch (error) {
      console.error('Translation error:', error);
      return text; // Return original text if translation fails
    }
  };

  const initializeSpeechRecognition = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert('Speech recognition is not supported in your browser. Please try Chrome or Edge.');
      return null;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setInputValue(transcript);
      setIsListening(false);
    };

    recognition.onerror = (event: any) => {
      console.error('Speech recognition error', event.error);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    return recognition;
  };

  const toggleVoiceInput = () => {
    if (isListening) {
      // Stop listening
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      setIsListening(false);
    } else {
      // Start listening
      const recognition = initializeSpeechRecognition();
      if (recognition) {
        recognitionRef.current = recognition;
        recognition.start();
        setIsListening(true);
      }
    }
  };


  return (
    <div className={styles.chatbotContainer}>
      {!isOpen && (
        <button
          className={styles.chatButton}
          onClick={toggleChat}
          aria-label="Open AI Assistant"
        >
          <div className={styles.chatButtonInner}>
            <svg className={styles.chatIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              <path d="M8 10h8M8 14h4" strokeLinecap="round" />
            </svg>
            <div className={styles.pulseRing}></div>
            <div className={styles.pulseRing2}></div>
          </div>
          {selectedText && <span className={styles.notificationBadge}>1</span>}
          <div className={styles.tooltip}>Physical AI Humanoid Robotics</div>
        </button>
      )}

      {isOpen && (
        <div className={styles.chatWindow}>
          <div className={styles.chatHeader}>
            <div className={styles.headerLeft}>
              <div className={styles.botAvatarContainer}>
                <div className={styles.botAvatar}>
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2M7.5 13A2.5 2.5 0 0 0 5 15.5 2.5 2.5 0 0 0 7.5 18a2.5 2.5 0 0 0 2.5-2.5A2.5 2.5 0 0 0 7.5 13m9 0a2.5 2.5 0 0 0-2.5 2.5 2.5 2.5 0 0 0 2.5 2.5 2.5 2.5 0 0 0 2.5-2.5 2.5 2.5 0 0 0-2.5-2.5z"/>
                  </svg>
                </div>
                <div className={styles.statusIndicator}></div>
              </div>
              <div className={styles.headerInfo}>
                <h3 className={styles.botName}>Physical AI Humanoid Robotics</h3>
                <p className={styles.botStatus}>
                  {isAuthenticated ? (
                    <>
                      <span className={styles.statusDot}></span>
                      {user?.name || 'User'}
                    </>
                  ) : (
                    <>
                      <span className={styles.statusDot}></span>
                      Online & Ready
                    </>
                  )}
                </p>
              </div>
            </div>
            <div className={styles.headerActions}>
              <button
                className={styles.iconButton}
                onClick={clearChat}
                aria-label="Clear chat"
                title="Clear chat"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
              </button>
              <button
                className={styles.iconButton}
                onClick={toggleUrduMode}
                aria-label={isUrduMode ? "English Mode" : "Urdu Mode"}
                title={isUrduMode ? "Switch to English" : "Switch to Urdu"}
                style={isUrduMode ? { backgroundColor: 'rgba(0, 255, 255, 0.3)', border: '1px solid #00ffff' } : {}}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M10 13h4M12 5V2l6 3v12a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h2z" />
                </svg>
              </button>
              <button
                className={styles.iconButton}
                onClick={toggleChat}
                aria-label="Close chat"
                title="Close"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            </div>
          </div>

          <>
            {selectedText && (
              <div className={styles.selectedTextBanner}>
                <div className={styles.selectedTextIcon}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
                  </svg>
                </div>
                <div className={styles.selectedTextContent}>
                  <span className={styles.selectedTextLabel}>🎯 Selected Text</span>
                  <span className={styles.selectedTextPreview}>
                    {selectedText.substring(0, 80)}
                    {selectedText.length > 80 ? '...' : ''}
                  </span>
                </div>
                <button
                  className={styles.clearSelectionButton}
                  onClick={() => setSelectedText(null)}
                  aria-label="Clear selection"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                </button>
              </div>
            )}

            <div className={styles.messagesContainer}>
              {currentConversation.messages.map((message, index) => (
                <div
                  key={message.id}
                  className={`${styles.message} ${
                    message.sender === 'user' ? styles.userMessage : styles.botMessage
                  }`}
                >
                  {message.sender === 'bot' && (
                    <div className={styles.messageAvatar}>
                      <svg viewBox="0 0 24 24" fill="currentColor">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M8 14s1.5 2 4 2 4-2 4-2" stroke="white" strokeWidth="1.5" fill="none" strokeLinecap="round"/>
                        <circle cx="9" cy="10" r="1.5" fill="white"/>
                        <circle cx="15" cy="10" r="1.5" fill="white"/>
                      </svg>
                    </div>
                  )}
                  <div className={styles.messageContent}>
                    <p className={styles.messageText}>{message.text}</p>
                    {message.sources && message.sources.length > 0 && (
                      <div className={styles.sources}>
                        <div className={styles.sourcesHeader}>
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
                          </svg>
                          <span>Sources</span>
                        </div>
                        <ul className={styles.sourcesList}>
                          {message.sources.map((source, idx) => (
                            <li key={idx}>
                              <a
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className={styles.sourceLink}
                              >
                                <span className={styles.sourceDot}></span>
                                {source.title}
                              </a>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    <span className={styles.messageTime}>
                      {message.timestamp.toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </span>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className={`${styles.message} ${styles.botMessage}`}>
                  <div className={styles.messageAvatar}>
                    <svg viewBox="0 0 24 24" fill="currentColor">
                      <circle cx="12" cy="12" r="10"/>
                    </svg>
                  </div>
                  <div className={styles.messageContent}>
                    <div className={styles.typingIndicator}>
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
              {showSuggestions && currentConversation.messages.length === 1 && (
                <div className={styles.suggestionsContainer}>
                  <p className={styles.suggestionsTitle}>Suggestions</p>
                  <div className={styles.suggestionsList}>
                    {getRandomSuggestions().map((suggestion, idx) => (
                      <button
                        key={idx}
                        className={styles.suggestionChip}
                        onClick={() => handleSuggestionClick(suggestion)}
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className={styles.inputContainer}>
              <div className={styles.inputWrapper}>
                <textarea
                  ref={inputRef}
                  className={styles.messageInput}
                  value={inputValue}
                  onChange={handleInput}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask me anything about Physical AI & Humanoid Robotics..."
                  disabled={isLoading}
                />
                <div className={styles.inputActions}>
                  <button
                    className={`${styles.voiceButton} ${isListening ? styles.listening : ''}`}
                    onClick={toggleVoiceInput}
                    aria-label={isListening ? "Stop voice input" : "Start voice input"}
                    title={isListening ? "Stop voice input" : "Start voice input"}
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill={isListening ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2">
                      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                      <path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8"/>
                    </svg>
                  </button>
                  <button
                    className={styles.sendButton}
                    onClick={() => handleSendMessage()}
                    disabled={!inputValue.trim() || isLoading}
                    aria-label="Send message"
                  >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                    </svg>
                  </button>
                </div>
              </div>
              <div className={styles.inputFooter}>
                <span className={styles.footerText}>
                  Physical AI Humanoid Robotics
                </span>
              </div>
            </div>
          </>
        </div>
      )}
    </div>
  );
};

export default LuxuryChatbotWidget;