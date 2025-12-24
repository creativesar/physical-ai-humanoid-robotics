import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '@site/src/hooks/useAuth';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
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
  reaction?: 'thumbsUp' | 'thumbsDown' | null;
  type?: 'text' | 'image' | 'file' | 'code';
  attachments?: Array<{
    name: string;
    url: string;
    type: string;
    size: number;
  }>;
  threadId?: string;
  pinned?: boolean;
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

// Single theme configuration
const THEME = {
  primary: '#00ffff',
  secondary: '#00cccc',
  background: 'linear-gradient(145deg, #0a0a0a, #1a1a1a)',
  accent: '#0066ff',
  text: '#00ffff',
  border: 'rgba(0, 255, 255, 0.2)',
  glow: 'rgba(0, 255, 255, 0.5)',
};

// Utility functions
const getThemeColors = () => THEME;
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// Code Block Component with Copy Button
const CodeBlock = ({ language, value }: { language: string; value: string }) => {
  const [copied, setCopied] = useState(false);
  const themeColors = getThemeColors();

  const handleCopy = () => {
    navigator.clipboard.writeText(value);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{ position: 'relative', marginBottom: '1em' }}>
      <button
        onClick={handleCopy}
        className={styles.copyButton}
        title="Copy code"
        style={{
          background: `rgba(${themeColors.primary}, 0.1)`,
          borderColor: themeColors.border,
          color: themeColors.primary,
        }}
      >
        {copied ? (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        ) : (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
          </svg>
        )}
      </button>
      <SyntaxHighlighter
        language={language}
        style={vscDarkPlus}
        customStyle={{
          borderRadius: '8px',
          padding: '1em',
          margin: 0,
          fontSize: '13px',
          background: theme === 'minimal' ? '#f8fafc' : undefined,
        }}
      >
        {value}
      </SyntaxHighlighter>
    </div>
  );
};

// File Attachment Component
const FileAttachment = ({ attachment }: { attachment: any }) => {
  const themeColors = getThemeColors();

  return (
    <div className={styles.fileAttachment} style={{
      background: `rgba(${themeColors.primary}, 0.1)`,
      border: `1px solid ${themeColors.border}`,
      color: themeColors.text,
    }}>
      <div className={styles.fileIcon}>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14,2 14,8 20,8" />
        </svg>
      </div>
      <div className={styles.fileInfo}>
        <div className={styles.fileName}>{attachment.name}</div>
        <div className={styles.fileSize}>{formatFileSize(attachment.size)}</div>
      </div>
      <button className={styles.downloadButton} style={{ color: themeColors.primary }}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
      </button>
    </div>
  );
};

// Typing Indicator Component
const TypingIndicator = () => {
  const themeColors = getThemeColors();

  return (
    <div className={styles.typingIndicator} style={{
      background: `linear-gradient(135deg, rgba(64, 64, 64, 0.8), rgba(32, 32, 32, 0.9))`,
      border: `1px solid ${themeColors.border}`,
      boxShadow: `0 0 20px ${themeColors.glow}`,
    }}>
      <div className={styles.typingDots}>
        <span style={{ background: `linear-gradient(135deg, ${themeColors.primary}, ${themeColors.secondary})` }}></span>
        <span style={{ background: `linear-gradient(135deg, ${themeColors.primary}, ${themeColors.secondary})` }}></span>
        <span style={{ background: `linear-gradient(135deg, ${themeColors.primary}, ${themeColors.secondary})` }}></span>
      </div>
      <span style={{ color: themeColors.text }}>AI is thinking...</span>
    </div>
  );
};

const LuxuryChatbotWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [streamingText, setStreamingText] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [showParticles, setShowParticles] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(false);
  const currentTheme = 'default'; // Fixed theme
  const [showFileUpload, setShowFileUpload] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [messageTemplates, setMessageTemplates] = useState([
    { id: '1', title: 'Ask about ROS', content: 'Can you explain ROS 2 and its architecture?' },
    { id: '2', title: 'Gazebo help', content: 'How do I set up a Gazebo simulation environment?' },
    { id: '3', title: 'NVIDIA Isaac', content: 'What are the key features of NVIDIA Isaac platform?' },
  ]);
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
              timestamp: new Date(msg.timestamp),
              type: msg.type || 'text',
              pinned: msg.pinned || false
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
          text: 'AI Assistant! 👋\n\nWelcome to the Physical AI & Humanoid Robotics knowledge base.',
          sender: 'bot',
          timestamp: new Date(),
          type: 'text',
          pinned: false,
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
  const [pinnedMessages, setPinnedMessages] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { user, isAuthenticated } = useAuth();
  const { siteConfig } = useDocusaurusContext();

  const API_URL = (siteConfig.customFields?.backendUrl as string) || 'https://creativesar-face.hf.space';

  // Get current conversation and theme
  const currentConversation = conversations.find(conv => conv.id === currentConversationId) || conversations[0];
  const themeColors = getThemeColors(currentTheme);

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

    const userMessage: Message = {
      id: Date.now().toString(),
      text: text,
      sender: 'user',
      timestamp: new Date(),
      type: 'text',
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
            ? text.substring(0, 30) + (text.length > 30 ? '...' : '')
            : conv.title
        };
      }
      return conv;
    }));

    setInputValue('');
    setIsLoading(true);
    setShowSuggestions(false);

    // Simulate typing indicator
    simulateTyping();

    try {
      const response = await fetch(`${API_URL}/api/chat/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: text,
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

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: data.answer,
        sender: 'bot',
        timestamp: new Date(),
        sources: data.sources,
        type: 'text',
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
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: '⚠️ I\'m having trouble connecting right now. Please ensure the backend server is running.',
        sender: 'bot',
        timestamp: new Date(),
        type: 'text',
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


  const clearChat = () => {
    setConversations(prev => prev.map(conv => {
      if (conv.id === currentConversationId) {
        return {
          ...conv,
          messages: [
            {
              id: '1',
              text: 'AI Assistant! 👋\n\nWelcome to the Physical AI & Humanoid Robotics knowledge base.',
              sender: 'bot',
              timestamp: new Date(),
            },
          ],
          title: 'Welcome Conversation',
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

  const handleReaction = (messageId: string, reaction: 'thumbsUp' | 'thumbsDown') => {
    setConversations(prev => prev.map(conv => {
      if (conv.id === currentConversationId) {
        return {
          ...conv,
          messages: conv.messages.map(msg => {
            if (msg.id === messageId) {
              return {
                ...msg,
                reaction: msg.reaction === reaction ? null : reaction
              };
            }
            return msg;
          }),
          updatedAt: new Date()
        };
      }
      return conv;
    }));
  };

  // Theme change functionality has been removed

  const toggleMessagePin = (messageId: string) => {
    setConversations(prev => prev.map(conv => {
      if (conv.id === currentConversationId) {
        return {
          ...conv,
          messages: conv.messages.map(msg => {
            if (msg.id === messageId) {
              return { ...msg, pinned: !msg.pinned };
            }
            return msg;
          }),
          updatedAt: new Date()
        };
      }
      return conv;
    }));

    setPinnedMessages(prev => {
      const message = conversations
        .find(c => c.id === currentConversationId)
        ?.messages.find(m => m.id === messageId);

      if (message?.pinned) {
        return prev.filter(id => id !== messageId);
      } else {
        return [...prev, messageId];
      }
    });
  };

  // File upload functionality has been removed

  const insertTemplate = (template: { id: string; title: string; content: string }) => {
    setInputValue(template.content);
    setTimeout(() => inputRef.current?.focus(), 100);
  };

  const simulateTyping = () => {
    setIsTyping(true);
    setTimeout(() => setIsTyping(false), 2000);
  };

  // Streaming text effect
  const streamText = (text: string, callback: (fullText: string) => void) => {
    let index = 0;
    setIsStreaming(true);
    setStreamingText('');

    const interval = setInterval(() => {
      if (index < text.length) {
        setStreamingText(prev => prev + text[index]);
        index++;
      } else {
        clearInterval(interval);
        setIsStreaming(false);
        setStreamingText('');
        callback(text);
      }
    }, 20); // 20ms per character for smooth streaming
  };

  // Export chat history
  const exportChatHistory = () => {
    const exportData = {
      conversation: currentConversation,
      exportDate: new Date().toISOString(),
      totalMessages: currentConversation.messages.length
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-${currentConversation.id}-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Search messages
  const filteredMessages = searchQuery
    ? currentConversation.messages.filter(msg =>
        msg.text.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : currentConversation.messages;

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
    <div
      className={styles.chatbotContainer}
      style={{
        position: 'fixed',
        zIndex: 99999,
        display: 'block',
        visibility: 'visible',
        opacity: 1,
        pointerEvents: 'all'
      }}
    >
      {!isOpen && (
        <button
          className={styles.chatButton}
          onClick={toggleChat}
          aria-label="Open AI Assistant"
          style={{
            background: `linear-gradient(135deg, #0a0a0a, #1a1a1a, #000000)`,
            borderColor: themeColors.primary,
            boxShadow: `0 0 30px ${themeColors.glow}, 0 0 0 2px rgba(0, 0, 0, 0.8), inset 0 2px 10px rgba(0, 0, 0, 0.8)`,
            display: 'flex',
            visibility: 'visible',
            opacity: 1,
            pointerEvents: 'all'
          }}
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
        <div className={styles.chatWindow} style={{
          background: themeColors.background,
          borderColor: themeColors.border,
          boxShadow: `0 0 40px ${themeColors.glow}, 0 0 0 2px ${themeColors.border}, inset 0 0 20px rgba(0, 0, 0, 0.9)`
        }}>
          <div className={styles.chatHeader} style={{
            background: `linear-gradient(135deg, rgba(10, 10, 10, 0.95), rgba(26, 26, 26, 0.8))`,
            borderBottomColor: themeColors.border,
            color: themeColors.text
          }}>
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

            <div className={styles.messagesContainer} style={{
              background: `linear-gradient(rgba(10, 10, 10, 0.85), rgba(26, 26, 26, 0.85)), radial-gradient(circle at 10% 20%, ${themeColors.glow.replace('0.5', '0.05')} 0%, transparent 20%), radial-gradient(circle at 90% 80%, ${themeColors.glow.replace('0.5', '0.05')} 0%, transparent 20%)`,
              scrollbarColor: `${themeColors.border.replace('0.2', '0.3')} transparent`
            }}>
              {currentConversation.messages
                .filter(msg => !msg.pinned)
                .map((message, index) => (
                <div
                  key={message.id}
                  className={`${styles.message} ${
                    message.sender === 'user' ? styles.userMessage : styles.botMessage
                  } ${message.pinned ? styles.pinned : ''}`}
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
                    {message.pinned && (
                      <div className={styles.pinIndicator}>
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                          <path d="M17 4v7l2 3v2h-6v5l-1 1-1-1v-5H5v-2l2-3V4c0-1.1.9-2 2-2h6c1.11 0 2 .89 2 2zM9 4v7.58l-1.45 2.09L7 16h10l-.55-2.33L15 11.58V4H9z"/>
                        </svg>
                        Pinned
                      </div>
                    )}

                    {message.attachments && message.attachments.length > 0 && (
                      <div className={styles.attachments}>
                        {message.attachments.map((attachment, idx) => (
                          attachment.type.startsWith('image/') ? (
                            <img
                              key={idx}
                              src={attachment.url}
                              alt={attachment.name}
                              className={styles.imageAttachment}
                              onLoad={() => URL.revokeObjectURL(attachment.url)}
                            />
                          ) : (
                            <FileAttachment key={idx} attachment={attachment} theme={currentTheme} />
                          )
                        ))}
                      </div>
                    )}

                    {message.sender === 'bot' ? (
                      <div className={styles.messageText}>
                        <ReactMarkdown
                          components={{
                            code({ node, inline, className, children, ...props }) {
                              const match = /language-(\w+)/.exec(className || '');
                              return !inline && match ? (
                                <CodeBlock
                                  language={match[1]}
                                  value={String(children).replace(/\n$/, '')}
                                />
                              ) : (
                                <code className={styles.inlineCode} {...props}>
                                  {children}
                                </code>
                              );
                            },
                            p: ({ children }) => <p style={{ margin: '0.5em 0' }}>{children}</p>,
                            ul: ({ children }) => <ul style={{ marginLeft: '1.5em', marginTop: '0.5em', marginBottom: '0.5em' }}>{children}</ul>,
                            ol: ({ children }) => <ol style={{ marginLeft: '1.5em', marginTop: '0.5em', marginBottom: '0.5em' }}>{children}</ol>,
                            li: ({ children }) => <li style={{ marginBottom: '0.25em' }}>{children}</li>,
                            h1: ({ children }) => <h1 style={{ fontSize: '1.5em', marginTop: '0.5em', marginBottom: '0.5em' }}>{children}</h1>,
                            h2: ({ children }) => <h2 style={{ fontSize: '1.3em', marginTop: '0.5em', marginBottom: '0.5em' }}>{children}</h2>,
                            h3: ({ children }) => <h3 style={{ fontSize: '1.1em', marginTop: '0.5em', marginBottom: '0.5em' }}>{children}</h3>,
                            blockquote: ({ children }) => (
                              <blockquote className={styles.blockquote}>{children}</blockquote>
                            ),
                            a: ({ children, href }) => (
                              <a href={href} target="_blank" rel="noopener noreferrer" className={styles.markdownLink}>
                                {children}
                              </a>
                            ),
                          }}
                        >
                          {message.text}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <p className={styles.messageText}>{message.text}</p>
                    )}
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
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '8px' }}>
                      <span className={styles.messageTime}>
                        {message.timestamp.toLocaleTimeString([], {
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </span>
                      <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                        <button
                          className={`${styles.pinButton} ${message.pinned ? styles.pinned : ''}`}
                          onClick={() => toggleMessagePin(message.id)}
                          title={message.pinned ? "Unpin message" : "Pin message"}
                        >
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M17 4v7l2 3v2h-6v5l-1 1-1-1v-5H5v-2l2-3V4c0-1.1.9-2 2-2h6c1.11 0 2 .89 2 2zM9 4v7.58l-1.45 2.09L7 16h10l-.55-2.33L15 11.58V4H9z"/>
                          </svg>
                        </button>
                        {message.sender === 'bot' && (
                          <div className={styles.messageReactions}>
                            <button
                              className={`${styles.reactionButton} ${message.reaction === 'thumbsUp' ? styles.active : ''}`}
                              onClick={() => handleReaction(message.id, 'thumbsUp')}
                              title="Helpful"
                            >
                              👍
                            </button>
                            <button
                              className={`${styles.reactionButton} ${message.reaction === 'thumbsDown' ? styles.active : ''}`}
                              onClick={() => handleReaction(message.id, 'thumbsDown')}
                              title="Not helpful"
                            >
                              👎
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              {(isLoading || isTyping) && (
                <div className={`${styles.message} ${styles.botMessage}`}>
                  <div className={styles.messageAvatar}>
                    <svg viewBox="0 0 24 24" fill="currentColor">
                      <circle cx="12" cy="12" r="10"/>
                    </svg>
                  </div>
                  <div className={styles.messageContent}>
                    <TypingIndicator />
                  </div>
                </div>
              )}
              {showSuggestions && currentConversation.messages.length === 1 && (
                <div className={styles.suggestionsContainer}>
                  <p className={styles.suggestionsTitle}>💡 Quick Start</p>
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

                  <div className={styles.templatesContainer}>
                    <p className={styles.templatesTitle}>📝 Message Templates</p>
                    <div className={styles.templatesList}>
                      {messageTemplates.map((template) => (
                        <button
                          key={template.id}
                          className={styles.templateChip}
                          onClick={() => insertTemplate(template)}
                          title={template.content}
                        >
                          {template.title}
                        </button>
                      ))}
                    </div>
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