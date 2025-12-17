import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import { motion } from 'framer-motion';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import RAGChatbotSDK from '../sdk/rag-chatbot-sdk';

interface ModuleDetailPageProps {
  moduleData: {
    id: string;
    path: string;
    title: string;
    description: string;
    icon: string;
    learningObjectives: string[];
    content: string;
    exercises?: string[];
    glossaryTerms?: { term: string; definition: string }[];
    references?: string[];
  };
}

const ModuleDetailPage: React.FC<ModuleDetailPageProps> = ({ moduleData }) => {
  const context = useDocusaurusContext();
  const backendUrl = context.siteConfig.customFields?.backendUrl || 'http://localhost:8000';
  const [activeTab, setActiveTab] = useState<'content' | 'exercises' | 'glossary' | 'references'>('content');
  const [selectedText, setSelectedText] = useState<string | null>(null);

  // Mock content for demonstration - in a real implementation, this would come from the backend
  const mockContent = `
# ${moduleData.title}

${moduleData.description}

## Learning Objectives
${moduleData.learningObjectives.map(obj => `- ${obj}`).join('\n')}

## Introduction

Physical AI represents a paradigm shift in artificial intelligence research, moving beyond traditional data-centric approaches to embrace the principles of embodied intelligence. Unlike classical AI which operates primarily in virtual spaces, Physical AI grounds intelligence in the physical world through sensorimotor interactions.

This module introduces the fundamental concepts that differentiate Physical AI from traditional AI systems, exploring how embodiment and environmental interaction contribute to intelligent behavior.

## Key Concepts

1. **Embodied Cognition**: The theory that cognitive processes are deeply rooted in the body's interactions with the world.
2. **Sensorimotor Contingencies**: The lawful relationships between motor commands and sensory feedback.
3. **Morphological Computation**: The idea that body structure contributes to computation and control.

## Applications

Physical AI has broad applications in robotics, from industrial automation to assistive technologies. In humanoid robotics specifically, these principles enable more natural and adaptive interactions with the physical world.

## Conclusion

Understanding Physical AI fundamentals is crucial for developing robots that can interact effectively with the physical world. This foundation will support deeper exploration of humanoid robotics concepts in subsequent modules.

## Diagrams and Tables

### Table: Comparison of Traditional vs Physical AI
| Aspect | Traditional AI | Physical AI |
|--------|----------------|-------------|
| Interaction Model | Input-Process-Output | Sensorimotor Loop |
| Learning Method | Data-driven | Embodied Learning |
| Environmental Integration | External | Intrinsic |

### Example Diagram Description
[Diagram showing the Physical AI perception-action cycle with arrows indicating continuous interaction between the agent, its body, and the environment.]
`;

  // Mock exercises for demonstration
  const mockExercises = [
    'Explain the difference between traditional AI and Physical AI in your own words.',
    'Identify three examples of morphological computation in biological systems.',
    'Design a simple experiment to demonstrate sensorimotor contingencies.',
    'Research and summarize a recent paper on embodied AI (within last 2 years).'
  ];

  // Mock glossary terms for demonstration
  const mockGlossaryTerms = [
    { term: 'Embodied Intelligence', definition: 'Intelligence that emerges from the interaction between an agent and its environment through a physical body.' },
    { term: 'Sensorimotor Loop', definition: 'The continuous cycle of sensing, processing, and acting that characterizes embodied agents.' },
    { term: 'Morphological Computation', definition: 'The contribution of body structure and environmental interaction to behavioral control.' }
  ];

  // Mock references for demonstration
  const mockReferences = [
    'Pfeifer, R., & Bongard, J. (2006). How the body shapes the way we think: A new view of intelligence. MIT Press.',
    'Brooks, R. A. (1991). Intelligence without representation. Artificial intelligence, 47(1-3), 139-159.',
    'Clark, A. (2008). Supersizing the mind: Embodiment, action, and cognitive extension. Oxford University Press.'
  ];

  // Get content based on active tab
  const getContent = () => {
    switch(activeTab) {
      case 'content':
        if (moduleData.content) {
          return moduleData.content;
        }
        return mockContent;
      case 'exercises':
        if (moduleData.exercises) {
          return moduleData.exercises.map((ex, i) => `${i + 1}. ${ex}`).join('\n\n');
        }
        return mockExercises.map((ex, i) => `${i + 1}. ${ex}`).join('\n\n');
      case 'glossary':
        if (moduleData.glossaryTerms) {
          return moduleData.glossaryTerms.map(term => `**${term.term}**: ${term.definition}`).join('\n\n');
        }
        return mockGlossaryTerms.map(term => `**${term.term}**: ${term.definition}`).join('\n\n');
      case 'references':
        if (moduleData.references) {
          return moduleData.references.join('\n\n');
        }
        return mockReferences.join('\n\n');
      default:
        return mockContent;
    }
  };

  // Function to render content with proper formatting
  const renderContent = (content: string) => {
    const lines = content.split('\n');
    const elements = [];
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      if (line.startsWith('# ')) {
        elements.push(
          <h1 key={`h1-${i}`} style={{ fontFamily: 'Sora, sans-serif', color: '#fff', fontSize: '2rem', marginTop: '2rem', marginBottom: '1rem' }}>
            {line.substring(2)}
          </h1>
        );
      } else if (line.startsWith('## ')) {
        elements.push(
          <h2 key={`h2-${i}`} style={{ fontFamily: 'Sora, sans-serif', color: '#e0e0e0', fontSize: '1.6rem', marginTop: '1.8rem', marginBottom: '0.8rem' }}>
            {line.substring(3)}
          </h2>
        );
      } else if (line.startsWith('### ')) {
        elements.push(
          <h3 key={`h3-${i}`} style={{ fontFamily: 'Sora, sans-serif', color: '#d0d0d0', fontSize: '1.4rem', marginTop: '1.5rem', marginBottom: '0.6rem' }}>
            {line.substring(4)}
          </h3>
        );
      } else if (line.startsWith('- ')) {
        elements.push(
          <li key={`li-${i}`} style={{ fontFamily: 'Inter, sans-serif', color: '#c0c0c0', margin: '0.5rem 0', lineHeight: '1.7' }}>
            {line.substring(2)}
          </li>
        );
      } else if (line.trim() === '') {
        elements.push(<br key={`br-${i}`} />);
      } else if (line.startsWith('|')) {
        // Start of a table - find all table rows
        const tableRows = [];
        while (i < lines.length && lines[i].trim() !== '' && lines[i].startsWith('|')) {
          const currentLine = lines[i];
          if (!currentLine.includes('--')) { // Skip separator line
            const cells = currentLine.split('|').filter(cell => cell.trim() !== '');
            tableRows.push(
              <tr key={`tr-${i}`} style={{ fontFamily: 'Inter, sans-serif', color: '#c0c0c0' }}>
                {cells.map((cell, j) => (
                  <td
                    key={`td-${i}-${j}`}
                    style={{
                      padding: '0.8rem',
                      borderBottom: '1px solid #444',
                      color: '#c0c0c0',
                      borderRight: j < cells.length - 1 ? '1px solid #444' : 'none',
                      backgroundColor: tableRows.length === 0 ? 'rgba(136, 136, 136, 0.1)' : 'transparent' // Header row styling
                    }}
                  >
                    {cell.trim()}
                  </td>
                ))}
              </tr>
            );
          }
          i++;
        }
        i--; // Adjust for the while loop increment

        elements.push(
          <div key={`table-${i}`} style={{ overflowX: 'auto', margin: '1.5rem 0' }}>
            <table style={{
              width: '100%',
              borderCollapse: 'collapse',
              border: '1px solid #444',
              borderRadius: '4px',
              overflow: 'hidden'
            }}>
              <tbody>{tableRows}</tbody>
            </table>
          </div>
        );
      } else if (line.startsWith('> ')) {
        // Callout/Blockquote
        const calloutLines = [];
        while (i < lines.length && lines[i].startsWith('>')) {
          calloutLines.push(lines[i].substring(2)); // Remove '> '
          i++;
        }
        i--; // Adjust for the while loop increment

        elements.push(
          <div
            key={`callout-${i}`}
            style={{
              margin: '1.5rem 0',
              padding: '1.2rem',
              borderLeft: '4px solid #888888',
              backgroundColor: 'rgba(136, 136, 136, 0.08)',
              borderRadius: '0 8px 8px 0',
              fontFamily: 'Inter, sans-serif',
              color: '#c0c0c0',
              lineHeight: '1.7'
            }}
          >
            {calloutLines.join(' ')}
          </div>
        );
      } else if (line.startsWith('### Diagram:')) {
        // Diagram description
        const diagramTitle = line.substring(12); // Remove '### Diagram:'
        let diagramDescription = '';
        i++; // Move to next line

        // Capture description until next heading or end of relevant content
        while (i < lines.length &&
               !lines[i].startsWith('# ') &&
               !lines[i].startsWith('## ') &&
               !lines[i].startsWith('### ') &&
               lines[i].trim() !== '') {
          diagramDescription += lines[i] + ' ';
          i++;
        }
        i--; // Adjust for the while loop increment

        elements.push(
          <div
            key={`diagram-${i}`}
            style={{
              margin: '2rem 0',
              padding: '1.5rem',
              border: '1px solid #555',
              borderRadius: '8px',
              backgroundColor: 'rgba(30, 30, 30, 0.6)',
              textAlign: 'center'
            }}
          >
            <h4 style={{
              fontFamily: 'Sora, sans-serif',
              color: '#888888',
              margin: '0 0 1rem 0',
              fontSize: '1.1rem',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}>
              {diagramTitle}
            </h4>
            <p style={{
              fontFamily: 'Inter, sans-serif',
              color: '#a8a8a8',
              margin: '0',
              fontStyle: 'italic'
            }}>
              {diagramDescription.trim() || 'Diagram placeholder'}
            </p>
            <div style={{
              margin: '1rem auto 0',
              width: 'fit-content',
              padding: '1rem',
              border: '1px dashed #666',
              borderRadius: '4px',
              color: '#777',
              fontSize: '0.9rem'
            }}>
              [Diagram Visualization Area]
            </div>
          </div>
        );
      } else if (line.startsWith('**') && line.endsWith('**')) {
        elements.push(
          <strong key={`strong-${i}`} style={{ color: '#888888', fontWeight: 700 }}>
            {line.substring(2, line.length - 2)}
          </strong>
        );
      } else {
        elements.push(
          <p key={`p-${i}`} style={{ fontFamily: 'Inter, sans-serif', color: '#c0c0c0', lineHeight: '1.8', marginBottom: '1rem' }}>
            {line}
          </p>
        );
      }

      i++;
    }

    return elements;
  };

  // Function to specifically render exercises
  const renderExercises = () => {
    if (!moduleData.exercises || moduleData.exercises.length === 0) {
      return (
        <div style={{
          textAlign: 'center',
          padding: '2rem',
          color: '#a0a0a0',
          fontFamily: 'Inter, sans-serif',
        }}>
          <p>No exercises available for this module.</p>
        </div>
      );
    }

    return (
      <div>
        <div style={{
          fontFamily: 'Sora, sans-serif',
          color: '#888888',
          fontSize: '1.2rem',
          marginBottom: '1.5rem',
          fontWeight: 700,
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
        }}>
          📝 Module Exercises
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {moduleData.exercises.map((exercise, index) => (
            <div
              key={index}
              style={{
                background: 'rgba(30, 30, 30, 0.6)',
                borderRadius: '12px',
                padding: '1.5rem',
                border: '1px solid rgba(136, 136, 136, 0.2)',
                transition: 'all 0.3s ease',
              }}
            >
              <div style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: '1rem'
              }}>
                <div style={{
                  flexShrink: 0,
                  width: '30px',
                  height: '30px',
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, #0FE3C0 0%, #0a8f7a 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#000',
                  fontWeight: 'bold',
                  fontFamily: 'Sora, sans-serif',
                  fontSize: '0.9rem',
                  marginTop: '0.2rem'
                }}>
                  {index + 1}
                </div>
                <div style={{ flex: 1 }}>
                  <p style={{
                    fontFamily: 'Inter, sans-serif',
                    color: '#c0c0c0',
                    lineHeight: '1.7',
                    margin: 0,
                    fontSize: '1.05rem'
                  }}>
                    {exercise}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Function to specifically render glossary terms
  const renderGlossary = () => {
    if (!moduleData.glossaryTerms || moduleData.glossaryTerms.length === 0) {
      return (
        <div style={{
          textAlign: 'center',
          padding: '2rem',
          color: '#a0a0a0',
          fontFamily: 'Inter, sans-serif',
        }}>
          <p>No glossary terms available for this module.</p>
        </div>
      );
    }

    return (
      <div>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '1.5rem',
        }}>
          <div style={{
            fontFamily: 'Sora, sans-serif',
            color: '#888888',
            fontSize: '1.2rem',
            fontWeight: 700,
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}>
            📘 Glossary Terms
          </div>
          <Link
            to="/docs/19-glossary-references"  // Link to the glossary module
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.3rem',
              padding: '0.5rem 1rem',
              background: 'rgba(15, 227, 192, 0.1)',
              color: '#0FE3C0',
              fontFamily: 'Sora, sans-serif',
              fontWeight: 600,
              fontSize: '0.9rem',
              borderRadius: '6px',
              textDecoration: 'none',
              border: '1px solid rgba(15, 227, 192, 0.3)',
              transition: 'all 0.3s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(15, 227, 192, 0.2)';
              e.currentTarget.style.color = '#fff';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'rgba(15, 227, 192, 0.1)';
              e.currentTarget.style.color = '#0FE3C0';
            }}
          >
            Full Glossary →
          </Link>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
          {moduleData.glossaryTerms.map((termObj, index) => (
            <div
              key={index}
              style={{
                background: 'rgba(30, 30, 30, 0.6)',
                borderRadius: '12px',
                padding: '1.3rem',
                border: '1px solid rgba(136, 136, 136, 0.2)',
              }}
            >
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
                gap: '1rem'
              }}>
                <div style={{ flex: 1 }}>
                  <h4 style={{
                    fontFamily: 'Sora, sans-serif',
                    color: '#0FE3C0',
                    fontSize: '1.1rem',
                    fontWeight: 700,
                    margin: '0 0 0.5rem 0',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px'
                  }}>
                    {termObj.term}
                  </h4>
                  <p style={{
                    fontFamily: 'Inter, sans-serif',
                    color: '#c0c0c0',
                    lineHeight: '1.6',
                    margin: 0,
                    fontSize: '1rem'
                  }}>
                    {termObj.definition}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Function to specifically render references
  const renderReferences = () => {
    if (!moduleData.references || moduleData.references.length === 0) {
      return (
        <div style={{
          textAlign: 'center',
          padding: '2rem',
          color: '#a0a0a0',
          fontFamily: 'Inter, sans-serif',
        }}>
          <p>No references available for this module.</p>
        </div>
      );
    }

    return (
      <div>
        <div style={{
          fontFamily: 'Sora, sans-serif',
          color: '#888888',
          fontSize: '1.2rem',
          marginBottom: '1.5rem',
          fontWeight: 700,
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
        }}>
          📚 Academic References
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {moduleData.references.map((reference, index) => (
            <div
              key={index}
              style={{
                background: 'rgba(30, 30, 30, 0.6)',
                borderRadius: '10px',
                padding: '1.2rem',
                border: '1px solid rgba(136, 136, 136, 0.2)',
              }}
            >
              <p style={{
                fontFamily: 'Inter, sans-serif',
                color: '#c0c0c0',
                lineHeight: '1.7',
                margin: 0,
                fontSize: '0.95rem'
              }}>
                {index + 1}. {reference}
              </p>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Function to render related modules
  const renderRelatedModules = () => {
    // Define some example related modules - in a real implementation, this would come from the backend
    const relatedModules = [
      { id: '02', path: '02-robotics-mechatronics-fundamentals', title: 'Robotics Fundamentals', description: 'Mechanical structures, actuators, and control systems' },
      { id: '05', path: '05-kinematics', title: 'Kinematics', description: 'Forward/inverse kinematics and transformations' },
      { id: '07', path: '07-sensors-perception', title: 'Sensors & Perception', description: 'Sensor technologies and data processing' },
    ];

    return (
      <div>
        <div style={{
          fontFamily: 'Sora, sans-serif',
          color: '#888888',
          fontSize: '1.2rem',
          marginBottom: '1.5rem',
          fontWeight: 700,
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
        }}>
          🔗 Related Modules
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
          {relatedModules.map((module, index) => (
            <div
              key={index}
              style={{
                background: 'rgba(30, 30, 30, 0.6)',
                borderRadius: '12px',
                padding: '1.3rem',
                border: '1px solid rgba(136, 136, 136, 0.2)',
                transition: 'transform 0.3s ease, border-color 0.3s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-3px)';
                e.currentTarget.style.borderColor = 'rgba(136, 136, 136, 0.4)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.borderColor = 'rgba(136, 136, 136, 0.2)';
              }}
            >
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
                gap: '1rem'
              }}>
                <div style={{ flex: 1 }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    marginBottom: '0.3rem'
                  }}>
                    <span style={{
                      fontFamily: 'Sora, sans-serif',
                      color: '#888888',
                      fontSize: '0.85rem',
                      textTransform: 'uppercase',
                      letterSpacing: '1px',
                      fontWeight: 700,
                    }}>
                      Module {module.id}
                    </span>
                  </div>
                  <h4 style={{
                    fontFamily: 'Sora, sans-serif',
                    color: '#0FE3C0',
                    fontSize: '1.1rem',
                    fontWeight: 700,
                    margin: '0 0 0.5rem 0',
                  }}>
                    {module.title}
                  </h4>
                  <p style={{
                    fontFamily: 'Inter, sans-serif',
                    color: '#c0c0c0',
                    lineHeight: '1.6',
                    margin: '0 0 0.8rem 0',
                    fontSize: '0.95rem'
                  }}>
                    {module.description}
                  </p>
                  <Link
                    to={`/docs/${module.path}`}
                    style={{
                      display: 'inline-block',
                      padding: '0.6rem 1.2rem',
                      background: 'rgba(15, 227, 192, 0.1)',
                      color: '#0FE3C0',
                      fontFamily: 'Sora, sans-serif',
                      fontWeight: 600,
                      fontSize: '0.9rem',
                      borderRadius: '6px',
                      textDecoration: 'none',
                      border: '1px solid rgba(15, 227, 192, 0.3)',
                      transition: 'all 0.3s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'rgba(15, 227, 192, 0.2)';
                      e.currentTarget.style.color = '#fff';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'rgba(15, 227, 192, 0.1)';
                      e.currentTarget.style.color = '#0FE3C0';
                    }}
                  >
                    Explore Module →
                  </Link>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Function to handle text selection for the RAG chatbot
  const handleTextSelection = () => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim() !== '') {
      setSelectedText(selection.toString().trim());
    }
  };

  useEffect(() => {
    document.addEventListener('mouseup', handleTextSelection);
    return () => {
      document.removeEventListener('mouseup', handleTextSelection);
    };
  }, []);

  // Function to simulate a typing effect for streaming responses (frontend representation)
  const simulateStreamingResponse = (responseText) => {
    const responseElement = document.getElementById('chat-response');
    const typingIndicator = document.getElementById('typing-indicator');
    const responseContainer = responseElement?.parentElement;

    if (responseContainer) responseContainer.style.display = 'block';
    if (typingIndicator) typingIndicator.style.display = 'block';
    if (responseElement) responseElement.textContent = '';

    // Simulate streaming by adding characters one by one
    let i = 0;
    const timer = setInterval(() => {
      if (i < responseText.length) {
        if (responseElement) responseElement.textContent += responseText.charAt(i);
        i++;
      } else {
        clearInterval(timer);
        if (typingIndicator) typingIndicator.style.display = 'none';
      }
    }, 20); // Adjust speed as needed (20ms per character)
  };

  // Function to display citations
  const displayCitations = (citationList) => {
    const citationsContainer = document.getElementById('citations');
    const citationListElement = document.getElementById('citation-list');

    if (citationListElement) {
      citationListElement.innerHTML = '';
      citationList.forEach((citation, index) => {
        const li = document.createElement('li');
        li.textContent = `${index + 1}. ${citation}`;
        citationListElement.appendChild(li);
      });
    }

    if (citationsContainer) citationsContainer.style.display = 'block';
  };

  // Function to handle question submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    // Get the question from the textarea
    const textarea = e.target.querySelector('textarea');
    const question = textarea.value.trim();
    if (!question) return;

    // Hide any previous error messages and reset UI
    const errorElement = document.getElementById('error-message');
    if (errorElement) {
      errorElement.style.display = 'none';
    }

    // Show typing indicator while processing the request
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
      typingIndicator.style.display = 'block';
    }

    try {
      // Initialize the RAG SDK with the backend URL
      const ragSDK = new RAGChatbotSDK(backendUrl);

      // Send the chat request
      const response = await ragSDK.chat({
        question: question,
        session_id: null, // Will be generated by backend if null
        selected_text: selectedText || null // Send selected text if available
      });

      // Hide typing indicator
      if (typingIndicator) typingIndicator.style.display = 'none';

      // Show response
      simulateStreamingResponse(response.answer);

      // Display citations using the SDK's processed format
      const processedCitations = ragSDK.processCitations(response.citations);
      displayCitations(processedCitations);

    } catch (error) {
      // Hide typing indicator
      if (typingIndicator) typingIndicator.style.display = 'none';

      // Show error message
      if (errorElement) {
        const errorMessage = error.message || 'An error occurred while processing your request.';
        errorElement.textContent = `Error: ${errorMessage}`;
        errorElement.style.display = 'block';
      }

      // Hide other response elements
      const responseElement = document.getElementById('chat-response');
      const citationsElement = document.getElementById('citations');
      if (responseElement) responseElement.textContent = '';
      if (citationsElement) citationsElement.style.display = 'none';
    }
  };

  return (
    <Layout title={moduleData.title} description={moduleData.description}>
      <div style={{
        background: 'linear-gradient(135deg, #000000 0%, #0a0a0a 100%)',
        minHeight: '100vh',
        padding: '3rem 0',
      }}>
        <div className="container">
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: '1fr 300px', 
            gap: '2.5rem',
            marginBottom: '3rem'
          }}>
            {/* Main Content Area */}
            <div style={{ 
              background: 'rgba(20, 20, 20, 0.8)',
              backdropFilter: 'blur(10px)',
              borderRadius: '20px',
              border: '1px solid rgba(176, 224, 230, 0.1)',
              padding: '2.5rem',
              boxShadow: '0 10px 40px rgba(0, 0, 0, 0.3)',
              position: 'relative',
              overflow: 'hidden',
            }}>
              {/* Top accent */}
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '3px',
                background: 'linear-gradient(90deg, #888888 0%, #e0e0e0 100%)',
              }} />
              
              {/* Module Header */}
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1.5rem', marginBottom: '2rem' }}>
                <div style={{
                  fontSize: '3rem',
                  minWidth: '60px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}>
                  {moduleData.icon}
                </div>
                <div>
                  <h3 style={{
                    fontFamily: 'Sora, sans-serif',
                    color: '#888888',
                    marginBottom: '0.3rem',
                    fontSize: '1rem',
                    textTransform: 'uppercase',
                    letterSpacing: '1.5px',
                    fontWeight: 700,
                  }}>
                    Module {moduleData.id}
                  </h3>
                  <h1 style={{
                    fontFamily: 'Sora, sans-serif',
                    color: '#fff',
                    fontSize: '2.2rem',
                    marginBottom: '0.5rem',
                    fontWeight: 800,
                    lineHeight: '1.2',
                  }}>
                    {moduleData.title}
                  </h1>
                  <p style={{
                    fontFamily: 'Inter, sans-serif',
                    color: '#a8a8a8',
                    fontSize: '1.1rem',
                    margin: 0,
                    lineHeight: '1.6',
                  }}>
                    {moduleData.description}
                  </p>
                </div>
              </div>
              
              {/* Learning Objectives */}
              <div style={{ 
                background: 'rgba(30, 30, 30, 0.6)',
                borderRadius: '12px',
                padding: '1.5rem',
                marginBottom: '2rem',
                border: '1px solid rgba(136, 136, 136, 0.2)',
              }}>
                <h3 style={{
                  fontFamily: 'Sora, sans-serif',
                  color: '#888888',
                  fontSize: '1.1rem',
                  textTransform: 'uppercase',
                  letterSpacing: '1px',
                  fontWeight: 700,
                  marginBottom: '1rem',
                }}>
                  Learning Objectives
                </h3>
                <ul style={{
                  listStyle: 'none',
                  padding: 0,
                  margin: 0,
                }}>
                  {moduleData.learningObjectives.map((objective, i) => (
                    <li key={i} style={{
                      fontFamily: 'Inter, sans-serif',
                      color: '#c0c0c0',
                      fontSize: '1rem',
                      marginBottom: '0.7rem',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '0.8rem',
                    }}>
                      <span style={{ 
                        color: '#888888', 
                        fontSize: '1.2rem', 
                        fontWeight: 700,
                        marginTop: '0.2rem'
                      }}>✓</span>
                      <span>{objective}</span>
                    </li>
                  ))}
                </ul>
              </div>
              
              {/* Tabs */}
              <div style={{
                display: 'flex',
                borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                marginBottom: '2rem',
              }}>
                {['content', 'exercises', 'glossary', 'references', 'related'].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab as any)}
                    style={{
                      padding: '0.8rem 1.5rem',
                      background: activeTab === tab ? 'rgba(136, 136, 136, 0.2)' : 'transparent',
                      color: activeTab === tab ? '#e0e0e0' : '#888888',
                      border: 'none',
                      fontFamily: 'Sora, sans-serif',
                      fontSize: '0.95rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px',
                      borderBottom: activeTab === tab ? '2px solid #888888' : 'none',
                    }}
                  >
                    {tab === 'related' ? 'Related Modules' : tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                ))}
              </div>
              
              {/* Content Area */}
              <div
                style={{
                  color: '#c0c0c0',
                  fontFamily: 'Inter, sans-serif',
                  lineHeight: '1.8',
                  fontSize: '1.05rem',
                }}
                onMouseUp={handleTextSelection}
              >
                {activeTab === 'exercises'
                  ? renderExercises()
                  : activeTab === 'glossary'
                  ? renderGlossary()
                  : activeTab === 'references'
                  ? renderReferences()
                  : activeTab === 'related'
                  ? renderRelatedModules()
                  : renderContent(getContent())}
              </div>
            </div>
            
            {/* Sidebar */}
            <div>
              {/* RAG Chatbot Section */}
              <div style={{
                background: 'rgba(20, 20, 20, 0.8)',
                backdropFilter: 'blur(10px)',
                borderRadius: '20px',
                border: '1px solid rgba(176, 224, 230, 0.1)',
                padding: '1.8rem',
                boxShadow: '0 10px 40px rgba(0, 0, 0, 0.3)',
                marginBottom: '2rem',
                position: 'sticky',
                top: '3rem',
              }}>
                <div style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: '3px',
                  background: 'linear-gradient(90deg, #0FE3C0 0%, #0a8f7a 100%)',
                }} />

                <h3 style={{
                  fontFamily: 'Sora, sans-serif',
                  color: '#0FE3C0',
                  fontSize: '1.2rem',
                  marginBottom: '1.2rem',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                }}>
                  🤖 AI Study Assistant
                </h3>

                {selectedText && (
                  <div style={{
                    background: 'rgba(15, 227, 192, 0.1)',
                    border: '1px solid rgba(15, 227, 192, 0.2)',
                    borderRadius: '8px',
                    padding: '0.8rem',
                    marginBottom: '1rem',
                    fontSize: '0.85rem',
                  }}>
                    <div style={{ color: '#0FE3C0', fontWeight: 600, marginBottom: '0.3rem' }}>Selected text:</div>
                    <div style={{ color: '#d0d0d0', fontStyle: 'italic' }}>
                      "{selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}"
                    </div>
                  </div>
                )}

                <div style={{
                  marginBottom: '1.2rem',
                }}>
                  <label style={{
                    display: 'block',
                    fontFamily: 'Sora, sans-serif',
                    color: '#c0c0c0',
                    fontSize: '0.9rem',
                    marginBottom: '0.5rem',
                    fontWeight: 600,
                  }}>
                    Ask about this module:
                  </label>
                  <textarea
                    placeholder={selectedText
                      ? `Ask about the selected text...`
                      : `Ask a question about ${moduleData.title}...`}
                    rows={4}
                    style={{
                      width: '100%',
                      padding: '0.8rem',
                      borderRadius: '8px',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      background: 'rgba(10, 10, 10, 0.7)',
                      color: '#fff',
                      fontFamily: 'Inter, sans-serif',
                      fontSize: '0.95rem',
                      resize: 'vertical',
                      outline: 'none',
                    }}
                  />
                </div>

                <button
                  onClick={handleSubmit}
                  style={{
                    width: '100%',
                    padding: '0.8rem',
                    background: 'linear-gradient(135deg, #0FE3C0 0%, #0a8f7a 100%)',
                    color: '#000',
                    fontFamily: 'Sora, sans-serif',
                    fontWeight: 700,
                    fontSize: '1rem',
                    borderRadius: '8px',
                    border: 'none',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                  }}>
                  Submit Question →
                </button>

                {/* Chatbot Response Area - prepared for streaming */}
                <div style={{
                  marginTop: '1.5rem',
                  padding: '1.2rem',
                  background: 'rgba(10, 10, 10, 0.7)',
                  borderRadius: '10px',
                  border: '1px solid rgba(15, 227, 192, 0.2)',
                  minHeight: '100px',
                }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    marginBottom: '0.8rem',
                    color: '#0FE3C0',
                    fontFamily: 'Sora, sans-serif',
                    fontWeight: 700,
                  }}>
                    <span style={{ marginRight: '0.5rem' }}>🤖</span>
                    AI Response
                  </div>

                  {/* Error Message Display */}
                  <div id="error-message" style={{
                    display: 'none',
                    color: '#ff6b6b',
                    fontFamily: 'Sora, sans-serif',
                    fontWeight: 600,
                    fontSize: '0.95rem',
                    marginBottom: '0.8rem',
                    padding: '0.8rem',
                    background: 'rgba(255, 107, 107, 0.1)',
                    borderRadius: '6px',
                    border: '1px solid rgba(255, 107, 107, 0.3)'
                  }}>
                    <!-- Error messages will appear here -->
                  </div>

                  {/* Typing Indicator */}
                  <div id="typing-indicator" style={{
                    display: 'none',
                    color: '#0FE3C0',
                    fontFamily: 'Inter, sans-serif',
                    fontSize: '0.95rem',
                    marginBottom: '0.5rem',
                    fontStyle: 'italic'
                  }}>
                    AI is thinking...
                  </div>

                  {/* Response Content */}
                  <div id="chat-response" style={{
                    fontFamily: 'Inter, sans-serif',
                    color: '#c0c0c0',
                    lineHeight: '1.7',
                    fontSize: '1rem'
                  }}>
                    <!-- Response content will appear here -->
                  </div>

                  {/* Citations */}
                  <div id="citations" style={{
                    marginTop: '1rem',
                    paddingTop: '1rem',
                    borderTop: '1px solid rgba(150, 150, 150, 0.2)',
                    display: 'none'
                  }}>
                    <div style={{
                      fontFamily: 'Sora, sans-serif',
                      color: '#888888',
                      fontSize: '0.9rem',
                      fontWeight: 600,
                      marginBottom: '0.5rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px'
                    }}>
                      Sources:
                    </div>
                    <ul id="citation-list" style={{
                      fontFamily: 'Inter, sans-serif',
                      color: '#a0a0a0',
                      fontSize: '0.85rem',
                      lineHeight: '1.6',
                      margin: 0,
                      padding: 0,
                      listStyle: 'none'
                    }}>
                      <!-- Citations will appear here -->
                    </ul>
                  </div>
                </div>

                <div style={{
                  marginTop: '1rem',
                  padding: '0.8rem',
                  background: 'rgba(30, 30, 30, 0.6)',
                  borderRadius: '8px',
                  fontSize: '0.8rem',
                  color: '#a0a0a0',
                  border: '1px solid rgba(160, 160, 160, 0.1)',
                }}>
                  This AI assistant only uses content from this textbook. Answers are grounded in module content with proper citations.
                </div>
              </div>
              
              {/* Module Navigation */}
              <div style={{ 
                background: 'rgba(20, 20, 20, 0.8)',
                backdropFilter: 'blur(10px)',
                borderRadius: '20px',
                border: '1px solid rgba(176, 224, 230, 0.1)',
                padding: '1.8rem',
                boxShadow: '0 10px 40px rgba(0, 0, 0, 0.3)',
              }}>
                <h3 style={{
                  fontFamily: 'Sora, sans-serif',
                  color: '#888888',
                  fontSize: '1.2rem',
                  marginBottom: '1.2rem',
                  fontWeight: 700,
                }}>
                  Module Navigation
                </h3>
                
                <nav>
                  <ul style={{ 
                    listStyle: 'none', 
                    padding: 0,
                    margin: 0,
                  }}>
                    {['Introduction', 'Key Concepts', 'Applications', 'Diagrams', 'Conclusion'].map((section, i) => (
                      <li key={i} style={{ marginBottom: '0.8rem' }}>
                        <Link
                          to={`#${section.toLowerCase().replace(/\s+/g, '-')}`}
                          style={{
                            display: 'block',
                            padding: '0.6rem 0.8rem',
                            color: '#c0c0c0',
                            fontFamily: 'Inter, sans-serif',
                            fontSize: '0.95rem',
                            textDecoration: 'none',
                            borderRadius: '6px',
                            transition: 'all 0.2s ease',
                            borderLeft: '2px solid transparent',
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.color = '#e0e0e0';
                            e.currentTarget.style.background = 'rgba(136, 136, 136, 0.1)';
                            e.currentTarget.style.borderLeft = '2px solid #888888';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.color = '#c0c0c0';
                            e.currentTarget.style.background = 'transparent';
                            e.currentTarget.style.borderLeft = '2px solid transparent';
                          }}
                        >
                          {section}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </nav>
              </div>
            </div>
          </div>
          
          {/* Back to Modules Link */}
          <div style={{ textAlign: 'center', marginTop: '2rem' }}>
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Link
                to="/modules"
                style={{
                  display: 'inline-block',
                  padding: '1rem 2.5rem',
                  background: 'rgba(200, 200, 200, 0.08)',
                  color: '#c0c0c0',
                  fontFamily: 'Sora, sans-serif',
                  fontWeight: 700,
                  fontSize: '1rem',
                  borderRadius: '50px',
                  textDecoration: 'none',
                  boxShadow: '0 10px 30px rgba(200, 200, 200, 0.05)',
                  transition: 'all 0.3s ease',
                  border: '1px solid rgba(200, 200, 200, 0.1)',
                }}
              >
                ← Browse All Modules
              </Link>
            </motion.div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default ModuleDetailPage;