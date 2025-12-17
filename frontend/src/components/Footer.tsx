import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

const Footer = () => {
  const { siteConfig } = useDocusaurusContext();

  return (
    <footer
      className="footer"
      style={{
        backgroundColor: '#000000',
        borderTop: '1px solid rgba(176, 224, 230, 0.2)',
        color: '#e0f0f0',
        padding: '4rem 0 2rem',
        fontFamily: 'Inter, sans-serif'
      }}
    >
      <div className="container container--fluid padding-horiz--md">
        <div className="row">
          {/* Textbook Section */}
          <div className="col col--3">
            <h4 style={{
              fontFamily: 'Sora, sans-serif',
              color: '#b0e0e6',
              marginBottom: '1rem',
              fontSize: '1.1rem',
              textTransform: 'uppercase',
              letterSpacing: '1px'
            }}>
              Textbook
            </h4>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li style={{ marginBottom: '0.5rem' }}>
                <Link
                  to="/docs/intro"
                  style={{
                    color: '#e0f0f0',
                    textDecoration: 'none',
                    fontSize: '0.9rem',
                    transition: 'color 0.3s ease'
                  }}
                  onMouseOver={(e) => e.target.style.color = '#b0e0e6'}
                  onMouseOut={(e) => e.target.style.color = '#e0f0f0'}
                >
                  Introduction
                </Link>
              </li>
              <li style={{ marginBottom: '0.5rem' }}>
                <Link
                  to="/docs/robotics-mechatronics-fundamentals"
                  style={{
                    color: '#e0f0f0',
                    textDecoration: 'none',
                    fontSize: '0.9rem',
                    transition: 'color 0.3s ease'
                  }}
                  onMouseOver={(e) => e.target.style.color = '#b0e0e6'}
                  onMouseOut={(e) => e.target.style.color = '#e0f0f0'}
                >
                  Robotics Fundamentals
                </Link>
              </li>
              <li style={{ marginBottom: '0.5rem' }}>
                <Link
                  to="/docs/ros2-foundations"
                  style={{
                    color: '#e0f0f0',
                    textDecoration: 'none',
                    fontSize: '0.9rem',
                    transition: 'color 0.3s ease'
                  }}
                  onMouseOver={(e) => e.target.style.color = '#b0e0e6'}
                  onMouseOut={(e) => e.target.style.color = '#e0f0f0'}
                >
                  ROS 2 Foundations
                </Link>
              </li>
            </ul>
          </div>

          {/* Textbook Section */}
          <div className="col col--3">
            <h4 style={{
              fontFamily: 'Sora, sans-serif',
              color: '#b0e0e6',
              marginBottom: '1rem',
              fontSize: '1.1rem',
              textTransform: 'uppercase',
              letterSpacing: '1px'
            }}>
              Textbook
            </h4>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li style={{ marginBottom: '0.5rem' }}>
                <Link
                  to="/docs/intro"
                  style={{
                    color: '#e0f0f0',
                    textDecoration: 'none',
                    fontSize: '0.9rem',
                    transition: 'color 0.3s ease'
                  }}
                  onMouseOver={(e) => e.target.style.color = '#b0e0e6'}
                  onMouseOut={(e) => e.target.style.color = '#e0f0f0'}
                >
                  Getting Started
                </Link>
              </li>
              <li style={{ marginBottom: '0.5rem' }}>
                <Link
                  to="/modules"
                  style={{
                    color: '#e0f0f0',
                    textDecoration: 'none',
                    fontSize: '0.9rem',
                    transition: 'color 0.3s ease'
                  }}
                  onMouseOver={(e) => e.target.style.color = '#b0e0e6'}
                  onMouseOut={(e) => e.target.style.color = '#e0f0f0'}
                >
                  Complete Curriculum
                </Link>
              </li>
              <li style={{ marginBottom: '0.5rem' }}>
                <Link
                  to="/docs/glossary-references"
                  style={{
                    color: '#e0f0f0',
                    textDecoration: 'none',
                    fontSize: '0.9rem',
                    transition: 'color 0.3s ease'
                  }}
                  onMouseOver={(e) => e.target.style.color = '#b0e0e6'}
                  onMouseOut={(e) => e.target.style.color = '#e0f0f0'}
                >
                  Glossary & References
                </Link>
              </li>
            </ul>
          </div>

          {/* Community Section */}
          <div className="col col--3">
            <h4 style={{
              fontFamily: 'Sora, sans-serif',
              color: '#b0e0e6',
              marginBottom: '1rem',
              fontSize: '1.1rem',
              textTransform: 'uppercase',
              letterSpacing: '1px'
            }}>
              Community
            </h4>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li style={{ marginBottom: '0.5rem' }}>
                <Link
                  href="https://github.com/creativesar/Physical-AI-Humanoid-Robotics-Textbook/discussions"
                  style={{
                    color: '#e0f0f0',
                    textDecoration: 'none',
                    fontSize: '0.9rem',
                    transition: 'color 0.3s ease'
                  }}
                  onMouseOver={(e) => e.target.style.color = '#b0e0e6'}
                  onMouseOut={(e) => e.target.style.color = '#e0f0f0'}
                >
                  Discussion Forum
                </Link>
              </li>
              <li style={{ marginBottom: '0.5rem' }}>
                <Link
                  href="https://github.com/creativesar/Physical-AI-Humanoid-Robotics-Textbook"
                  style={{
                    color: '#e0f0f0',
                    textDecoration: 'none',
                    fontSize: '0.9rem',
                    transition: 'color 0.3s ease'
                  }}
                  onMouseOver={(e) => e.target.style.color = '#b0e0e6'}
                  onMouseOut={(e) => e.target.style.color = '#e0f0f0'}
                >
                  Contribute on GitHub
                </Link>
              </li>
              <li style={{ marginBottom: '0.5rem' }}>
                <Link
                  href="https://discord.gg/physical-ai"
                  style={{
                    color: '#e0f0f0',
                    textDecoration: 'none',
                    fontSize: '0.9rem',
                    transition: 'color 0.3s ease'
                  }}
                  onMouseOver={(e) => e.target.style.color = '#b0e0e6'}
                  onMouseOut={(e) => e.target.style.color = '#e0f0f0'}
                >
                  Discord Community
                </Link>
              </li>
            </ul>
          </div>

          {/* Connect Section */}
          <div className="col col--3">
            <h4 style={{
              fontFamily: 'Sora, sans-serif',
              color: '#b0e0e6',
              marginBottom: '1rem',
              fontSize: '1.1rem',
              textTransform: 'uppercase',
              letterSpacing: '1px'
            }}>
              Connect
            </h4>
            <div style={{ marginBottom: '1rem' }}>
              <p style={{
                color: '#e0f0f0',
                fontSize: '0.9rem',
                lineHeight: '1.6',
                marginBottom: '1rem'
              }}>
                Access supplementary materials, code examples, and updates for the Physical AI & Humanoid Robotics textbook.
              </p>
              <Link
                href="mailto:textbook@physicalai-humanoid-robotics.org"
                style={{
                  color: '#b0e0e6',
                  textDecoration: 'none',
                  fontSize: '0.9rem',
                  display: 'inline-block',
                  padding: '0.5rem 1rem',
                  border: '1px solid #b0e0e6',
                  borderRadius: '4px',
                  transition: 'all 0.3s ease'
                }}
                onMouseOver={(e) => {
                  e.target.style.backgroundColor = '#b0e0e6';
                  e.target.style.color = '#000';
                }}
                onMouseOut={(e) => {
                  e.target.style.backgroundColor = 'transparent';
                  e.target.style.color = '#b0e0e6';
                }}
              >
                Textbook Support
              </Link>
            </div>
          </div>
        </div>

        <div className="footer__bottom text--center" style={{
          borderTop: '1px solid rgba(176, 224, 230, 0.1)',
          paddingTop: '2rem',
          marginTop: '2rem',
          color: 'rgba(224, 240, 240, 0.7)',
          fontSize: '0.85rem'
        }}>
          {/* Footer Logo and Brand */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            marginBottom: '1.5rem',
          }}>
            <Link to="/" style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              textDecoration: 'none',
              marginBottom: '1rem',
            }}>
              <div style={{
                position: 'relative',
                width: '50px',
                height: '50px',
              }}>
                {/* Glow effect */}
                <div style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: '60px',
                  height: '60px',
                  background: 'radial-gradient(circle, rgba(15, 227, 192, 0.2) 0%, transparent 70%)',
                  borderRadius: '50%',
                }}></div>
                <img
                  src="/img/logo.svg"
                  alt="Physical AI Logo"
                  style={{
                    width: '50px',
                    height: '50px',
                    position: 'relative',
                    zIndex: 1,
                  }}
                />
              </div>
              <span style={{
                fontFamily: 'Sora, sans-serif',
                fontSize: '1.2rem',
                fontWeight: 600,
                background: 'linear-gradient(90deg, #0FE3C0, #6366F1, #EC4899)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}>
                Physical AI & Robotics
              </span>
            </Link>
            <p style={{
              color: 'rgba(224, 240, 240, 0.6)',
              fontSize: '0.9rem',
              maxWidth: '400px',
              textAlign: 'center',
              margin: 0,
            }}>
              Master foundational principles and technical frameworks of modern intelligent robotics systems.
            </p>
          </div>

          <div className="footer__copyright" style={{
            fontFamily: 'Sora, sans-serif',
            textTransform: 'uppercase',
            letterSpacing: '1px',
            marginBottom: '0.5rem',
            fontSize: '0.75rem',
            color: 'rgba(224, 240, 240, 0.5)',
          }}>
            {siteConfig.title}
          </div>
          <div style={{ color: 'rgba(224, 240, 240, 0.4)' }}>
            {`Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`}
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;