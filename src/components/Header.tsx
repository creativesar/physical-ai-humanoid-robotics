import React, { useState, useRef, useEffect } from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import AuthButton from './AuthButton';

const Header = () => {
  const { siteConfig } = useDocusaurusContext();
  const [hoveredLink, setHoveredLink] = useState<string | null>(null);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const mobileMenuRef = useRef<HTMLDivElement>(null);

  // Close mobile menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (mobileMenuRef.current && !mobileMenuRef.current.contains(event.target as Node)) {
        setIsMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Handle scroll effect
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header style={{
      background: 'linear-gradient(135deg, rgba(10, 10, 10, 0.3) 0%, rgba(5, 5, 5, 0.2) 100%)',
      padding: '0.8rem 1.5rem',
      boxShadow: '0 4px 30px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(176, 224, 230, 0.1)',
      position: 'sticky',
      top: 0,
      zIndex: 1000,
      backdropFilter: 'blur(20px)',
      WebkitBackdropFilter: 'blur(20px)',
      border: '1px solid rgba(176, 224, 230, 0.18)',
      borderBottom: '1px solid rgba(176, 224, 230, 0.2)',
      transition: 'all 0.4s cubic-bezier(0.23, 1, 0.320, 1)',
      borderRadius: '0 0 20px 20px',
      borderRight: '1px solid rgba(176, 224, 230, 0.15)',
      borderLeft: '1px solid rgba(176, 224, 230, 0.15)',
    }}>
      <div style={{
        maxWidth: '1400px',
        margin: '0 auto',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '1.2rem',
      }}>
        {/* Logo Section with Premium Glass Effects */}
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <Link to="/" style={{
            textDecoration: 'none',
            display: 'flex',
            alignItems: 'center',
            position: 'relative',
            padding: '0.4rem',
            borderRadius: '12px',
            background: 'rgba(176, 224, 230, 0.05)',
            border: '1px solid rgba(176, 224, 230, 0.1)',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
            transition: 'all 0.3s ease',
          }}
          onMouseEnter={(e) => {
            const target = e.currentTarget;
            target.style.background = 'rgba(176, 224, 230, 0.1)';
            target.style.transform = 'translateY(-2px)';
            target.style.boxShadow = '0 5px 20px rgba(176, 224, 230, 0.1)';
          }}
          onMouseLeave={(e) => {
            const target = e.currentTarget;
            target.style.background = 'rgba(176, 224, 230, 0.05)';
            target.style.transform = 'translateY(0)';
            target.style.boxShadow = 'none';
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '36px',
              height: '36px',
              position: 'relative',
              transition: 'all 0.3s ease',
              background: 'rgba(176, 224, 230, 0.1)',
              borderRadius: '10px',
              border: '1px solid rgba(176, 224, 230, 0.15)',
            }}>
              <img
                src="/img/logo.svg"
                alt="Physical AI Logo"
                style={{
                  height: '28px',
                  width: '28px',
                  position: 'relative',
                  zIndex: 2,
                  transition: 'transform 0.3s ease',
                }}
                onMouseEnter={(e) => {
                  const target = e.currentTarget;
                  target.style.transform = 'scale(1.15)';
                }}
                onMouseLeave={(e) => {
                  const target = e.currentTarget;
                  target.style.transform = 'scale(1)';
                }}
              />
            </div>
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              marginLeft: '12px',
            }}>
              <h1 className="header-logo-title" style={{
                color: '#ffffff',
                fontSize: '1.15rem',
                margin: '0 0 2px 0',
                fontFamily: 'Sora, sans-serif',
                fontWeight: 600,
                letterSpacing: '0.4px',
                background: 'linear-gradient(90deg, #b0e0e6, #e0ffff, #b0e0e6)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundSize: '200% auto',
                animation: 'shine 4s linear infinite',
                lineHeight: '1.2',
                textShadow: '0 0 10px rgba(176, 224, 230, 0.2)',
              }}>
                {siteConfig.title}
              </h1>
              <div style={{
                color: 'rgba(224, 240, 240, 0.8)',
                fontSize: '0.65rem',
                fontFamily: 'Inter, sans-serif',
                fontWeight: 400,
                letterSpacing: '0.5px',
                textTransform: 'uppercase',
                marginTop: '1px',
                opacity: 0.8,
              }}>
                Advanced Robotics & AI Education
              </div>
            </div>
          </Link>
        </div>

        {/* Mobile Menu Toggle */}
        <button
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          className="header-menu-toggle"
          style={{
            background: 'rgba(176, 224, 230, 0.05)',
            border: '1px solid rgba(176, 224, 230, 0.15)',
            color: '#e0f0f0',
            fontSize: '1.4rem',
            cursor: 'pointer',
            padding: '0.6rem',
            borderRadius: '10px',
            transition: 'all 0.3s ease',
            minWidth: '44px',
            minHeight: '44px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
          }}
          onMouseEnter={(e) => {
            const target = e.currentTarget;
            target.style.background = 'rgba(176, 224, 230, 0.15)';
            target.style.borderColor = 'rgba(176, 224, 230, 0.25)';
            target.style.transform = 'scale(1.05)';
            target.style.boxShadow = '0 0 15px rgba(176, 224, 230, 0.1)';
          }}
          onMouseLeave={(e) => {
            const target = e.currentTarget;
            target.style.background = 'rgba(176, 224, 230, 0.05)';
            target.style.borderColor = 'rgba(176, 224, 230, 0.15)';
            target.style.transform = 'scale(1)';
            target.style.boxShadow = 'none';
          }}
          aria-label={isMenuOpen ? "Close menu" : "Open menu"}
        >
          {isMenuOpen ? '✕' : '☰'}
        </button>

        {/* Navigation Links with Premium Glass Hover Effects */}
        <nav className="header-nav-desktop" style={{
          display: 'flex',
          gap: '1.5rem',
          alignItems: 'center',
        }}>
          <Link
            to="/docs/intro"
            style={{
              color: hoveredLink === 'textbook' ? '#b0e0e6' : '#e0f0f0',
              textDecoration: 'none',
              fontSize: '1rem',
              fontWeight: 500,
              fontFamily: 'Inter, sans-serif',
              textTransform: 'none',
              letterSpacing: '0.3px',
              transition: 'all 0.3s cubic-bezier(0.23, 1, 0.320, 1)',
              position: 'relative',
              padding: '0.5rem 0.8rem',
              borderRadius: '8px',
              background: hoveredLink === 'textbook' ? 'rgba(176, 224, 230, 0.1)' : 'transparent',
              border: hoveredLink === 'textbook' ? '1px solid rgba(176, 224, 230, 0.2)' : '1px solid transparent',
              backdropFilter: 'blur(5px)',
              WebkitBackdropFilter: 'blur(5px)',
            }}
            onMouseEnter={() => setHoveredLink('textbook')}
            onMouseLeave={() => setHoveredLink(null)}
          >
            Textbook
            <span style={{
              content: '',
              position: 'absolute',
              bottom: '2px',
              left: 0,
              width: hoveredLink === 'textbook' ? '100%' : '0',
              height: '2px',
              background: 'linear-gradient(90deg, #b0e0e6, #87ceeb)',
              transition: 'width 0.4s cubic-bezier(0.23, 1, 0.320, 1)',
              borderRadius: '1px',
            }} />
          </Link>
          <Link
            to="/modules"
            style={{
              color: hoveredLink === 'modules' ? '#b0e0e6' : '#e0f0f0',
              textDecoration: 'none',
              fontSize: '1rem',
              fontWeight: 500,
              fontFamily: 'Inter, sans-serif',
              textTransform: 'none',
              letterSpacing: '0.3px',
              transition: 'all 0.3s cubic-bezier(0.23, 1, 0.320, 1)',
              position: 'relative',
              padding: '0.5rem 0.8rem',
              borderRadius: '8px',
              background: hoveredLink === 'modules' ? 'rgba(176, 224, 230, 0.1)' : 'transparent',
              border: hoveredLink === 'modules' ? '1px solid rgba(176, 224, 230, 0.2)' : '1px solid transparent',
              backdropFilter: 'blur(5px)',
              WebkitBackdropFilter: 'blur(5px)',
            }}
            onMouseEnter={() => setHoveredLink('modules')}
            onMouseLeave={() => setHoveredLink(null)}
          >
            Modules
            <span style={{
              content: '',
              position: 'absolute',
              bottom: '2px',
              left: 0,
              width: hoveredLink === 'modules' ? '100%' : '0',
              height: '2px',
              background: 'linear-gradient(90deg, #b0e0e6, #87ceeb)',
              transition: 'width 0.4s cubic-bezier(0.23, 1, 0.320, 1)',
              borderRadius: '1px',
            }} />
          </Link>


          <Link
            href="https://github.com/creativesar/Physical-AI-Humanoid-Robotics-Textbook"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: hoveredLink === 'github' ? '#b0e0e6' : '#e0f0f0',
              textDecoration: 'none',
              fontSize: '1rem',
              fontWeight: 500,
              fontFamily: 'Inter, sans-serif',
              textTransform: 'none',
              letterSpacing: '0.3px',
              transition: 'all 0.3s cubic-bezier(0.23, 1, 0.320, 1)',
              position: 'relative',
              padding: '0.5rem 0.8rem',
              borderRadius: '8px',
              background: hoveredLink === 'github' ? 'rgba(176, 224, 230, 0.1)' : 'transparent',
              border: hoveredLink === 'github' ? '1px solid rgba(176, 224, 230, 0.2)' : '1px solid transparent',
              backdropFilter: 'blur(5px)',
              WebkitBackdropFilter: 'blur(5px)',
            }}
            onMouseEnter={() => setHoveredLink('github')}
            onMouseLeave={() => setHoveredLink(null)}
          >
            GitHub
            <span style={{
              content: '',
              position: 'absolute',
              bottom: '2px',
              left: 0,
              width: hoveredLink === 'github' ? '100%' : '0',
              height: '2px',
              background: 'linear-gradient(90deg, #b0e0e6, #87ceeb)',
              transition: 'width 0.4s cubic-bezier(0.23, 1, 0.320, 1)',
              borderRadius: '1px',
            }} />
          </Link>
          {/* Auth Button */}
          <div style={{
            marginLeft: '1rem',
            padding: '0.2rem',
            borderRadius: '8px',
            background: 'rgba(176, 224, 230, 0.05)',
            border: '1px solid rgba(176, 224, 230, 0.1)',
            backdropFilter: 'blur(5px)',
            WebkitBackdropFilter: 'blur(5px)',
          }}>
            <AuthButton />
          </div>
        </nav>
      </div>

      {/* Mobile Menu */}
      <div
        ref={mobileMenuRef}
        className="header-mobile-menu"
        style={{
          display: isMenuOpen ? 'block' : 'none',
          position: 'fixed',
          top: '65px',
          left: 0,
          right: 0,
          background: 'rgba(10, 10, 10, 0.3)',
          padding: '1.6rem 1.2rem',
          boxShadow: '0 15px 50px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(176, 224, 230, 0.1)',
          backdropFilter: 'blur(25px)',
          WebkitBackdropFilter: 'blur(25px)',
          border: '1px solid rgba(176, 224, 230, 0.18)',
          borderTop: '1px solid rgba(176, 224, 230, 0.2)',
          zIndex: 999,
          maxHeight: 'calc(100vh - 65px)',
          overflowY: 'auto',
          borderRadius: '0 0 20px 20px',
          borderRight: '1px solid rgba(176, 224, 230, 0.15)',
          borderLeft: '1px solid rgba(176, 224, 230, 0.15)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '0.8rem',
        }}>
          <Link
            to="/docs/intro"
            style={{
              color: '#e0f0f0',
              textDecoration: 'none',
              fontSize: '1.15rem',
              fontWeight: 500,
              fontFamily: 'Inter, sans-serif',
              padding: '1.1rem',
              borderRadius: '12px',
              background: 'rgba(176, 224, 230, 0.08)',
              transition: 'all 0.3s ease',
              border: '1px solid rgba(176, 224, 230, 0.12)',
              backdropFilter: 'blur(10px)',
              WebkitBackdropFilter: 'blur(10px)',
            }}
            onClick={() => setIsMenuOpen(false)}
            onMouseEnter={(e) => {
              const target = e.currentTarget;
              target.style.background = 'rgba(176, 224, 230, 0.15)';
              target.style.transform = 'translateX(4px)';
              target.style.boxShadow = '0 0 20px rgba(176, 224, 230, 0.1)';
            }}
            onMouseLeave={(e) => {
              const target = e.currentTarget;
              target.style.background = 'rgba(176, 224, 230, 0.08)';
              target.style.transform = 'translateX(0)';
              target.style.boxShadow = 'none';
            }}
          >
            Textbook
          </Link>
          <Link
            to="/modules"
            style={{
              color: '#e0f0f0',
              textDecoration: 'none',
              fontSize: '1.15rem',
              fontWeight: 500,
              fontFamily: 'Inter, sans-serif',
              padding: '1.1rem',
              borderRadius: '12px',
              background: 'rgba(176, 224, 230, 0.08)',
              transition: 'all 0.3s ease',
              border: '1px solid rgba(176, 224, 230, 0.12)',
              backdropFilter: 'blur(10px)',
              WebkitBackdropFilter: 'blur(10px)',
            }}
            onClick={() => setIsMenuOpen(false)}
            onMouseEnter={(e) => {
              const target = e.currentTarget;
              target.style.background = 'rgba(176, 224, 230, 0.15)';
              target.style.transform = 'translateX(4px)';
              target.style.boxShadow = '0 0 20px rgba(176, 224, 230, 0.1)';
            }}
            onMouseLeave={(e) => {
              const target = e.currentTarget;
              target.style.background = 'rgba(176, 224, 230, 0.08)';
              target.style.transform = 'translateX(0)';
              target.style.boxShadow = 'none';
            }}
          >
            Modules
          </Link>
          <Link
            href="https://github.com/creativesar/Physical-AI-Humanoid-Robotics-Textbook"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: '#e0f0f0',
              textDecoration: 'none',
              fontSize: '1.15rem',
              fontWeight: 500,
              fontFamily: 'Inter, sans-serif',
              padding: '1.1rem',
              borderRadius: '12px',
              background: 'rgba(176, 224, 230, 0.08)',
              transition: 'all 0.3s ease',
              border: '1px solid rgba(176, 224, 230, 0.12)',
              backdropFilter: 'blur(10px)',
              WebkitBackdropFilter: 'blur(10px)',
            }}
            onClick={() => setIsMenuOpen(false)}
            onMouseEnter={(e) => {
              const target = e.currentTarget;
              target.style.background = 'rgba(176, 224, 230, 0.15)';
              target.style.transform = 'translateX(4px)';
              target.style.boxShadow = '0 0 20px rgba(176, 224, 230, 0.1)';
            }}
            onMouseLeave={(e) => {
              const target = e.currentTarget;
              target.style.background = 'rgba(176, 224, 230, 0.08)';
              target.style.transform = 'translateX(0)';
              target.style.boxShadow = 'none';
            }}
          >
            GitHub
          </Link>
          {/* Auth Button in Mobile Menu */}
          <div style={{
            padding: '1.1rem',
            borderRadius: '12px',
            background: 'rgba(176, 224, 230, 0.08)',
            border: '1px solid rgba(176, 224, 230, 0.12)',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
          }}>
            <AuthButton />
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;