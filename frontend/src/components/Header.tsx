import React, { useState, useRef, useEffect } from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { motion, AnimatePresence } from 'framer-motion';
import AuthButton from './AuthButton';

const Header = () => {
  const { siteConfig } = useDocusaurusContext();
  const [hoveredLink, setHoveredLink] = useState<string | null>(null);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const mobileMenuRef = useRef<HTMLDivElement>(null);
  const headerRef = useRef<HTMLElement>(null);

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

  // Track mouse position for interactive effects
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (headerRef.current) {
        const rect = headerRef.current.getBoundingClientRect();
        setMousePosition({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
        });
      }
    };

    const header = headerRef.current;
    if (header) {
      header.addEventListener('mousemove', handleMouseMove);
      return () => header.removeEventListener('mousemove', handleMouseMove);
    }
  }, []);

  return (
    <motion.header
      ref={headerRef}
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      style={{
      background: isScrolled
        ? 'linear-gradient(135deg, rgba(10, 10, 10, 0.95) 0%, rgba(5, 5, 5, 0.9) 100%)'
        : 'linear-gradient(135deg, rgba(10, 10, 10, 0.85) 0%, rgba(5, 5, 5, 0.75) 100%)',
      padding: isScrolled ? '0.6rem 1.5rem' : '0.8rem 1.5rem',
      boxShadow: isScrolled
        ? '0 8px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(176, 224, 230, 0.15), 0 0 80px rgba(176, 224, 230, 0.05)'
        : '0 4px 30px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(176, 224, 230, 0.1)',
      position: 'sticky',
      top: 0,
      zIndex: 1000,
      backdropFilter: 'blur(25px) saturate(180%)',
      WebkitBackdropFilter: 'blur(25px) saturate(180%)',
      border: '1px solid rgba(176, 224, 230, 0.2)',
      borderBottom: '1px solid rgba(176, 224, 230, 0.25)',
      transition: 'all 0.4s cubic-bezier(0.23, 1, 0.320, 1)',
      borderRadius: '0 0 24px 24px',
      borderRight: '1px solid rgba(176, 224, 230, 0.18)',
      borderLeft: '1px solid rgba(176, 224, 230, 0.18)',
      overflow: 'hidden',
      position: 'relative' as const,
    }}>
      {/* Animated gradient orb following mouse */}
      <motion.div
        animate={{
          x: mousePosition.x - 150,
          y: mousePosition.y - 150,
        }}
        transition={{ type: 'spring', damping: 30, stiffness: 200 }}
        style={{
          position: 'absolute',
          width: '300px',
          height: '300px',
          background: 'radial-gradient(circle, rgba(176, 224, 230, 0.15) 0%, transparent 70%)',
          borderRadius: '50%',
          pointerEvents: 'none',
          filter: 'blur(40px)',
          zIndex: 0,
        }}
      />

      {/* Floating particles */}
      {[...Array(5)].map((_, i) => (
        <motion.div
          key={i}
          animate={{
            y: [-20, 20, -20],
            x: [0, Math.sin(i) * 10, 0],
            opacity: [0.3, 0.6, 0.3],
          }}
          transition={{
            duration: 3 + i,
            repeat: Infinity,
            delay: i * 0.5,
          }}
          style={{
            position: 'absolute',
            width: '4px',
            height: '4px',
            background: 'rgba(176, 224, 230, 0.6)',
            borderRadius: '50%',
            left: `${20 + i * 20}%`,
            top: `${30 + i * 10}%`,
            pointerEvents: 'none',
            boxShadow: '0 0 10px rgba(176, 224, 230, 0.8)',
            zIndex: 0,
          }}
        />
      ))}

      <div style={{ position: 'relative', zIndex: 1 }}>
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
            onMouseEnter={(e: React.MouseEvent<HTMLAnchorElement>) => {
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

        {/* Navigation Links with Premium Glass Hover Effects */}
        <nav className="header-nav-desktop" style={{
          display: 'flex',
          gap: '1rem',
          alignItems: 'center',
        }}>
          {/* Textbook Link */}
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Link
              to="/docs/module-1/"
              style={{
                color: hoveredLink === 'textbook' ? '#ffffff' : '#e0f0f0',
                textDecoration: 'none',
                fontSize: '1rem',
                fontWeight: 500,
                fontFamily: 'Inter, sans-serif',
                textTransform: 'none',
                letterSpacing: '0.3px',
                position: 'relative',
                padding: '0.6rem 1rem',
                borderRadius: '12px',
                background: hoveredLink === 'textbook'
                  ? 'linear-gradient(135deg, rgba(176, 224, 230, 0.2) 0%, rgba(135, 206, 235, 0.15) 100%)'
                  : 'transparent',
                border: hoveredLink === 'textbook'
                  ? '1px solid rgba(176, 224, 230, 0.3)'
                  : '1px solid rgba(176, 224, 230, 0.1)',
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                display: 'inline-block',
                boxShadow: hoveredLink === 'textbook'
                  ? '0 8px 32px rgba(176, 224, 230, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
                  : 'none',
                transition: 'all 0.4s cubic-bezier(0.23, 1, 0.320, 1)',
              }}
              onMouseEnter={() => setHoveredLink('textbook')}
              onMouseLeave={() => setHoveredLink(null)}
            >
              {/* Glow effect on hover */}
              <motion.span
                animate={{
                  opacity: hoveredLink === 'textbook' ? 1 : 0,
                  scale: hoveredLink === 'textbook' ? 1 : 0.8,
                }}
                style={{
                  position: 'absolute',
                  inset: '-2px',
                  background: 'linear-gradient(135deg, rgba(176, 224, 230, 0.3), rgba(135, 206, 235, 0.2))',
                  borderRadius: '12px',
                  filter: 'blur(8px)',
                  zIndex: -1,
                  pointerEvents: 'none',
                }}
              />
              Textbook
              {/* Animated underline */}
              <motion.span
                animate={{
                  width: hoveredLink === 'textbook' ? '100%' : '0%',
                }}
                transition={{ duration: 0.3 }}
                style={{
                  position: 'absolute',
                  bottom: '4px',
                  left: 0,
                  height: '2px',
                  background: 'linear-gradient(90deg, #b0e0e6, #87ceeb, #b0e0e6)',
                  borderRadius: '2px',
                  boxShadow: '0 0 8px rgba(176, 224, 230, 0.8)',
                }}
              />
            </Link>
          </motion.div>

          {/* GitHub Link */}
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Link
              href="https://github.com/creativesar/physical-ai-humanoid-robotics"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                color: hoveredLink === 'github' ? '#ffffff' : '#e0f0f0',
                textDecoration: 'none',
                fontSize: '1rem',
                fontWeight: 500,
                fontFamily: 'Inter, sans-serif',
                textTransform: 'none',
                letterSpacing: '0.3px',
                position: 'relative',
                padding: '0.6rem 1rem',
                borderRadius: '12px',
                background: hoveredLink === 'github'
                  ? 'linear-gradient(135deg, rgba(176, 224, 230, 0.2) 0%, rgba(135, 206, 235, 0.15) 100%)'
                  : 'transparent',
                border: hoveredLink === 'github'
                  ? '1px solid rgba(176, 224, 230, 0.3)'
                  : '1px solid rgba(176, 224, 230, 0.1)',
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                display: 'inline-block',
                boxShadow: hoveredLink === 'github'
                  ? '0 8px 32px rgba(176, 224, 230, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
                  : 'none',
                transition: 'all 0.4s cubic-bezier(0.23, 1, 0.320, 1)',
              }}
              onMouseEnter={() => setHoveredLink('github')}
              onMouseLeave={() => setHoveredLink(null)}
            >
              <motion.span
                animate={{
                  opacity: hoveredLink === 'github' ? 1 : 0,
                  scale: hoveredLink === 'github' ? 1 : 0.8,
                }}
                style={{
                  position: 'absolute',
                  inset: '-2px',
                  background: 'linear-gradient(135deg, rgba(176, 224, 230, 0.3), rgba(135, 206, 235, 0.2))',
                  borderRadius: '12px',
                  filter: 'blur(8px)',
                  zIndex: -1,
                  pointerEvents: 'none',
                }}
              />
              GitHub
              <motion.span
                animate={{
                  width: hoveredLink === 'github' ? '100%' : '0%',
                }}
                transition={{ duration: 0.3 }}
                style={{
                  position: 'absolute',
                  bottom: '4px',
                  left: 0,
                  height: '2px',
                  background: 'linear-gradient(90deg, #b0e0e6, #87ceeb, #b0e0e6)',
                  borderRadius: '2px',
                  boxShadow: '0 0 8px rgba(176, 224, 230, 0.8)',
                }}
              />
            </Link>
          </motion.div>
          {/* Auth Button */}
          <div style={{
            marginLeft: '1rem',
            padding: '0.4rem',
            borderRadius: '10px',
            background: 'rgba(176, 224, 230, 0.1)',
            border: '1px solid rgba(176, 224, 230, 0.2)',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
            display: 'flex',
            alignItems: 'center',
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
            to="/docs/module-1/"
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
            href="https://github.com/creativesar/physical-ai-humanoid-robotics"
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
            padding: '1.2rem',
            borderRadius: '12px',
            background: 'rgba(176, 224, 230, 0.1)',
            border: '1px solid rgba(176, 224, 230, 0.2)',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
            display: 'flex',
            alignItems: 'center',
          }}>
            <AuthButton />
          </div>
        </div>
      </div>
      </div>
    </motion.header>
  );
};

export default Header;