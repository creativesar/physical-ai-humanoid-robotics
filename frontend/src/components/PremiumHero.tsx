import React, { useState } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { motion, easeOut } from 'framer-motion';
import HeroBackground from './HeroBackground';
import styles from './PremiumHero.module.css';

// Premium Hero Button Component
function PremiumButton({
  to,
  children,
  variant = 'primary',
  icon
}: {
  to: string;
  children: React.ReactNode;
  variant?: 'primary' | 'secondary';
  icon?: React.ReactNode;
}) {
  const [isHovered, setIsHovered] = useState(false);

  const isPrimary = variant === 'primary';

  return (
    <motion.div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      style={{ position: 'relative' }}
    >
      {/* Outer glow effect */}
      <motion.div
        animate={{
          opacity: isHovered ? 1 : 0,
          scale: isHovered ? 1 : 0.8,
        }}
        transition={{ duration: 0.3 }}
        style={{
          position: 'absolute',
          inset: '-4px',
          background: isPrimary
            ? 'linear-gradient(135deg, #0FE3C0, #6366F1, #EC4899)'
            : 'linear-gradient(135deg, rgba(255,255,255,0.3), rgba(255,255,255,0.1))',
          borderRadius: '16px',
          filter: 'blur(12px)',
          zIndex: 0,
        }}
      />

      {/* Animated border */}
      <div style={{
        position: 'absolute',
        inset: 0,
        borderRadius: '12px',
        padding: '2px',
        background: isPrimary
          ? 'linear-gradient(135deg, #0FE3C0, #6366F1, #EC4899, #0FE3C0)'
          : 'linear-gradient(135deg, rgba(255,255,255,0.5), rgba(255,255,255,0.2), rgba(255,255,255,0.5))',
        backgroundSize: '300% 300%',
        animation: 'borderGlow 4s ease infinite',
        zIndex: 1,
      }}>
        <div style={{
          width: '100%',
          height: '100%',
          background: isPrimary ? '#000' : 'transparent',
          borderRadius: '10px',
        }} />
      </div>

      <Link
        to={to}
        style={{
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '12px',
          padding: '16px 36px',
          fontFamily: 'Sora, sans-serif',
          fontWeight: 700,
          fontSize: '15px',
          textTransform: 'uppercase',
          letterSpacing: '2px',
          textDecoration: 'none',
          color: isPrimary ? '#fff' : '#000',
          background: isPrimary
            ? 'linear-gradient(135deg, rgba(15, 227, 192, 0.15), rgba(99, 102, 241, 0.15))'
            : 'linear-gradient(135deg, #fff, #e0e0e0)',
          borderRadius: '12px',
          border: 'none',
          overflow: 'hidden',
          zIndex: 2,
          minWidth: '200px',
        }}
      >
        {/* Shimmer effect */}
        <motion.div
          animate={{
            x: isHovered ? ['-100%', '200%'] : '-100%',
          }}
          transition={{
            duration: 0.8,
            ease: 'easeInOut',
          }}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '50%',
            height: '100%',
            background: isPrimary
              ? 'linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)'
              : 'linear-gradient(90deg, transparent, rgba(0,0,0,0.1), transparent)',
            transform: 'skewX(-20deg)',
          }}
        />

        {/* Icon */}
        {icon && (
          <motion.span
            animate={{
              x: isHovered ? 3 : 0,
            }}
            transition={{ duration: 0.2 }}
          >
            {icon}
          </motion.span>
        )}

        {/* Text */}
        <span style={{ position: 'relative', zIndex: 1 }}>{children}</span>

        {/* Arrow indicator for primary */}
        {isPrimary && (
          <motion.svg
            animate={{
              x: isHovered ? 5 : 0,
              opacity: isHovered ? 1 : 0.7,
            }}
            transition={{ duration: 0.2 }}
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M5 12h14M12 5l7 7-7 7" />
          </motion.svg>
        )}
      </Link>
    </motion.div>
  );
}

function PremiumHero() {
  const { siteConfig } = useDocusaurusContext();

  // Animation variants for staggered entrance
  const containerVariants = {
    hidden: { opacity: 1 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.6,
        ease: easeOut
      }
    }
  };

  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <HeroBackground>
        <motion.div
          className={styles.textContainer}
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.h1
            className={clsx('hero__title', styles.heroTitle)}
            variants={itemVariants}
            style={{
              fontFamily: 'Sora, sans-serif',
              fontWeight: 800,
              letterSpacing: '-0.03em',
              fontSize: 'clamp(2rem, 6vw, 4rem)', // Responsive font size
              lineHeight: 1.1,
            }}
          >
            {siteConfig.title}
          </motion.h1>
          <motion.p
            className={clsx('hero__subtitle', styles.heroSubtitle)}
            variants={itemVariants}
            style={{
              fontFamily: 'Inter, sans-serif',
              fontSize: 'clamp(1rem, 3vw, 1.4rem)', // Responsive font size
              lineHeight: '1.6',
              color: '#e0f0f0',
              textShadow: '0 0 10px rgba(176, 224, 230, 0.5)',
              maxWidth: '90%', // Increased max width for mobile
              margin: '1.5rem auto',
            }}
          >
            Master the convergence of artificial intelligence and robotics with our comprehensive textbook covering ROS 2, NVIDIA Isaac, Gazebo simulations, and cutting-edge Vision-Language-Action systems.
          </motion.p>

          {/* Premium Buttons Container - Left and Right Layout */}
          <motion.div
            className={styles.buttonContainer}
            variants={itemVariants}
          >
            <div style={{
              display: 'flex',
              flexDirection: 'row',
              gap: '24px',
              flexWrap: 'wrap',
              justifyContent: 'center',
              alignItems: 'center',
              width: '100%',
              maxWidth: '600px',
              margin: '0 auto',
            }}>
              {/* Left Button - Primary CTA */}
              <PremiumButton
                to="/docs/intro"
                variant="primary"
                icon={
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                  </svg>
                }
              >
                Start Learning
              </PremiumButton>

              {/* Right Button - Secondary CTA */}
              <PremiumButton
                to="/docs/module-1/introduction-to-ros2"
                variant="secondary"
                icon={
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                  </svg>
                }
              >
                Begin Module 1
              </PremiumButton>
            </div>

            {/* Animated particles around buttons */}
            <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', overflow: 'hidden' }}>
              {[...Array(6)].map((_, i) => (
                <motion.div
                  key={i}
                  animate={{
                    y: [0, -30, 0],
                    x: [0, Math.sin(i) * 20, 0],
                    opacity: [0, 0.6, 0],
                    scale: [0, 1, 0],
                  }}
                  transition={{
                    duration: 3,
                    repeat: Infinity,
                    delay: i * 0.5,
                    ease: 'easeInOut',
                  }}
                  style={{
                    position: 'absolute',
                    left: `${15 + i * 15}%`,
                    bottom: '0',
                    width: '4px',
                    height: '4px',
                    background: i % 2 === 0 ? '#0FE3C0' : '#6366F1',
                    borderRadius: '50%',
                    boxShadow: `0 0 10px ${i % 2 === 0 ? '#0FE3C0' : '#6366F1'}`,
                  }}
                />
              ))}
            </div>
          </motion.div>
          <motion.div
            className={styles.scrollIndicator}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2, duration: 0.8 }}
            style={{
              position: 'absolute',
              bottom: '30px',
              left: '50%',
              transform: 'translateX(-50%)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              zIndex: 20
            }}
          >
            <motion.div
              animate={{
                y: [0, 10, 0]
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                repeatType: 'loop',
                ease: easeOut
              }}
              style={{
                width: '2px',
                height: '40px',
                background: 'linear-gradient(to bottom, transparent, #b0e0e6, transparent)',
              }}
            />
          </motion.div>
        </motion.div>
      </HeroBackground>
    </header>
  );
}

export default PremiumHero;