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
          padding: '18px 42px',
          fontFamily: "'Plus Jakarta Sans', sans-serif",
          fontWeight: 700,
          fontSize: '14px',
          textTransform: 'uppercase',
          letterSpacing: '0.15em',
          textDecoration: 'none',
          color: isPrimary ? '#fff' : '#ffffff',
          background: isPrimary
            ? 'rgba(255, 255, 255, 0.05)'
            : 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(10px)',
          borderRadius: '12px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          overflow: 'hidden',
          zIndex: 2,
          minWidth: '220px',
          transition: 'all 0.4s cubic-bezier(0.23, 1, 0.32, 1)',
        }}
      >
        {/* Shimmer effect */}
        <motion.div
          animate={{
            x: isHovered ? ['-100%', '200%'] : '-100%',
          }}
          transition={{
            duration: 1.2,
            ease: 'easeInOut',
            repeat: Infinity,
            repeatDelay: 3
          }}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '50%',
            height: '100%',
            background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)',
            transform: 'skewX(-20deg)',
          }}
        />

        {/* Icon */}
        {icon && (
          <motion.span
            animate={{
              scale: isHovered ? 1.1 : 1,
              rotate: isHovered ? 5 : 0
            }}
            transition={{ duration: 0.3 }}
            style={{ display: 'flex' }}
          >
            {icon}
          </motion.span>
        )}

        {/* Text */}
        <span style={{ position: 'relative', zIndex: 1, textShadow: '0 2px 10px rgba(0,0,0,0.3)' }}>{children}</span>
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
    hidden: { opacity: 0 },
    visible: {
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
          >
            Physical AI & Humanoid <br /> Robotics Textbook
          </motion.h1>
          <motion.p
            className={clsx('hero__subtitle', styles.heroSubtitle)}
            variants={itemVariants}
          >
            Master the convergence of artificial intelligence and robotics with our comprehensive textbook covering ROS 2, NVIDIA Isaac, Gazebo simulations, and cutting-edge Vision-Language-Action systems.
          </motion.p>

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
              <PremiumButton
                to="/docs/module-1/"
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

              <PremiumButton
                to="/docs/module-1/"
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
          </motion.div>
        </motion.div>
      </HeroBackground>
    </header>
  );
}

export default PremiumHero;