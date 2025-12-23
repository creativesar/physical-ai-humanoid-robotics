import React from 'react';
import { motion } from 'framer-motion';
import styles from './WhatWeDoCompact.module.css';

const WhatWeDoCompact = () => {
  const coreThinking = [
    {
      title: "Physical AI Systems",
      description: "Advanced frameworks for sensorimotor intelligence and embodied cognition in robotics.",
      icon: "🤖",
      gradient: "linear-gradient(135deg, #22d3ee 0%, #0ea5e9 100%)"
    },
    {
      title: "Humanoid Engineering",
      description: "Practical development for locomotion, perception, control, and robot embodiment.",
      icon: "⚡",
      gradient: "linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%)"
    },
    {
      title: "AI Learning Tools",
      description: "Interactive textbooks and modules for robotics education and skill development.",
      icon: "🧠",
      gradient: "linear-gradient(135deg, #3b82f6 0%, #6366f1 100%)"
    },
    {
      title: "Research Curriculum",
      description: "University-level programs aligned with cutting-edge robotics research and innovation.",
      icon: "📚",
      gradient: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)"
    }
  ];

  // Generate floating particles
  const particles = Array.from({ length: 20 }, (_, i) => i);

  return (
    <section className={styles.section}>
      {/* Holographic Background Effects */}
      <div className={styles.backgroundEffects}>
        {/* Plasma Effect Layers */}
        <div className={styles.plasmaLayer1} />
        <div className={styles.plasmaLayer2} />
        <div className={styles.plasmaLayer3} />
        <div className={styles.plasmaLayer4} />
        <div className={styles.plasmaOverlay} />

        {/* Plasma Waves */}
        <motion.div
          className={styles.plasmaWave1}
          animate={{
            scale: [1, 1.5, 1],
            rotate: [0, 180, 360],
            opacity: [0.2, 0.4, 0.2]
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        <motion.div
          className={styles.plasmaWave2}
          animate={{
            scale: [1, 1.3, 1],
            rotate: [360, 180, 0],
            opacity: [0.15, 0.35, 0.15]
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />

        {/* Animated gradient orbs */}
        <div className={styles.orb1} />
        <div className={styles.orb2} />
        <div className={styles.orb3} />

        {/* Holographic grid */}
        <div className={styles.holoGrid} />

        {/* Floating particles */}
        <div className={styles.particleContainer}>
          {particles.map((i) => (
            <motion.div
              key={i}
              className={styles.particle}
              animate={{
                y: [0, -150 - Math.random() * 100, 0],
                x: [0, Math.sin(i) * 80, 0],
                opacity: [0, 0.6, 0],
                scale: [0.5, 1, 0.5],
              }}
              transition={{
                duration: 6 + i * 0.3,
                repeat: Infinity,
                delay: i * 0.3,
                ease: 'easeInOut',
              }}
              style={{
                left: `${5 + i * 5}%`,
                top: `${10 + (i % 5) * 18}%`,
              }}
            />
          ))}
        </div>

        {/* Scanning lines */}
        <div className={styles.scanLine1} />
        <div className={styles.scanLine2} />
      </div>

      <div className={styles.container}>
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className={styles.header}
        >
          <div className={styles.badge}>
            <span className={styles.badgeDot} />
            OUR APPROACH
          </div>

          <h2 className={styles.title}>
            What We Do
          </h2>

          <p className={styles.subtitle}>
            We create comprehensive frameworks that combine theoretical AI with practical robotics applications.
            Our approach bridges academic research with real-world implementation in humanoid robotics.
          </p>
        </motion.div>

        {/* Unified Content Block */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className={styles.contentBlock}
        >
          {/* Holographic border */}
          <div className={styles.holoBorder} />

          <div className={styles.contentGrid}>
            {coreThinking.map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -30, rotateY: -15 }}
                whileInView={{ opacity: 1, x: 0, rotateY: 0 }}
                viewport={{ once: true }}
                transition={{
                  duration: 0.6,
                  delay: index * 0.15,
                  type: "spring",
                  stiffness: 100
                }}
                whileHover={{
                  scale: 1.02,
                  x: 10,
                  transition: { duration: 0.3 }
                }}
                className={styles.thinkingItem}
              >
                {/* Gradient background on hover */}
                <motion.div
                  className={styles.itemGradientBg}
                  style={{ background: item.gradient }}
                  whileHover={{ opacity: 0.15 }}
                  initial={{ opacity: 0 }}
                  transition={{ duration: 0.3 }}
                />

                <div className={styles.itemIndicator}>
                  <motion.div
                    className={styles.indicatorDot}
                    animate={{
                      boxShadow: [
                        '0 0 15px rgba(34, 211, 238, 0.8)',
                        '0 0 25px rgba(34, 211, 238, 1)',
                        '0 0 15px rgba(34, 211, 238, 0.8)'
                      ]
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      delay: index * 0.3
                    }}
                  />
                  <div className={styles.indicatorLine} />
                </div>

                <div className={styles.itemContent}>
                  <div className={styles.itemHeader}>
                    <motion.span
                      className={styles.itemIcon}
                      whileHover={{
                        scale: 1.2,
                        rotate: 360,
                        transition: { duration: 0.5 }
                      }}
                    >
                      {item.icon}
                    </motion.span>
                    <h3 className={styles.itemTitle}>{item.title}</h3>
                  </div>
                  <p className={styles.itemDescription}>{item.description}</p>

                  {/* Number indicator */}
                  <motion.div
                    className={styles.itemNumber}
                    initial={{ opacity: 0 }}
                    whileHover={{ opacity: 0.3 }}
                  >
                    {String(index + 1).padStart(2, '0')}
                  </motion.div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Bottom accent */}
          <div className={styles.bottomAccent} />
        </motion.div>
      </div>
    </section>
  );
};

export default WhatWeDoCompact;
