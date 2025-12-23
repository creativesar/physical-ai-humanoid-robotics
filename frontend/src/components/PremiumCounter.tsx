import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import styles from './PremiumCounter.module.css';

interface CounterItem {
  value: number;
  label: string;
  description: string;
  prefix?: string;
  suffix?: string;
}

const PremiumCounter = () => {
  const [counters] = useState<CounterItem[]>([
    {
      value: 10,
      label: "Research Modules",
      description: "Comprehensive modules covering the latest advancements in Physical AI and Humanoid Robotics"
    },
    {
      value: 50,
      label: "Hands-on Labs",
      description: "Interactive laboratory exercises designed to reinforce theoretical concepts with practical implementation"
    },
    {
      value: 20,
      label: "Industry Experts",
      description: "Leading researchers and engineers contributing their expertise to our curriculum development"
    },
    {
      value: 100,
      label: "Practical Focus",
      description: "Curriculum emphasizing real-world applications and industry-relevant skills development",
      suffix: "%"
    }
  ]);

  const [animatedValues, setAnimatedValues] = useState<number[]>(Array(counters.length).fill(0));

  useEffect(() => {
    const timers: NodeJS.Timeout[] = [];
    
    counters.forEach((_, index) => {
      const timer = setTimeout(() => {
        let start = 0;
        const end = counters[index].value;
        const duration = 2000; // 2 seconds
        const increment = end / (duration / 16); // 60fps approximation
        
        const animate = () => {
          start += increment;
          if (start >= end) {
            setAnimatedValues(prev => {
              const newValues = [...prev];
              newValues[index] = end;
              return newValues;
            });
          } else {
            setAnimatedValues(prev => {
              const newValues = [...prev];
              newValues[index] = Math.floor(start);
              return newValues;
            });
            requestAnimationFrame(animate);
          }
        };
        
        animate();
      }, index * 300); // Stagger animations
      
      timers.push(timer);
    });
    
    return () => {
      timers.forEach(timer => clearTimeout(timer));
    };
  }, []);

  return (
    <section className={styles.section}>
      {/* Gold Accent Background Elements */}
      <div className={styles.goldAccent1}></div>
      <div className={styles.goldAccent2}></div>
      <div className={styles.goldAccent3}></div>
      
      {/* Plasma Wave Effects */}
      <div className={`${styles.plasmaWave} ${styles.wave1}`}></div>
      <div className={`${styles.plasmaWave} ${styles.wave2}`}></div>
      <div className={`${styles.plasmaWave} ${styles.wave3}`}></div>
      
      {/* Subtle Grain Texture Overlay */}
      <div className={styles.grainOverlay}></div>

      <div className={styles.container}>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className={styles.header}
        >
          <h2 className={styles.title}>Impact & Excellence</h2>
          <p className={styles.subtitle}>
            Our commitment to excellence is reflected in the breadth and depth of our educational offerings
          </p>
        </motion.div>

        <div className={styles.countersGrid}>
          {counters.map((counter, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              whileHover={{ y: -8 }}
              className={styles.counterCard}
            >
              {/* Gold Border Effect */}
              <div className={styles.goldBorder}></div>
              
              {/* Glowing Orbs */}
              <div className={`${styles.glowOrb} ${styles.orb1}`}></div>
              <div className={`${styles.glowOrb} ${styles.orb2}`}></div>
              
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                whileInView={{ scale: 1, opacity: 1 }}
                viewport={{ once: true }}
                transition={{ 
                  type: "spring", 
                  stiffness: 300, 
                  damping: 20,
                  delay: index * 0.1 + 0.3
                }}
                className={styles.counterValue}
              >
                {counter.prefix || ''}{animatedValues[index]}{counter.suffix || ''}
              </motion.div>
              
              <h3 className={styles.counterLabel}>{counter.label}</h3>
              <p className={styles.counterDescription}>{counter.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default PremiumCounter;