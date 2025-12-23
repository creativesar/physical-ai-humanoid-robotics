import React from 'react';
import { motion } from 'framer-motion';
import styles from './CoreThinking.module.css';

const CoreThinking = () => {
  const corePoints = [
    {
      title: 'Progressive Learning Approach',
      description: 'We build understanding from fundamentals to advanced applications, ensuring you develop both technical skills and conceptual clarity.'
    },
    {
      title: 'Integrated Curriculum Design',
      description: 'Each module connects theory, simulation, and real-world application to create a unified learning experience.'
    },
    {
      title: 'Hands-On Implementation',
      description: 'Every concept is reinforced through practical projects that mirror actual challenges in the robotics industry.'
    },
    {
      title: 'Future-Focused Content',
      description: 'Stay ahead with curriculum that anticipates emerging trends and prepares you for next-generation robotics.'
    },
    {
      title: 'Systematic Problem Solving',
      description: 'Develop analytical abilities needed to tackle complex robotics challenges through structured methodologies.'
    }
  ];

  return (
    <section className={styles.section}>
      {/* Plasma Burst Background Effects */}
      <div className={styles.plasmaBurst1}></div>
      <div className={styles.plasmaBurst2}></div>
      <div className={styles.plasmaBurst3}></div>
      
      {/* Plasma Wave Effects */}
      <div className={`${styles.plasmaWave} ${styles.wave1}`}></div>
      <div className={`${styles.plasmaWave} ${styles.wave2}`}></div>
      <div className={`${styles.plasmaWave} ${styles.wave3}`}></div>
      
      {/* Subtle Grain Texture Overlay */}
      <div className={styles.grainOverlay}></div>

      <div className={styles.container}>
        <div className={styles.contentWrapper}>
          <div className={styles.leftColumn}>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className={styles.header}
            >
              <h2 className={styles.title}>Advanced Educational Philosophy</h2>
              <p className={styles.description}>
                Our approach to teaching Physical AI and Humanoid Robotics emphasizes deep understanding, practical application, and systematic thinking to prepare you for the future of robotics.
              </p>
            </motion.div>
          </div>
          
          <div className={styles.rightColumn}>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              <ul className={styles.pointsList}>
                {corePoints.map((point, index) => (
                  <motion.li
                    key={index}
                    initial={{ opacity: 0, y: 15 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.4, delay: index * 0.05 }}
                    whileHover={{ x: 2 }}
                    className={styles.pointItem}
                  >
                    <h3 className={styles.pointTitle}>{point.title}</h3>
                    <p className={styles.pointDescription}>{point.description}</p>
                  </motion.li>
                ))}
              </ul>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default CoreThinking;