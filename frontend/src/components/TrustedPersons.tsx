import React from 'react';
import { motion } from 'framer-motion';
import styles from './TrustedPersons.module.css';

const TrustedPersons = () => {
  const trustedPersons = [
    {
      name: "Dr. Sarah Chen",
      role: "Lead Robotics Researcher",
      bio: "PhD in Artificial Intelligence from MIT, specializing in humanoid robotics and machine learning algorithms."
    },
    {
      name: "Prof. Michael Rodriguez",
      role: "Computer Vision Expert",
      bio: "Former NASA engineer with extensive experience in autonomous systems and sensor fusion technologies."
    },
    {
      name: "Dr. Aisha Patel",
      role: "NLP & VLA Systems Specialist",
      bio: "Researcher at Stanford AI Lab, focusing on vision-language models and multimodal perception systems."
    },
    {
      name: "Eng. James Wu",
      role: "Simulation & Digital Twin Architect",
      bio: "Senior engineer at NVIDIA, leading development of realistic physics engines for robotic simulations."
    }
  ];

  return (
    <section className={styles.section}>
      {/* Cyan/Purple Accent Background Elements */}
      <div className={styles.cyanAccent1}></div>
      <div className={styles.cyanAccent2}></div>
      <div className={styles.purpleAccent}></div>

      {/* Plasma Wave Effects */}
      <div className={`${styles.plasmaWave} ${styles.wave1}`}></div>
      <div className={`${styles.plasmaWave} ${styles.wave2}`}></div>
      <div className={`${styles.plasmaWave} ${styles.wave3}`}></div>

      {/* Energy Field */}
      <div className={styles.energyField}></div>

      <div className={styles.container}>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className={styles.header}
        >
          <h2 className={styles.title}>Trusted Experts</h2>
          <p className={styles.subtitle}>
            Learn from industry leaders and pioneers in Physical AI and Humanoid Robotics
          </p>
        </motion.div>

        <div className={styles.trustedPersonsGrid}>
          {trustedPersons.map((person, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              whileHover={{ y: -8 }}
              className={styles.personCard}
            >
              {/* Electric Border Effect */}
              <div className={styles.electricBorder}></div>

              {/* HUD Corner Brackets */}
              <div className={styles.cornerBracket} style={{ top: '10px', left: '10px' }}></div>
              <div className={styles.cornerBracket} style={{ top: '10px', right: '10px', transform: 'scaleX(-1)' }}></div>
              <div className={styles.cornerBracket} style={{ bottom: '10px', left: '10px', transform: 'scaleY(-1)' }}></div>
              <div className={styles.cornerBracket} style={{ bottom: '10px', right: '10px', transform: 'scale(-1)' }}></div>

              {/* Data Stream Indicator */}
              <div className={styles.dataStream}></div>

              <div className={styles.personAvatar}>👤</div>
              <h3 className={styles.personName}>{person.name}</h3>
              <p className={styles.personRole}>{person.role}</p>
              <p className={styles.personBio}>{person.bio}</p>

              {/* Tech Grid Overlay */}
              <div className={styles.techGrid}></div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TrustedPersons;