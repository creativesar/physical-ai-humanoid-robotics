import React from 'react';
import { motion } from 'framer-motion';
import styles from './PremiumModules.module.css';
// SVG Icons for each module
const ModuleIcons = {
  module1: (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="#22D3EE" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M2 17L12 22L22 17" stroke="#22D3EE" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M2 12L12 17L22 12" stroke="#22D3EE" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  module2: (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="#22D3EE" strokeWidth="2"/>
      <path d="M12 7V12L15 15" stroke="#22D3EE" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  module3: (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2L13.09 8.26L22 9L13.09 9.74L12 16L10.91 9.74L2 9L10.91 8.26L12 2Z" stroke="#22D3EE" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M19 14L19.5 17L22 18L19.5 19L19 22L18.5 19L16 18L18.5 17L19 14Z" stroke="#22D3EE" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M5 10L5.5 12L7 13L5.5 14L5 16L4.5 14L3 13L4.5 12L5 10Z" stroke="#22D3EE" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  module4: (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M21 15C21 15.5304 20.7893 16.0391 20.4142 16.4142C20.0391 16.7893 19.5304 17 19 17H7L3 21V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H19C19.5304 3 20.0391 3.21071 20.4142 3.58579C20.7893 3.96086 21 4.46957 21 5V15Z" stroke="#22D3EE" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M16 7L14 9L16 11" stroke="#22D3EE" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M8 7L10 9L8 11" stroke="#22D3EE" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  )
};

const modules = [
  {
    id: 'module1',
    number: 'MODULE 1',
    title: 'The Robotic Nervous System (ROS 2)',
    description: 'Master the Robot Operating System 2, the communication framework that powers the next generation of humanoid robots.',
    objectives: [
      'Understand the architecture and core concepts of ROS 2',
      'Create and manage ROS 2 packages and workspaces',
      'Implement communication between robotic components',
      'Design distributed robotic systems using ROS 2\'s middleware'
    ],
    link: '/docs/module-1'
  },
  {
    id: 'module2',
    number: 'MODULE 2',
    title: 'The Digital Twin (Gazebo & Unity)',
    description: 'Build virtual replicas of robots for safe testing and development in advanced simulation environments.',
    objectives: [
      'Set up and configure Gazebo simulation environments',
      'Create realistic robot models and environments',
      'Integrate Unity for advanced visualization',
      'Connect simulation to real robotic systems'
    ],
    link: '/docs/module-2'
  },
  {
    id: 'module3',
    number: 'MODULE 3',
    title: 'The AI-Robot Brain (NVIDIA Isaac™)',
    description: 'Develop intelligent robotic systems using NVIDIA\'s cutting-edge AI and robotics platform.',
    objectives: [
      'Understand the NVIDIA Isaac™ platform architecture',
      'Implement perception systems using NVIDIA\'s AI frameworks',
      'Develop planning and control algorithms',
      'Deploy AI models on NVIDIA hardware platforms'
    ],
    link: '/docs/module-3'
  },
  {
    id: 'module4',
    number: 'MODULE 4',
    title: 'Vision-Language-Action (VLA)',
    description: 'Enable robots to perceive, understand, and interact with the world using advanced multimodal capabilities.',
    objectives: [
      'Understand VLA architectures and applications',
      'Implement vision-language models for perception',
      'Develop action planning systems',
      'Create multimodal systems integrating vision, language, and action'
    ],
    link: '/docs/module-4'
  }
];

const PremiumModules = () => {
  return (
    <section className={styles.section}>
      {/* Background Effects */}
      <div className={styles.backgroundEffects}>
        {/* Subtle Gradient Overlay */}
        <div className={styles.gradientOverlay}></div>
        
        {/* Minimal Grid Lines */}
        <div className={styles.minimalGrid}></div>

        {/* Subtle Particles */}
        <div className={`${styles.subtleParticle} ${styles.particle1}`}></div>
        <div className={`${styles.subtleParticle} ${styles.particle2}`}></div>
        <div className={`${styles.subtleParticle} ${styles.particle3}`}></div>
        <div className={`${styles.subtleParticle} ${styles.particle4}`}></div>
        <div className={`${styles.subtleParticle} ${styles.particle5}`}></div>
      </div>
      <div className={styles.container}>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className={styles.header}
        >
          <h2 className={styles.title}>Advanced Curriculum</h2>
          <p className={styles.subtitle}>
            Master the complete learning path designed to transform you into a world-class Physical AI and Humanoid Robotics expert
          </p>
        </motion.div>

        <div className={styles.modulesGrid}>
          {modules.map((module, index) => (
            <motion.a
              href={module.link}
              key={module.id}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              whileHover={{ y: -10 }}
              className={styles.moduleCard}
            >
              <div className={styles.moduleIcon}>
                {ModuleIcons[module.id as keyof typeof ModuleIcons]}
              </div>

              <div className={styles.moduleNumber}>{module.number}</div>
              <h3 className={styles.moduleTitle}>{module.title}</h3>
              <p className={styles.moduleDescription}>{module.description}</p>
            </motion.a>
          ))}
        </div>
      </div>
    </section>
  );
};

export default PremiumModules;