import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';
import { motion } from 'framer-motion';

type FeatureItem = {
  title: string;
  description: React.ReactNode;
  imageUrl?: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Physics-Based Foundation',
    description: (
      <>
        Realistic humanoid robot behaviors grounded in accurate physics simulation. All algorithms work with real-world physical constraints, momentum, friction, and dynamics.
      </>
    ),
  },
  {
    title: 'Safety-First Robotics',
    description: (
      <>
        Human safety is paramount in all robotics implementations. Multiple fail-safe mechanisms built into every system, with all robot movements incorporating collision avoidance protocols.
      </>
    ),
  },
  {
    title: 'Real-World Implementations',
    description: (
      <>
        Practical applications bridging the gap between theoretical knowledge and physical implementation. Includes hardware integration, deployment strategies, and safety considerations.
      </>
    ),
  },
  {
    title: 'Advanced AI Integration',
    description: (
      <>
        Cutting-edge artificial intelligence techniques specifically adapted for robotic systems. Machine learning, computer vision, and natural language processing for embodied intelligence.
      </>
    ),
  },
];

function Feature({title, description}: FeatureItem) {
  return (
    <motion.div
      className={clsx('col col--3')}
      whileHover={{ y: -10 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <div className={styles.featureCard}>
        <div className={styles.featureTitle}>{title}</div>
        <p className={styles.featureDescription}>{description}</p>
      </div>
    </motion.div>
  );
}

export default function HomepageFeatures(): React.ReactElement {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}