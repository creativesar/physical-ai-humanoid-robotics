import React from 'react';
import { motion } from 'framer-motion';
import Link from '@docusaurus/Link';
import styles from './LuxuryAboutUs.module.css';

interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  delay?: number;
}

function FeatureCard({ icon, title, description, delay = 0 }: FeatureCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
      className={styles.featureCard}
      whileHover={{ y: -8 }}
    >
      <div className={styles.featureIcon}>
        {icon}
      </div>
      <h3 className={styles.featureTitle}>{title}</h3>
      <p className={styles.featureDescription}>{description}</p>
    </motion.div>
  );
}

// Premium Button Component
function PremiumButton({
  to,
  children,
  variant = 'primary'
}: {
  to: string;
  children: React.ReactNode;
  variant?: 'primary' | 'secondary';
}) {
  return (
    <Link
      to={to}
      className={variant === 'primary' ? styles.premiumBtn : styles.premiumBtnSecondary}
    >
      {children}
    </Link>
  );
}

export default function LuxuryAboutUs() {
  // Generate particles for background effect
  const particles = Array.from({ length: 12 }, (_, i) => i);

  return (
    <section className={styles.section}>
      {/* Background effects */}
      <div className={styles.backgroundGlow} />
      <div className={styles.backgroundParticles}>
        {particles.map((i) => (
          <motion.div
            key={i}
            className={styles.particle}
            animate={{
              y: [0, -100 - Math.random() * 100, 0],
              x: [0, Math.sin(i) * 50, 0],
              opacity: [0, 0.8, 0],
            }}
            transition={{
              duration: 5 + i,
              repeat: Infinity,
              delay: i * 0.5,
              ease: 'easeInOut',
            }}
            style={{
              left: `${10 + i * 8}%`,
              top: `${20 + (i % 4) * 20}%`,
            }}
          />
        ))}
      </div>

      <div className={styles.container}>
        <div className={styles.contentGrid}>
          {/* Left content - Heading, text, and buttons */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className={styles.leftContent}
          >
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              <span className={styles.badge}>About This Textbook</span>
            </motion.div>

            <motion.h2 
              className={styles.mainHeading}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              About <span className={styles.highlight}>Us</span>
            </motion.h2>
            
            <motion.p
              className={styles.description}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              We are pioneering the future of embodied intelligence through a world-class, research-driven textbook that seamlessly integrates cutting-edge AI theory with real-world humanoid robotics engineering. Our vision is to democratize access to elite-level knowledge, empowering the next generation of innovators to build intelligent machines that learn, reason, and act in the physical world.
            </motion.p>

            <motion.div
              className={styles.statsGrid}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <motion.div
                className={styles.statCard}
                whileHover={{ scale: 1.05, y: -5 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <h3 className={styles.statNumber}>15+</h3>
                <p className={styles.statLabel}>Advanced Modules</p>
              </motion.div>
              <motion.div
                className={styles.statCard}
                whileHover={{ scale: 1.05, y: -5 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <h3 className={styles.statNumber}>100+</h3>
                <p className={styles.statLabel}>Practical Exercises</p>
              </motion.div>
              <motion.div
                className={styles.statCard}
                whileHover={{ scale: 1.05, y: -5 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <h3 className={styles.statNumber}>50+</h3>
                <p className={styles.statLabel}>World-Class Contributors</p>
              </motion.div>
              <motion.div
                className={styles.statCard}
                whileHover={{ scale: 1.05, y: -5 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <h3 className={styles.statNumber}>∞</h3>
                <p className={styles.statLabel}>Innovation Potential</p>
              </motion.div>
            </motion.div>

            <motion.div 
              className={styles.buttonGroup}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <PremiumButton
                to="/docs/module-1/"
                variant="primary"
              >
                Read Textbook
              </PremiumButton>
              
              <PremiumButton
                to="/our-vision"
                variant="secondary"
              >
                Our Vision
              </PremiumButton>
            </motion.div>
          </motion.div>

          {/* Right content - Grid cards */}
          <motion.div 
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className={styles.rightContent}
          >
            <div className={styles.featuresGrid}>
              <FeatureCard
                icon={
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <circle cx="12" cy="12" r="6" />
                    <circle cx="12" cy="12" r="2" />
                  </svg>
                }
                title="Our Mission"
                description="Transforming the landscape of embodied AI education by delivering world-class, research-grade content that merges theoretical rigor with hands-on engineering excellence, making advanced robotics knowledge accessible to all."
                delay={0.1}
              />

              <FeatureCard
                icon={
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                  </svg>
                }
                title="Our Vision"
                description="Shaping a future where machines possess true embodied intelligence—systems that perceive, learn, reason, and execute complex tasks in dynamic real-world environments with human-level adaptability and precision."
                delay={0.2}
              />

              <FeatureCard
                icon={
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                    <circle cx="9" cy="7" r="4" />
                    <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                  </svg>
                }
                title="Who We Are"
                description="A global collective of leading AI researchers, robotics engineers, and visionary educators united at the forefront of embodied intelligence—bridging simulation, theory, and revolutionary real-world deployment."
                delay={0.3}
              />

              <FeatureCard
                icon={
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.04-3.22L6.31 15H5a3 3 0 0 1-3-3V6a3 3 0 0 1 3-3h4.5Z" />
                    <path d="M22 12h-5" />
                    <path d="M18 9l3 3-3 3" />
                  </svg>
                }
                title="Innovation First"
                description="We champion breakthrough methodologies that seamlessly unite cutting-edge theoretical frameworks with battle-tested engineering practices—accelerating the journey from concept to deployment at scale."
                delay={0.4}
              />
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}