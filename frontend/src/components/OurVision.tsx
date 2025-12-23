import React from 'react';
import { motion } from 'framer-motion';
import Link from '@docusaurus/Link';
import styles from './OurVision.module.css';

interface VisionCardProps {
  number: string;
  title: string;
  description: string;
  delay?: number;
}

function VisionCard({ number, title, description, delay = 0 }: VisionCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 0.7, delay, ease: [0.25, 0.1, 0.25, 1] }}
      className={styles.visionCard}
      whileHover={{ y: -12, scale: 1.02 }}
    >
      <div className={styles.cardGlow} />
      <div className={styles.cardNumber}>{number}</div>
      <h3 className={styles.cardTitle}>{title}</h3>
      <p className={styles.cardDescription}>{description}</p>
      <div className={styles.cardBorder} />
    </motion.div>
  );
}

interface PillarCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  delay?: number;
}

function PillarCard({ icon, title, description, delay = 0 }: PillarCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      whileInView={{ opacity: 1, scale: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6, delay }}
      className={styles.pillarCard}
      whileHover={{ y: -8 }}
    >
      <div className={styles.pillarIcon}>{icon}</div>
      <h4 className={styles.pillarTitle}>{title}</h4>
      <p className={styles.pillarDescription}>{description}</p>
    </motion.div>
  );
}

export default function OurVision() {
  // Generate floating particles
  const particles = Array.from({ length: 20 }, (_, i) => i);

  return (
    <div className={styles.visionPage}>
      {/* Animated background effects */}
      <div className={styles.backgroundWrapper}>
        <div className={styles.gradientOrb1} />
        <div className={styles.gradientOrb2} />
        <div className={styles.gradientOrb3} />
        <div className={styles.meshGrid} />

        {/* Floating particles */}
        {particles.map((i) => (
          <motion.div
            key={i}
            className={styles.particle}
            animate={{
              y: [0, -150 - Math.random() * 100, 0],
              x: [0, Math.sin(i * 0.5) * 80, 0],
              opacity: [0, 0.6, 0],
              scale: [0.5, 1, 0.5],
            }}
            transition={{
              duration: 8 + i * 0.3,
              repeat: Infinity,
              delay: i * 0.3,
              ease: 'easeInOut',
            }}
            style={{
              left: `${5 + i * 4.5}%`,
              top: `${10 + (i % 5) * 18}%`,
            }}
          />
        ))}
      </div>

      {/* Hero Section */}
      <section className={styles.heroSection}>
        <div className={styles.container}>
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className={styles.heroContent}
          >
            <motion.span
              className={styles.badge}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              Our Vision
            </motion.span>

            <motion.h1
              className={styles.heroTitle}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              Pioneering the Future of
              <span className={styles.highlight}> Embodied Intelligence</span>
            </motion.h1>

            <motion.p
              className={styles.heroDescription}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              We envision a world where intelligent machines seamlessly integrate into society,
              augmenting human capabilities and solving humanity's greatest challenges through
              the convergence of artificial intelligence and physical robotics.
            </motion.p>

            <motion.div
              className={styles.heroButtons}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.5 }}
            >
              <Link to="/docs/module-1/" className={styles.primaryBtn}>
                Explore Curriculum
              </Link>
              <Link to="/about" className={styles.secondaryBtn}>
                About Us
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Vision Statement Section */}
      <section className={styles.visionSection}>
        <div className={styles.container}>
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className={styles.sectionHeader}
          >
            <h2 className={styles.sectionTitle}>
              The <span className={styles.highlight}>Future</span> We're Building
            </h2>
            <p className={styles.sectionSubtitle}>
              A comprehensive roadmap for the next generation of embodied AI systems
            </p>
          </motion.div>

          <div className={styles.visionGrid}>
            <VisionCard
              number="01"
              title="Democratizing Advanced Robotics Education"
              description="Making world-class robotics and AI knowledge accessible to learners worldwide, breaking down barriers to entry and fostering a global community of innovators who will shape the future of embodied intelligence."
              delay={0.1}
            />
            <VisionCard
              number="02"
              title="Bridging Theory and Practice"
              description="Creating seamless pathways from theoretical foundations to real-world deployment, ensuring students master both the mathematical rigor of AI algorithms and the engineering excellence required for physical systems."
              delay={0.2}
            />
            <VisionCard
              number="03"
              title="Accelerating Innovation Cycles"
              description="Empowering researchers and engineers with cutting-edge tools, methodologies, and frameworks that dramatically reduce the time from concept to deployment in humanoid robotics and embodied AI systems."
              delay={0.3}
            />
            <VisionCard
              number="04"
              title="Ethical AI Development"
              description="Instilling principles of responsible AI development, ensuring future generations of roboticists prioritize safety, transparency, and human-centered design in every intelligent system they create."
              delay={0.4}
            />
          </div>
        </div>
      </section>

      {/* Core Pillars Section */}
      <section className={styles.pillarsSection}>
        <div className={styles.container}>
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className={styles.sectionHeader}
          >
            <h2 className={styles.sectionTitle}>
              Our <span className={styles.highlight}>Core Pillars</span>
            </h2>
            <p className={styles.sectionSubtitle}>
              The foundational principles guiding our educational mission
            </p>
          </motion.div>

          <div className={styles.pillarsGrid}>
            <PillarCard
              icon={
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 2L2 7l10 5 10-5-10-5z" />
                  <path d="M2 17l10 5 10-5" />
                  <path d="M2 12l10 5 10-5" />
                </svg>
              }
              title="Research Excellence"
              description="Grounded in peer-reviewed research and validated by leading institutions in robotics and AI."
              delay={0.1}
            />
            <PillarCard
              icon={
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M8 14s1.5 2 4 2 4-2 4-2" />
                  <line x1="9" y1="9" x2="9.01" y2="9" />
                  <line x1="15" y1="9" x2="15.01" y2="9" />
                </svg>
              }
              title="Hands-On Learning"
              description="Practical projects and simulations that mirror real-world robotics engineering challenges."
              delay={0.2}
            />
            <PillarCard
              icon={
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                  <circle cx="9" cy="7" r="4" />
                  <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                  <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                </svg>
              }
              title="Global Community"
              description="Building a vibrant ecosystem of learners, educators, and researchers collaborating across borders."
              delay={0.3}
            />
            <PillarCard
              icon={
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </svg>
              }
              title="Continuous Evolution"
              description="Regular updates incorporating the latest breakthroughs in AI, robotics, and embodied intelligence."
              delay={0.4}
            />
          </div>
        </div>
      </section>

      {/* Impact Section */}
      <section className={styles.impactSection}>
        <div className={styles.container}>
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className={styles.impactContent}
          >
            <h2 className={styles.impactTitle}>
              The <span className={styles.highlight}>Impact</span> We Aspire To Create
            </h2>
            <div className={styles.impactGrid}>
              <motion.div
                className={styles.impactCard}
                whileHover={{ scale: 1.05 }}
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.1 }}
              >
                <div className={styles.impactNumber}>10K+</div>
                <div className={styles.impactLabel}>Future Engineers</div>
              </motion.div>
              <motion.div
                className={styles.impactCard}
                whileHover={{ scale: 1.05 }}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                <div className={styles.impactNumber}>100+</div>
                <div className={styles.impactLabel}>Global Institutions</div>
              </motion.div>
              <motion.div
                className={styles.impactCard}
                whileHover={{ scale: 1.05 }}
                initial={{ opacity: 0, x: 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.3 }}
              >
                <div className={styles.impactNumber}>∞</div>
                <div className={styles.impactLabel}>Possibilities Unlocked</div>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className={styles.ctaSection}>
        <div className={styles.container}>
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className={styles.ctaContent}
          >
            <h2 className={styles.ctaTitle}>
              Ready to Shape the Future?
            </h2>
            <p className={styles.ctaDescription}>
              Join us on this transformative journey to master embodied intelligence and
              build the intelligent machines of tomorrow.
            </p>
            <div className={styles.ctaButtons}>
              <Link to="/docs/module-1/" className={styles.ctaPrimary}>
                Start Learning Now
              </Link>
              <Link to="/contact" className={styles.ctaSecondary}>
                Get In Touch
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
