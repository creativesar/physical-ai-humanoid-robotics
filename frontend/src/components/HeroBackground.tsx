import React, { useEffect, useRef } from 'react';
import styles from './HeroBackground.module.css';
import { motion } from 'framer-motion';

interface HeroBackgroundProps {
  children: React.ReactNode;
}

const HeroBackground: React.FC<HeroBackgroundProps> = ({ children }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  

  // Function to create thunder bolts
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const boltInterval = setInterval(() => {
      if (container.children.length > 10) return; // Limit elements

      const bolt = document.createElement('div');
      bolt.className = styles.thunderBolt;

      // Random horizontal position
      const posX = Math.random() * 100;
      bolt.style.left = `${posX}%`;

      // Random size and animation
      const width = 2 + Math.random() * 4;
      const height = 50 + Math.random() * 100;
      const duration = 2 + Math.random() * 3; // 2-5 seconds
      bolt.style.width = `${width}px`;
      bolt.style.height = `${height}px`;
      bolt.style.animation = `${styles.thunderBoltFlash} ${duration}s ease-in-out infinite`;
      bolt.style.animationDelay = `${Math.random() * 5}s`;

      container.appendChild(bolt);

      // Remove element after animation completes
      setTimeout(() => {
        bolt.remove();
      }, duration * 1000);
    }, 3000); // Create a new thunder bolt every 3 seconds

    return () => clearInterval(boltInterval);
  }, []);

  // Function to create falling galaxy stars
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const starInterval = setInterval(() => {
      if (container.children.length > 50) return; // Limit elements

      const star = document.createElement('div');

      // Randomly assign star class for different types
      const starTypes = [
        styles.galaxyStar,
        styles.starSmall,
        styles.starMedium,
        styles.starLarge,
        styles.starBright,
        styles.starDim,
        styles.starPulsating,
        styles.starColorful,
        styles.starDistant
      ];
      const randomStarType = starTypes[Math.floor(Math.random() * starTypes.length)];
      star.className = randomStarType;

      // Random horizontal position
      const posX = Math.random() * 100;
      star.style.left = `${posX}%`;

      // Random size based on star type
      const baseSize = Math.random() < 0.7 ? 1 : 1.5 + Math.random() * 2; // Most stars are small
      star.style.width = `${baseSize}px`;
      star.style.height = `${baseSize}px`;

      // Random animation duration for falling speed
      const duration = 8 + Math.random() * 15; // 8-23 seconds
      star.style.animation = `${styles.starFall} ${duration}s linear infinite, ${styles.starTwinkle} ${2 + Math.random() * 6}s ease-in-out infinite`;
      star.style.animationDelay = `${Math.random() * 5}s, ${Math.random() * 3}s`;

      // Random initial opacity
      const initialOpacity = 0.2 + Math.random() * 0.6;
      star.style.opacity = `${initialOpacity}`;

      container.appendChild(star);

      // Remove element after some time
      setTimeout(() => {
        star.remove();
      }, duration * 1000);
    }, 400); // Create a new star every 0.4 seconds

    return () => clearInterval(starInterval);
  }, []);

  // Function to create floating particles
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const particleInterval = setInterval(() => {
      if (container.children.length > 20) return; // Limit elements

      const particle = document.createElement('div');
      particle.className = styles.floatingParticle;

      // Random position
      const posX = Math.random() * 100;
      const posY = Math.random() * 100;
      particle.style.left = `${posX}%`;
      particle.style.top = `${posY}%`;

      // Random size
      const size = 1 + Math.random() * 3;
      particle.style.width = `${size}px`;
      particle.style.height = `${size}px`;

      // Random movement variables
      const tx = (Math.random() - 0.5) * 100; // -50 to 50
      const ty = (Math.random() - 0.5) * 100;
      const txEnd = (Math.random() - 0.5) * 200; // -100 to 100
      const tyEnd = (Math.random() - 0.5) * 200;

      particle.style.setProperty('--tx', `${tx}px`);
      particle.style.setProperty('--ty', `${ty}px`);
      particle.style.setProperty('--tx-end', `${txEnd}px`);
      particle.style.setProperty('--ty-end', `${tyEnd}px`);

      // Random animation duration and delay
      const duration = 8 + Math.random() * 10; // 8-18 seconds
      particle.style.animation = `${styles.particleFloat} ${duration}s ease-in-out forwards`;
      particle.style.animationDelay = `${Math.random() * 5}s`;

      container.appendChild(particle);

      // Remove element after animation completes
      setTimeout(() => {
        particle.remove();
      }, duration * 1000);
    }, 800); // Create a new particle every 0.8 seconds

    return () => clearInterval(particleInterval);
  }, []);

  // Function to create gradient orbs
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const orbInterval = setInterval(() => {
      if (container.children.length > 5) return; // Limit elements

      const orb = document.createElement('div');
      orb.className = styles.gradientOrb;

      // Random position
      const posX = Math.random() * 80 + 10; // 10-90% to keep within bounds
      const posY = Math.random() * 80 + 10;
      orb.style.left = `${posX}%`;
      orb.style.top = `${posY}%`;

      // Random size
      const size = 50 + Math.random() * 150; // 50-200px
      orb.style.width = `${size}px`;
      orb.style.height = `${size}px`;

      // Random color
      const colors = [
        'radial-gradient(circle, #4f46e5, #7c3aed)',  // indigo
        'radial-gradient(circle, #ec4899, #f43f5e)',  // pink
        'radial-gradient(circle, #06b6d4, #0ea5e9)',  // cyan
        'radial-gradient(circle, #10b981, #22c55e)',  // emerald
        'radial-gradient(circle, #f59e0b, #f97316)',  // amber
      ];
      const randomColor = colors[Math.floor(Math.random() * colors.length)];
      orb.style.background = randomColor;

      // Random animation duration
      const duration = 10 + Math.random() * 20; // 10-30 seconds
      orb.style.animation = `${styles.orbPulse} ${duration}s ease-in-out infinite`;

      container.appendChild(orb);

      // Remove element after some time
      setTimeout(() => {
        orb.remove();
      }, (duration + 5) * 1000); // Remove after animation duration + buffer
    }, 6000); // Create a new orb every 6 seconds

    return () => clearInterval(orbInterval);
  }, []);

  // Function to create spark effects
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const sparkInterval = setInterval(() => {
      if (container.children.length > 30) return; // Limit elements

      const spark = document.createElement('div');
      spark.className = styles.spark;

      // Random position
      const posX = Math.random() * 100;
      const posY = Math.random() * 100;
      spark.style.left = `${posX}%`;
      spark.style.top = `${posY}%`;

      // Random animation duration and delay
      const duration = 1 + Math.random() * 2; // 1-3 seconds
      spark.style.animation = `${styles.sparkTwinkle} ${duration}s ease-in-out forwards`;
      spark.style.animationDelay = `${Math.random() * 2}s`;

      container.appendChild(spark);

      // Remove element after animation completes
      setTimeout(() => {
        spark.remove();
      }, duration * 1000);
    }, 200); // Create a new spark every 0.2 seconds

    return () => clearInterval(sparkInterval);
  }, []);

  // Function to create blue neon particles
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const particleInterval = setInterval(() => {
      if (container.children.length > 15) return; // Limit elements

      const particle = document.createElement('div');
      particle.className = styles.blueNeonParticle;

      // Random position
      const posX = Math.random() * 100;
      const posY = Math.random() * 100;
      particle.style.left = `${posX}%`;
      particle.style.top = `${posY}%`;

      // Random size
      const size = 0.5 + Math.random() * 1.5;
      particle.style.width = `${size}px`;
      particle.style.height = `${size}px`;

      // Random animation duration
      const duration = 3 + Math.random() * 4; // 3-7 seconds
      particle.style.animation = `${styles.sparkTwinkle} ${duration}s ease-in-out infinite`;
      particle.style.animationDelay = `${Math.random() * 3}s`;

      container.appendChild(particle);

      // Remove element after animation completes
      setTimeout(() => {
        particle.remove();
      }, duration * 1000);
    }, 600); // Create a new particle every 0.6 seconds

    return () => clearInterval(particleInterval);
  }, []);

  // Function to create blue neon orbs
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const orbInterval = setInterval(() => {
      if (container.children.length > 3) return; // Limit elements

      const orb = document.createElement('div');
      orb.className = styles.blueNeonOrb;

      // Random position
      const posX = Math.random() * 80 + 10; // 10-90% to keep within bounds
      const posY = Math.random() * 80 + 10;
      orb.style.left = `${posX}%`;
      orb.style.top = `${posY}%`;

      // Random size
      const size = 60 + Math.random() * 100; // 60-160px
      orb.style.width = `${size}px`;
      orb.style.height = `${size}px`;

      // Random animation duration
      const duration = 15 + Math.random() * 15; // 15-30 seconds
      orb.style.animation = `${styles.orbPulse} ${duration}s ease-in-out infinite`;

      container.appendChild(orb);

      // Remove element after some time
      setTimeout(() => {
        orb.remove();
      }, (duration + 5) * 1000); // Remove after animation duration + buffer
    }, 8000); // Create a new orb every 8 seconds

    return () => clearInterval(orbInterval);
  }, []);

  // Function to create thunder spark effects
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const thunderSparkInterval = setInterval(() => {
      if (container.children.length > 15) return; // Limit elements

      const thunderSpark = document.createElement('div');
      thunderSpark.className = styles.thunderSpark;

      // Random horizontal position
      const posX = Math.random() * 100;
      thunderSpark.style.left = `${posX}%`;

      // Random size and animation
      const width = 1 + Math.random() * 3; // 1-4px width
      const height = 20 + Math.random() * 40; // 20-60px height
      const duration = 1.5 + Math.random() * 1.5; // 1.5-3 seconds
      thunderSpark.style.width = `${width}px`;
      thunderSpark.style.height = `${height}px`;
      thunderSpark.style.animation = `${styles.thunderSparkFlash} ${duration}s ease-in-out forwards`;
      thunderSpark.style.animationDelay = `${Math.random() * 2}s`;

      container.appendChild(thunderSpark);

      // Remove element after animation completes
      setTimeout(() => {
        thunderSpark.remove();
      }, duration * 1000);
    }, 500); // Create a new thunder spark every 0.5 seconds

    return () => clearInterval(thunderSparkInterval);
  }, []);

  // Function to create zigzag thunder sparks
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const zigzagInterval = setInterval(() => {
      if (container.children.length > 10) return; // Limit elements

      const zigzagSpark = document.createElement('div');
      zigzagSpark.className = styles.zigzagSpark;

      // Random position
      const posX = Math.random() * 100;
      const posY = Math.random() * 30; // Start from top 30% of screen
      zigzagSpark.style.left = `${posX}%`;
      zigzagSpark.style.top = `${posY}%`;

      // Random animation
      const duration = 1.2 + Math.random() * 1.8; // 1.2-3 seconds
      zigzagSpark.style.animation = `${styles.zigzagSparkFlash} ${duration}s ease-in-out forwards`;
      zigzagSpark.style.animationDelay = `${Math.random() * 3}s`;

      container.appendChild(zigzagSpark);

      // Remove element after animation completes
      setTimeout(() => {
        zigzagSpark.remove();
      }, duration * 1000);
    }, 600); // Create a new zigzag spark every 0.6 seconds

    return () => clearInterval(zigzagInterval);
  }, []);

  // Function to create electric spark effects
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const electricSparkInterval = setInterval(() => {
      if (container.children.length > 25) return; // Limit elements

      const electricSpark = document.createElement('div');
      electricSpark.className = styles.electricSpark;

      // Random position
      const posX = Math.random() * 100;
      const posY = Math.random() * 100;
      electricSpark.style.left = `${posX}%`;
      electricSpark.style.top = `${posY}%`;

      // Random animation
      const duration = 0.8 + Math.random() * 0.4; // 0.8-1.2 seconds
      electricSpark.style.animation = `${styles.electricSparkFlash} ${duration}s ease-out forwards`;
      electricSpark.style.animationDelay = `${Math.random() * 1.5}s`;

      container.appendChild(electricSpark);

      // Remove element after animation completes
      setTimeout(() => {
        electricSpark.remove();
      }, duration * 1000);
    }, 150); // Create a new electric spark every 0.15 seconds

    return () => clearInterval(electricSparkInterval);
  }, []);

  // Function to create electric arc effects
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const electricArcInterval = setInterval(() => {
      if (container.children.length > 15) return; // Limit elements

      const electricArc = document.createElement('div');
      electricArc.className = styles.electricArc;

      // Random position and angle
      const posX = Math.random() * 100;
      const posY = Math.random() * 100;
      const angle = Math.random() * 360; // Random angle
      electricArc.style.left = `${posX}%`;
      electricArc.style.top = `${posY}%`;
      electricArc.style.transform = `rotate(${angle}deg)`;

      // Random size and animation
      const duration = 0.5 + Math.random() * 0.7; // 0.5-1.2 seconds
      electricArc.style.animation = `${styles.electricArcFlash} ${duration}s ease-out forwards`;
      electricArc.style.animationDelay = `${Math.random() * 2}s`;

      container.appendChild(electricArc);

      // Remove element after animation completes
      setTimeout(() => {
        electricArc.remove();
      }, duration * 1000);
    }, 200); // Create a new electric arc every 0.2 seconds

    return () => clearInterval(electricArcInterval);
  }, []);

  // Function to create horizontal electric belts
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const electricBeltInterval = setInterval(() => {
      if (container.children.length > 8) return; // Limit elements

      const electricBelt = document.createElement('div');
      electricBelt.className = styles.electricBelt;

      // Random vertical position
      const posY = Math.random() * 100;
      electricBelt.style.top = `${posY}%`;

      // Random animation
      const duration = 2 + Math.random() * 3; // 2-5 seconds
      electricBelt.style.animation = `${styles.electricBeltMove} ${duration}s linear forwards`;
      electricBelt.style.animationDelay = `${Math.random() * 4}s`;

      container.appendChild(electricBelt);

      // Remove element after animation completes
      setTimeout(() => {
        electricBelt.remove();
      }, duration * 1000);
    }, 2000); // Create a new electric belt every 2 seconds

    return () => clearInterval(electricBeltInterval);
  }, []);

  // Function to create vertical electric belts
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const electricBeltVerticalInterval = setInterval(() => {
      if (container.children.length > 8) return; // Limit elements

      const electricBeltVertical = document.createElement('div');
      electricBeltVertical.className = styles.electricBeltVertical;

      // Random horizontal position
      const posX = Math.random() * 100;
      electricBeltVertical.style.left = `${posX}%`;

      // Random animation
      const duration = 2 + Math.random() * 3; // 2-5 seconds
      electricBeltVertical.style.animation = `${styles.electricBeltVerticalMove} ${duration}s linear forwards`;
      electricBeltVertical.style.animationDelay = `${Math.random() * 4}s`;

      container.appendChild(electricBeltVertical);

      // Remove element after animation completes
      setTimeout(() => {
        electricBeltVertical.remove();
      }, duration * 1000);
    }, 2500); // Create a new vertical electric belt every 2.5 seconds

    return () => clearInterval(electricBeltVerticalInterval);
  }, []);

  // Function to create white electric arcs
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const whiteArcInterval = setInterval(() => {
      if (container.children.length > 12) return; // Limit elements

      const whiteArc = document.createElement('div');
      whiteArc.className = styles.whiteElectricArc;

      // Random position and angle
      const posX = Math.random() * 100;
      const posY = Math.random() * 100;
      const angle = Math.random() * 360; // Random angle
      whiteArc.style.left = `${posX}%`;
      whiteArc.style.top = `${posY}%`;
      whiteArc.style.transform = `rotate(${angle}deg)`;

      // Random animation
      const duration = 0.8 + Math.random() * 0.7; // 0.8-1.5 seconds
      whiteArc.style.animation = `${styles.whiteElectricArcFlash} ${duration}s ease-out forwards`;
      whiteArc.style.animationDelay = `${Math.random() * 2}s`;

      container.appendChild(whiteArc);

      // Remove element after animation completes
      setTimeout(() => {
        whiteArc.remove();
      }, duration * 1000);
    }, 300); // Create a new white electric arc every 0.3 seconds

    return () => clearInterval(whiteArcInterval);
  }, []);

  // Function to create white zigzag electric arcs
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const whiteZigzagInterval = setInterval(() => {
      if (container.children.length > 10) return; // Limit elements

      const whiteZigzag = document.createElement('div');
      whiteZigzag.className = styles.whiteElectricArcZigzag;

      // Random position
      const posX = Math.random() * 100;
      const posY = Math.random() * 100;
      whiteZigzag.style.left = `${posX}%`;
      whiteZigzag.style.top = `${posY}%`;

      // Random animation
      const duration = 0.6 + Math.random() * 0.9; // 0.6-1.5 seconds
      whiteZigzag.style.animation = `${styles.whiteElectricArcZigzagFlash} ${duration}s ease-out forwards`;
      whiteZigzag.style.animationDelay = `${Math.random() * 2.5}s`;

      container.appendChild(whiteZigzag);

      // Remove element after animation completes
      setTimeout(() => {
        whiteZigzag.remove();
      }, duration * 1000);
    }, 400); // Create a new white zigzag electric arc every 0.4 seconds

    return () => clearInterval(whiteZigzagInterval);
  }, []);

  return (
    <div className={styles.heroContainer} ref={containerRef}>
      {/* Animated gradient background */}
      <div className={styles.animatedGradientBg}></div>

      {/* Particle Effects */}
      <div className={styles.particleEffects}>
        <div className={styles.floatingParticle}></div>
        <div className={styles.floatingParticle}></div>
        <div className={styles.floatingParticle}></div>
        <div className={styles.gradientOrb}></div>
        <div className={styles.gradientOrb}></div>
        <div className={styles.blueNeonOrb}></div>
        <div className={styles.blueNeonParticle}></div>
        <div className={styles.blueNeonParticle}></div>
        <div className={styles.spark}></div>
        <div className={styles.spark}></div>
        <div className={styles.thunderSpark}></div>
        <div className={styles.thunderSpark}></div>
        <div className={styles.zigzagSpark}></div>
        <div className={styles.electricSpark}></div>
        <div className={styles.electricSpark}></div>
        <div className={styles.electricArc}></div>
        <div className={styles.electricBelt}></div>
        <div className={styles.electricBeltVertical}></div>
        <div className={styles.whiteElectricArc}></div>
        <div className={styles.whiteElectricArcZigzag}></div>
      </div>

      {/* Thunder Bolts */}
      <div className={styles.thunderBolts}>
        <div className={styles.thunderBolt}></div>
        <div className={styles.thunderBolt}></div>
        <div className={styles.thunderBolt}></div>
      </div>

      {/* Falling Galaxy Stars */}
      <div className={styles.particleLayer}>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.galaxyStar}></div>
        <div className={styles.shootingStar}></div>
        <div className={styles.shootingStar}></div>
        <div className={styles.shootingStar}></div>
      </div>

      {/* Content layer - children will be displayed here */}
      <div className={styles.content}>
        {children}
      </div>
    </div>
  );
};

export default HeroBackground;