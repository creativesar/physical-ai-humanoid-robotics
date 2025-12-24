import React, { useEffect, useRef } from 'react';
import styles from './HeroBackground.module.css';

interface HeroBackgroundProps {
  children: React.ReactNode;
}

const HeroBackground: React.FC<HeroBackgroundProps> = ({ children }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Optimized: Only create stars (reduced from 18 effects to 1)
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    let starCount = 0;
    const maxStars = 15; // Reduced from 50

    const starInterval = setInterval(() => {
      if (starCount >= maxStars) return;

      const star = document.createElement('div');
      star.className = styles.galaxyStar;

      const posX = Math.random() * 100;
      star.style.left = `${posX}%`;
      star.style.width = '1px';
      star.style.height = '1px';

      const duration = 10 + Math.random() * 10; // Slower animation
      star.style.animation = `${styles.starFall} ${duration}s linear infinite`;
      star.style.animationDelay = `${Math.random() * 5}s`;
      star.style.opacity = `${0.3 + Math.random() * 0.4}`;

      container.appendChild(star);
      starCount++;

      setTimeout(() => {
        star.remove();
        starCount--;
      }, duration * 1000);
    }, 1500); // Much slower creation rate (was 400ms)

    return () => clearInterval(starInterval);
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