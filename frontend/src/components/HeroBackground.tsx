import React, { useEffect, useRef } from 'react';
import styles from './HeroBackground.module.css';
import { motion } from 'framer-motion';

interface HeroBackgroundProps {
  children: React.ReactNode;
}

const HeroBackground: React.FC<HeroBackgroundProps> = ({ children }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Function to create plasma pulses
  useEffect(() => {
    if (!containerRef.current) return;
    
    const container = containerRef.current;
    const pulseInterval = setInterval(() => {
      if (container.children.length > 20) return; // Limit elements
      
      const pulse = document.createElement('div');
      pulse.className = styles.plasmaPulse;
      
      // Random position
      const posX = Math.random() * 100;
      const posY = Math.random() * 100;
      pulse.style.left = `${posX}%`;
      pulse.style.top = `${posY}%`;
      
      // Random size
      const size = 50 + Math.random() * 150;
      pulse.style.width = `${size}px`;
      pulse.style.height = `${size}px`;
      
      container.appendChild(pulse);
      
      // Remove element after animation completes
      setTimeout(() => {
        pulse.remove();
      }, 3000);
    }, 3000); // Create a new pulse every 3 seconds
    
    return () => clearInterval(pulseInterval);
  }, []);
  
  // Function to create ionized energy drops
  useEffect(() => {
    if (!containerRef.current) return;
    
    const container = containerRef.current;
    const dropInterval = setInterval(() => {
      if (container.children.length > 30) return; // Limit elements
      
      const drop = document.createElement('div');
      drop.className = styles.ionizedDrop;
      
      // Random horizontal position
      const posX = Math.random() * 100;
      drop.style.left = `${posX}%`;
      
      // Random size and animation duration
      const height = 10 + Math.random() * 15;
      const duration = 3 + Math.random() * 4;
      drop.style.height = `${height}px`;
      drop.style.animation = `${styles.ionizedDropFall} ${duration}s linear infinite`;
      drop.style.animationDelay = `${Math.random() * 2}s`;
      
      container.appendChild(drop);
      
      // Remove element after some time
      setTimeout(() => {
        drop.remove();
      }, duration * 1000);
    }, 500); // Create a new drop every 0.5 seconds
    
    return () => clearInterval(dropInterval);
  }, []);

  return (
    <div className={styles.heroContainer} ref={containerRef}>
      {/* Electric Arcs */}
      <div className={styles.electricArcs}>
        <div className={styles.arc}></div>
        <div className={styles.arc}></div>
        <div className={styles.arc}></div>
        <div className={styles.arc}></div>
        <div className={styles.arc}></div>
      </div>
      
      {/* Content layer - children will be displayed here */}
      <div className={styles.content}>
        {children}
      </div>
    </div>
  );
};

export default HeroBackground;