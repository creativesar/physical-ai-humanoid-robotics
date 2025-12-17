import React from 'react';
import clsx from 'clsx';
import styles from './WhatWeDoCompact.module.css';
import { motion } from 'framer-motion';

const WhatWeDoPremium = () => {
  const features = [
    {
      title: "Physical AI Systems",
      description: "Advanced frameworks for sensorimotor intelligence and embodied cognition in robotics.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
        </svg>
      )
    },
    {
      title: "Humanoid Engineering",
      description: "Practical development for locomotion, perception, control, and robot embodiment.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
        </svg>
      )
    },
    {
      title: "AI Learning Tools",
      description: "Interactive textbooks and modules for robotics education and skill development.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
      )
    },
    {
      title: "Research Curriculum",
      description: "University-level programs aligned with cutting-edge robotics research and innovation.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
        </svg>
      )
    }
  ];

  return (
    <section style={{
      padding: '8rem 0',
      background: 'transparent',
      position: 'relative'
    }}>
      <div className="container">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          style={{ textAlign: 'center', marginBottom: '5rem' }}
        >
          <h2 style={{
            fontFamily: 'Sora, sans-serif',
            fontSize: 'clamp(2.5rem, 5vw, 3.5rem)',
            fontWeight: 800,
            background: 'linear-gradient(135deg, #ffffff 0%, #b0e0e6 100%)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            marginBottom: '1.5rem',
            lineHeight: 1.2
          }}>
            What We Do
          </h2>

          <p style={{
            fontFamily: 'Inter, sans-serif',
            color: '#c0c0c0',
            lineHeight: '1.8',
            fontSize: '1.2rem',
            fontWeight: 400,
            maxWidth: '800px',
            margin: '0 auto'
          }}>
            We create comprehensive frameworks that combine theoretical AI with practical robotics applications.
            Our approach bridges academic research with real-world implementation in humanoid robotics.
          </p>
        </motion.div>

        {/* Feature Cards Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: '2rem',
          maxWidth: '1400px',
          margin: '0 auto'
        }}>
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: index * 0.08 }}
              whileHover={{
                y: -8,
                transition: { duration: 0.3 }
              }}
              style={{
                background: 'linear-gradient(145deg, rgba(15, 15, 15, 0.9), rgba(10, 10, 10, 0.8))',
                backdropFilter: 'blur(20px)',
                padding: '2.5rem',
                borderRadius: '20px',
                border: '1px solid rgba(176, 224, 230, 0.15)',
                boxShadow: '0 10px 40px rgba(0, 0, 0, 0.3)',
                position: 'relative',
                overflow: 'hidden',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
            >
              {/* Top accent line */}
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '3px',
                background: 'linear-gradient(90deg, rgba(176, 224, 230, 0.5) 0%, transparent 100%)',
              }} />

              {/* Icon container */}
              <motion.div
                whileHover={{ scale: 1.05, rotate: 3 }}
                transition={{ type: "spring", stiffness: 300 }}
                style={{
                  width: '60px',
                  height: '60px',
                  borderRadius: '16px',
                  background: 'linear-gradient(135deg, rgba(176, 224, 230, 0.2) 0%, rgba(176, 224, 230, 0.05) 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '1.5rem',
                  border: '1px solid rgba(176, 224, 230, 0.2)',
                  color: '#b0e0e6'
                }}
              >
                {feature.icon}
              </motion.div>

              {/* Title */}
              <h3 style={{
                fontFamily: 'Sora, sans-serif',
                margin: '0 0 1rem 0',
                fontSize: '1.4rem',
                fontWeight: 700,
                color: '#ffffff',
                lineHeight: 1.3
              }}>
                {feature.title}
              </h3>

              {/* Description */}
              <p style={{
                fontFamily: 'Inter, sans-serif',
                fontSize: '1rem',
                color: '#b0c0c0',
                margin: 0,
                lineHeight: '1.7'
              }}>
                {feature.description}
              </p>

              {/* Background decoration */}
              <div style={{
                position: 'absolute',
                bottom: '-50px',
                right: '-50px',
                width: '150px',
                height: '150px',
                borderRadius: '50%',
                background: 'radial-gradient(circle, rgba(176, 224, 230, 0.08) 0%, transparent 70%)',
                pointerEvents: 'none'
              }} />
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default WhatWeDoPremium;