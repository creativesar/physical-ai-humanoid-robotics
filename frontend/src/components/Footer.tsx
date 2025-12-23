import React, { useState } from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { motion } from 'framer-motion';

const Footer = () => {
  const { siteConfig } = useDocusaurusContext();
  const [hoveredLink, setHoveredLink] = useState<string | null>(null);

  const FooterLink = ({ to, href, children, id }: { to?: string; href?: string; children: React.ReactNode; id: string }) => (
    <motion.li
      style={{ marginBottom: '0.75rem', listStyle: 'none' }}
      whileHover={{ x: 5 }}
      transition={{ type: 'spring', stiffness: 300 }}
    >
      <Link
        to={to}
        href={href}
        style={{
          color: hoveredLink === id ? '#ffffff' : 'rgba(224, 240, 240, 0.9)',
          textDecoration: 'none',
          fontSize: '0.95rem',
          transition: 'all 0.3s ease',
          position: 'relative',
          display: 'inline-block',
          fontWeight: 400,
        }}
        onMouseEnter={() => setHoveredLink(id)}
        onMouseLeave={() => setHoveredLink(null)}
      >
        <span style={{ position: 'relative' }}>
          {/* Underline glow effect */}
          {hoveredLink === id && (
            <motion.span
              layoutId="footer-underline"
              style={{
                position: 'absolute',
                bottom: '-2px',
                left: 0,
                right: 0,
                height: '2px',
                background: 'linear-gradient(90deg, #b0e0e6, #87ceeb)',
                boxShadow: '0 0 8px rgba(176, 224, 230, 0.8)',
                borderRadius: '2px',
              }}
            />
          )}
          {children}
        </span>
      </Link>
    </motion.li>
  );

  return (
    <motion.footer
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
      className="footer"
      style={{
        background: 'linear-gradient(180deg, rgba(10, 10, 10, 0.95) 0%, rgba(0, 0, 0, 1) 100%)',
        borderTop: '1px solid rgba(176, 224, 230, 0.25)',
        color: '#e0f0f0',
        padding: '4rem 0 2rem',
        fontFamily: 'Inter, sans-serif',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Animated gradient background orbs */}
      <motion.div
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.1, 0.2, 0.1],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        style={{
          position: 'absolute',
          top: '-200px',
          right: '10%',
          width: '400px',
          height: '400px',
          background: 'radial-gradient(circle, rgba(176, 224, 230, 0.15) 0%, transparent 70%)',
          borderRadius: '50%',
          filter: 'blur(60px)',
          pointerEvents: 'none',
        }}
      />

      <motion.div
        animate={{
          scale: [1, 1.3, 1],
          opacity: [0.08, 0.15, 0.08],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: 'easeInOut',
          delay: 2,
        }}
        style={{
          position: 'absolute',
          bottom: '-150px',
          left: '15%',
          width: '350px',
          height: '350px',
          background: 'radial-gradient(circle, rgba(135, 206, 235, 0.12) 0%, transparent 70%)',
          borderRadius: '50%',
          filter: 'blur(50px)',
          pointerEvents: 'none',
        }}
      />

      {/* Floating particles */}
      {[...Array(8)].map((_, i) => (
        <motion.div
          key={i}
          animate={{
            y: [-30, 30, -30],
            x: [0, Math.sin(i) * 15, 0],
            opacity: [0.2, 0.5, 0.2],
          }}
          transition={{
            duration: 4 + i,
            repeat: Infinity,
            delay: i * 0.7,
          }}
          style={{
            position: 'absolute',
            width: '3px',
            height: '3px',
            background: 'rgba(176, 224, 230, 0.5)',
            borderRadius: '50%',
            left: `${10 + i * 12}%`,
            top: `${20 + (i % 4) * 20}%`,
            pointerEvents: 'none',
            boxShadow: '0 0 8px rgba(176, 224, 230, 0.6)',
          }}
        />
      ))}

      <div className="container container--fluid padding-horiz--md" style={{ position: 'relative', zIndex: 1 }}>
        <div className="row">
          {/* Curriculum Section */}
          <motion.div
            className="col col--3"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <h4 style={{
              fontFamily: 'Sora, sans-serif',
              color: '#b0e0e6',
              marginBottom: '1.5rem',
              fontSize: '1.15rem',
              textTransform: 'uppercase',
              letterSpacing: '1.5px',
              fontWeight: 600,
              background: 'linear-gradient(90deg, #b0e0e6, #87ceeb)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              textShadow: '0 0 20px rgba(176, 224, 230, 0.3)',
            }}>
              Curriculum
            </h4>
            <ul style={{ padding: 0 }}>
              <FooterLink to="/docs/module-1/" id="module1">Module 1: ROS 2 Foundations</FooterLink>
              <FooterLink to="/docs/module-2/" id="module2">Module 2: Digital Twin</FooterLink>
              <FooterLink to="/docs/module-3/" id="module3">Module 3: AI-Robot Brain</FooterLink>
              <FooterLink to="/docs/module-4/" id="module4">Module 4: Vision-Language-Action</FooterLink>
            </ul>
          </motion.div>

          {/* Resources Section */}
          <motion.div
            className="col col--3"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <h4 style={{
              fontFamily: 'Sora, sans-serif',
              color: '#b0e0e6',
              marginBottom: '1.5rem',
              fontSize: '1.15rem',
              textTransform: 'uppercase',
              letterSpacing: '1.5px',
              fontWeight: 600,
              background: 'linear-gradient(90deg, #b0e0e6, #87ceeb)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              textShadow: '0 0 20px rgba(176, 224, 230, 0.3)',
            }}>
              Resources
            </h4>
            <ul style={{ padding: 0 }}>
              <FooterLink to="/docs/module-1/" id="getting-started">Getting Started</FooterLink>
              <FooterLink to="/docs/module-1/" id="curriculum">Complete Curriculum</FooterLink>
              <FooterLink to="/docs/conclusion" id="conclusion">Conclusion</FooterLink>
              <FooterLink to="/docs/hardware-requirements/" id="hardware">Hardware Requirements</FooterLink>
            </ul>
          </motion.div>

          {/* Community Section */}
          <motion.div
            className="col col--3"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <h4 style={{
              fontFamily: 'Sora, sans-serif',
              color: '#b0e0e6',
              marginBottom: '1.5rem',
              fontSize: '1.15rem',
              textTransform: 'uppercase',
              letterSpacing: '1.5px',
              fontWeight: 600,
              background: 'linear-gradient(90deg, #b0e0e6, #87ceeb)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              textShadow: '0 0 20px rgba(176, 224, 230, 0.3)',
            }}>
              Community
            </h4>
            <ul style={{ padding: 0 }}>
              <FooterLink href="https://github.com/creativesar/physical-ai-humanoid-robotics/discussions" id="discussions">Discussion Forum</FooterLink>
              <FooterLink href="https://github.com/creativesar/physical-ai-humanoid-robotics" id="github">Contribute on GitHub</FooterLink>
              <FooterLink href="https://discord.gg/physical-ai" id="discord">Discord Community</FooterLink>
            </ul>
          </motion.div>

          {/* Connect Section */}
          <motion.div
            className="col col--3"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <h4 style={{
              fontFamily: 'Sora, sans-serif',
              color: '#b0e0e6',
              marginBottom: '1.5rem',
              fontSize: '1.15rem',
              textTransform: 'uppercase',
              letterSpacing: '1.5px',
              fontWeight: 600,
              background: 'linear-gradient(90deg, #b0e0e6, #87ceeb)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              textShadow: '0 0 20px rgba(176, 224, 230, 0.3)',
            }}>
              Connect
            </h4>
            <p style={{
              color: 'rgba(224, 240, 240, 0.8)',
              fontSize: '0.95rem',
              lineHeight: '1.7',
              marginBottom: '1.5rem',
            }}>
              Access supplementary materials, code examples, and updates for the Physical AI & Humanoid Robotics textbook.
            </p>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link
                href="mailto:textbook@physicalai-humanoid-robotics.org"
                style={{
                  color: '#ffffff',
                  textDecoration: 'none',
                  fontSize: '0.95rem',
                  display: 'inline-block',
                  padding: '0.75rem 1.5rem',
                  borderRadius: '12px',
                  background: 'linear-gradient(135deg, rgba(176, 224, 230, 0.2) 0%, rgba(135, 206, 235, 0.15) 100%)',
                  border: '1px solid rgba(176, 224, 230, 0.3)',
                  backdropFilter: 'blur(10px)',
                  WebkitBackdropFilter: 'blur(10px)',
                  transition: 'all 0.4s cubic-bezier(0.23, 1, 0.320, 1)',
                  boxShadow: '0 4px 20px rgba(176, 224, 230, 0.15)',
                  fontWeight: 500,
                }}
                onMouseOver={(e: React.MouseEvent<HTMLAnchorElement>) => {
                  (e.currentTarget as HTMLElement).style.background = 'linear-gradient(135deg, rgba(176, 224, 230, 0.3) 0%, rgba(135, 206, 235, 0.25) 100%)';
                  (e.currentTarget as HTMLElement).style.boxShadow = '0 8px 32px rgba(176, 224, 230, 0.25)';
                }}
                onMouseOut={(e: React.MouseEvent<HTMLAnchorElement>) => {
                  (e.currentTarget as HTMLElement).style.background = 'linear-gradient(135deg, rgba(176, 224, 230, 0.2) 0%, rgba(135, 206, 235, 0.15) 100%)';
                  (e.currentTarget as HTMLElement).style.boxShadow = '0 4px 20px rgba(176, 224, 230, 0.15)';
                }}
              >
                Textbook Support
              </Link>
            </motion.div>
          </motion.div>
        </div>

        {/* Bottom Section */}
        <motion.div
          className="footer__bottom text--center"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.5 }}
          style={{
            borderTop: '1px solid rgba(176, 224, 230, 0.2)',
            paddingTop: '2.5rem',
            marginTop: '3rem',
            color: 'rgba(224, 240, 240, 0.7)',
            fontSize: '0.9rem',
          }}
        >
          {/* Logo and Brand */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            marginBottom: '2rem',
          }}>
            <motion.div
              whileHover={{ scale: 1.05, rotate: 5 }}
              transition={{ type: 'spring', stiffness: 300 }}
            >
              <Link to="/" style={{
                display: 'flex',
                alignItems: 'center',
                gap: '15px',
                textDecoration: 'none',
                marginBottom: '1.2rem',
              }}>
                <div style={{
                  position: 'relative',
                  width: '55px',
                  height: '55px',
                }}>
                  {/* Animated glow effect */}
                  <motion.div
                    animate={{
                      scale: [1, 1.2, 1],
                      opacity: [0.3, 0.6, 0.3],
                    }}
                    transition={{
                      duration: 3,
                      repeat: Infinity,
                      ease: 'easeInOut',
                    }}
                    style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      width: '70px',
                      height: '70px',
                      background: 'radial-gradient(circle, rgba(176, 224, 230, 0.4) 0%, transparent 70%)',
                      borderRadius: '50%',
                      filter: 'blur(15px)',
                    }}
                  />
                  <img
                    src="/img/logo.svg"
                    alt="Physical AI Logo"
                    style={{
                      width: '55px',
                      height: '55px',
                      position: 'relative',
                      zIndex: 1,
                    }}
                  />
                </div>
                <span style={{
                  fontFamily: 'Sora, sans-serif',
                  fontSize: '1.3rem',
                  fontWeight: 700,
                  background: 'linear-gradient(90deg, #b0e0e6, #87ceeb, #b0e0e6)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundSize: '200% auto',
                  animation: 'shine 4s linear infinite',
                }}>
                  Physical AI & Robotics
                </span>
              </Link>
            </motion.div>
            <p style={{
              color: 'rgba(224, 240, 240, 0.65)',
              fontSize: '0.95rem',
              maxWidth: '500px',
              textAlign: 'center',
              margin: 0,
              lineHeight: '1.6',
            }}>
              Master foundational principles and technical frameworks of modern intelligent robotics systems.
            </p>
          </div>

          {/* Copyright */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '0.5rem',
          }}>
            <div style={{
              fontFamily: 'Sora, sans-serif',
              textTransform: 'uppercase',
              letterSpacing: '1.5px',
              fontSize: '0.8rem',
              color: 'rgba(224, 240, 240, 0.5)',
              fontWeight: 600,
            }}>
              {siteConfig.title}
            </div>
            <div style={{
              color: 'rgba(224, 240, 240, 0.45)',
              fontSize: '0.85rem',
            }}>
              {`Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`}
            </div>
          </div>
        </motion.div>
      </div>
    </motion.footer>
  );
};

export default Footer;
