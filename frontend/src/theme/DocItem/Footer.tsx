import React from 'react';
import { useDoc } from '@docusaurus/plugin-content-docs/client';
import { motion } from 'framer-motion';

interface NavLink {
  title: string;
  permalink: string;
}

export default function DocItemFooter() {
  const { metadata } = useDoc();
  const { previous, next } = metadata as { previous?: NavLink; next?: NavLink };

  // If no navigation links, don't render anything
  if (!previous && !next) {
    return null;
  }

  return (
    <motion.nav
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="luxury-pagination-nav"
      aria-label="Documentation pages navigation"
      style={{
        display: 'grid',
        gridTemplateColumns: previous && next ? '1fr 1fr' : '1fr',
        gap: '1.5rem',
        marginTop: '4rem',
        padding: '2.5rem 0',
        borderTop: '1px solid rgba(176, 224, 230, 0.2)',
        position: 'relative',
      }}
    >
      {/* Previous Button */}
      {previous && (
        <motion.a
          href={previous.permalink}
          whileHover={{ scale: 1.02, x: -5 }}
          whileTap={{ scale: 0.98 }}
          className="luxury-nav-button luxury-nav-button--prev"
          style={{
            display: 'flex',
            flexDirection: 'column',
            padding: '1.8rem',
            borderRadius: '16px',
            background: 'linear-gradient(135deg, rgba(176, 224, 230, 0.08) 0%, rgba(135, 206, 235, 0.05) 100%)',
            backdropFilter: 'blur(20px) saturate(180%)',
            WebkitBackdropFilter: 'blur(20px) saturate(180%)',
            border: '1px solid rgba(176, 224, 230, 0.2)',
            textDecoration: 'none',
            transition: 'all 0.4s cubic-bezier(0.23, 1, 0.320, 1)',
            position: 'relative',
            overflow: 'hidden',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(176, 224, 230, 0.15) 0%, rgba(135, 206, 235, 0.1) 100%)';
            e.currentTarget.style.borderColor = 'rgba(176, 224, 230, 0.4)';
            e.currentTarget.style.boxShadow = '0 8px 32px rgba(176, 224, 230, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(176, 224, 230, 0.08) 0%, rgba(135, 206, 235, 0.05) 100%)';
            e.currentTarget.style.borderColor = 'rgba(176, 224, 230, 0.2)';
            e.currentTarget.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.1)';
          }}
        >
          {/* Glow effect on hover */}
          <motion.span
            initial={{ opacity: 0 }}
            whileHover={{ opacity: 1 }}
            style={{
              position: 'absolute',
              inset: '-2px',
              background: 'linear-gradient(135deg, rgba(176, 224, 230, 0.2), rgba(135, 206, 235, 0.15))',
              borderRadius: '16px',
              filter: 'blur(8px)',
              zIndex: -1,
              pointerEvents: 'none',
            }}
          />
          <div
            style={{
              fontSize: '0.8rem',
              color: 'rgba(176, 224, 230, 0.8)',
              marginBottom: '0.6rem',
              textTransform: 'uppercase',
              letterSpacing: '1.2px',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.6rem',
            }}
          >
            <span style={{ fontSize: '1.2rem' }}>←</span>
            Previous
          </div>
          <div
            style={{
              fontSize: '1.1rem',
              color: '#e0f0f0',
              fontWeight: 600,
              lineHeight: '1.4',
            }}
          >
            {previous.title}
          </div>
        </motion.a>
      )}

      {/* Next Button */}
      {next && (
        <motion.a
          href={next.permalink}
          whileHover={{ scale: 1.02, x: 5 }}
          whileTap={{ scale: 0.98 }}
          className="luxury-nav-button luxury-nav-button--next"
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'flex-end',
            padding: '1.8rem',
            borderRadius: '16px',
            background: 'linear-gradient(135deg, rgba(176, 224, 230, 0.08) 0%, rgba(135, 206, 235, 0.05) 100%)',
            backdropFilter: 'blur(20px) saturate(180%)',
            WebkitBackdropFilter: 'blur(20px) saturate(180%)',
            border: '1px solid rgba(176, 224, 230, 0.2)',
            textDecoration: 'none',
            transition: 'all 0.4s cubic-bezier(0.23, 1, 0.320, 1)',
            gridColumn: previous ? 'auto' : '1',
            justifySelf: previous ? 'auto' : 'end',
            position: 'relative',
            overflow: 'hidden',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(176, 224, 230, 0.15) 0%, rgba(135, 206, 235, 0.1) 100%)';
            e.currentTarget.style.borderColor = 'rgba(176, 224, 230, 0.4)';
            e.currentTarget.style.boxShadow = '0 8px 32px rgba(176, 224, 230, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(176, 224, 230, 0.08) 0%, rgba(135, 206, 235, 0.05) 100%)';
            e.currentTarget.style.borderColor = 'rgba(176, 224, 230, 0.2)';
            e.currentTarget.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.1)';
          }}
        >
          {/* Glow effect on hover */}
          <motion.span
            initial={{ opacity: 0 }}
            whileHover={{ opacity: 1 }}
            style={{
              position: 'absolute',
              inset: '-2px',
              background: 'linear-gradient(135deg, rgba(176, 224, 230, 0.2), rgba(135, 206, 235, 0.15))',
              borderRadius: '16px',
              filter: 'blur(8px)',
              zIndex: -1,
              pointerEvents: 'none',
            }}
          />
          <div
            style={{
              fontSize: '0.8rem',
              color: 'rgba(176, 224, 230, 0.8)',
              marginBottom: '0.6rem',
              textTransform: 'uppercase',
              letterSpacing: '1.2px',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.6rem',
            }}
          >
            Next
            <span style={{ fontSize: '1.2rem' }}>→</span>
          </div>
          <div
            style={{
              fontSize: '1.1rem',
              color: '#e0f0f0',
              fontWeight: 600,
              lineHeight: '1.4',
              textAlign: 'right',
            }}
          >
            {next.title}
          </div>
        </motion.a>
      )}
    </motion.nav>
  );
}
