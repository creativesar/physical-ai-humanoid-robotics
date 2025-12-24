import React, { lazy, Suspense } from 'react';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

// OPTIMIZED: Only load hero immediately, lazy load all other sections
import PremiumHero from '../components/PremiumHero';

const LuxuryAboutUs = lazy(() => import('../components/LuxuryAboutUs'));
const WhatWeDoCompact = lazy(() => import('../components/WhatWeDoCompact'));
const TrustedPersons = lazy(() => import('../components/TrustedPersons'));
const PremiumModules = lazy(() => import('../components/PremiumModules'));
const CoreThinking = lazy(() => import('../components/CoreThinking'));
const PremiumCounter = lazy(() => import('../components/PremiumCounter'));
const PremiumGetInTouch = lazy(() => import('../components/PremiumGetInTouch'));

// ============================================
// MAIN HOMEPAGE COMPONENT
// ============================================
export default function Home() {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title="Home"
      description="Master Physical AI & Humanoid Robotics with ROS 2, NVIDIA Isaac, Gazebo, and Vision-Language-Action systems. Comprehensive, hands-on curriculum for the next generation of robotics."
    >
      {/* Premium Hero Section - Load immediately */}
      <PremiumHero />

      {/* OPTIMIZED: Lazy load all sections below the fold */}
      <Suspense fallback={<div style={{ minHeight: '100vh' }} />}>
        <LuxuryAboutUs />
      </Suspense>

      <Suspense fallback={<div style={{ minHeight: '80vh' }} />}>
        <WhatWeDoCompact />
      </Suspense>

      <Suspense fallback={<div style={{ minHeight: '60vh' }} />}>
        <PremiumCounter />
      </Suspense>

      <Suspense fallback={<div style={{ minHeight: '100vh' }} />}>
        <PremiumModules />
      </Suspense>

      <Suspense fallback={<div style={{ minHeight: '80vh' }} />}>
        <CoreThinking />
      </Suspense>

      <Suspense fallback={<div style={{ minHeight: '80vh' }} />}>
        <TrustedPersons />
      </Suspense>

      <Suspense fallback={<div style={{ minHeight: '80vh' }} />}>
        <PremiumGetInTouch />
      </Suspense>
    </Layout>
  );
}