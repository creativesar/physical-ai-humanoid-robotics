import React from 'react';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

// Import all components from components folder
import PremiumHero from '../components/PremiumHero';
import LuxuryAboutUs from '../components/LuxuryAboutUs';
import WhatWeDoCompact from '../components/WhatWeDoCompact';
import TrustedPersons from '../components/TrustedPersons';
import PremiumModules from '../components/PremiumModules';
import CoreThinking from '../components/CoreThinking';
import PremiumCounter from '../components/PremiumCounter';
import PremiumGetInTouch from '../components/PremiumGetInTouch';

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
      {/* Premium Hero Section with glassmorphism effects */}
      <PremiumHero />

      {/* About Us Section - Premium Luxury Design */}
      <LuxuryAboutUs />

      {/* What We Do - Compact Premium Version */}
      <WhatWeDoCompact />

      {/* Premium Animated Counter */}
      <PremiumCounter />


      {/* Premium Modules Section */}
      <PremiumModules />

      {/* Core Thinking Section */}
      <CoreThinking />

      {/* Trusted Persons & Education Section */}
      <TrustedPersons />

      {/* Get In Touch Section */}
      <PremiumGetInTouch />
    </Layout>
  );
}