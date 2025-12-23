import React from 'react';
import Layout from '@theme/Layout';
import LuxuryAboutUs from '../components/LuxuryAboutUs';

export default function AboutPage() {
  return (
    <Layout
      title="About Us"
      description="Learn about our Physical AI & Humanoid Robotics Textbook project">
      <main>
        <LuxuryAboutUs />
      </main>
    </Layout>
  );
}