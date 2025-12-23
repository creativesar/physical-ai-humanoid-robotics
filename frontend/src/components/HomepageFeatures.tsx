import React, { ReactElement } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './index.module.css';
import HeroSection from './HeroSection'; // Import the new HeroSection component
import WhatWeDo from './WhatWeDoPremium'; // Import the new premium WhatWeDo component

type FeatureItem = {
  title: string;
  description: ReactElement;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'About Us',
    description: (
      <>
        We are a team of robotics researchers and educators dedicated to advancing the field of Physical AI
        and Humanoid Robotics. Our mission is to provide academically rigorous and accessible educational
        resources for students, researchers, and professionals worldwide.
      </>
    ),
  },
  {
    title: 'Our Services',
    description: (
      <>
        Through our platform, we offer structured learning paths, interactive content, expert-reviewed
        modules, AI-powered educational support, and industry connections to advance your robotics knowledge.
      </>
    ),
  },
];

function Feature({title, description}: FeatureItem) {
  return (
    <div className={clsx('col col--12 col--md-6 col--lg-4')}> {/* Responsive: full width on mobile, 6 cols on medium, 4 cols on large */}
      <div className="text--center padding-horiz--md">
        <h3 style={{
          fontFamily: 'Sora, sans-serif',
          fontSize: 'clamp(1.2rem, 3vw, 1.5rem)' // Responsive font size
        }}>{title}</h3>
        <p style={{
          fontSize: 'clamp(0.9rem, 2vw, 1rem)' // Responsive font size
        }}>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactElement {
  return (
    <>
      {/* Hero Section with Plasma Fire Blast effect */}
      <HeroSection />

      {/* Rest of the homepage content starts here */}
      <section className={styles.features}>
        <div className="container padding-vert--lg">
          {/* About Us, Our Services Sections */}
          <div className="row">
            {FeatureList.map((props, idx) => (
              <Feature key={idx} {...props} />
            ))}
          </div>

          {/* What We Do Section - New Premium Component */}
          <WhatWeDo />

          {/* Module Grid Placeholder */}
          <div className="row padding-vert--md" id="module-grid">
            <div className="col col--12">
              <h2 className="module-grid" style={{fontFamily: 'Sora, sans-serif', textAlign: 'center'}}>Textbook Modules</h2>
              <div className="text--center">
                <p>Coming Soon: Explore the complete 19-module curriculum</p>
              </div>
            </div>
          </div>

          {/* Counter Section Placeholder */}
          <div className="row padding-vert--md counter-section" style={{backgroundColor: 'transparent', borderRadius: '8px', margin: '20px 0'}}>
            <div className="col col--6 col--md-3 text--center"> {/* Responsive: 2 cols on mobile, 3 on medium+ */}
              <h3 style={{fontFamily: 'Sora, sans-serif', color: '#25c19f', fontSize: 'clamp(1.5rem, 4vw, 2.5rem)'}}>19</h3>
              <p style={{fontSize: 'clamp(0.8rem, 2vw, 1rem)'}}>Modules</p>
            </div>
            <div className="col col--6 col--md-3 text--center"> {/* Responsive: 2 cols on mobile, 3 on medium+ */}
              <h3 style={{fontFamily: 'Sora, sans-serif', color: '#25c19f', fontSize: 'clamp(1.5rem, 4vw, 2.5rem)'}}>50+</h3>
              <p style={{fontSize: 'clamp(0.8rem, 2vw, 1rem)'}}>Subpages</p>
            </div>
            <div className="col col--6 col--md-3 text--center"> {/* Responsive: 2 cols on mobile, 3 on medium+ */}
              <h3 style={{fontFamily: 'Sora, sans-serif', color: '#25c19f', fontSize: 'clamp(1.5rem, 4vw, 2.5rem)'}}>10k+</h3>
              <p style={{fontSize: 'clamp(0.8rem, 2vw, 1rem)'}}>Students</p>
            </div>
            <div className="col col--6 col--md-3 text--center"> {/* Responsive: 2 cols on mobile, 3 on medium+ */}
              <h3 style={{fontFamily: 'Sora, sans-serif', color: '#25c19f', fontSize: 'clamp(1.5rem, 4vw, 2.5rem)'}}>24/7</h3>
              <p style={{fontSize: 'clamp(0.8rem, 2vw, 1rem)'}}>Support</p>
            </div>
          </div>

          {/* Testimonials Section Placeholder */}
          <div className="row padding-vert--md" id="testimonials">
            <div className="col col--12">
              <h2 className="testimonials" style={{fontFamily: 'Sora, sans-serif', textAlign: 'center', fontSize: 'clamp(1.5rem, 4vw, 2rem)'}}>What Our Users Say</h2>
              <div className="row">
                <div className="col col--12 col--md-4"> {/* Responsive: full width on mobile, 4 cols on medium+ */}
                  <div style={{padding: '20px', border: '1px solid #555', borderRadius: '8px', backgroundColor: 'transparent', margin: '10px 0'}}>
                    <p style={{fontSize: 'clamp(0.9rem, 2vw, 1rem)'}}>"This textbook provided the comprehensive foundation I needed to start my robotics research."</p>
                    <p style={{fontSize: 'clamp(0.7rem, 1.8vw, 0.8rem)'}}><em>- Graduate Student</em></p>
                  </div>
                </div>
                <div className="col col--12 col--md-4"> {/* Responsive: full width on mobile, 4 cols on medium+ */}
                  <div style={{padding: '20px', border: '1px solid #555', borderRadius: '8px', backgroundColor: 'transparent', margin: '10px 0'}}>
                    <p style={{fontSize: 'clamp(0.9rem, 2vw, 1rem)'}}>"The practical examples and theoretical background perfectly complement each other."</p>
                    <p style={{fontSize: 'clamp(0.7rem, 1.8vw, 0.8rem)'}}><em>- Robotics Engineer</em></p>
                  </div>
                </div>
                <div className="col col--12 col--md-4"> {/* Responsive: full width on mobile, 4 cols on medium+ */}
                  <div style={{padding: '20px', border: '1px solid #555', borderRadius: '8px', backgroundColor: 'transparent', margin: '10px 0'}}>
                    <p style={{fontSize: 'clamp(0.9rem, 2vw, 1rem)'}}>"An invaluable resource for anyone serious about Physical AI and Humanoid Robotics."</p>
                    <p style={{fontSize: 'clamp(0.7rem, 1.8vw, 0.8rem)'}}><em>- Professor</em></p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Trusted Partners Placeholder */}
          <div className="row padding-vert--md" id="trusted-partners">
            <div className="col col--12">
              <h2 className="partners" style={{fontFamily: 'Sora, sans-serif', textAlign: 'center'}}>Trusted By</h2>
              <div className="text--center">
                <p>Coming Soon: Academic and Industry Partners</p>
              </div>
            </div>
          </div>

          {/* FAQ Section Placeholder */}
          <div className="row padding-vert--md">
            <div className="col col--12">
              <h2 style={{fontFamily: 'Sora, sans-serif', textAlign: 'center', fontSize: 'clamp(1.5rem, 4vw, 2rem)'}}>Frequently Asked Questions</h2>
              <div style={{padding: '20px', border: '1px solid #555', borderRadius: '8px', backgroundColor: 'transparent', marginBottom: '10px'}}>
                <h3 style={{fontSize: 'clamp(1rem, 2.5vw, 1.2rem)'}}>Who is this textbook for?</h3>
                <p style={{fontSize: 'clamp(0.85rem, 2vw, 1rem)'}}>This textbook is designed for students, researchers, and professionals working in robotics, AI, or related fields who want to understand the principles of Physical AI and Humanoid Robotics.</p>
              </div>
              <div style={{padding: '20px', border: '1px solid #555', borderRadius: '8px', backgroundColor: 'transparent', marginBottom: '10px'}}>
                <h3 style={{fontSize: 'clamp(1rem, 2.5vw, 1.2rem)'}}>Do I need prior robotics experience?</h3>
                <p style={{fontSize: 'clamp(0.85rem, 2vw, 1rem)'}}>While some background in programming and basic robotics concepts is helpful, this textbook is structured to be accessible to motivated learners at various levels.</p>
              </div>
              <div style={{padding: '20px', border: '1px solid #555', borderRadius: '8px', backgroundColor: 'transparent', marginBottom: '10px'}}>
                <h3 style={{fontSize: 'clamp(1rem, 2.5vw, 1.2rem)'}}>How is this textbook different?</h3>
                <p style={{fontSize: 'clamp(0.85rem, 2vw, 1rem)'}}>Unlike traditional robotics textbooks, this resource specifically focuses on the intersection of AI and physical systems, with emphasis on embodied intelligence and humanoid systems.</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}