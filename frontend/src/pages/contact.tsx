import React from 'react';
import Layout from '@theme/Layout';
import ContactUs from '../components/ContactUs';

export default function Contact(): JSX.Element {
  return (
    <Layout
      title="Contact Us"
      description="Get in touch with our premium consulting services. We're here to help with your business inquiries and partnership opportunities."
    >
      <ContactUs />
    </Layout>
  );
}