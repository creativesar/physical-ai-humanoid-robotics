import React, { useState } from 'react';
import { motion } from 'framer-motion';
import styles from './ContactUs.module.css';

const ContactUs = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Form submission logic would go here
    console.log('Form submitted:', formData);
    alert('Thank you for your message! We will get back to you soon.');
    setFormData({ name: '', email: '', subject: '', message: '' });
  };

  const contactInfo = [
    {
      icon: '🤖',
      title: 'Research Inquiries',
      content: 'research@physical-ai-robotics.org'
    },
    {
      icon: '📍',
      title: 'Research Facility',
      content: 'Advanced Robotics Lab, University Campus, Tech District'
    },
    {
      icon: '🕒',
      title: 'Lab Hours',
      content: 'Monday - Friday: 8:00 AM - 8:00 PM, Saturday: 10:00 AM - 4:00 PM'
    }
  ];

  const socialLinks = [
    { name: 'GitHub', icon: '💻', url: '#' },
    { name: 'ResearchGate', icon: '🔬', url: '#' },
    { name: 'LinkedIn', icon: '💼', url: '#' },
    { name: 'YouTube', icon: '🎥', url: '#' }
  ];

  return (
    <section className={styles.section}>
      {/* Tech Accent Background Elements */}
      <div className={styles.techAccent1}></div>
      <div className={styles.techAccent2}></div>
      <div className={styles.techAccent3}></div>
      
      {/* Subtle Grain Texture Overlay */}
      <div className={styles.grainOverlay}></div>

      <div className={styles.container}>
        <div className={styles.contentWrapper}>
          <div className={styles.leftColumn}>
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.7 }}
              className={styles.header}
            >
              <h1 className={styles.title}>Get In Touch</h1>
              <p className={styles.subtitle}>
                Ready to explore the future of Physical AI and Humanoid Robotics?
              </p>
              <p className={styles.description}>
                Whether you're a researcher, student, or industry professional interested in our work, we'd love to hear from you. Reach out for collaboration opportunities, research inquiries, or questions about our educational resources.
              </p>
            </motion.div>

            <motion.ul
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.7, delay: 0.2 }}
              className={styles.contactInfo}
            >
              {contactInfo.map((info, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  whileHover={{ x: 5 }}
                  className={styles.infoItem}
                >
                  <span className={styles.infoIcon}>{info.icon}</span>
                  <div className={styles.infoContent}>
                    <h3>{info.title}</h3>
                    <p>{info.content}</p>
                  </div>
                </motion.li>
              ))}
            </motion.ul>
          </div>
          
          <div className={styles.rightColumn}>
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.7 }}
            >
              <form onSubmit={handleSubmit} className={styles.form}>
                <h2 className={styles.formTitle}>Send a Message</h2>
                
                <div className={styles.formGroup}>
                  <label htmlFor="name" className={styles.formLabel}>Full Name</label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    className={styles.formInput}
                    placeholder="Enter your full name"
                    required
                  />
                </div>
                
                <div className={styles.formGroup}>
                  <label htmlFor="email" className={styles.formLabel}>Email Address</label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    className={styles.formInput}
                    placeholder="Enter your email address"
                    required
                  />
                </div>
                
                <div className={styles.formGroup}>
                  <label htmlFor="subject" className={styles.formLabel}>Subject</label>
                  <input
                    type="text"
                    id="subject"
                    name="subject"
                    value={formData.subject}
                    onChange={handleChange}
                    className={styles.formInput}
                    placeholder="What is this regarding?"
                    required
                  />
                </div>
                
                <div className={styles.formGroup}>
                  <label htmlFor="message" className={styles.formLabel}>Message</label>
                  <textarea
                    id="message"
                    name="message"
                    value={formData.message}
                    onChange={handleChange}
                    className={styles.formTextarea}
                    placeholder="How can we assist you?"
                    required
                  ></textarea>
                </div>
                
                <button type="submit" className={styles.submitButton}>
                  Send Message
                </button>
              </form>
              
              <div className={styles.socialLinks}>
                {socialLinks.map((link, index) => (
                  <motion.a
                    key={index}
                    href={link.url}
                    className={styles.socialLink}
                    whileHover={{ y: -5 }}
                    whileTap={{ scale: 0.95 }}
                    aria-label={link.name}
                  >
                    {link.icon}
                  </motion.a>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ContactUs;