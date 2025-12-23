import React, { useState } from 'react';
import { motion } from 'framer-motion';
import styles from './PremiumGetInTouch.module.css';

const PremiumGetInTouch = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    // Form submission logic would go here
    console.log('Form submitted:', formData);
    // Simulate API call
    setTimeout(() => {
      alert('Thank you for your message! We will get back to you soon.');
      setFormData({ name: '', email: '', subject: '', message: '' });
      setIsSubmitting(false);
    }, 1000);
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
    { name: 'YouTube', icon: '🎥', url: '#' }
  ];

  return (
    <section className={styles.section}>
      {/* Animated gradient background */}
      <div className={styles.animatedGradientBg}></div>

      {/* Particle Effects */}
      <div className={styles.particleEffects}>
        <div className={styles.floatingParticle}></div>
        <div className={styles.floatingParticle}></div>
        <div className={styles.floatingParticle}></div>
        <div className={styles.gradientOrb}></div>
        <div className={styles.gradientOrb}></div>
        <div className={styles.blueNeonOrb}></div>
        <div className={styles.blueNeonParticle}></div>
        <div className={styles.blueNeonParticle}></div>
        <div className={styles.spark}></div>
        <div className={styles.spark}></div>
        <div className={styles.electricSpark}></div>
        <div className={styles.electricSpark}></div>
        <div className={styles.electricArc}></div>
        <div className={styles.electricArc}></div>
      </div>

      <div className={styles.container}>
        <div className={styles.contentWrapper}>
          <div className={styles.leftColumn}>
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className={styles.header}
            >
              <h1 className={styles.title}>Join the Future</h1>
              <p className={styles.subtitle}>
                Ready to explore the future of Physical AI and Humanoid Robotics?
              </p>
              <p className={styles.description}>
                Whether you're a researcher, student, or industry professional interested in our work, we'd love to hear from you. Reach out for collaboration opportunities, research inquiries, or questions about our educational resources.
              </p>
            </motion.div>

            <motion.ul
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
              className={styles.contactInfo}
            >
              {contactInfo.map((info, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: index * 0.15, ease: "easeOut" }}
                  whileHover={{ scale: 1.02 }}
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
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, ease: "easeOut" }}
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

                <button
                  type="submit"
                  className={styles.submitButton}
                  disabled={isSubmitting}
                >
                  {isSubmitting ? 'Sending...' : 'Send Message'}
                </button>
              </form>

              <div className={styles.socialLinks}>
                {socialLinks.map((link, index) => (
                  <motion.a
                    key={index}
                    href={link.url}
                    className={styles.socialLink}
                    whileHover={{ scale: 1.1, y: -8 }}
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

export default PremiumGetInTouch;