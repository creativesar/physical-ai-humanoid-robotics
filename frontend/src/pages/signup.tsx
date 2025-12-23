import React, { useState } from 'react';
import Layout from '@theme/Layout';
import { signUp } from '../client';
import { usePersonalization } from '../contexts/PersonalizationContext';
import { motion, AnimatePresence } from 'framer-motion';

export default function SignUp() {
  const { updateUserProfile, setDifficultyLevel, setLanguage } = usePersonalization();
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    softwareBackground: '',
    hardwareBackground: '',
    difficultyLevel: 'intermediate',
    language: 'english'
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const nextStep = () => {
    if (step === 1) {
      if (!formData.name || !formData.email || !formData.password) {
        setError('Please fill in all account details');
        return;
      }
      if (formData.password !== formData.confirmPassword) {
        setError('Passwords do not match');
        return;
      }
    }
    setError('');
    setStep(step + 1);
  };

  const prevStep = () => {
    setError('');
    setStep(step - 1);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      await signUp.email({
        email: formData.email,
        password: formData.password,
        name: formData.name,
      });

      // Update personalization profile
      updateUserProfile({
        name: formData.name,
        email: formData.email,
        softwareBackground: formData.softwareBackground,
        hardwareBackground: formData.hardwareBackground,
      });

      // @ts-ignore
      setDifficultyLevel(formData.difficultyLevel);
      // @ts-ignore
      setLanguage(formData.language);

      // Redirect to home
      if (typeof window !== 'undefined') {
        window.location.href = '/';
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to create account');
      console.error('Sign up error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const steps = [
    { id: 1, title: 'Identity' },
    { id: 2, title: 'Background' },
    { id: 3, title: 'Settings' }
  ];

  const renderStepIcon = (id: number) => {
    if (step > id) return '✓';
    return id;
  };

  return (
    <Layout title="Join the Future" description="Create your Robotics Masterclass account">
      <div style={{
        minHeight: 'calc(100vh - 60px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '2rem',
        background: 'radial-gradient(circle at top right, rgba(0, 242, 254, 0.05), transparent), radial-gradient(circle at bottom left, rgba(79, 172, 254, 0.05), transparent), #050505',
        position: 'relative',
        overflow: 'hidden',
        color: '#fff'
      }}>
        {/* Background Decorative Elements */}
        <div style={{
          position: 'absolute',
          top: '20%',
          left: '10%',
          width: '300px',
          height: '300px',
          background: 'rgba(0, 242, 254, 0.03)',
          borderRadius: '50%',
          filter: 'blur(80px)',
          zIndex: 0
        }} />
        <div style={{
          position: 'absolute',
          bottom: '10%',
          right: '5%',
          width: '400px',
          height: '400px',
          background: 'rgba(79, 172, 254, 0.03)',
          borderRadius: '50%',
          filter: 'blur(100px)',
          zIndex: 0
        }} />

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          style={{
            width: '100%',
            maxWidth: '520px',
            padding: '3rem',
            borderRadius: '24px',
            background: 'rgba(15, 15, 15, 0.7)',
            backdropFilter: 'blur(24px)',
            WebkitBackdropFilter: 'blur(24px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
            position: 'relative',
            zIndex: 1
          }}
        >
          {/* Header */}
          <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
            <h1 style={{
              fontSize: '2.25rem',
              fontWeight: 800,
              marginBottom: '0.75rem',
              background: 'linear-gradient(to right, #ffffff, #a0a0a0)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '-0.02em',
              margin: 0
            }}>
              Join the Academy
            </h1>
            <p style={{ color: 'rgba(255, 255, 255, 0.5)', fontSize: '0.95rem', margin: 0 }}>
              Your journey into Humanoid Robotics starts here
            </p>
          </div>

          {/* Stepper */}
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            marginBottom: '3.5rem',
            position: 'relative'
          }}>
            <div style={{
              position: 'absolute',
              top: '18px',
              left: '15%',
              right: '15%',
              height: '2px',
              background: 'rgba(255, 255, 255, 0.05)',
              zIndex: 0
            }} />
            <div style={{
              position: 'absolute',
              top: '18px',
              left: '15%',
              width: `${(step - 1) * 35}%`,
              height: '2px',
              background: 'linear-gradient(to right, #00f2fe, #4facfe)',
              boxShadow: '0 0 10px rgba(0, 242, 254, 0.5)',
              zIndex: 0,
              transition: 'width 0.5s ease'
            }} />

            {steps.map((s) => (
              <div key={s.id} style={{ 
                flex: 1, 
                display: 'flex', 
                flexDirection: 'column', 
                alignItems: 'center',
                zIndex: 1
              }}>
                <div style={{
                  width: '36px',
                  height: '36px',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  transition: 'all 0.3s ease',
                  background: step >= s.id 
                    ? 'linear-gradient(135deg, #00f2fe 0%, #4facfe 100%)' 
                    : 'rgba(255, 255, 255, 0.05)',
                  color: step >= s.id ? '#000' : 'rgba(255, 255, 255, 0.3)',
                  border: step === s.id ? '4px solid rgba(0, 242, 254, 0.2)' : 'none',
                  boxShadow: step >= s.id ? '0 0 15px rgba(0, 242, 254, 0.3)' : 'none',
                }}>
                  {renderStepIcon(s.id)}
                </div>
                <span style={{ 
                  marginTop: '0.75rem', 
                  fontSize: '0.75rem', 
                  fontWeight: 500,
                  color: step >= s.id ? 'rgba(255, 255, 255, 0.9)' : 'rgba(255, 255, 255, 0.3)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em'
                }}>
                  {s.title}
                </span>
              </div>
            ))}
          </div>

          {error && (
            <motion.div 
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              style={{
                padding: '1rem',
                marginBottom: '2rem',
                borderRadius: '12px',
                background: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.2)',
                color: '#ef4444',
                fontSize: '0.85rem',
                textAlign: 'center'
              }}>
              {error}
            </motion.div>
          )}

          <form onSubmit={handleSubmit}>
            <AnimatePresence mode="wait">
              {step === 1 && (
                <motion.div
                  key="step1"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div style={{ marginBottom: '1.5rem' }}>
                    <label style={labelStyle}>Full Name</label>
                    <input
                      type="text"
                      name="name"
                      placeholder="Enter your name"
                      value={formData.name}
                      onChange={handleChange}
                      style={inputStyle}
                      required
                    />
                  </div>
                  <div style={{ marginBottom: '1.5rem' }}>
                    <label style={labelStyle}>Email Address</label>
                    <input
                      type="email"
                      name="email"
                      placeholder="name@example.com"
                      value={formData.email}
                      onChange={handleChange}
                      style={inputStyle}
                      required
                    />
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '2.5rem' }}>
                    <div>
                      <label style={labelStyle}>Password</label>
                      <input
                        type="password"
                        name="password"
                        placeholder="••••••••"
                        value={formData.password}
                        onChange={handleChange}
                        style={inputStyle}
                        required
                        minLength={8}
                      />
                    </div>
                    <div>
                      <label style={labelStyle}>Confirm</label>
                      <input
                        type="password"
                        name="confirmPassword"
                        placeholder="••••••••"
                        value={formData.confirmPassword}
                        onChange={handleChange}
                        style={inputStyle}
                        required
                      />
                    </div>
                  </div>

                  <div style={{ marginBottom: '2rem' }}>
                    <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.8rem', textAlign: 'center', marginBottom: '1rem' }}>Or continue with</p>
                    <div style={{ display: 'flex', gap: '1rem' }}>
                      <button type="button" style={socialButtonStyle}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/></svg>
                      </button>
                      <button type="button" style={socialButtonStyle}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>
                      </button>
                    </div>
                  </div>

                  <button type="button" onClick={nextStep} style={primaryButtonStyle}>
                    Continue Building Profile
                  </button>
                </motion.div>
              )}

              {step === 2 && (
                <motion.div
                  key="step2"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div style={{ marginBottom: '2rem' }}>
                    <label style={labelStyle}>Software Experience</label>
                    <textarea
                      name="softwareBackground"
                      placeholder="e.g. Python, ROS, C++ development..."
                      value={formData.softwareBackground}
                      onChange={handleChange}
                      rows={3}
                      style={{ ...inputStyle, resize: 'none' }}
                    />
                    <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem', flexWrap: 'wrap' }}>
                      {['Python', 'ROS2', 'C++', 'PyTorch'].map(tag => (
                        <button 
                          key={tag}
                          type="button"
                          onClick={() => setFormData(prev => ({ ...prev, softwareBackground: prev.softwareBackground + (prev.softwareBackground ? ', ' : '') + tag }))}
                          style={tagStyle}
                        >
                          +{tag}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div style={{ marginBottom: '2.5rem' }}>
                    <label style={labelStyle}>Hardware Experience</label>
                    <textarea
                      name="hardwareBackground"
                      placeholder="e.g. PCB design, Actuators, Sensors..."
                      value={formData.hardwareBackground}
                      onChange={handleChange}
                      rows={3}
                      style={{ ...inputStyle, resize: 'none' }}
                    />
                    <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem', flexWrap: 'wrap' }}>
                      {['Arduino', 'ESP32', 'CAD', 'PCB'].map(tag => (
                        <button 
                          key={tag}
                          type="button"
                          onClick={() => setFormData(prev => ({ ...prev, hardwareBackground: prev.hardwareBackground + (prev.hardwareBackground ? ', ' : '') + tag }))}
                          style={tagStyle}
                        >
                          +{tag}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '1rem' }}>
                    <button type="button" onClick={prevStep} style={secondaryButtonStyle}>Back</button>
                    <button type="button" onClick={nextStep} style={primaryButtonStyle}>Almost Done</button>
                  </div>
                </motion.div>
              )}

              {step === 3 && (
                <motion.div
                  key="step3"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div style={{ marginBottom: '2rem' }}>
                    <label style={labelStyle}>Preferred Language</label>
                    <select 
                      name="language" 
                      value={formData.language} 
                      onChange={handleChange} 
                      style={inputStyle}
                    >
                      <option value="english">English (Standard)</option>
                      <option value="urdu">Urdu (Regional)</option>
                    </select>
                  </div>

                  <div style={{ marginBottom: '2.5rem' }}>
                    <label style={labelStyle}>Starting Level</label>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem' }}>
                      {(['beginner', 'intermediate', 'advanced'] as const).map(level => (
                        <button
                          key={level}
                          type="button"
                          onClick={() => setFormData(prev => ({ ...prev, difficultyLevel: level }))}
                          style={{
                            ...tagStyle,
                            padding: '1rem 0.5rem',
                            fontSize: '0.75rem',
                            background: formData.difficultyLevel === level ? 'rgba(0, 242, 254, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                            border: formData.difficultyLevel === level ? '1px solid rgba(0, 242, 254, 0.4)' : '1px solid rgba(255, 255, 255, 0.05)',
                            color: formData.difficultyLevel === level ? '#00f2fe' : 'rgba(255, 255, 255, 0.5)',
                            flex: 1
                          }}
                        >
                          {level.charAt(0).toUpperCase() + level.slice(1)}
                        </button>
                      ))}
                    </div>
                  </div>

                  <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.8rem', textAlign: 'center', marginBottom: '2rem' }}>
                    By creating an account, you agree to our <a href="/terms" style={{ color: '#4facfe' }}>Terms of Service</a>.
                  </p>

                  <div style={{ display: 'flex', gap: '1rem' }}>
                    <button type="button" onClick={prevStep} style={secondaryButtonStyle}>Back</button>
                    <button type="submit" disabled={isLoading} style={primaryButtonStyle}>
                      {isLoading ? 'Processing...' : 'Complete Registration'}
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </form>

          <div style={{ 
            marginTop: '2.5rem', 
            textAlign: 'center', 
            borderTop: '1px solid rgba(255, 255, 255, 0.05)',
            paddingTop: '2rem'
          }}>
            <span style={{ color: 'rgba(255, 255, 255, 0.4)', fontSize: '0.9rem' }}>
              Member already? <a href="/signin" style={{ color: '#00f2fe', textDecoration: 'none', fontWeight: 600 }}>Sign In</a>
            </span>
          </div>
        </motion.div>
      </div>
    </Layout>
  );
}

// Styles
const labelStyle: React.CSSProperties = {
  display: 'block',
  marginBottom: '0.75rem',
  color: 'rgba(255, 255, 255, 0.6)',
  fontSize: '0.85rem',
  fontWeight: 600,
  textTransform: 'uppercase',
  letterSpacing: '0.05em'
};

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '1rem 1.25rem',
  borderRadius: '12px',
  background: 'rgba(255, 255, 255, 0.03)',
  border: '1px solid rgba(255, 255, 255, 0.08)',
  color: '#fff',
  fontSize: '1rem',
  outline: 'none',
  transition: 'border-color 0.2s ease, background-color 0.2s ease',
  boxSizing: 'border-box'
};

const primaryButtonStyle: React.CSSProperties = {
  flex: 1,
  padding: '1.1rem',
  borderRadius: '12px',
  background: 'linear-gradient(135deg, #00f2fe 0%, #4facfe 100%)',
  border: 'none',
  color: '#000',
  fontSize: '1rem',
  fontWeight: 700,
  cursor: 'pointer',
  transition: 'transform 0.2s ease, opacity 0.2s ease, box-shadow 0.2s ease',
  boxShadow: '0 10px 20px -10px rgba(0, 242, 254, 0.5)'
};

const secondaryButtonStyle: React.CSSProperties = {
  padding: '1.1rem 2rem',
  borderRadius: '12px',
  background: 'rgba(255, 255, 255, 0.05)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  color: '#fff',
  fontSize: '1rem',
  fontWeight: 600,
  cursor: 'pointer',
  transition: 'background 0.2s ease'
};

const socialButtonStyle: React.CSSProperties = {
  flex: 1,
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  padding: '0.9rem',
  borderRadius: '12px',
  background: 'rgba(255, 255, 255, 0.03)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  cursor: 'pointer',
  transition: 'all 0.2s ease'
};

const tagStyle: React.CSSProperties = {
  background: 'rgba(255, 255, 255, 0.05)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  color: 'rgba(255, 255, 255, 0.6)',
  padding: '0.4rem 0.8rem',
  borderRadius: '8px',
  fontSize: '0.8rem',
  cursor: 'pointer',
  transition: 'all 0.2s ease'
};
