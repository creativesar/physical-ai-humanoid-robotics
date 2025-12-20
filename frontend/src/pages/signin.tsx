import React, { useState } from 'react';
import Layout from '@theme/Layout';
import { signIn } from '../client';
import { motion } from 'framer-motion';

export default function SignIn() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      await signIn.email({
        email,
        password,
      });

      // Redirect on success
      if (typeof window !== 'undefined') {
        window.location.href = '/';
      }
    } catch (err: any) {
      setError(err?.message || 'Invalid email or password');
      console.error('Sign in error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Layout title="Sign In" description="Welcome back to the Robotics Academy">
      <div style={{
        minHeight: 'calc(100vh - 60px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '2rem',
        background: 'radial-gradient(circle at top left, rgba(0, 242, 254, 0.05), transparent), radial-gradient(circle at bottom right, rgba(79, 172, 254, 0.05), transparent), #050505',
        position: 'relative',
        overflow: 'hidden',
        color: '#fff'
      }}>
        {/* Background Decorative Elements */}
        <div style={{
          position: 'absolute',
          top: '10%',
          right: '10%',
          width: '300px',
          height: '300px',
          background: 'rgba(0, 242, 254, 0.03)',
          borderRadius: '50%',
          filter: 'blur(80px)',
          zIndex: 0
        }} />

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          style={{
            width: '100%',
            maxWidth: '450px',
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
              Welcome Back
            </h1>
            <p style={{ color: 'rgba(255, 255, 255, 0.5)', fontSize: '0.95rem', margin: 0 }}>
              Continue your journey into the world of AI
            </p>
          </div>

          {error && (
            <div style={{
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
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <div style={{ marginBottom: '1.5rem' }}>
              <label style={labelStyle}>Email Address</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                style={inputStyle}
                placeholder="name@example.com"
              />
            </div>

            <div style={{ marginBottom: '2.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <label style={{ ...labelStyle, marginBottom: 0 }}>Password</label>
                <a href="#" style={{ fontSize: '0.8rem', color: '#00f2fe', textDecoration: 'none' }}>Forgot?</a>
              </div>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                style={inputStyle}
                placeholder="••••••••"
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              style={primaryButtonStyle}
            >
              {isLoading ? 'Authenticating...' : 'Sign In to Dashboard'}
            </button>

            <div style={{ marginTop: '2rem', textAlign: 'center' }}>
              <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.8rem', marginBottom: '1rem' }}>Or sign in with</p>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <button type="button" style={socialButtonStyle}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" /><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" /><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05" /><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" /></svg>
                </button>
                <button type="button" style={socialButtonStyle}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" /></svg>
                </button>
              </div>
            </div>
          </form>

          <div style={{
            marginTop: '2.5rem',
            textAlign: 'center',
            borderTop: '1px solid rgba(255, 255, 255, 0.05)',
            paddingTop: '2rem'
          }}>
            <span style={{ color: 'rgba(255, 255, 255, 0.4)', fontSize: '0.9rem' }}>
              New here? <a href="/signup" style={{ color: '#00f2fe', textDecoration: 'none', fontWeight: 600 }}>Create Account</a>
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
  transition: 'border-color 0.2s ease',
  boxSizing: 'border-box'
};

const primaryButtonStyle: React.CSSProperties = {
  width: '100%',
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
