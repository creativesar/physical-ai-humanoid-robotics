import React, { useState, useEffect, useRef } from 'react';
import Link from '@docusaurus/Link';
import { useAuth } from '../hooks/useAuth';

const AuthButton = () => {
  // Try to use auth, but provide fallback if AuthProvider not available
  let authData;
  try {
    authData = useAuth();
  } catch (error) {
    // AuthProvider not available yet, show sign in button
    return (
      <Link
        to="/signin"
        style={{
          textDecoration: 'none',
        }}
      >
        <button
          style={{
            padding: '0.6rem 1.2rem',
            borderRadius: '8px',
            background: 'linear-gradient(135deg, #00D0FF, #0088FF)',
            border: 'none',
            color: '#FFF',
            fontSize: '0.9rem',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-2px)';
            e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 208, 255, 0.4)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = 'none';
          }}
        >
          Sign In
        </button>
      </Link>
    );
  }

  const { user, isAuthenticated, isLoading, signOut: authSignOut } = authData;
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Debug log to check if component is rendering
  // console.log('AuthButton rendering', { session, isPending });

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSignOut = async () => {
    try {
      await authSignOut();
      setIsDropdownOpen(false);
      if (typeof window !== 'undefined') {
        window.location.href = '/';
      }
    } catch (error) {
      console.error('Sign out error:', error);
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <div style={{
        padding: '0.6rem 1.2rem',
        borderRadius: '8px',
        background: 'rgba(176, 224, 230, 0.1)',
        border: '1px solid rgba(176, 224, 230, 0.2)',
        color: '#b0e0e6',
        fontSize: '0.9rem',
        fontWeight: 500,
        opacity: 0.7
      }}>
        Loading...
      </div>
    );
  }

  // Not authenticated - show Sign In button
  if (!isAuthenticated) {
    return (
      <Link
        to="/signin"
        style={{
          textDecoration: 'none',
        }}
      >
        <button
          style={{
            padding: '0.6rem 1.2rem',
            borderRadius: '8px',
            background: 'linear-gradient(135deg, #b0e0e6, #87ceeb)',
            border: 'none',
            color: '#000',
            fontSize: '0.9rem',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-2px)';
            e.currentTarget.style.boxShadow = '0 4px 12px rgba(176, 224, 230, 0.4)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = 'none';
          }}
        >
          Sign In
        </button>
      </Link>
    );
  }

  // Authenticated - show user menu
  const authenticatedUser = user;

  return (
    <div ref={dropdownRef} style={{ position: 'relative' }}>
      <button
        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.6rem',
          padding: '0.6rem 1rem',
          borderRadius: '8px',
          background: 'rgba(176, 224, 230, 0.1)',
          border: '1px solid rgba(176, 224, 230, 0.2)',
          color: '#b0e0e6',
          fontSize: '0.9rem',
          fontWeight: 500,
          cursor: 'pointer',
          transition: 'all 0.2s ease',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'rgba(176, 224, 230, 0.15)';
          e.currentTarget.style.borderColor = 'rgba(176, 224, 230, 0.3)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'rgba(176, 224, 230, 0.1)';
          e.currentTarget.style.borderColor = 'rgba(176, 224, 230, 0.2)';
        }}
      >
        <div style={{
          width: '28px',
          height: '28px',
          borderRadius: '50%',
          background: 'linear-gradient(135deg, #b0e0e6, #87ceeb)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#000',
          fontWeight: 700,
          fontSize: '0.85rem',
        }}>
          {authenticatedUser.name?.charAt(0).toUpperCase() || authenticatedUser.email?.charAt(0).toUpperCase() || 'U'}
        </div>
        <span style={{ maxWidth: '120px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {authenticatedUser.name || authenticatedUser.email}
        </span>
        <span style={{ fontSize: '0.7rem', opacity: 0.7 }}>▼</span>
      </button>

      {/* Dropdown Menu */}
      {isDropdownOpen && (
        <div style={{
          position: 'absolute',
          top: 'calc(100% + 8px)',
          right: 0,
          minWidth: '220px',
          padding: '0.5rem',
          borderRadius: '12px',
          background: 'rgba(20, 20, 20, 0.98)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          border: '1px solid rgba(176, 224, 230, 0.2)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
          zIndex: 1000,
        }}>
          {/* User Info */}
          <div style={{
            padding: '0.8rem',
            borderBottom: '1px solid rgba(176, 224, 230, 0.1)',
            marginBottom: '0.5rem',
          }}>
            <div style={{
              color: '#b0e0e6',
              fontWeight: 600,
              fontSize: '0.95rem',
              marginBottom: '0.3rem',
            }}>
              {authenticatedUser.name || 'User'}
            </div>
            <div style={{
              color: 'rgba(224, 240, 240, 0.6)',
              fontSize: '0.8rem',
            }}>
              {authenticatedUser.email}
            </div>
          </div>

          {/* Menu Items */}
          <Link
            to="/profile"
            onClick={() => setIsDropdownOpen(false)}
            style={{
              display: 'block',
              padding: '0.8rem',
              borderRadius: '8px',
              color: '#e0f0f0',
              textDecoration: 'none',
              fontSize: '0.9rem',
              transition: 'all 0.2s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(176, 224, 230, 0.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'transparent';
            }}
          >
            👤 Profile
          </Link>

          <Link
            to="/settings"
            onClick={() => setIsDropdownOpen(false)}
            style={{
              display: 'block',
              padding: '0.8rem',
              borderRadius: '8px',
              color: '#e0f0f0',
              textDecoration: 'none',
              fontSize: '0.9rem',
              transition: 'all 0.2s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(176, 224, 230, 0.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'transparent';
            }}
          >
            ⚙️ Settings
          </Link>

          <div style={{
            borderTop: '1px solid rgba(176, 224, 230, 0.1)',
            marginTop: '0.5rem',
            paddingTop: '0.5rem',
          }}>
            <button
              onClick={handleSignOut}
              style={{
                width: '100%',
                padding: '0.8rem',
                borderRadius: '8px',
                background: 'transparent',
                border: 'none',
                color: '#fca5a5',
                fontSize: '0.9rem',
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(220, 38, 38, 0.1)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
              }}
            >
              🚪 Sign Out
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AuthButton;
