import React, { useState, useEffect } from 'react';

const AuthButton = () => {
  // Check if we're in the browser environment before using hooks that require it
  const [isClient, setIsClient] = useState(false);
  const [session, setSession] = useState(null);
  const [isPending, setIsPending] = useState(true);

  useEffect(() => {
    setIsClient(true);
    // Dynamically import the auth client to avoid SSR issues
    import('../client').then(({ useAuth }) => {
      // Since useAuth is a React hook, we need to create a custom solution
      // For now, we'll just return a placeholder for static generation
      setIsPending(false);
    }).catch(() => {
      // If the auth client fails to load, just proceed without auth
      setIsPending(false);
    });
  }, []);

  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  // Close dropdown when clicking outside
  useEffect(() => {
    if (!isClient) return;

    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (isDropdownOpen && !target.closest?.('.auth-dropdown')) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isDropdownOpen, isClient]);

  const handleSignIn = () => {
    if (typeof window !== 'undefined') {
      import('../client').then(({ signIn }) => {
        signIn();
      });
    }
  };

  const handleSignOut = async () => {
    if (typeof window !== 'undefined') {
      import('../client').then(async ({ signOut }) => {
        await signOut();
        setSession(null);
        setIsDropdownOpen(false);
      });
    }
  };

  if (!isClient) {
    // Return a placeholder during static generation
    return (
      <div style={{
        padding: '0.6rem 1.2rem',
        borderRadius: '8px',
        background: 'rgba(176, 224, 230, 0.1)',
        border: '1px solid rgba(176, 224, 230, 0.2)',
        color: '#b0e0e6',
        fontSize: '0.9rem',
        fontWeight: 500,
        cursor: 'not-allowed',
        opacity: 0.7
      }}>
        Loading...
      </div>
    );
  }

  if (isPending) {
    return (
      <div style={{
        padding: '0.6rem 1.2rem',
        borderRadius: '8px',
        background: 'rgba(176, 224, 230, 0.1)',
        border: '1px solid rgba(176, 224, 230, 0.2)',
        color: '#b0e0e6',
        fontSize: '0.9rem',
        fontWeight: 500,
        cursor: 'not-allowed',
        opacity: 0.7
      }}>
        Loading...
      </div>
    );
  }

  // For now, just return a placeholder button since the auth context isn't available during build
  return (
    <div style={{ display: 'flex', gap: '8px' }}>
      <button
        onClick={handleSignIn}
        style={{
          padding: '0.6rem 1.2rem',
          borderRadius: '8px',
          background: 'linear-gradient(135deg, #888888, #e0e0e0)',
          border: 'none',
          color: '#000',
          fontSize: '0.9rem',
          fontWeight: 600,
          cursor: 'pointer',
          transition: 'all 0.2s ease',
        }}
        onMouseEnter={(e) => {
          const target = e.currentTarget;
          target.style.transform = 'translateY(-2px)';
          target.style.boxShadow = '0 4px 12px rgba(136, 136, 136, 0.3)';
        }}
        onMouseLeave={(e) => {
          const target = e.currentTarget;
          target.style.transform = 'translateY(0)';
          target.style.boxShadow = 'none';
        }}
      >
        Sign In
      </button>
    </div>
  );
};

export default AuthButton;