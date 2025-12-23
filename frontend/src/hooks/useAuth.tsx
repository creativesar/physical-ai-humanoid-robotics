import { useState, useEffect, createContext, useContext, ReactNode } from 'react';
import {
  signIn as authSignIn,
  signOut as authSignOut,
  signUp as authSignUp,
  getSession,
  getCurrentUser,
  type User
} from '../lib/auth';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  signUp: (data: SignUpData) => Promise<any>;
  signIn: (data: SignInData) => Promise<any>;
  signOut: () => void;
  refreshSession: () => void;
}

interface SignUpData {
  email: string;
  password: string;
  name: string;
  softwareBackground?: string;
  hardwareBackground?: string;
}

interface SignInData {
  email: string;
  password: string;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const loadSession = () => {
    if (typeof window !== 'undefined') {
      const currentUser = getCurrentUser();
      setUser(currentUser);
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadSession();
  }, []);

  const refreshSession = () => {
    loadSession();
  };

  const signUp = async (data: SignUpData) => {
    try {
      const result = await authSignUp({
        email: data.email,
        password: data.password,
        name: data.name,
        softwareBackground: data.softwareBackground,
        hardwareBackground: data.hardwareBackground,
      });

      if (result.success) {
        setUser(result.session?.user || null);
      }

      return result;
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'Sign up failed' };
    }
  };

  const signIn = async (data: SignInData) => {
    try {
      const result = await authSignIn({
        email: data.email,
        password: data.password
      });

      if (result.success) {
        setUser(result.session?.user || null);
      }

      return result;
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'Sign in failed' };
    }
  };

  const signOut = () => {
    try {
      authSignOut();
      setUser(null);
      // Redirect to home
      if (typeof window !== 'undefined') {
        window.location.href = '/';
      }
    } catch (error) {
      console.error('Sign out error:', error);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        signUp,
        signIn,
        signOut,
        refreshSession,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
