import { useState, useEffect, createContext, useContext, ReactNode } from 'react';
import {
  getSession,
  signUp as authSignUp,
  signIn as authSignIn,
  signOut as authSignOut,
  signInWithGoogle as authSignInWithGoogle,
  updateProfile as authUpdateProfile,
  getCurrentUser,
  type User,
  type AuthSession
} from '../lib/auth';

interface AuthContextType {
  user: User | null;
  session: AuthSession | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  signUp: (data: SignUpData) => Promise<{ success: boolean; error?: string }>;
  signIn: (data: SignInData) => Promise<{ success: boolean; error?: string }>;
  signOut: () => void;
  signInWithGoogle: () => Promise<{ success: boolean; error?: string }>;
  updateProfile: (updates: Partial<User>) => Promise<{ success: boolean; error?: string }>;
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
  const [session, setSession] = useState<AuthSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load session on mount
  useEffect(() => {
    refreshSession();
  }, []);

  const refreshSession = () => {
    const currentSession = getSession();
    setSession(currentSession);
    setUser(currentSession?.user || null);
    setIsLoading(false);
  };

  const signUp = async (data: SignUpData) => {
    const result = await authSignUp(data);
    if (result.success && result.session) {
      setSession(result.session);
      setUser(result.session.user);
    }
    return result;
  };

  const signIn = async (data: SignInData) => {
    const result = await authSignIn(data);
    if (result.success && result.session) {
      setSession(result.session);
      setUser(result.session.user);
    }
    return result;
  };

  const signOut = () => {
    authSignOut();
    setSession(null);
    setUser(null);
  };

  const signInWithGoogle = async () => {
    const result = await authSignInWithGoogle();
    if (result.success && result.session) {
      setSession(result.session);
      setUser(result.session.user);
    }
    return result;
  };

  const updateProfile = async (updates: Partial<User>) => {
    const result = await authUpdateProfile(updates);
    if (result.success) {
      const updatedUser = getCurrentUser();
      setUser(updatedUser);
    }
    return result;
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        session,
        isLoading,
        isAuthenticated: !!user,
        signUp,
        signIn,
        signOut,
        signInWithGoogle,
        updateProfile,
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
