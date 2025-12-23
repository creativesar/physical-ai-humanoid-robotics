import { useState, useEffect, createContext, useContext, ReactNode } from 'react';
import {
  signIn as authSignIn,
  signOut as authSignOut,
  signUp as authSignUp,
  useSession,
  type User
} from '../client';

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
  const { data: session, mutate } = useSession();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Set loading to false once session data is available
    setIsLoading(!session);
  }, [session]);

  const refreshSession = () => {
    mutate(); // This will refresh the session
  };

  const signUp = async (data: SignUpData) => {
    try {
      const result = await authSignUp({
        email: data.email,
        password: data.password,
        name: data.name
      });
      return { success: true, result };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'Sign up failed' };
    }
  };

  const signIn = async (data: SignInData) => {
    try {
      const result = await authSignIn('credentials', {
        email: data.email,
        password: data.password
      });
      return { success: true, result };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'Sign in failed' };
    }
  };

  const signOut = async () => {
    try {
      await authSignOut();
    } catch (error) {
      console.error('Sign out error:', error);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user: session?.user || null,
        isLoading,
        isAuthenticated: !!session?.user,
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
