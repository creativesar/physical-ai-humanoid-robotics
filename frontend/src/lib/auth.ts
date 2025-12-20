/**
 * Frontend-Only Authentication Service
 *
 * This is a complete frontend authentication system without any backend.
 * Uses localStorage for persistence and simple token-based auth.
 *
 * NOTE: This is for development/demo purposes. In production, you should:
 * - Use proper backend authentication
 * - Never store passwords in localStorage
 * - Use secure token generation
 */

export interface User {
  id: string;
  email: string;
  name: string;
  softwareBackground?: string;
  hardwareBackground?: string;
  createdAt: string;
}

export interface AuthSession {
  user: User;
  token: string;
  expiresAt: string;
}

const USERS_KEY = 'physical-ai-users';
const SESSION_KEY = 'physical-ai-session';
const SESSION_DURATION = 7 * 24 * 60 * 60 * 1000; // 7 days

// Simple hash function (NOT secure - for demo only!)
function simpleHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(36);
}

// Generate random token
function generateToken(): string {
  return Math.random().toString(36).substring(2) + Date.now().toString(36);
}

// Get all users from localStorage
function getUsers(): Record<string, { email: string; passwordHash: string; user: User }> {
  const users = localStorage.getItem(USERS_KEY);
  return users ? JSON.parse(users) : {};
}

// Save users to localStorage
function saveUsers(users: Record<string, any>): void {
  localStorage.setItem(USERS_KEY, JSON.stringify(users));
}

// Get current session
export function getSession(): AuthSession | null {
  const session = localStorage.getItem(SESSION_KEY);
  if (!session) return null;

  const parsed: AuthSession = JSON.parse(session);

  // Check if session expired
  if (new Date(parsed.expiresAt) < new Date()) {
    localStorage.removeItem(SESSION_KEY);
    return null;
  }

  return parsed;
}

// Save session
function saveSession(user: User): AuthSession {
  const session: AuthSession = {
    user,
    token: generateToken(),
    expiresAt: new Date(Date.now() + SESSION_DURATION).toISOString(),
  };

  localStorage.setItem(SESSION_KEY, JSON.stringify(session));
  return session;
}

// Sign up new user
export async function signUp(data: {
  email: string;
  password: string;
  name: string;
  softwareBackground?: string;
  hardwareBackground?: string;
}): Promise<{ success: boolean; error?: string; session?: AuthSession }> {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 500));

  const users = getUsers();

  // Check if user already exists
  if (users[data.email]) {
    return { success: false, error: 'User with this email already exists' };
  }

  // Validate email
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(data.email)) {
    return { success: false, error: 'Invalid email address' };
  }

  // Validate password
  if (data.password.length < 8) {
    return { success: false, error: 'Password must be at least 8 characters' };
  }

  // Create user
  const user: User = {
    id: generateToken(),
    email: data.email,
    name: data.name,
    softwareBackground: data.softwareBackground,
    hardwareBackground: data.hardwareBackground,
    createdAt: new Date().toISOString(),
  };

  // Save user
  users[data.email] = {
    email: data.email,
    passwordHash: simpleHash(data.password),
    user,
  };
  saveUsers(users);

  // Create session
  const session = saveSession(user);

  return { success: true, session };
}

// Sign in existing user
export async function signIn(data: {
  email: string;
  password: string;
}): Promise<{ success: boolean; error?: string; session?: AuthSession }> {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 500));

  const users = getUsers();
  const userRecord = users[data.email];

  // Check if user exists
  if (!userRecord) {
    return { success: false, error: 'Invalid email or password' };
  }

  // Check password
  const passwordHash = simpleHash(data.password);
  if (userRecord.passwordHash !== passwordHash) {
    return { success: false, error: 'Invalid email or password' };
  }

  // Create session
  const session = saveSession(userRecord.user);

  return { success: true, session };
}

// Sign out
export function signOut(): void {
  localStorage.removeItem(SESSION_KEY);
}

// Update user profile
export async function updateProfile(updates: Partial<User>): Promise<{ success: boolean; error?: string }> {
  const session = getSession();
  if (!session) {
    return { success: false, error: 'Not authenticated' };
  }

  const users = getUsers();
  const userRecord = users[session.user.email];

  if (!userRecord) {
    return { success: false, error: 'User not found' };
  }

  // Update user data
  const updatedUser = { ...userRecord.user, ...updates };
  userRecord.user = updatedUser;
  users[session.user.email] = userRecord;
  saveUsers(users);

  // Update session
  session.user = updatedUser;
  localStorage.setItem(SESSION_KEY, JSON.stringify(session));

  return { success: true };
}

// Check if user is authenticated
export function isAuthenticated(): boolean {
  return getSession() !== null;
}

// Get current user
export function getCurrentUser(): User | null {
  const session = getSession();
  return session?.user || null;
}

// Mock Google Sign In (for demo)
export async function signInWithGoogle(): Promise<{ success: boolean; error?: string; session?: AuthSession }> {
  // Simulate OAuth flow
  await new Promise(resolve => setTimeout(resolve, 1000));

  const mockGoogleUser = {
    email: 'user@gmail.com',
    name: 'Google User',
    password: 'google-oauth-' + Math.random(),
  };

  // Check if user exists
  const users = getUsers();
  if (users[mockGoogleUser.email]) {
    // Sign in existing user
    return signIn({ email: mockGoogleUser.email, password: users[mockGoogleUser.email].passwordHash });
  }

  // Create new user
  return signUp(mockGoogleUser);
}

// Password reset (frontend-only mock)
export async function requestPasswordReset(email: string): Promise<{ success: boolean; error?: string }> {
  await new Promise(resolve => setTimeout(resolve, 500));

  const users = getUsers();
  if (!users[email]) {
    return { success: false, error: 'User not found' };
  }

  // In real app, this would send email
  console.log(`Password reset link would be sent to: ${email}`);
  alert(`Password reset link sent to ${email} (check console for demo)`);

  return { success: true };
}
