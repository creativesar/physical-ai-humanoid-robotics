import { createAuthClient } from "better-auth/react";

export const { signIn, signOut, useAuth } = createAuthClient({
  // Configure your backend URL - adjust this to match your actual backend
  baseURL: process.env.NODE_ENV === 'production'
    ? 'https://your-backend-url.com' // Replace with your actual production backend URL
    : 'http://localhost:8000',       // Replace with your actual development backend URL
});