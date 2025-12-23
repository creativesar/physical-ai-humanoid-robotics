import { createAuthClient } from "better-auth/react";

// Better-Auth client - same domain (port 3000)
export const authClient = createAuthClient({
  baseURL: "http://localhost:3000", // Same as frontend
});

// Export auth methods
export const { signIn, signOut, signUp, useSession } = authClient;