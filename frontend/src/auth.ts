import { betterAuth } from "better-auth";

export const auth = betterAuth({
  // Stateless mode - no database needed!
  // Sessions stored in encrypted JWT tokens

  emailAndPassword: {
    enabled: true,
    requireEmailVerification: false, // Set true in production
  },

  // Optional: Add social providers
  // socialProviders: {
  //   google: {
  //     clientId: process.env.GOOGLE_CLIENT_ID as string,
  //     clientSecret: process.env.GOOGLE_CLIENT_SECRET as string,
  //   },
  // },

  // Session configuration
  session: {
    expiresIn: 60 * 60 * 24 * 7, // 7 days
    updateAge: 60 * 60 * 24, // 1 day
  },

  // Secret for JWT signing (in production, use env variable)
  secret: process.env.BETTER_AUTH_SECRET || "your-secret-key-change-in-production-minimum-32-characters-long",

  // Base URL (same as frontend)
  baseURL: process.env.BASE_URL || "http://localhost:3000",

  // Database not required in stateless mode
  // But user data will not persist between server restarts

  // Optional: Add database for persistence
  // database: new Pool({
  //   connectionString: process.env.DATABASE_URL,
  // }),
});

export type Auth = typeof auth;
