const path = require('path');

module.exports = function betterAuthPlugin(context, options) {
  return {
    name: 'better-auth-plugin',

    configureWebpack(config, isServer) {
      if (!isServer) return {};

      return {
        devServer: {
          setupMiddlewares: (middlewares, devServer) => {
            if (!devServer) {
              throw new Error('webpack-dev-server is not defined');
            }

            // Import and setup better-auth
            devServer.app.all('/api/auth/*', async (req, res) => {
              try {
                // Dynamically import better-auth
                const { betterAuth } = await import('better-auth');

                // Create auth instance
                const auth = betterAuth({
                  emailAndPassword: {
                    enabled: true,
                    requireEmailVerification: false,
                  },
                  session: {
                    expiresIn: 60 * 60 * 24 * 7, // 7 days
                  },
                  secret: process.env.BETTER_AUTH_SECRET || 'your-dev-secret-min-32-characters-long',
                  baseURL: 'http://localhost:3000',
                });

                // Handle the request
                const response = await auth.handler(req);

                // Send response
                res.status(response.status);
                response.headers.forEach((value, key) => {
                  res.setHeader(key, value);
                });
                const body = await response.text();
                res.send(body);
              } catch (error) {
                console.error('Better-auth error:', error);
                res.status(500).json({
                  error: 'Authentication error',
                  message: error.message
                });
              }
            });

            return middlewares;
          },
        },
      };
    },
  };
};
