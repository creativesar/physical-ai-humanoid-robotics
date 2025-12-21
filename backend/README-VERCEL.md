# Deploying to Vercel

This document explains how to deploy your Physical AI & Humanoid Robotics Textbook API to Vercel.

## Prerequisites

1. Install the Vercel CLI:
```bash
npm install -g vercel
```

Or create an account at [vercel.com](https://vercel.com) and connect your GitHub repository.

## Deployment Steps

### Option 1: Using Vercel CLI

1. Navigate to the backend directory:
```bash
cd backend
```

2. Login to Vercel:
```bash
vercel login
```

3. Deploy the project:
```bash
vercel
```

4. When prompted, set the following:
   - Scope: Your Vercel account
   - Project name: Choose a name for your project
   - Framework: Press Enter to detect automatically (or select "Other")
   - Root Directory: Press Enter to use current directory
   - Build Command: `pip install -r requirements.txt`
   - Output Directory: Leave blank for Python projects

### Option 2: Using Vercel Dashboard

1. Push your code to a GitHub repository
2. Go to [vercel.com](https://vercel.com) and create a new project
3. Import your repository
4. In the configuration:
   - Framework: Select "Other" or let it auto-detect
   - Root Directory: Set to the backend directory
   - Build Command: `pip install -r requirements.txt`
   - Install Command: Leave blank or `pip install -r requirements.txt`
   - Output Directory: Leave blank

## Environment Variables

After deployment, you'll need to set the following environment variables in your Vercel project settings:

- `MISTRAL_API_KEY`: Your Mistral AI API key
- `QDRANT_URL`: Your Qdrant database URL
- `QDRANT_API_KEY`: Your Qdrant API key
- `DATABASE_URL`: Your PostgreSQL database URL

## Important Notes

1. **External Dependencies**: This deployment assumes that external services (PostgreSQL, Qdrant) are available. If these services are not accessible from Vercel, you may need to:
   - Use Vercel's database solutions
   - Configure Vercel's private networks
   - Modify the application to work without these services temporarily

2. **Build Time**: Python deployments with many dependencies can take several minutes to build.

3. **Cold Starts**: Serverless functions may have cold start delays. Consider using Vercel's Pro or Enterprise plans for dedicated instances if needed.

## Troubleshooting

- If deployment fails due to build timeout, try reducing dependencies in requirements.txt
- If external services are not accessible, the application will start but some features may not work
- Check Vercel logs in your dashboard for detailed error information
- If you get timeout errors during build, try using a lighter requirements.txt file
- For connection errors, ensure your external services allow connections from Vercel's IP ranges

## Testing Your Deployment

Once deployed, test these endpoints:
- `GET /` - Should return a welcome message
- `GET /health` - Should return health status of services
- `GET /docs` - Should return the API documentation

## Common Issues and Solutions

### 1. Service Dependencies Not Available
If external services (Mistral, Qdrant, PostgreSQL) are not accessible from Vercel:
- Update your services to allow connections from external sources
- Use Vercel's Private Networks feature if available on your plan
- Consider using Vercel's database solutions

### 2. Cold Start Issues
First requests to serverless functions can be slow. Consider:
- Using Vercel's Pro plan for dedicated instances
- Implementing client-side caching
- Adding a warm-up endpoint that gets called periodically

### 3. Build Failures
If the build fails due to large dependencies:
- Consider using a lighter requirements file for Vercel deployment
- Use Vercel's build cache to speed up subsequent builds
- Check Vercel's build timeout limits (typically 10 minutes)

## Alternative: Git Integration

The easiest way is to:
1. Push this code to a GitHub repository
2. Go to vercel.com/dashboard/new
3. Import your GitHub repository
4. Vercel will automatically detect the configuration and deploy your app