# 🚀 Hugging Face Spaces Deployment Guide

Complete guide to deploy this FastAPI backend to Hugging Face Spaces.

## Prerequisites

1. Hugging Face account: https://huggingface.co/join
2. Git installed on your system
3. Hugging Face CLI (optional): `pip install huggingface-hub`

## Deployment Steps

### Method 1: Git Clone and Push (Recommended)

```bash
# 1. Clone your Hugging Face Space
git clone https://huggingface.co/spaces/creativesar/hackathonnew
cd hackathonnew

# 2. Copy backend files to the Space directory
# Copy these files from the backend folder:
#   - main.py
#   - Dockerfile
#   - README.md (with YAML frontmatter)
#   - requirements.txt
#   - api/ (entire folder)
#   - .gitignore

# 3. Stage and commit files
git add main.py Dockerfile README.md requirements.txt api/ .gitignore
git commit -m "Deploy FastAPI backend to Hugging Face Spaces"

# 4. Push to Hugging Face
git push
```

### Method 2: Direct Upload via Web Interface

1. Go to https://huggingface.co/spaces/creativesar/hackathonnew
2. Click on **Files** tab
3. Click **Upload files** button
4. Upload these files:
   - `main.py`
   - `Dockerfile`
   - `README.md`
   - `requirements.txt`
   - All files from `api/` folder
5. Commit changes

## Environment Variables Setup

After deployment, configure these secrets in Space Settings:

1. Go to: https://huggingface.co/spaces/creativesar/hackathonnew/settings
2. Scroll to **Repository secrets** section
3. Add these variables:

```
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
PORT=7860
```

**Important:** Never commit `.env` file with actual secrets!

## Verify Deployment

Once deployed, your API will be available at:

```
Base URL: https://creativesar-hackathonnew.hf.space
```

### Test Endpoints:

1. **Root**: https://creativesar-hackathonnew.hf.space/
2. **Health Check**: https://creativesar-hackathonnew.hf.space/health
3. **API Docs**: https://creativesar-hackathonnew.hf.space/docs
4. **Chat Query**: POST https://creativesar-hackathonnew.hf.space/api/chat/query

## Dockerfile Configuration

The Dockerfile is already configured for Hugging Face Spaces:

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port 7860 (HF Spaces default)
EXPOSE 7860
ENV PORT=7860

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

## README.md Configuration

The README.md must have YAML frontmatter for Hugging Face:

```yaml
---
title: Physical AI Robotics Backend API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---
```

## Troubleshooting

### Space not starting?
- Check **Logs** tab in your Space
- Verify all dependencies in `requirements.txt`
- Ensure port 7860 is configured

### API endpoints not working?
- Verify environment variables are set
- Check if external services (Qdrant, OpenAI) are accessible
- Review application logs

### Build failing?
- Check Dockerfile syntax
- Verify Python version compatibility
- Ensure all imports are in `requirements.txt`

## Update Deployment

To update your deployment:

```bash
cd hackathonnew

# Make changes to files
# Then commit and push

git add .
git commit -m "Update backend configuration"
git push
```

Hugging Face will automatically rebuild and redeploy.

## Monitoring

- **Build Logs**: Check the Space page for build status
- **Runtime Logs**: Available in the Logs tab
- **Metrics**: View in Space settings

## Free Tier Limitations

- CPU-only (no GPU)
- Space may sleep after inactivity (cold start ~10-30s)
- Limited compute resources

For production, consider upgrading to:
- **Persistent Spaces** (always on)
- **GPU Spaces** (for heavy ML workloads)

## Support

- Hugging Face Docs: https://huggingface.co/docs/hub/spaces
- Community Forum: https://discuss.huggingface.co/

---

**Deployment Status**: Ready ✅
**API Version**: 1.0.0
**Last Updated**: 2025-12-24
