#!/bin/bash

# Hugging Face Spaces Deployment Script
# This script automates the deployment process

echo "🚀 Hugging Face Spaces Deployment"
echo "=================================="
echo ""

# Check if we're in the backend directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this from the backend directory."
    exit 1
fi

# Set variables
SPACE_REPO="https://huggingface.co/spaces/creativesar/hackathonnew"
TEMP_DIR="../hackathonnew"

echo "📦 Cloning Hugging Face Space..."
cd ..
git clone $SPACE_REPO 2>/dev/null || cd hackathonnew

cd hackathonnew

echo "📋 Copying backend files..."
cp ../backend/main.py .
cp ../backend/Dockerfile .
cp ../backend/README.md .
cp ../backend/requirements.txt .
cp ../backend/.gitignore .
cp -r ../backend/api .

echo "✅ Files copied successfully!"
echo ""
echo "📝 Files to be deployed:"
ls -lh main.py Dockerfile README.md requirements.txt

echo ""
echo "🔧 Next steps:"
echo "1. Review the changes: git status"
echo "2. Add files: git add ."
echo "3. Commit: git commit -m 'Deploy FastAPI backend'"
echo "4. Push: git push"
echo ""
echo "⚙️  Don't forget to set environment variables in Space Settings!"
echo "   - OPENAI_API_KEY"
echo "   - QDRANT_URL"
echo "   - QDRANT_API_KEY"
echo ""
