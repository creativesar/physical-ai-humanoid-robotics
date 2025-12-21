#!/bin/bash

# Vercel build script for lighter deployment
echo "Installing lighter dependencies for Vercel deployment..."

# Install lighter requirements
pip install -r requirements-vercel-light.txt

# If the basic installation succeeds, try to install additional lightweight packages
# that don't exceed the size limit
echo "Lightweight installation complete"