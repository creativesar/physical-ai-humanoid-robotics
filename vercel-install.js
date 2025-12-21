#!/usr/bin/env node
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

// Find the repository root by looking for frontend directory
function findRepoRoot() {
  let currentDir = __dirname;
  const maxDepth = 10;
  let depth = 0;
  
  while (depth < maxDepth) {
    const frontendPath = path.join(currentDir, 'frontend');
    if (fs.existsSync(frontendPath) && fs.existsSync(path.join(frontendPath, 'package.json'))) {
      return currentDir;
    }
    const parentDir = path.dirname(currentDir);
    if (parentDir === currentDir) break; // Reached filesystem root
    currentDir = parentDir;
    depth++;
  }
  
  // Fallback: try current working directory
  if (fs.existsSync(path.join(process.cwd(), 'frontend', 'package.json'))) {
    return process.cwd();
  }
  
  throw new Error('Could not find repository root with frontend directory');
}

try {
  const repoRoot = findRepoRoot();
  const frontendDir = path.join(repoRoot, 'frontend');
  
  console.log(`Repository root: ${repoRoot}`);
  console.log(`Frontend directory: ${frontendDir}`);
  
  if (!fs.existsSync(frontendDir)) {
    throw new Error(`Frontend directory not found at: ${frontendDir}`);
  }
  
  console.log('Installing dependencies in frontend directory...');
  process.chdir(frontendDir);
  execSync('npm ci', { stdio: 'inherit' });
  console.log('✓ Dependencies installed successfully!');
} catch (error) {
  console.error('Error:', error.message);
  process.exit(1);
}
