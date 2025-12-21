#!/usr/bin/env node
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

// Navigate up from backend to find frontend
const backendDir = __dirname;
const repoRoot = path.dirname(backendDir);
const frontendDir = path.join(repoRoot, 'frontend');

console.log(`Backend directory: ${backendDir}`);
console.log(`Repository root: ${repoRoot}`);
console.log(`Frontend directory: ${frontendDir}`);

if (!fs.existsSync(frontendDir)) {
  console.error(`Error: Frontend directory not found at: ${frontendDir}`);
  process.exit(1);
}

if (!fs.existsSync(path.join(frontendDir, 'package.json'))) {
  console.error(`Error: package.json not found in frontend directory`);
  process.exit(1);
}

console.log('Building project in frontend directory...');
process.chdir(frontendDir);
execSync('npm run build', { stdio: 'inherit' });
console.log('✓ Build completed successfully!');

