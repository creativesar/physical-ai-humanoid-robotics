const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const frontendDir = path.join(__dirname, 'frontend');

// Check if frontend directory exists
if (!fs.existsSync(frontendDir)) {
  console.error('Error: frontend directory not found!');
  process.exit(1);
}

console.log('Building project in frontend directory...');
process.chdir(frontendDir);
execSync('npm run build', { stdio: 'inherit' });
console.log('Build completed successfully!');
