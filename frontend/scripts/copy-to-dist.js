const fs = require('fs');
const path = require('path');

const sourceDir = path.join(__dirname, '..', 'build');
const destDir = path.join(__dirname, '..', 'dist');

// Remove existing dist folder if it exists
if (fs.existsSync(destDir)) {
  fs.rmSync(destDir, { recursive: true, force: true });
  console.log('✓ Removed existing dist folder');
}

// Copy build to dist
if (fs.existsSync(sourceDir)) {
  fs.mkdirSync(destDir, { recursive: true });
  
  const copyRecursive = (src, dest) => {
    const entries = fs.readdirSync(src, { withFileTypes: true });
    
    for (const entry of entries) {
      const srcPath = path.join(src, entry.name);
      const destPath = path.join(dest, entry.name);
      
      if (entry.isDirectory()) {
        fs.mkdirSync(destPath, { recursive: true });
        copyRecursive(srcPath, destPath);
      } else {
        fs.copyFileSync(srcPath, destPath);
      }
    }
  };
  
  copyRecursive(sourceDir, destDir);
  console.log('✓ Build copied to dist folder successfully!');
} else {
  console.error('✗ Build folder not found. Please run "npm run build" first.');
  process.exit(1);
}

