# Frontend-Only Authentication System

Complete authentication system that works **entirely in the browser** - NO backend required!

## 🎯 Features

✅ **Sign Up / Sign In** with email & password
✅ **User Profiles** with custom fields (software/hardware background)
✅ **Session Management** (7-day expiry)
✅ **LocalStorage** based persistence
✅ **Personalization** integration
✅ **Password validation**
✅ **User menu dropdown**
✅ **Mock Google authentication** (demo)

## 📁 Files Structure

```
frontend/src/
├── lib/
│   └── auth.ts                    # Core auth logic (localStorage)
├── hooks/
│   └── useAuth.tsx                # React auth hook & context
├── components/
│   └── AuthButton.tsx             # Navbar auth button
├── pages/
│   ├── signin.tsx                 # Sign in page
│   └── signup.tsx                 # Sign up page
└── theme/
    └── Root.tsx                   # AuthProvider wrapper
```

## 🚀 How It Works

### 1. **User Storage**
All users are stored in browser's `localStorage`:
```javascript
localStorage.getItem('physical-ai-users')
```

### 2. **Session Management**
Active session stored in:
```javascript
localStorage.getItem('physical-ai-session')
```

Session includes:
- User data
- Auth token
- Expiry date (7 days)

### 3. **Password Hashing**
Simple hash function (for demo - NOT production secure):
```javascript
simpleHash(password) // Creates hash from password
```

## 💻 Usage

### In Any Component:

```tsx
import { useAuth } from '@/hooks/useAuth';

function MyComponent() {
  const {
    user,              // Current user object
    isAuthenticated,   // Boolean: is user logged in?
    signIn,            // Function to sign in
    signOut,           // Function to sign out
    signUp,            // Function to sign up
    isLoading          // Boolean: auth state loading?
  } = useAuth();

  if (isLoading) return <div>Loading...</div>;

  if (!isAuthenticated) {
    return <button onClick={() => signIn({email, password})}>
      Sign In
    </button>;
  }

  return (
    <div>
      <p>Hello {user.name}!</p>
      <button onClick={signOut}>Sign Out</button>
    </div>
  );
}
```

### Sign Up:

```tsx
const result = await signUp({
  email: 'user@example.com',
  password: 'password123',
  name: 'John Doe',
  softwareBackground: 'Python, ROS',
  hardwareBackground: 'Arduino, Raspberry Pi'
});

if (result.success) {
  // User created and logged in
  console.log(result.session);
} else {
  // Error occurred
  console.error(result.error);
}
```

### Sign In:

```tsx
const result = await signIn({
  email: 'user@example.com',
  password: 'password123'
});

if (result.success) {
  // User logged in
} else {
  // Invalid credentials
}
```

### Sign Out:

```tsx
signOut(); // Clears session
```

### Check Auth Status:

```tsx
const { user, isAuthenticated } = useAuth();

if (isAuthenticated) {
  console.log('User:', user.email);
}
```

## 🔐 Security Notes

**⚠️ IMPORTANT: This is for development/demo purposes!**

### What This System Does NOT Have:

❌ Secure password hashing (uses simple hash)
❌ Encrypted storage
❌ Protection against XSS attacks
❌ Token refresh mechanism
❌ Rate limiting
❌ HTTPS enforcement
❌ CSRF protection
❌ Backend validation

### For Production:

You MUST:
1. Use proper backend authentication
2. Use bcrypt/argon2 for password hashing
3. Store tokens securely (httpOnly cookies)
4. Implement proper session management
5. Add rate limiting
6. Use HTTPS only
7. Validate on server-side
8. Add 2FA/MFA
9. Monitor for suspicious activity
10. Regular security audits

## 🎨 UI Components

### AuthButton

Shows in navbar:
- **Not logged in**: "Sign In" button → redirects to `/signin`
- **Logged in**: User avatar + dropdown menu
  - Shows user name & email
  - Profile link
  - Settings link
  - Sign out button

### Sign In Page (`/signin`)

- Email & password fields
- Error messages
- Loading states
- "Continue with Google" (demo)
- Link to sign up

### Sign Up Page (`/signup`)

- Name, email, password fields
- Confirm password
- Background information (optional):
  - Software experience
  - Hardware experience
- Creates user + auto-login
- Integrates with personalization

## 📊 Data Structure

### User Object:

```typescript
interface User {
  id: string;
  email: string;
  name: string;
  softwareBackground?: string;
  hardwareBackground?: string;
  createdAt: string;
}
```

### Session Object:

```typescript
interface AuthSession {
  user: User;
  token: string;
  expiresAt: string;
}
```

## 🧪 Testing

### Create Test User:

1. Go to http://localhost:3000/signup
2. Fill form:
   - Name: Test User
   - Email: test@example.com
   - Password: Test123456
3. Click "Create Account"
4. Should auto-login and redirect home

### Check Storage:

Open browser DevTools → Application → Local Storage:

```javascript
// View all users
JSON.parse(localStorage.getItem('physical-ai-users'))

// View current session
JSON.parse(localStorage.getItem('physical-ai-session'))
```

### Clear All Data:

```javascript
localStorage.removeItem('physical-ai-users');
localStorage.removeItem('physical-ai-session');
localStorage.removeItem('physical-ai-personalization');
```

## 🔧 Configuration

### Session Duration:

Edit `frontend/src/lib/auth.ts`:

```typescript
const SESSION_DURATION = 7 * 24 * 60 * 60 * 1000; // 7 days
// Change to: 1 * 24 * 60 * 60 * 1000 for 1 day
```

### Password Requirements:

```typescript
if (data.password.length < 8) {
  return { success: false, error: 'Password must be at least 8 characters' };
}
```

## 🐛 Troubleshooting

### "User already exists" error

User with that email is already in localStorage.

**Solution**: Clear storage or use different email.

### Session expires immediately

Check browser's date/time settings.

**Solution**: Ensure system time is correct.

### Can't sign in after signup

Password might not match.

**Solution**: Try password reset or recreate user.

### Data lost after closing browser

localStorage cleared by browser settings.

**Solution**: Check browser privacy settings.

## ✨ Integration with Personalization

When user signs up, their background is automatically:
1. Saved in auth system
2. Passed to personalization context
3. Used for difficulty auto-detection:
   - Both backgrounds filled → Advanced
   - One background filled → Intermediate
   - None → Beginner

## 📱 Mobile Support

Fully responsive:
- Touch-friendly buttons
- Mobile dropdown menus
- Optimized forms

## 🎉 That's It!

Your frontend-only authentication is ready to use!

**No backend needed. Everything works in the browser.** 🚀

Start the app:
```bash
cd frontend
npm run start
```

Visit:
- http://localhost:3000/signup
- http://localhost:3000/signin
