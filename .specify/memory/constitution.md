# Project Constitution: Physical AI & Humanoid Robotics Interactive Textbook Platform

## Mission
Create an AI-native, interactive textbook platform for teaching Physical AI & Humanoid Robotics, featuring Docusaurus-based content delivery, embedded RAG chatbot with text selection queries, frontend-only authentication, content personalization, and multilingual translation support.

## Core Values
- **Education-First Design**: Clarity, interactivity, and accessibility for learners with diverse backgrounds
- **Technical Excellence**: Type-safe code, modern architecture, performance optimization
- **User Experience**: Intuitive navigation, contextual help, responsive design
- **AI-Powered Learning**: Intelligent content adaptation and multilingual support
- **Open & Transparent**: Open-source tools, clear documentation, privacy-focused

## Technology Stack

### Frontend
- **Framework**: Docusaurus 3.x with TypeScript 5.x
- **Styling**: TailwindCSS 4.x for consistent, responsive design
- **Authentication**: Frontend-only localStorage system (better-auth inspired, no backend)
- **UI Enhancements**: Framer Motion, GSAP, React Three Fiber
- **State Management**: React hooks + Context API

### Backend
- **Framework**: FastAPI (Python 3.9+)
- **AI Provider**: OpenRouter API (unified access to GPT-4, Claude, etc.)
- **Vector Database**: Qdrant Cloud Free Tier
- **Deployment**: Vercel Serverless Functions
- **No Auth Backend**: All authentication handled on frontend

### Key Decision: OpenRouter as Single Provider
Instead of integrating Cohere, Mistral, OpenAI, and Gemini separately, we use OpenRouter API for unified model access. This simplifies integration, reduces code complexity, and provides flexibility to switch models without code changes.

### Key Decision: Frontend-Only Authentication
For MVP/hackathon scope, we implement localStorage-based authentication without backend. This enables rapid development, eliminates server costs, and provides sufficient functionality for demo purposes.

## Quality Standards

### Code Quality
- **Type Safety**: Strict TypeScript throughout frontend
- **Component Modularity**: Small, reusable React components
- **Separation of Concerns**: Clear boundaries between UI, logic, and data
- **Error Handling**: Graceful fallbacks with user-friendly messages
- **Performance**: Optimized for fast loads and smooth interactions

### User Experience
- **Responsive Design**: Mobile-first approach, works across all devices
- **Accessibility**: WCAG 2.1 AA compliance
- **Loading States**: Clear feedback during AI operations
- **Visual Hierarchy**: Consistent spacing and typography with TailwindCSS

### Security & Privacy
- **Input Validation**: Validate all user inputs
- **Session Management**: 7-day sessions with automatic cleanup
- **No Plain Text Passwords**: Use hashing (even for demo)
- **HTTPS Only**: Enforce secure connections in production
- **Minimal Data Collection**: Collect only necessary information

## Architectural Principles

### Frontend Architecture
- **Static Site Generation**: Docusaurus for fast, SEO-friendly pages
- **Component-Based**: Reusable React components with TypeScript
- **State Management**: Context API for global state (user, preferences)
- **Route-Based Code Splitting**: Optimize bundle sizes

### Backend Architecture
- **Microservices**: Separate APIs for chat, content, translation
- **Serverless**: Deploy as Vercel functions for scalability
- **RAG Pipeline**: Chunk content → embed → store in Qdrant → retrieve → generate
- **Error Resilience**: Circuit breakers for external API calls

### Data Management
- **Vector Embeddings**: Store document chunks in Qdrant for semantic search
- **User Preferences**: Store in localStorage (background, personalization)
- **Content Versioning**: Track updates in Docusaurus
- **No User Database**: All user data in localStorage

## Development Practices

### Workflow
- **Spec-Kit Plus**: Use for AI-driven development
- **Claude Code**: Primary development assistant
- **Git Practices**: Feature branches, clear commits, PR reviews
- **Documentation**: Keep README, API docs, and inline comments updated

### Testing Strategy
- **Unit Tests**: Test functions and components
- **Integration Tests**: Test API endpoints and services
- **E2E Tests**: Test critical flows (signup, chat, personalization)
- **Manual QA**: Test UI/UX before releases

### Deployment
- **GitHub Pages**: Static frontend deployment
- **Vercel Functions**: Serverless backend
- **Environment Variables**: Secure API key management
- **CI/CD**: Automated builds with GitHub Actions

## Success Metrics

### Core Deliverables (100 points)
1. ✅ Docusaurus textbook deployed to GitHub Pages
2. ✅ RAG chatbot embedded and functional
3. ✅ Text selection-based queries working
4. ✅ FastAPI backend with OpenRouter integration
5. ✅ Qdrant vector search operational

### Bonus Features (up to 200 points)
1. ⚡ Claude Code Subagents and Skills (50 points) - In Progress
2. ✅ Better-auth signup/signin with background questions (50 points)
3. ✅ Content personalization per chapter (50 points)
4. ✅ Urdu translation per chapter (50 points)

## Technical Constraints
- **No backend database**: Vector DB for content, localStorage for users
- **Free tier services**: Qdrant Cloud free tier, serverless limits
- **Browser compatibility**: Modern browsers only (Chrome, Firefox, Safari, Edge)
- **OpenRouter dependency**: Single API for all LLM operations

## Non-Goals
- ❌ Full backend authentication system (frontend-only for MVP)
- ❌ Multi-tenant SaaS platform (single deployment)
- ❌ Content management system (Docusaurus markdown)
- ❌ Real-time collaboration
- ❌ Mobile native app (responsive web only)

## Code Standards

### TypeScript
```typescript
// Use explicit types
interface User {
  id: string;
  name: string;
  email: string;
  softwareBackground?: string;
  hardwareBackground?: string;
}

// Prefer async/await
async function fetchUser(id: string): Promise<User> {
  // implementation
}

// Functional components
export default function Component({ prop }: { prop: string }) {
  return <div>{prop}</div>;
}
```

### Python
```python
# Use type hints
from typing import List, Optional

async def process_query(query: str, user_id: Optional[str] = None) -> dict:
    """Process RAG query with optional user context"""
    pass
```

### CSS/TailwindCSS
```tsx
// Prefer Tailwind utilities
<div className="flex items-center justify-between p-4 bg-blue-500">

// Custom CSS only when necessary
<div className="custom-gradient">
```

## Error Handling

### Frontend
- User-friendly error messages
- Retry mechanisms for transient failures
- Loading states during async operations
- Console logging for debugging

### Backend
- Structured error responses with status codes
- Sufficient logging context
- Graceful API rate limit handling
- Circuit breakers for external services

## Documentation Requirements
- JSDoc comments for complex functions
- README files for major modules
- API documentation for endpoints
- Type definitions for interfaces
- User guides and troubleshooting

## Version Control
- Semantic versioning (MAJOR.MINOR.PATCH)
- Changelog in CHANGELOG.md
- Git tags for releases
- Feature branches for development

---

**Last Updated**: December 24, 2025
**Version**: 1.0.0
**Project**: Hackathon I - Physical AI & Humanoid Robotics Textbook