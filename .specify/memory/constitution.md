# Project Constitution: Physical AI & Humanoid Robotics Textbook

## Mission
Create a comprehensive, interactive textbook for teaching Physical AI & Humanoid Robotics, combining Docusaurus-based content delivery with an integrated RAG chatbot, user authentication, and personalized learning experiences.

## Core Values
- **Educational Excellence**: Deliver high-quality, accessible content that bridges digital AI and physical robotics
- **Technical Innovation**: Leverage cutting-edge technologies (OpenRouter API, Docusaurus, better-auth, etc.)
- **User-Centric Design**: Prioritize intuitive navigation and personalized learning paths
- **Open Standards**: Use open-source tools and maintain accessibility standards

## Technology Stack
- **Frontend**: Docusaurus with TypeScript and TailwindCSS
- **Authentication**: Better-Auth (frontend-only, no separate backend needed)
- **AI Integration**: OpenRouter API instead of OpenAI/Gemini/Mistral/Cohere for RAG functionality
- **Database**: Neon Serverless Postgres
- **Vector Storage**: Qdrant Cloud Free Tier
- **Backend**: FastAPI for RAG services only (no auth backend needed)
- **Deployment**: GitHub Pages with Docker support

## Quality Standards
- All code must be properly typed with TypeScript
- Components must be styled with TailwindCSS for consistency
- All user-facing features must be responsive
- Security best practices must be followed for authentication
- Performance optimizations must be implemented for large content

## Architectural Principles
- Maintain separation of concerns between frontend and backend services
- Use component-based architecture for Docusaurus
- Implement proper error handling and user feedback
- Follow DRY principles while maintaining code readability
- Ensure scalability for future content additions

## Development Practices
- Use Spec-Kit Plus methodology for structured development
- Maintain comprehensive documentation
- Implement proper testing strategies
- Follow accessibility standards (WCAG)
- Ensure cross-browser compatibility

## Success Metrics
- Functional RAG chatbot that answers textbook content questions
- Successful user authentication and personalization features
- Responsive, accessible textbook interface
- Proper deployment to GitHub Pages
- Docker containerization for development consistency