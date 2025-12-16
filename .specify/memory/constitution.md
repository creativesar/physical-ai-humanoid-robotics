# Physical AI & Humanoid Robotics Textbook Constitution

## Core Principles

### I. Embodied Intelligence Focus
The primary mission is to bridge the gap between digital AI and physical robotics. All content and features must emphasize the application of AI knowledge to control humanoid robots in simulated and real-world environments, fostering understanding of AI systems functioning in the physical world.

### II. AI-Agent Partnership Approach
Prioritize the development of learning materials that reflect the future partnership between people, intelligent agents (AI software), and robots. Content should prepare students for a world where human-AI-robot collaboration is the norm.

### III. Practical-First Learning (NON-NEGOTIABLE)
All theoretical concepts must be paired with practical implementations. Every chapter should include hands-on exercises using ROS 2, Gazebo, NVIDIA Isaac, or other relevant platforms. Students learn best by doing, not just reading.

### IV. Technology Stack Integration
Focus on integrating the core robotics technologies: ROS 2 (middleware), Gazebo/Unity (simulation), NVIDIA Isaac (AI-powered robotics), and Vision-Language-Action systems. All components must work cohesively.

### V. Accessibility and Inclusivity
Ensure the textbook is accessible to diverse audiences. Implement multi-language support (English and Urdu), adaptive content based on user background, and clear explanations for different skill levels. Democratize access to advanced robotics education.

### VI. Cohere-Centric AI Integration
Utilize Cohere's AI capabilities instead of OpenAI, OpenAPI, or Gemini for all AI-driven features including the RAG chatbot, content personalization, and translation services. This creates a unified AI experience throughout the platform.

## Technical Requirements

### Platform Standards
- **Framework**: Docusaurus for documentation and textbook presentation
- **Deployment**: GitHub Pages for public access
- **Authentication**: Better-Auth for secure user management
- **Database**: Neon Serverless Postgres for user data
- **Vector Database**: Qdrant Cloud Free Tier for semantic search
- **AI Provider**: Cohere for all AI-powered features

### Core Components
- Interactive textbook with 4 modules (ROS 2, Gazebo, NVIDIA Isaac, VLA)
- Integrated RAG chatbot for answering textbook content questions
- User background assessment and content personalization
- Multi-language support with Urdu translation
- Claude Code Subagents for enhanced learning experiences

### Performance Standards
- Page load time under 3 seconds
- AI response time under 5 seconds
- Support for concurrent users during peak usage
- Mobile-responsive design for learning anywhere

## Development Workflow

### Spec-Driven Development
- All features must begin with clear specifications in the spec.md files
- Follow the Spec-Kit Plus methodology with structured plan.md and tasks.md
- Maintain Prompt History Records (PHRs) for all major decisions and implementations
- Document significant architectural decisions in ADRs

### Quality Assurance
- Test all integrations thoroughly, especially AI services and database connections
- Verify all 4 course modules have complete, working content
- Ensure the RAG chatbot accurately responds based on textbook content
- Validate user authentication and personalization features
- Test Urdu translation functionality across all chapters

### Implementation Phases
1. Core textbook structure and content creation
2. Cohere AI integration for chatbot functionality
3. User authentication and background assessment
4. Personalization engine development
5. Multi-language support implementation
6. Bonus features (Subagents, advanced personalization)

## Governance

This constitution guides all development decisions for the Physical AI & Humanoid Robotics textbook project. All implementations must align with these principles. Changes to this constitution require explicit approval and proper documentation of rationale.

**Version**: 1.0.0 | **Ratified**: 2025-12-17 | **Last Amended**: 2025-12-17
