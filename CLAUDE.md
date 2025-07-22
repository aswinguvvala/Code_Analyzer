# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Development mode
streamlit run app.py

# Production mode with Docker
docker build -t ai-code-analyzer .
docker run -p 8501:8501 ai-code-analyzer

# Using docker-compose
docker-compose up
```

### Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Core dependencies are defined in requirements.txt with optimization comments
```

### Environment Setup
```bash
# Gemini API keys should be configured in secrets.toml:
# gemini_api_key_1 = "your_first_key"
# gemini_api_key_2 = "your_second_key"  
# gemini_api_key_3 = "your_third_key"
# gemini_api_key_4 = "your_fourth_key"

# Optional Ollama host configuration  
export OLLAMA_HOST="http://localhost:11434"

# Recommended Ollama models for development
ollama pull llama3.2:3b           # Balanced performance
ollama pull qwen2.5-coder:7b      # Best for code analysis
ollama pull deepseek-coder:6.7b   # Excellent code understanding
```

## Architecture Overview

This is a **hybrid AI-powered code analysis platform** built with Streamlit that combines multiple AI models for comprehensive repository analysis.

### Core Components

- **app.py** - Main Streamlit application with tabbed interface (Dashboard, Architecture, Files, Visuals, Mentor)
- **model_service.py** - AI model abstraction layer with multi-key Gemini rotation and Ollama fallback
- **code_quality_analyzer.py** - Code metrics, complexity analysis, and quality scoring
- **visual_code_analyzer.py** - Mermaid diagram generation and architecture visualization  
- **code_mentor.py** - Interactive AI mentor with repository context awareness

### Hybrid AI Architecture

The system implements intelligent fallback with multi-key rotation:
1. **Primary**: Multiple Gemini API keys with automatic rotation when rate limits hit
2. **Secondary**: Local Ollama models (privacy-focused fallback when all Gemini keys exhausted)

### Data Flow

1. Repository URL input → Git cloning to temp directory
2. File discovery and filtering (excludes common non-code files)
3. Parallel content extraction with smart caching (hashlib-based)
4. Batch AI analysis requests with progress tracking and automatic key rotation
5. Multi-dimensional analysis across 5 specialized tabs (Dashboard, Architecture, Files, Visuals, Mentor)
6. Interactive mentor system with repository-specific context

### Key Design Patterns

- **Async/await** for non-blocking operations
- **Concurrent.futures** for parallel file processing
- **Smart caching** with content hashing to avoid redundant analysis
- **Progressive loading** with real-time Streamlit updates
- **Context-aware responses** in mentor system using repository knowledge base
- **Multi-key rotation** with automatic fallback ensuring analysis always completes
- **Rate limit handling** with intelligent key cycling

### File Processing Strategy

- Prioritizes key files (main entry points, configs, core modules)
- Intelligent file filtering based on extensions and patterns
- Batch processing for AI requests to optimize API usage
- Content truncation for large files while preserving structure

## Working with This Codebase

### Adding New Analysis Features

1. Create new analyzer class following the pattern in existing analyzers
2. Add import and integration in app.py tabs
3. Implement async methods for non-blocking execution
4. Use the model_service abstraction for AI calls
5. Add progress indicators for user feedback

### Modifying AI Models

- All AI interactions go through model_service.py abstraction
- Add new providers by implementing ModelService base class
- Configure fallback order in the service selection logic
- Test with different models using the sidebar model selector

### Working with Streamlit State

- Repository analysis results stored in st.session_state
- Cache analysis between tab switches for performance  
- Use st.rerun() carefully to avoid infinite loops
- Progress indicators use st.progress() and st.status()

### Performance Considerations

- File processing is parallelized using ThreadPoolExecutor
- Content is cached using SHA-256 hashes of file content
- Large repositories are handled with smart file prioritization
- AI requests are batched to reduce API calls

### Error Handling

- Graceful fallback between AI providers
- Repository access errors are caught and displayed clearly
- File processing errors don't stop entire analysis
- Network timeouts handled with retry logic

### Security Notes

- Temporary directories are cleaned up after analysis
- No sensitive data is logged or cached permanently
- Environment variables used for API keys
- Git operations are read-only (no modifications to analyzed repos)