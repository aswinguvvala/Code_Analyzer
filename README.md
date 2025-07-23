# CodeScope

A GitHub repository analysis tool that provides comprehensive insights into code structure, quality metrics, and interactive guidance for understanding codebases.

## Overview

CodeScope analyzes GitHub repositories and presents detailed information through four main analysis sections:

- **Overview**: Repository statistics, file distribution, and quality metrics
- **Files**: Individual file analysis with AI-generated explanations  
- **Charts**: Visual representations of code structure and metrics
- **Chat**: Interactive mentor for asking questions about the codebase

## Features

### Code Analysis
- Repository structure analysis
- Code quality metrics and complexity scoring
- File type distribution and statistics
- Technology stack detection

### AI Integration
- Hybrid AI system supporting Google Gemini and local Ollama models
- Intelligent file analysis with context-aware explanations
- Interactive chat interface for codebase questions
- Automatic fallback between cloud and local AI services

### Visualization
- Code quality charts and metrics
- File structure diagrams
- Technology distribution graphs
- Performance and complexity visualizations

## Installation

### Requirements
- Python 3.8 or higher
- Git for repository cloning
- Ollama (for local AI processing)

### Quick Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/codescope.git
cd codescope
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install and configure Ollama:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Download recommended models
ollama pull llama3.2:3b
ollama pull qwen2.5-coder:7b
ollama pull deepseek-coder:6.7b
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open your browser to `http://localhost:8501`

## Usage

### Basic Analysis
1. Enter a GitHub repository URL in the input field
2. Click the "Analyze" button
3. Wait for analysis to complete
4. Explore results in the four tabs

### Supported Repository Formats
- Full GitHub URLs: `https://github.com/owner/repo`
- Short format: `owner/repo`
- Any public GitHub repository

### AI Configuration

#### Google Gemini API (Optional)
For faster analysis, you can configure Gemini API keys:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

#### Ollama Models
The application supports various Ollama models:
- `llama3.2:3b` - Balanced performance and quality
- `qwen2.5-coder:7b` - Optimized for code analysis
- `deepseek-coder:6.7b` - Excellent code understanding

## Technology Stack

### Core Framework
- **Streamlit** - Web application framework
- **Python** - Primary programming language
- **Git** - Repository cloning and management

### AI Services
- **Google Gemini** - Cloud-based AI analysis
- **Ollama** - Local AI model execution
- **Multiple Model Support** - Fallback and redundancy

### Data Processing
- **Pandas** - Data manipulation and analysis
- **Plotly** - Interactive charts and visualizations
- **AST** - Python code parsing and analysis

### Development Tools
- **Docker** - Containerization support
- **Asyncio** - Asynchronous processing
- **Concurrent.futures** - Parallel file processing

## Configuration

### Environment Variables
```bash
# Optional Google Gemini API key
GEMINI_API_KEY=your_api_key_here

# Optional Ollama host configuration  
OLLAMA_HOST=http://localhost:11434
```

### Model Management
```bash
# List available Ollama models
ollama list

# Download new models
ollama pull model_name

# Remove unused models
ollama rm model_name
```

## Project Structure

```
codescope/
├── app.py                          # Main Streamlit application
├── code_mentor.py                  # Interactive chat functionality
├── code_quality_analyzer.py        # Code metrics and quality analysis
├── performance_predictor.py        # Performance analysis module
├── visual_code_analyzer.py         # Chart generation and visualization
├── code_evolution_analyzer.py      # Code change analysis
├── model_service.py               # AI model management
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container configuration
├── docker-compose.yml             # Multi-container setup
└── deployment/                    # Deployment configurations
```

## Deployment

### Docker
```bash
docker build -t codescope .
docker run -p 8501:8501 codescope
```

### Docker Compose
```bash
docker-compose up
```

### Cloud Platforms
The application includes configuration files for:
- Railway (`railway.toml`)
- Vercel (`deployment/vercel.json`)
- AWS (`deployment/aws-sagemaker.yml`)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/new-feature`)
6. Create a Pull Request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Check existing documentation
- Review the code structure for implementation details