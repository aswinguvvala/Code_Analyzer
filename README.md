# AI Code Analyzer

A comprehensive GitHub repository analysis tool powered by **hybrid AI models** (Gemini + Ollama) that provides deep insights into code structure, quality metrics, and interactive mentoring.

## 📋 Documentation

- **[Complete Codebase Documentation](CODEBASE_DOCUMENTATION.md)** - Comprehensive technical documentation
- **[Developer Quick Reference](DEVELOPER_QUICK_REFERENCE.md)** - Quick start guide and API reference  
- **[Technical Analysis](TECHNICAL_ANALYSIS.md)** - Performance analysis and architecture insights
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production deployment instructions

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-lightgrey.svg)

## ✨ Key Features

### 🧠 **Intelligent Code Analysis**
- 🔍 **Deep Repository Scanning** - Comprehensive analysis of file structure, dependencies, and patterns
- 📊 **Code Quality Metrics** - Complexity analysis, duplication detection, and quality scoring
- 🏗️ **Architecture Visualization** - Interactive diagrams and system flow charts
- ⚡ **Performance Prediction** - AI-powered performance bottleneck detection

### 🤖 **Hybrid AI System**
- 🚀 **Gemini API Integration** - Fast analysis with Google's latest models
- 🏠 **Local Ollama Support** - Privacy-focused local AI processing
- 🔄 **Smart Fallback** - Automatic switching between cloud and local models
- 🎯 **Specialized Models** - Code-focused models like CodeLlama, Qwen2.5-Coder, DeepSeek-Coder

### 🧠 **Interactive Code Mentor**
- 💬 **Context-Aware Responses** - Understands your specific codebase
- 📚 **Educational Guidance** - Learn from your code with personalized explanations
- 🔍 **Repository Intelligence** - Deep understanding of file relationships and architecture
- 💡 **Smart Question Classification** - Automatically detects repository-specific vs general questions

### 📈 **Comprehensive Analytics**
- 📄 **File-by-File Explanations** - AI-powered descriptions of each component
- 🎨 **Visual Code Maps** - Mermaid diagrams showing system architecture
- 📊 **Quality Dashboard** - Metrics, trends, and improvement suggestions
- 🔧 **Performance Insights** - Bottleneck identification and optimization tips

## 🛠 Installation & Setup

### Prerequisites
- **Python 3.8+**
- **Ollama** (for local AI processing)
- **Git** (for repository cloning)

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ai-code-analyzer.git
cd ai-code-analyzer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install and setup Ollama:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Download recommended models
ollama pull llama3.2:3b           # Balanced performance
ollama pull qwen2.5-coder:7b      # Best for code analysis
ollama pull codellama:7b          # Code-focused model
ollama pull deepseek-coder:6.7b   # Excellent code understanding
```

4. **Run the application:**
```bash
streamlit run app.py
```

5. **Open your browser** to `http://localhost:8501`

## 🚀 Usage

### Basic Analysis
1. Enter a GitHub repository URL (e.g., `https://github.com/user/repo` or `user/repo`)
2. Choose your preferred Ollama model from the sidebar
3. Optionally add a Gemini API key for faster processing
4. Click "🧠 Analyze Repository"
5. Explore results across different tabs

### Advanced Configuration

#### Gemini API (Optional - for fastest analysis)
```bash
# Get free API key from https://makersuite.google.com/app/apikey
export GEMINI_API_KEY="your_api_key_here"
```

#### Model Selection Guide
| Model | Best For | Speed | Quality | RAM Required |
|-------|----------|-------|---------|--------------|
| `llama3.2:1b` | Quick insights | ⚡⚡⚡ | ⭐⭐ | 2GB |
| `llama3.2:3b` | Balanced analysis | ⚡⚡ | ⭐⭐⭐ | 4GB |
| `qwen2.5-coder:7b` | Code analysis | ⚡ | ⭐⭐⭐⭐⭐ | 8GB |
| `codellama:7b` | Programming focus | ⚡ | ⭐⭐⭐⭐ | 8GB |
| `deepseek-coder:6.7b` | Code understanding | ⚡ | ⭐⭐⭐⭐⭐ | 8GB |

## 📊 Analysis Features

### 🏠 Dashboard Tab
- Repository overview and statistics
- File type distribution
- Technology stack detection
- Quality score and metrics

### 🏗️ Architecture Tab
- System architecture diagrams
- Component relationships
- Data flow visualization
- Dependency analysis

### 📄 Files Tab
- File-by-file AI explanations
- Code structure analysis
- Import/export relationships
- Function and class inventories

### 🎨 Visuals Tab
- Interactive Mermaid diagrams
- System flow charts
- Architecture visualizations
- Component interaction maps

### 🧠 Mentor Tab
- Interactive code mentor chat
- Context-aware responses about your codebase
- Programming guidance and best practices
- Learning progress tracking

### ⚡ Performance Tab
- Performance bottleneck detection
- Optimization recommendations
- Code complexity analysis
- Performance prediction scoring

## 🤖 Interactive Code Mentor

The AI mentor provides intelligent responses based on your specific codebase:

### Context-Aware Features
- **Repository Intelligence**: Understands your project's architecture and purpose
- **File Relationships**: Knows how components interact and depend on each other
- **Domain Detection**: Identifies if your project is ML, web service, CLI tool, etc.
- **Smart Classification**: Automatically detects repository-specific vs general questions

### Example Questions
- "How does this codebase work from start to finish?"
- "What is the main workflow in this project?"
- "Explain the architecture of this application"
- "How do the files interact with each other?"
- "What are the key components and their purposes?"

### Enhanced Fallback Responses
Even without repository context, the mentor provides helpful guidance on:
- Software development workflows
- Architecture patterns and best practices
- Code quality principles
- Debugging strategies
- Learning approaches

## 🔧 Configuration

### Environment Variables
```bash
# Optional Gemini API key for faster analysis
export GEMINI_API_KEY="your_api_key_here"

# Optional Ollama host configuration
export OLLAMA_HOST="http://localhost:11434"
```

### Model Management
```bash
# List available models
ollama list

# Pull new models
ollama pull codegemma:7b

# Remove unused models
ollama rm model_name
```

## 📈 Performance Optimization

### For Fastest Analysis:
- Use Gemini API key
- Choose `llama3.2:1b` for quick insights
- Analyze smaller repositories first
- Close unnecessary applications

### For Best Quality:
- Use `qwen2.5-coder:7b` or `deepseek-coder:6.7b`
- Ensure sufficient RAM (8GB+ recommended)
- Use SSD storage for model files

### Hybrid Mode (Recommended):
- Configure both Gemini API and Ollama
- Gets speed of cloud AI with reliability of local models
- Automatic fallback ensures analysis always works

## 🛡️ Privacy & Security

- **Local Processing**: Ollama models run entirely on your machine
- **Optional Cloud**: Gemini API usage is optional and configurable
- **No Data Storage**: Code analysis happens in real-time, no permanent storage
- **Open Source**: Full transparency of analysis methods

## 📁 Project Structure

```
ai-code-analyzer/
├── app.py                      # Main Streamlit application
├── code_mentor.py              # Interactive AI mentor
├── code_quality_analyzer.py    # Code quality metrics
├── performance_predictor.py    # Performance analysis
├── visual_code_analyzer.py     # Visualization engine
├── model_service.py           # AI model management
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── deployment/               # Deployment configurations
└── cache/                   # Analysis cache
```

## 🚀 Deployment

### Docker
```bash
docker build -t ai-code-analyzer .
docker run -p 8501:8501 ai-code-analyzer
```

### Cloud Platforms
- **Railway**: Use included `railway.toml`
- **Vercel**: Use included `vercel.json`
- **AWS**: Use CloudFormation templates in `deployment/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** - Local AI model management
- **Google Gemini** - Fast cloud AI processing
- **Streamlit** - Beautiful web interface
- **Mermaid** - Diagram visualization
- **Community** - Open source AI models and tools

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/ai-code-analyzer/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-code-analyzer/discussions)
- 📧 **Email**: your.email@example.com

---

⭐ **Star this repository** if you find it helpful!

🔄 **Follow** for updates on new features and improvements!
