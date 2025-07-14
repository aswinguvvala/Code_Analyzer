# 🚀 Free LLM Repository Analyzer

A powerful tool to analyze any GitHub repository using **free local AI models** - no API costs! Get detailed insights about code structure, file explanations, and project analysis using Ollama.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-lightgrey.svg)

## ✨ Features

- 🔍 **Deep Repository Analysis** - Analyze file structure, technologies, and patterns
- 📄 **File-by-File Explanations** - AI-powered explanations of what each file does
- 🤖 **Smart Insights** - Overall project assessment and recommendations
- �� **Completely Free** - Uses local Ollama models, no API costs
- 🔒 **Private** - All analysis happens locally, code never leaves your machine
- ⚡ **Fast** - Optimized for quick analysis of repositories

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama** (for AI analysis)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/code_analyzer.git
cd code_analyzer
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

# Pull a model (choose one)
ollama pull llama3.2:3b      # Faster, good quality
ollama pull llama3.2:7b      # Slower, better quality
ollama pull codellama:7b     # Code-focused model
```

### Usage

1. **Start the application:**
```bash
streamlit run app.py
```

2. **Open your browser** to `http://localhost:8501`

3. **Analyze a repository:**
   - Enter a GitHub URL like `https://github.com/karpathy/micrograd`
   - Or use owner/repo format like `karpathy/micrograd`
   - Click "🧠 Analyze Repository"
   - Explore the results in different tabs

## �� Example Analysis

### For `karpathy/micrograd`:

**📄 micrograd/engine.py**
> "This file implements the core automatic differentiation engine with the Value class that tracks gradients for backpropagation in neural networks."

**📄 micrograd/nn.py** 
> "This file provides neural network building blocks including Neuron, Layer, and MLP classes for creating trainable networks using the micrograd engine."

**🤖 Overall Insights:**
> "This is a minimal deep learning library focused on educational purposes. The codebase demonstrates clean architecture with separation between the autodiff engine and neural network components..."

## 🛠️ Supported Languages

- **Python** (.py)
- **JavaScript** (.js, .jsx)
- **TypeScript** (.ts, .tsx)
- **Java** (.java)
- **C/C++** (.c, .cpp)
- **Go** (.go)
- **Rust** (.rs)
- **PHP** (.php)
- **Ruby** (.rb)
- **Configuration** files (.json, .yaml, .toml)
- **Documentation** (.md, .txt)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** for providing free local LLM infrastructure
- **Streamlit** for the amazing web framework
- **The open source community** for inspiration and tools

---

Made with ❤️ for the developer community. Analyze any repository, understand any codebase, completely free!
