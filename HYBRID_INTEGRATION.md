# Hybrid Gemini/Ollama Integration

## 🚀 Overview

Your Code Analyzer now supports a **hybrid approach** that uses **Gemini API first** for fast analysis and **falls back to Ollama** when needed. This gives you the best of both worlds: speed when available, and reliability when not.

## ✨ Key Features

### 🔄 Smart Fallback System
- **Primary**: Gemini API (Fast, requires API key)
- **Fallback**: Local Ollama (Reliable, no API key needed)
- **Automatic**: Seamlessly switches between them

### 📊 Status Tracking
- Real-time status of both systems
- API quota tracking
- Performance recommendations
- Usage metrics

### 🎯 Intelligent Routing
- Tries Gemini first for speed
- Falls back to Ollama if Gemini fails
- Handles API limits gracefully
- Provides clear user feedback

## 🛠 How to Use

### 1. **With Gemini API (Recommended for Speed)**
```bash
# Get a free API key from https://makersuite.google.com/app/apikey
export GEMINI_API_KEY="your_api_key_here"
streamlit run app.py
```

### 2. **With Ollama Only (No API Key Needed)**
```bash
# Make sure Ollama is running
ollama serve
ollama pull llama3.2:3b
streamlit run app.py
```

### 3. **Hybrid Mode (Best Experience)**
- Enter your Gemini API key in the sidebar
- Keep Ollama running as backup
- Enjoy fast analysis with reliable fallback

## 🔧 Configuration

### In the Streamlit App:
1. Open the sidebar
2. Enter your Gemini API key (optional)
3. Check system status
4. See current approach and recommendations

### Environment Variables:
```bash
# Optional: Set API key as environment variable
export GEMINI_API_KEY="your_api_key_here"
```

## 📈 Performance Comparison

| Method | Speed | Cost | Reliability | Setup |
|--------|-------|------|-------------|--------|
| **Gemini API** | ⚡ Very Fast | 💰 Free (with quota) | ✅ High | 🔑 API Key |
| **Ollama** | 🐌 Slower | 💰 Free | ✅ High | 🔧 Local Install |
| **Hybrid** | ⚡ Fast + Reliable | 💰 Free | ✅ Very High | 🔑 + 🔧 Both |

## 🎯 Benefits

### For Users:
- **Faster Analysis**: Gemini API is significantly faster than local models
- **No Interruptions**: Automatic fallback ensures analysis always works
- **Cost Effective**: Use free Gemini quota, fallback to free Ollama
- **Transparent**: Clear status and approach information

### For Developers:
- **Robust**: Handles API failures gracefully
- **Scalable**: Easy to add more LLM providers
- **Maintainable**: Clean separation of concerns
- **Extensible**: Status system for future enhancements

## 🔍 Technical Details

### Architecture:
```python
HybridAnalyzer
├── GeminiAPI (Primary)
│   ├── Fast responses
│   ├── Quota tracking
│   └── Error handling
├── OllamaFallback (Secondary)
│   ├── Local processing
│   ├── Reliable backup
│   └── No API dependency
└── StatusManager
    ├── Real-time monitoring
    ├── Smart routing
    └── User feedback
```

### Key Classes:
- **RepositoryAnalyzer**: Main analysis class with hybrid support
- **_call_gemini()**: Handles Gemini API calls with retry logic
- **_call_ollama()**: Handles local Ollama calls
- **_call_llm_hybrid()**: Smart routing between providers
- **get_status()**: System status and recommendations

## 🚨 Error Handling

### Gemini API Issues:
- **Quota Exceeded**: Automatically switches to Ollama
- **Rate Limits**: Implements retry logic with backoff
- **Network Errors**: Falls back to local processing
- **Invalid API Key**: Clear error messages and fallback

### Ollama Issues:
- **Not Running**: Clear installation instructions
- **Model Missing**: Guidance on model installation
- **Resource Limits**: Performance optimization tips

## 📊 Usage Examples

### Basic Analysis:
```python
# With API key
analyzer = RepositoryAnalyzer(gemini_api_key="your_key")

# Without API key (Ollama only)
analyzer = RepositoryAnalyzer()

# Check status
status = analyzer.get_status()
print(f"Approach: {status['recommended_approach']}")
```

### Custom Analysis:
```python
# Direct hybrid call
messages = [
    {"role": "system", "content": "You are a code analyst."},
    {"role": "user", "content": "Analyze this code..."}
]

response = await analyzer._call_llm_hybrid(messages, "code_analysis")
```

## 🎉 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install google-generativeai
   ```

2. **Get API Key** (optional):
   - Visit https://makersuite.google.com/app/apikey
   - Create a free account
   - Generate API key

3. **Install Ollama** (recommended):
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama serve
   ollama pull llama3.2:3b
   ```

4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

5. **Configure**:
   - Enter API key in sidebar (optional)
   - Check system status
   - Start analyzing repositories!

## 🔮 Future Enhancements

- **More Providers**: Add support for OpenAI, Claude, etc.
- **Load Balancing**: Distribute requests across multiple providers
- **Caching**: Cache responses for faster repeated analysis
- **Analytics**: Detailed usage analytics and optimization
- **Custom Models**: Support for fine-tuned models

---

**Happy Analyzing!** 🚀

Your hybrid analyzer is now ready to provide the fastest and most reliable code analysis experience possible. 